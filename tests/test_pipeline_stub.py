"""pipeline.answer / index_corpus against the sample corpus and the generated suite, with eval/stub_reader."""

import json
import time
from pathlib import Path

import pytest

import config
import pipeline
import textnorm
from budget import Deadline
from conftest import ROOT, SAMPLE_CORPUS, SAMPLE_QUESTIONS
from eval import run_suite, score
from eval.stub_reader import StubReader, parse_package
from schema import ReaderOut

QDATA = json.loads(SAMPLE_QUESTIONS.read_text(encoding="utf-8"))
S1 = ROOT / "eval" / "suites" / "s1"
Q4 = "Which firmware version fixed ticket ORR-1847?"
Q6 = next(q["query"] for q in QDATA["queries"] if q["n"] == 6)
COMMENT_ONLY = "2026-08"       # "Raised from 60 after the 2026-08 backlog incident." in ingest_service.py
Q8 = next(q["query"] for q in QDATA["queries"] if q["n"] == 8)


@pytest.fixture(scope="module")
def sample_idx(tmp_path_factory):
    return pipeline.index_corpus(SAMPLE_CORPUS, tmp_path_factory.mktemp("pipe_idx"))


def run_all(idx, reader, out_dir: Path):
    results = {}
    for q in QDATA["queries"]:
        res = pipeline.answer(idx, reader, q["query"], Deadline.after(20), SAMPLE_CORPUS)
        (out_dir / f"query_{q['n']:02d}_output.json").write_text(json.dumps(res.to_output()), encoding="utf-8")
        results[q["n"]] = res
    return results


class CountingReader(StubReader):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_read = 0

    def read(self, *a, **kw):
        self.n_read += 1
        return super().read(*a, **kw)


class ScriptedReader(StubReader):
    """Returns the scripted answers in order, with the support rid of the first package line containing them."""

    def __init__(self, answers, **kw):
        super().__init__("refuse", **kw)
        self.answers = list(answers)
        self.packages: list[str] = []

    def read(self, package_text, question, images, deadline, max_new_tokens=96):
        self.packages.append(package_text)
        ans = self.answers.pop(0) if self.answers else ""
        rids = [rid for rid, _, _, text in parse_package(package_text) if ans and ans in text]
        return self._out(ans, rids[:1], [])


# ---------------------------------------------------------------- sample corpus

def test_oracle_scores_200(sample_idx, tmp_path):
    results = run_all(sample_idx, StubReader.from_questions(QDATA, "oracle"), tmp_path)
    sc = score.score(QDATA, tmp_path)
    assert sc["total"] == sc["max"] == 200, [r for r in sc["rows"] if not r["pts"]]
    assert results[7].confidence == config.CONF_IMAGE and results[8].confidence == config.CONF_IMAGE
    assert results[9].confidence == config.CONF_CHAIN
    assert results[1].trace["reason"] == "OK" and results[1].trace["attempts"] == 1
    assert results[1].trace["support_rid"] is not None and results[1].trace["stages"]


def test_adversarial_answers_are_grounded_and_canary_refused(sample_idx, tmp_path):
    results = run_all(sample_idx, StubReader.from_questions(QDATA, "adversarial"), tmp_path)
    sc = score.score(QDATA, tmp_path)
    false = [r["id"] for r in sc["rows"] if r["got"]["answer"] != "" and not r["pts"]]
    print(f"adversarial: {len(false)} answers differ from gold (verifier cannot see they are decoys): {false}")
    assert results[10].answer == "" and results[10].citations == []        # the canary: price only in the encrypted pdf
    # whatever it answers must still be a verbatim value of a cited, eligible file
    for q in QDATA["queries"]:
        res = results[q["n"]]
        if not res.answer:
            assert res.citations == [] and res.confidence == 0.0
            continue
        assert res.citations
        texts = [sample_idx.record(r).text for c in res.citations for r in sample_idx.by_file[c]]
        assert any(textnorm.find_span(t, res.answer) for t in texts), (q["n"], res.answer, res.citations)
        assert all(sample_idx.file_by_path[c].eligible for c in res.citations)


def test_refuse_reader_refuses_everything(sample_idx, tmp_path):
    results = run_all(sample_idx, StubReader("refuse"), tmp_path)
    assert all(r.answer == "" and r.citations == [] and r.confidence == 0.0 for r in results.values())
    assert score.score(QDATA, tmp_path)["total"] == 20          # only the canary scores


def test_false_premise_refuses_without_calling_reader(sample_idx):
    rd = CountingReader("oracle", {})
    res = pipeline.answer(sample_idx, rd, "Which firmware version fixed ticket ORR-9999?", Deadline.after(20),
                          SAMPLE_CORPUS)
    assert (res.answer, res.citations, res.confidence) == ("", [], 0.0)
    assert res.trace["reason"] == "FALSE_PREMISE"
    assert rd.n_read == 0 and rd.calls == []


def test_expired_deadline_refuses_fast(sample_idx):
    rd = CountingReader("oracle", {Q4: {"expected_answer": "4.3.2", "expected_citations": ["support/bug_database.csv"]}})
    t = time.time()
    res = pipeline.answer(sample_idx, rd, Q4, Deadline(time.time() - 1), SAMPLE_CORPUS)
    assert time.time() - t < 0.5
    assert res.answer == "" and res.trace["reason"] == "TIMEOUT" and rd.n_read == 0


def test_reader_exception_is_a_refusal(sample_idx):
    class Boom(StubReader):
        def read(self, *a, **kw):
            raise RuntimeError("gpu on fire")

    res = pipeline.answer(sample_idx, Boom("refuse"), Q4, Deadline.after(20), SAMPLE_CORPUS)
    assert res.answer == "" and res.trace["reason"] == "ERROR" and "gpu on fire" in res.trace["notes"]["error"]


def test_no_reader_refuses(sample_idx):
    assert pipeline.answer(sample_idx, None, Q4, Deadline.after(20), SAMPLE_CORPUS).trace["reason"] == "NO_READER"


def test_retry_drops_violators_and_rereads(sample_idx):
    # first read quotes a value that only the stale .py comment carries; the comment loses to the literal
    rd = ScriptedReader([COMMENT_ONLY, "180"])
    res = pipeline.answer(sample_idx, rd, Q6, Deadline.after(20), SAMPLE_CORPUS)
    assert res.answer == "180" and res.citations == ["engineering/ingest_service.py"]
    assert res.trace["attempts"] == 2 and res.trace["notes"]["reason_1"] != "OK" and res.trace["notes"]["reason_2"] == "OK"
    assert "[R3] " in rd.packages[0] and "[R3] " not in rd.packages[1]   # the violating comment record is dropped


def test_no_retry_when_time_is_short(sample_idx):
    rd = ScriptedReader([COMMENT_ONLY, "180"])
    res = pipeline.answer(sample_idx, rd, Q6, Deadline.after(config.RETRY_MIN_REMAINING_S - 3), SAMPLE_CORPUS)
    assert res.answer == "" and res.trace["attempts"] == 1 and len(rd.packages) == 1


def test_retry_exhausted_keeps_last_reason(sample_idx):
    rd = ScriptedReader([COMMENT_ONLY, COMMENT_ONLY])
    res = pipeline.answer(sample_idx, rd, Q6, Deadline.after(20), SAMPLE_CORPUS)
    assert res.answer == "" and res.trace["attempts"] == 2
    assert res.trace["reason"] == res.trace["notes"]["reason_2"]


def test_gate_b_refuses_low_confidence_and_survives_scoring_errors(sample_idx, monkeypatch):
    monkeypatch.setattr(config, "GATE_B", True)
    gold = {Q4: {"expected_answer": "4.3.2", "expected_citations": ["support/bug_database.csv"]}}

    class Unsure(StubReader):
        def read(self, *a, **kw):
            out = super().read(*a, **kw)
            return ReaderOut(out.answer, out.support, out.bridge, out.raw, out.n_new_tokens, mean_logprob=-5.0)

    res = pipeline.answer(sample_idx, Unsure("oracle", gold), Q4, Deadline.after(20), SAMPLE_CORPUS)
    assert res.answer == "" and res.trace["reason"] == "GATE_B"       # gain 0 < theta2 and s1 < theta1

    class Broken(StubReader):
        def logprob(self, *a, **kw):
            raise RuntimeError("no logits")

    res = pipeline.answer(sample_idx, Broken("oracle", gold), Q4, Deadline.after(20), SAMPLE_CORPUS)
    assert res.answer == "4.3.2"                                       # error -> gain None -> s1 alone decides


# ---------------------------------------------------------------- VLM transcription + image self-consistency

class VisionStub(StubReader):
    has_vision = True

    def __init__(self, transcripts, vqa_says=None):
        super().__init__("refuse", transcripts=transcripts)
        self.vqa_says = vqa_says

    def read(self, package_text, question, images, deadline, max_new_tokens=96):
        self.calls.append("read")
        hit = [rid for rid, _, loc, text in parse_package(package_text) if "REV-ZZ99" in text and "vlm." in loc]
        return self._out("REV-ZZ99", hit[:1], []) if hit else self._out("", [], [])

    def vqa(self, question, image, deadline, max_new_tokens=48, longest_edge=None):
        self.calls.append("vqa")
        return self.vqa_says


LABEL = "support/asset_label.jpg"
TRANSCRIPT = {LABEL: "FIELD: Board revision\nValue: REV-ZZ99\n", "specs/backplane_pinout.png": "A1: GND"}


def test_transcribe_pending_appends_vlm_records_once(tmp_path):
    rd = VisionStub(TRANSCRIPT)
    idx = pipeline.index_corpus(SAMPLE_CORPUS, tmp_path, rd, Deadline.after(300))
    vlm = [r for r in idx.records if r.src == "vlm"]
    assert [r.loc for r in vlm if r.file == LABEL] == ["vlm.L1", "vlm.L2"]
    assert idx.has_id("REV-ZZ99")
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["vlm_done"] == 2 and status["vlm_pending"] == 0 and "t_vlm" in status
    assert list((tmp_path / "images").glob("*.vlm.txt"))
    n_calls = rd.calls.count("transcribe")
    assert n_calls >= 1
    # second run: loads the persisted index, finds the vlm records, transcribes nothing, adds nothing
    idx2 = pipeline.index_corpus(SAMPLE_CORPUS, tmp_path, rd, Deadline.after(300))
    assert len(idx2.records) == len(idx.records) and rd.calls.count("transcribe") == n_calls


def test_transcribe_pending_respects_deadline_margin(tmp_path):
    rd = VisionStub(TRANSCRIPT)
    idx = pipeline.index_corpus(SAMPLE_CORPUS, tmp_path, rd, Deadline.after(config.VLM_INDEX_MARGIN_S - 1))
    assert not [r for r in idx.records if r.src == "vlm"] and "transcribe" not in rd.calls
    assert json.loads((tmp_path / "status.json").read_text())["vlm_pending"] >= 1


@pytest.fixture(scope="module")
def vlm_idx(tmp_path_factory):
    return pipeline.index_corpus(SAMPLE_CORPUS, tmp_path_factory.mktemp("vlm_idx"), VisionStub(TRANSCRIPT),
                                 Deadline.after(300))


def test_image_disagreement_refuses(vlm_idx):
    rd = VisionStub(TRANSCRIPT, vqa_says="REV-C2")
    res = pipeline.answer(vlm_idx, rd, Q8, Deadline.after(20), SAMPLE_CORPUS)
    assert "vqa" in rd.calls
    assert res.answer == "" and res.trace["reason"] == "IMAGE_DISAGREE" and res.trace["notes"]["image_disagree"] == "REV-C2"


@pytest.mark.parametrize("says", ["", "REV-ZZ99", "The revision is REV-ZZ99."])
def test_image_agreement_or_silence_keeps_answer(vlm_idx, says):
    rd = VisionStub(TRANSCRIPT, vqa_says=says)
    res = pipeline.answer(vlm_idx, rd, Q8, Deadline.after(20), SAMPLE_CORPUS)
    assert "vqa" in rd.calls
    assert res.answer == "REV-ZZ99" and res.citations == [LABEL] and res.confidence == config.CONF_IMAGE


def test_vqa_skipped_without_time(vlm_idx):
    rd = VisionStub(TRANSCRIPT, vqa_says="REV-C2")
    res = pipeline.answer(vlm_idx, rd, Q8, Deadline.after(config.VQA_MIN_REMAINING_S - 0.5), SAMPLE_CORPUS)
    assert "vqa" not in rd.calls and res.answer == "REV-ZZ99"


def test_load_images_pixels(sample_idx):
    from retrieve import analyze, assemble
    pkg = assemble(sample_idx, analyze(Q8, sample_idx), 1)
    imgs = pipeline.load_images(pkg, SAMPLE_CORPUS)
    assert imgs and imgs[0][0] == LABEL and imgs[0][1].mode == "RGB" and len(imgs) <= config.MAX_IMAGES
    assert pipeline.load_images(pkg, None) == []
    assert pipeline.load_image(SAMPLE_CORPUS, "nope.png", ("file", "nope.png")) is None


# ---------------------------------------------------------------- generated suite

@pytest.mark.skipif(not (S1 / "questions.json").exists(), reason="run eval/gen_corpus.py --seed 1 --out eval/suites/s1")
def test_generated_suite_s1_oracle(tmp_path):
    rep = run_suite.run(S1 / "corpus", S1 / "questions.json", tmp_path, "oracle", verbose=True)
    print(f"s1 oracle: total {rep['total']}/{rep['max']} answerable {rep['answerable_correct']}/{rep['answerable']} "
          f"text {rep['text_correct']}/{rep['text_answerable']} canary false answers {rep['false_answers']}")
    assert rep["false_answers"] == 0
    assert rep["text_correct"] >= 0.9 * rep["text_answerable"]
    assert rep["seconds"]["p95"] < 5
