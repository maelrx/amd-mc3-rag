import json

import pytest

import parse
import index as ix
import textnorm
from conftest import SAMPLE_CORPUS, SAMPLE_QUESTIONS
from eval.stub_reader import StubReader, parse_package
from schema import ReaderOut

Q9 = "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?"
Q4 = "Which firmware version fixed ticket ORR-1847?"
GOLD = {
    Q9: {"expected_answer": "4.3.2",
         "expected_citations": ["logs/prod_inference_2026-09-02.log", "support/bug_database.csv"]},
    Q4: {"expected_answer": "4.3.2", "expected_citations": ["support/bug_database.csv"]},
    "What is the unit price of the TQ-40 at 10,000 unit volume?": {"expected_answer": "", "expected_citations": []},
}

PACKAGE = """\
## logs/prod_inference_2026-09-02.log (log)
[R29] logs/prod_inference_2026-09-02.log L6: 2026-09-02T03:14:22Z inference-node-07 meridian[2211]: ERROR E7731 thermal throttle engaged
[R30] logs/prod_inference_2026-09-02.log L7: 2026-09-02T03:19:03Z inference-node-07 meridian[2211]: INFO  incident logged against ORR-1847
## support/bug_database.csv (csv)
[R66] support/bug_database.csv header: columns: ticket | component | severity | summary | status | fixed_in
[R67] support/bug_database.csv row2: ticket=ORR-1847; component=power; severity=S2; summary=Die temperature sensor reports high under sustained small-batch load; status=closed; fixed_in=4.3.2
[R68] support/bug_database.csv row3: ticket=ORR-1802; component=pcie; severity=S3; summary=Doorbell hang under sustained small-batch load; status=closed; fixed_in=4.3.1
## engineering/meridian_release_notes.txt (txt)
[R50] engineering/meridian_release_notes.txt L3: Meridian 4.3.1 release notes: various fixes
"""


def test_parse_package():
    recs = parse_package(PACKAGE)
    assert [r[0] for r in recs] == [29, 30, 66, 67, 68, 50]
    assert recs[3][1:3] == ("support/bug_database.csv", "row2")
    assert recs[3][3].startswith("ticket=ORR-1847")


def test_oracle_q9_support_and_bridge():
    out = StubReader("oracle", GOLD).read(PACKAGE, Q9, [], None)
    assert isinstance(out, ReaderOut)
    assert out.answer == "4.3.2"
    assert 67 in out.support and 68 not in out.support
    assert out.bridge == [30]


def test_oracle_raw_is_real_json_and_parses_like_reader():
    import reader
    out = StubReader("oracle", GOLD).read(PACKAGE, Q9, [], None)
    assert json.loads(out.raw) == {"answer": "4.3.2", "support": ["R67"], "bridge": ["R30"]}
    assert reader.parse_output(out.raw) == ("4.3.2", [67], [30])


def test_oracle_single_file_has_no_bridge():
    out = StubReader("oracle", GOLD).read(PACKAGE, Q4, [], None)
    assert out.answer == "4.3.2" and out.support == [67] and out.bridge == []


def test_oracle_refusal_and_unknown_question():
    r = StubReader("oracle", GOLD)
    out = r.read(PACKAGE, "What is the unit price of the TQ-40 at 10,000 unit volume?", [], None)
    assert (out.answer, out.support, out.bridge) == ("", [], [])
    assert r.read(PACKAGE, "something not in gold", [], None).answer == ""


def test_adversarial_never_gold_and_outside_gold_files():
    out = StubReader("adversarial", GOLD).read(PACKAGE, Q4, [], None)
    assert out.answer and textnorm.norm(out.answer) != textnorm.norm("4.3.2")
    assert all(r in (29, 30, 50) for r in out.support)


def test_adversarial_on_unanswerable_still_answers():
    out = StubReader("adversarial", GOLD).read(PACKAGE, "What is the unit price of the TQ-40 at 10,000 unit volume?",
                                               [], None)
    assert out.answer


def test_lexical_returns_some_token():
    out = StubReader("lexical").read(PACKAGE, Q4, [], None)
    assert out.answer and out.support
    assert out.answer in PACKAGE


def test_lexical_no_overlap_is_empty():
    assert StubReader("lexical").read(PACKAGE, "zzz qqq", [], None).answer == ""


def test_refuse():
    out = StubReader("refuse", GOLD).read(PACKAGE, Q4, [], None)
    assert (out.answer, out.support, out.bridge) == ("", [], [])


def test_vqa_and_transcribe_and_misc():
    gold = {"Which backplane pin carries THERM_ALERT# on the TQ-40?":
            {"expected_answer": "B14", "expected_citations": ["specs/backplane_pinout.png"]}}
    q = next(iter(gold))
    r = StubReader("oracle", gold, transcripts={"specs/backplane_pinout.png": "B14: THERM_ALERT#"})
    assert r.vqa(q, ("specs/backplane_pinout.png", None), None) == "B14"
    assert r.vqa(q, ("specs/other.png", None), None) == ""
    assert r.vqa(q, object(), None) == "B14"          # unnamed image, gold cites an image file
    assert StubReader("lexical", gold).vqa(q, object(), None) == ""
    assert r.transcribe(("specs/backplane_pinout.png", None), None) == "B14: THERM_ALERT#"
    assert r.transcribe(("x.png", None), None) == ""
    assert r.logprob("", "", [], "x") == -1.0
    assert r.load() is None and r.warmup() is None and r.self_test_sdpa() is True
    assert r.force_math is False and r.attn == "stub" and r.timings == {}


def test_bad_mode():
    with pytest.raises(ValueError):
        StubReader("nope")


def test_from_questions_on_sample():
    data = json.loads(SAMPLE_QUESTIONS.read_text(encoding="utf-8"))
    idx = ix.build(parse.walk(SAMPLE_CORPUS))
    r = StubReader.from_questions(data)
    # package = every eligible record of the sample, as the pipeline would print them
    pkg = "\n".join(f"[R{x.rid}] {x.file} {x.loc}: {x.text}" for x in idx.records if idx.eligible(x.rid))
    for q in data["queries"]:
        out = r.read(pkg, q["query"], [], None)
        assert out.answer == q["expected_answer"]
        files = {idx.record(rid).file for rid in out.support + out.bridge}
        if q["expected_answer"] and q["n"] != 7 and q["n"] != 8:   # image answers live in OCR, absent here
            assert files == set(q["expected_citations"]), q["n"]


def test_parse_package_spaces_in_loc_and_path():
    pkg = ("## my docs/sheet one.xlsx (xlsx)\n"
           "[R5] my docs/sheet one.xlsx RMA Parts!R2: [sheet RMA Parts] part_number=ORR-FAN-2214-B\n"
           "## a/b.txt (txt)\n"
           "[R6] a/b.txt L3: plain: text\n")
    assert parse_package(pkg) == [
        (5, "my docs/sheet one.xlsx", "RMA Parts!R2", "[sheet RMA Parts] part_number=ORR-FAN-2214-B"),
        (6, "a/b.txt", "L3", "plain: text")]
