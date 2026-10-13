import json

import pytest

import config
import index as ix
import parse
import retrieve
import textnorm
from conftest import SAMPLE_CORPUS, SAMPLE_QUESTIONS

R1 = "specs/tq40_datasheet_r1_WITHDRAWN.pdf"
ENCRYPTED = "vendor/supplier_agreement_ENCRYPTED.pdf"
QUERIES = {q["n"]: q for q in json.loads(SAMPLE_QUESTIONS.read_text("utf-8"))["queries"]}


@pytest.fixture(scope="module")
def idx():
    return ix.build(parse.walk(SAMPLE_CORPUS))


@pytest.fixture(scope="module")
def pkgs(idx):
    """n -> (Q, Package) for the ten sample questions."""
    out = {}
    for n, q in QUERIES.items():
        Qn = retrieve.analyze(q["query"], idx)
        out[n] = (Qn, retrieve.assemble(idx, Qn))
    return out


def _gold_rids(idx, pkg, q):
    """Package records from a cited file whose text holds the expected answer (token-bounded)."""
    return [r for r in pkg.rids
            if idx.record(r).file in q["expected_citations"]
            and textnorm.find_span(idx.record(r).text, q["expected_answer"])]


# ---------------------------------------------------------------- analyze

def test_analyze_q4_ticket(idx):
    q = retrieve.analyze(QUERIES[4]["query"], idx)
    assert q.ids == ["ORR-1847"]
    assert "version" in q.attr_terms
    assert "csv" in q.type_hint
    assert q.rare_ids == ["ORR-1847"]  # df 2 <= RARE_DF
    assert not q.wants_old


def test_analyze_q6_unit_timeout_py(idx):
    q = retrieve.analyze(QUERIES[6]["query"], idx)
    assert q.unit == "s"
    assert "timeout" in q.attr_terms
    assert "py" in q.type_hint
    assert q.numbers == [] and q.ids == []


def test_analyze_q1_entity(idx):
    q = retrieve.analyze(QUERIES[1]["query"], idx)
    assert q.entities == ["TQ-40"]
    assert "temperature" in q.attr_terms
    assert q.numbers == []  # the 40 of TQ-40 is part of an id
    assert q.rare_ids == ["TQ-40"]  # TQ40 occurs in 4 files (df 4 <= RARE_DF 5)


def test_analyze_wants_old():
    assert retrieve.analyze("What was the value in the withdrawn revision 1?").wants_old
    assert retrieve.analyze("What did the previous datasheet say?").wants_old
    assert not retrieve.analyze("What is the temperature threshold of the hold timer?").wants_old
    assert not retrieve.analyze(QUERIES[1]["query"]).wants_old


def test_analyze_numbers():
    assert retrieve.analyze("at 10,000 unit volume").numbers == ["10,000"]
    assert retrieve.analyze("price at 10,000 unit volume for TQ-40").numbers == ["10,000"]
    assert retrieve.analyze("when does TQ-60 ship in Q3 FY27").numbers == []
    assert retrieve.analyze("how many fans were ordered in 2026 after 2026-09-02").numbers == []
    assert retrieve.analyze("is the limit 94.5 or more than 12 units").numbers == ["94.5", "12"]


def test_analyze_ids_and_rare():
    q = retrieve.analyze("Does version 4.3.2 or 94.5 apply to ORR-9999 and E7731?")
    assert q.ids == ["4.3.2", "ORR-9999", "E7731"]  # plain decimal 94.5 dropped
    assert retrieve.analyze("pin THERM_ALERT# on TQ-40").ids == ["THERM_ALERT#", "TQ-40"]
    idx_less = retrieve.analyze("ORR-9999 and TQ-40")
    assert idx_less.rare_ids == idx_less.ids


def test_analyze_rare_uses_index(idx):
    q = retrieve.analyze("Is ORR-9999 fixed in TQ-40?", idx)
    assert q.rare_ids == ["ORR-9999", "TQ-40"]  # absent from the corpus is rare; TQ-40 has df 4 <= RARE_DF


def test_analyze_units():
    assert retrieve.analyze("What is the limit in watts?").unit == "W"
    assert retrieve.analyze("timeout in milliseconds").unit == "ms"
    assert retrieve.analyze("limit in percent").unit == "%"
    assert retrieve.analyze("max 94 °C or so").unit == "C"
    assert retrieve.analyze("how long is the timeout").unit is None


def test_analyze_type_hints():
    assert retrieve.analyze("which release notes mention it").type_hint == {"txt"}
    assert retrieve.analyze("what is printed on the label").type_hint == {"image"}
    assert retrieve.analyze("see the lead time in the spreadsheet").type_hint == {"xlsx"}
    assert retrieve.analyze("what does the roadmap say").type_hint == {"docx"}
    assert retrieve.analyze("how many apples").type_hint == set()


# ---------------------------------------------------------------- sample questions

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 9])
def test_text_question_gold_in_package(idx, pkgs, n):
    q = QUERIES[n]
    _, pkg = pkgs[n]
    for f in q["expected_citations"]:
        assert f in pkg.files, f"{f} missing from {pkg.files}"
    assert _gold_rids(idx, pkg, q), f"no record with {q['expected_answer']!r} in package"


def test_q1_withdrawn_excluded(idx, pkgs):
    _, pkg = pkgs[1]
    assert R1 not in pkg.files
    assert not any(idx.record(r).file == R1 for r in pkg.rids)
    assert "105" not in pkg.text


def test_q4_ticket_in_question_is_not_hopped(idx, pkgs):
    q, pkg = pkgs[4]
    assert q.ids == ["ORR-1847"]
    # the ticket id is the question's own id: no hop may be driven by a record through it
    assert not any("ORR-1847" in idx.record(p).text for p in pkg.via.values())


def test_q9_hops_from_log_to_csv_row(idx, pkgs):
    _, pkg = pkgs[9]
    row2 = next(r.rid for r in idx.records if r.file == "support/bug_database.csv" and r.loc == "row2")
    assert row2 in pkg.rids
    assert row2 in pkg.via
    assert "ORR-1847" in idx.record(pkg.via[row2]).text
    assert idx.record(pkg.via[row2]).file == "logs/prod_inference_2026-09-02.log"
    assert pkg.via[row2] in pkg.rids


def test_q10_encrypted_not_in_files(pkgs):
    _, pkg = pkgs[10]
    assert ENCRYPTED not in pkg.files
    assert pkg.rids and pkg.text and pkg.token_est > 0


@pytest.mark.parametrize("n", [7, 8])
def test_image_questions_carry_descriptor(pkgs, n):
    q = QUERIES[n]
    _, pkg = pkgs[n]
    gold = q["expected_citations"][0]
    assert gold in pkg.files
    assert (gold, ("file", gold)) in pkg.images
    assert len(pkg.images) <= config.MAX_IMAGES


def test_package_format_and_order(idx, pkgs):
    _, pkg = pkgs[1]
    assert pkg.rids == sorted(pkg.rids, key=lambda r: (pkg.files.index(idx.record(r).file), r))
    lines = pkg.text.split("\n")
    assert lines[0].startswith("## ") and lines[0].endswith(")")
    for r in pkg.rids:
        rec = idx.record(r)
        assert f"[R{r}] {rec.file} {rec.loc}: " in pkg.text
        assert not rec.meta and rec.rid == r
    # name records are never rendered
    assert not any(idx.record(r).meta for r in pkg.rids)
    assert pkg.token_est == int(len(pkg.text) / config.CHARS_PER_TOKEN)


# ---------------------------------------------------------------- retrieve / hop / package

def test_retrieve_drops_ineligible_and_stale(idx):
    q = retrieve.analyze("junction temperature of the TQ-40 and fan assembly legacy single-rotor", idx)
    hits = retrieve.retrieve(idx, q)
    assert R1 not in {h.file for h in hits}
    stale = {r.rid for r in idx.records if r.stale}
    assert stale and not (stale & {r for h in hits for r in h.rids})
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_wants_old_keeps_withdrawn(idx):
    q = retrieve.analyze("What junction temperature does the withdrawn revision 1 datasheet of the TQ-40 give?", idx)
    assert q.wants_old
    pkg = retrieve.assemble(idx, q)
    assert R1 in pkg.files
    assert any(idx.record(r).file == R1 and "105" in idx.record(r).text for r in pkg.rids)
    assert "specs/tq40_datasheet_r2.pdf" in pkg.files  # eligible files are not dropped either
    stale_row = next(r.rid for r in idx.records if r.stale)
    q2 = retrieve.analyze("Which legacy single-rotor fan is in the parts spreadsheet of the TQ-20 (previous)?", idx)
    assert q2.wants_old and stale_row in retrieve.assemble(idx, q2).rids


def test_retrieve_attempt2_widens(idx):
    q = retrieve.analyze(QUERIES[1]["query"], idx)
    h1, h2 = retrieve.retrieve(idx, q, 1), retrieve.retrieve(idx, q, 2)
    assert len(h2) >= len(h1)
    assert max(len(h.rids) for h in h1) <= 6 and max(len(h.rids) for h in h2) <= 10


def test_exclude_rids(idx):
    q = retrieve.analyze(QUERIES[1]["query"], idx)
    gold = next(r.rid for r in idx.records if r.file == "specs/tq40_datasheet_r2.pdf" and "94 C" in r.text)
    assert gold in retrieve.assemble(idx, q).rids
    pkg = retrieve.assemble(idx, q, 2, exclude_rids={gold})
    assert gold not in pkg.rids
    assert gold not in {r for h in retrieve.retrieve(idx, q, 2, {gold}) for r in h.rids}


def test_hop_unit(idx):
    q = retrieve.analyze("The production log shows a thermal throttle incident. Who fixed it?", idx)
    hits = retrieve.retrieve(idx, q)
    log = next(h for h in hits if h.file.startswith("logs/"))
    assert "support/bug_database.csv" not in {h.file for h in hits[:1]}
    out = retrieve.hop(idx, q, hits)
    csv = next(h for h in out if h.file == "support/bug_database.csv")
    row2 = next(r.rid for r in idx.records if r.file == csv.file and r.loc == "row2")
    assert csv.via[row2] in log.rids and row2 in csv.rids
    assert csv.score >= 0.9 * log.score - 1e-12
    assert out == sorted(out, key=lambda h: (-h.score, h.file))
    assert all(not h.via for h in hits)  # inputs are not mutated


def test_hop_skips_ids_named_in_question(idx):
    q = retrieve.analyze("Which release closed ORR-1847?", idx)
    out = retrieve.hop(idx, q, retrieve.retrieve(idx, q))
    for h in out:
        assert not any("ORR-1847" in idx.record(p).text for p in h.via.values())


def test_hop_never_re_adds_parent(idx):
    q = retrieve.analyze(QUERIES[9]["query"], idx)
    out = retrieve.hop(idx, q, retrieve.retrieve(idx, q), max_hops=3)
    for h in out:
        for rid, parent in h.via.items():
            assert idx.record(rid).file != idx.record(parent).file
            assert rid != parent


def test_hop_disabled(idx):
    q = retrieve.analyze(QUERIES[9]["query"], idx)
    hits = retrieve.retrieve(idx, q)
    assert all(not h.via for h in retrieve.hop(idx, q, hits, max_hops=0))


def test_token_cap_and_forced_records(idx, monkeypatch):
    monkeypatch.setattr(config, "PKG_TOKENS", (60, 60))
    q = retrieve.analyze(QUERIES[1]["query"], idx)
    pkg = retrieve.assemble(idx, q)
    files = {idx.record(r).file for r in pkg.rids}
    assert len(files) >= 2  # top record of the top-2 files is always kept
    assert pkg.token_est < 60 + 120


def test_long_record_truncated():
    from schema import FileEntry, Record
    long_rec = Record(1, "a.txt", "L1", "word " * 400)
    index = ix.Index([FileEntry("a.txt", "txt", "ok", rid_range=[0, 2])],
                     [Record(0, "a.txt", "name", "a txt", meta=True, src="name"), long_rec])
    q = retrieve.analyze("word", index)
    pkg = retrieve.assemble(index, q)
    line = next(ln for ln in pkg.text.split("\n") if ln.startswith("[R1]"))
    assert line.endswith("…") and len(line) <= len("[R1] a.txt L1: ") + 600


def test_empty_query_and_no_match(idx):
    q = retrieve.analyze("zzzz qqqq", idx)
    pkg = retrieve.assemble(idx, q)
    assert pkg.rids == [] and pkg.text == "" and pkg.files == []
    assert retrieve.retrieve(idx, q) == []


def test_diagnose_and_table(idx, pkgs):
    print()
    print(f"{'n':>2} {'~tok':>5} {'recs':>4}  top files (score) | gold rid / via")
    for n, (q, pkg) in pkgs.items():
        hits = retrieve.hop(idx, q, retrieve.retrieve(idx, q))
        top = ", ".join(f"{h.file.rsplit('/', 1)[-1]}({h.score:.4f})" for h in hits[:4])
        gold = _gold_rids(idx, pkg, QUERIES[n])
        extra = f"gold={['R%d' % r for r in gold]} via={pkg.via} img={[f for f, _ in pkg.images]}"
        print(f"{n:>2} {pkg.token_est:>5} {len(pkg.rids):>4}  {top} | {extra}")
        d = retrieve.diagnose(idx, q, pkg)
        assert q.text in d and all(f in d for f in pkg.files)
