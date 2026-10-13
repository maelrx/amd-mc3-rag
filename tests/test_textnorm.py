import re
import sys
from pathlib import Path

import pytest

import textnorm as tn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))


def test_norm_matches_scorer():
    import score
    for s in ["TQ-40", "Q3 FY27", "a b·c_d.e", "4.3.2", 94, ""]:
        assert tn.norm(s) == score.norm(s)
    assert tn.nid("ORR-1847") == "ORR1847"


def test_tokens_expand_ids_and_drop_stopwords():
    assert tn.tokens("TQ-40") == ["tq-40", "tq", "40", "tq40"]
    assert tn.tokens("What is the default timeout of ORR-1847?") == [
        "timeout", "orr-1847", "orr", "1847", "orr1847"]
    assert tn.tokens("therm_alert#") == ["therm_alert#", "therm_alert", "therm", "alert", "thermalert"]
    assert tn.tokens("fixed fixed") == ["fixed", "fixed"]  # tf is kept
    assert "version" in tn.tokens("firmware version")


def test_ids():
    s = "Ticket ORR-1847. at 2026-09-02T03:14:22Z v4.3.2 Q3 FY27 THERM_ALERT# E7731, (B14) 4.3.2.1 in 2026"
    assert tn.ids(s) == ["ORR-1847", "v4.3.2", "Q3 FY27", "THERM_ALERT#", "E7731", "B14", "4.3.2.1"]
    assert tn.ids("fixed_in=4.3.2") == ["4.3.2"]
    assert tn.ids("ORR-1847 and ORR-1847") == ["ORR-1847"]
    assert tn.ids("12345 2026 10:30") == []


@pytest.mark.parametrize("s,expected", [
    ("94 C", False), ("94°C", False), ("$84.50", False), ("10,000 units", False), ("15%", False),
    ("84.50", False), ("REV-C2", True), ("v4.3.2", True), ("4.3.2", True), ("Q3 FY27", True),
    ("E7731", True), ("ORR-FAN-2214-B", True),
])
def test_is_identifier(s, expected):
    assert tn.is_identifier(s) is expected


def test_numbers():
    assert tn.numbers("94 C, 10,000 units, -3.5 and 7") == ["94", "10,000", "-3.5", "7"]
    assert tn.num_key("10,000") == "10000"


@pytest.mark.parametrize("hay,needle,found", [
    ("2026-09-02T03:14:22Z 4.3.2.1", "4.3.2", None),
    ("T03:14:22Z", "4.3.2", None),
    ("fixed_in=4.3.2", "4.3.2", "4.3.2"),
    ("release 4.3.2 shipped", "4.3.2", "4.3.2"),
    ("released in 4.3.2.", "4.3.2", "4.3.2"),
    ("fix v4.3.2 now", "4.3.2", "4.3.2"),   # a glued leading v is allowed for dotted versions
    ("fix xv4.3.2 now", "4.3.2", None),
    ("see ORR-1847, ok", "ORR-1847", "ORR-1847"),
    ("see ORR-1847-B ok", "ORR-1847", None),
    ('id "ORR-1847"', "ORR-1847", "ORR-1847"),
    ("Tj  ...  94 C", "94", "94"),
    ("temp (94) C", "(94)", "(94)"),
    ("part SN OS4-118823-77 end", "SN OS4-118823-77", "SN OS4-118823-77"),
    ("part SN OS4-118823-77 end", "SNOS4-118823-77", "SN OS4-118823-77"),
    ("model tq40 here", "TQ-40", "tq40"),
    ("model TQ-400 here", "TQ-40", None),
    ("1847", "ORR-1847", None),
])
def test_find_span(hay, needle, found):
    span = tn.find_span(hay, needle)
    assert (hay[span[0]:span[1]] if span else None) == found
    assert tn.word_boundary_contains(hay, needle) is (found is not None)


def test_find_span_loose_ocr_join():
    assert tn.find_span("SNOS4-118823-77", "OS4-118823-77") is None
    s, e = tn.find_span_loose("lbl SNOS4-118823-77", "OS4-118823-77")
    assert "lbl SNOS4-118823-77"[s:e] == "OS4-118823-77"
    # an ID-shaped needle (letter, digit, "-") accepts any alphabetic prefix; others keep the 3-char rule
    assert tn.find_span_loose("lbl LONGPREFIXOS4-118823-77", "OS4-118823-77") == (14, 27)
    assert tn.find_span_loose("lbl LONGPREFIXE7731", "E7731") is None
    assert tn.find_span_loose("a ORR-1847 b", "ORR-1847") == (2, 10)
