"""H7: printed field label stripped from the answer only when the question uses the label's words."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import config  # noqa: E402
import verify  # noqa: E402
from schema import Q, Verdict  # noqa: E402


def _q(text):
    return Q(text=text)


def _ans(span, qtext, on=True, monkeypatch=None):
    monkeypatch.setattr(config, "LABEL_STRIP", on)
    return verify.answer_text(Verdict(True, "OK", answer_span=span), _q(qtext))


def test_strips_label_named_in_question(monkeypatch):
    assert _ans("BIN D-14-2", "Which bin location is printed on the spares bin tag?", monkeypatch=monkeypatch) == "D-14-2"
    assert _ans("BOARD REVISION REV-C2", "What board revision is printed on the asset label?",
                monkeypatch=monkeypatch) == "REV-C2"


def test_keeps_when_word_not_in_question_or_no_digit(monkeypatch):
    assert _ans("BIN D-14-2", "Where is part TES-SPR-8320-G stored?", monkeypatch=monkeypatch) == "BIN D-14-2"
    assert _ans("Q3 FY27", "In which quarter does sampling start?", monkeypatch=monkeypatch) == "Q3 FY27"
    assert _ans("REV-C2", "What board revision is printed?", monkeypatch=monkeypatch) == "REV-C2"


def test_off_is_v5(monkeypatch):
    assert _ans("BIN D-14-2", "Which bin location is printed?", on=False, monkeypatch=monkeypatch) == "BIN D-14-2"


def test_keeps_month_and_period_words(monkeypatch):
    assert _ans("September 23, 2026", "On which date in September was it released?", monkeypatch=monkeypatch) \
        == "September 23, 2026"
    assert _ans("Week 12", "In which week does sampling start?", monkeypatch=monkeypatch) == "Week 12"
    assert _ans("Release 4.3.2", "Which firmware release fixed it?", monkeypatch=monkeypatch) == "4.3.2"
