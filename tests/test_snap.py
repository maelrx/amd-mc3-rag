import pytest

import snap


@pytest.mark.parametrize("span,expected", [
    ("94 C", "94"), ("94°C", "94"), ("$84.50", "84.50"), ("10,000 units", "10,000"),
    ("180 seconds", "180"), ("REV-C2", "REV-C2"), ("v4.3.2", "v4.3.2"), ("Q3 FY27", "Q3 FY27"),
    ("ORR-FAN-2214-B,", "ORR-FAN-2214-B"), ("15%", "15%"), ("E7731", "E7731"),
    ("91C", "91"), ("R$ 1,200.00", "1,200.00"), ('"B14"', "B14"), ("4.3.2.", "4.3.2"),
    ("USD 12 days", "12"), ("about five", "about five"),
])
def test_clean(span, expected):
    assert snap.clean(span) == expected


def test_clean_explicit_flag():
    assert snap.clean("94 C", is_identifier=True) == "94 C"
    assert snap.clean("3A", is_identifier=False) == "3"


def test_find_verbatim_span():
    assert snap.find("94", "Maximum junction temperature  ...  94 C") == "94"
    assert snap.find("ORR-1847", "ticket=ORR-1847; fixed_in=4.3.2") == "ORR-1847"
    assert snap.find("4.3.2", "2026-09-02T03:14:22Z 4.3.2.1") is None
    assert snap.find("tq40", "model TQ-40 x") == "TQ-40"  # equal under norm, returned as written
    assert snap.find("(94)", "temp (94) C") == "94"
    assert snap.find('"E7731"', "ERROR E7731: thermal throttle") == "E7731"
    assert snap.find("", "abc") is None


def test_find_loose_only_for_ocr():
    assert snap.find("OS4-118823-77", "SNOS4-118823-77") is None
    assert snap.find("OS4-118823-77", "SNOS4-118823-77", loose=True) == "OS4-118823-77"
