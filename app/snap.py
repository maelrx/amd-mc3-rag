"""Snap a reader answer to a verbatim source span and strip units/currency. Stdlib only."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

import textnorm

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_LABEL_RE = re.compile(r"^(?:SN|S/N|PN|P/N|MPN|ID|REF|LOT|SERIAL|PART|MODEL)\s*[:#]?\s+(?=\S*\d)", re.I)
_QUARTER_QF = re.compile(r"^(Q[1-4])\s*(FY\s*\d{2,4})$", re.I)
_QUARTER_FQ = re.compile(r"^(FY\s*\d{2,4})\s*(Q[1-4])$", re.I)
from textnorm import QTY_RE, UNIT_RE  # noqa: F401  (UNIT_RE is part of this module's public surface)

_CURRENCY_RE = re.compile(r"^(?:R\$|USD|EUR|[$€£])\s*", re.I)
_QUOTES = "\"'“”‘’"
_WRAP = _QUOTES + "()[]{}<>"


def find(answer: str, text: str, loose: bool = False) -> str | None:
    """Verbatim span of text that normalises like answer (loose allows OCR-glued prefixes)."""
    answer = str(answer).strip().strip(_QUOTES).strip()
    if not answer:
        return None
    finder = textnorm.find_span_loose if loose else textnorm.find_span
    span = finder(text, answer)
    if span is None:
        # a quarter written in the other order ("Q3 FY27" vs "FY27 Q3")
        m = _QUARTER_QF.match(answer) or _QUARTER_FQ.match(answer)
        if m:
            span = finder(text, f"{m.group(2)} {m.group(1)}")
    if span is None:
        # the model may add a currency sign or unit the record lacks ("$84.50" vs unit_cost_usd=84.50),
        # or write 84.5 for 84.50: retry with the bare value, then with numerically equal numbers
        bare = clean(answer)
        if bare != answer:
            span = finder(text, bare)
        if span is None and _NUMBER_RE.fullmatch(bare):
            try:
                target = Decimal(bare.replace(",", ""))
                for m in _NUMBER_RE.finditer(text):
                    if Decimal(m.group().replace(",", "")) == target and finder(text, m.group()):
                        span = finder(text, m.group())
                        break
            except (InvalidOperation, ValueError):
                span = None
    if span is None:
        return None
    return text[span[0]:span[1]].strip().strip(_WRAP).strip() or None


def clean(span: str, is_identifier: bool | None = None) -> str:
    """Final answer form: identifiers verbatim, quantities as the bare number, '%' kept.

    Unicode dashes/minus come out as ASCII "-" ("\u221220" -> "-20")."""
    s = textnorm.fold_dashes(str(span)).strip()
    if is_identifier is None:
        is_identifier = textnorm.is_identifier(s)
    if is_identifier:
        prev = None
        while prev != s:
            prev = s
            s = s.strip().strip(_QUOTES).lstrip("(").rstrip(".,;:)").strip()
        # a printed label copied along with the code: "SN PT1-783636-48" -> "PT1-783636-48"
        s = _LABEL_RE.sub("", s)
        return s
    body = _CURRENCY_RE.sub("", s).strip()
    for cand in (body, body.rstrip(".,;:")):
        value = textnorm.qty_value(cand)
        if value is not None:
            return value
    return body
