"""CPU stand-in for ``app/reader.Reader`` (protocol: see the app/reader.py docstring). No torch, no model.

Lets the pipeline, verifier and server be exercised end to end on a laptop:

    oracle       the gold answer, with support/bridge rids picked from the package like a perfect reader
    lexical      dumb baseline: the record with the most question-token overlap, its first value-looking token
    adversarial  a plausible WRONG value taken from a record outside the gold file(s) (the verifier must reject it)
    refuse       always ""

``gold`` maps question text -> {"expected_answer": str, "expected_citations": [file, ...]} (the questions.json
shape; ``StubReader.from_questions`` builds it from a loaded questions.json).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_APP = str(Path(__file__).resolve().parents[1] / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import config  # noqa: E402
import textnorm  # noqa: E402
from schema import ReaderOut  # noqa: E402

MODES = ("oracle", "lexical", "adversarial", "refuse")
_LINE_RE = re.compile(r"^\[R(\d+)\]\s+(\S+)\s+(.*?):\s?(.*)$")
_REC_RE = re.compile(r"^\[R(\d+)\]\s+(.*)$")
_HEAD_RE = re.compile(r"^##\s+(.+?)\s+\([^()]*\)\s*$")
_LOC_TEXT_RE = re.compile(r"^(.*?):\s?(.*)$")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")
_FALLBACK_WRONG = "99999"
_MAX_RIDS = 3


def parse_package(package_text: str) -> list[tuple[int, str, str, str]]:
    """``[R{rid}] file loc: text`` lines -> [(rid, file, loc, text)]; other lines are skipped.

    The file is taken from the preceding ``## file (kind)`` heading when the line starts with it (paths with
    spaces); locs may contain spaces ("RMA Parts!R2")."""
    out = []
    heading = None
    for line in str(package_text).splitlines():
        h = _HEAD_RE.match(line)
        if h:
            heading = h.group(1)
            continue
        m = _REC_RE.match(line)
        if not m:
            continue
        rid, rest = int(m.group(1)), m.group(2)
        if heading and rest.startswith(heading + " "):
            lt = _LOC_TEXT_RE.match(rest[len(heading) + 1:])
            if lt:
                out.append((rid, heading, lt.group(1), lt.group(2)))
                continue
        m2 = _LINE_RE.match(line)
        if m2:
            out.append((int(m2.group(1)), m2.group(2), m2.group(3), m2.group(4)))
    return out


def _values(text: str, question: str) -> list[str]:
    """Value-looking tokens (identifiers, then numbers) of text in order, minus those the question already has."""
    asked = {textnorm.norm(x) for x in textnorm.ids(question) + textnorm.numbers(question)}
    cands = textnorm.ids(text) + textnorm.numbers(text)
    seen, out = set(), []
    for c in sorted(set(cands), key=lambda c: text.find(c)):
        n = textnorm.norm(c)
        if n and n not in asked and n not in seen:
            seen.add(n)
            out.append(c)
    return out


def _overlap(question: str, text: str) -> int:
    return len(set(textnorm.tokens(question)) & set(textnorm.tokens(text)))


class StubReader:
    def __init__(self, mode: str = "oracle", gold: dict | None = None, transcripts: dict | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.gold = {str(k).strip(): v for k, v in (gold or {}).items()}
        self.transcripts = dict(transcripts or {})   # image file -> text returned by transcribe()
        self.timings: dict[str, float] = {}
        self.force_math = False
        self.attn = "stub"
        self.calls: list[str] = []                   # "read"/"vqa"/"transcribe"/"logprob"/"yes_no", for tests

    @classmethod
    def from_questions(cls, data: dict, mode: str = "oracle") -> "StubReader":
        """From a loaded questions.json ({"queries": [{"query", "expected_answer", "expected_citations"}]})."""
        gold = {q["query"]: {"expected_answer": q.get("expected_answer", ""),
                             "expected_citations": list(q.get("expected_citations", []))}
                for q in data["queries"]}
        return cls(mode, gold)

    # -- lifecycle: no-ops --------------------------------------------------------------------

    def load(self) -> None:
        return None

    def self_test_sdpa(self) -> bool:
        return True

    def warmup(self) -> None:
        return None

    # -- helpers ------------------------------------------------------------------------------

    def _gold(self, question: str) -> dict | None:
        return self.gold.get(str(question).strip())

    @staticmethod
    def _out(answer: str, support: list[int], bridge: list[int]) -> ReaderOut:
        raw = json.dumps({"answer": answer, "support": [f"R{r}" for r in support],
                          "bridge": [f"R{r}" for r in bridge]}, ensure_ascii=False, separators=(",", ":"))
        return ReaderOut(answer, list(support), list(bridge), raw=raw, n_new_tokens=max(1, len(raw) // 4),
                         mean_logprob=-0.1 if answer else -1.0)

    # -- modes --------------------------------------------------------------------------------

    def _oracle(self, recs, question: str) -> ReaderOut:
        g = self._gold(question)
        answer = str(g.get("expected_answer", "")) if g else ""
        cited = set(g.get("expected_citations", [])) if g else set()
        if not answer or not cited:
            return self._out("", [], [])
        support = [rid for rid, f, _, text in recs
                   if f in cited and textnorm.find_span(text, answer) is not None][:_MAX_RIDS]
        by_rid = {rid: (f, text) for rid, f, _, text in recs}
        support_files = {by_rid[r][0] for r in support}
        shared = {textnorm.norm(i) for r in support for i in textnorm.ids(by_rid[r][1])}
        bridge = []
        for rid, f, _, text in recs:
            if f in cited and f not in support_files and len(bridge) < _MAX_RIDS \
                    and shared & {textnorm.norm(i) for i in textnorm.ids(text)}:
                bridge.append(rid)
        return self._out(answer, support, bridge)

    def _lexical(self, recs, question: str) -> ReaderOut:
        best = None
        for rid, f, loc, text in recs:
            if loc == "name":
                continue
            score = _overlap(question, text)
            if score and (best is None or score > best[0]):
                best = (score, rid, text)
        if best is None:
            return self._out("", [], [])
        vals = _values(best[2], question)
        return self._out(vals[0], [best[1]], []) if vals else self._out("", [], [])

    def _adversarial(self, recs, question: str) -> ReaderOut:
        g = self._gold(question)
        cited = set(g.get("expected_citations", [])) if g else set()
        gold_norm = textnorm.norm(g.get("expected_answer", "")) if g else ""
        ranked = sorted((r for r in recs if r[1] not in cited and r[2] != "name"),
                        key=lambda r: (-_overlap(question, r[3]), r[0]))
        for rid, _, _, text in ranked:
            for v in _values(text, question):
                if textnorm.norm(v) != gold_norm:
                    return self._out(v, [rid], [])
        return self._out(_FALLBACK_WRONG, [], [])

    # -- reader protocol ----------------------------------------------------------------------

    def read(self, package_text: str, question: str, images, deadline,
             max_new_tokens: int = config.MAX_NEW_TOKENS) -> ReaderOut:
        self.calls.append("read")
        recs = parse_package(package_text)
        if self.mode == "oracle":
            return self._oracle(recs, question)
        if self.mode == "lexical":
            return self._lexical(recs, question)
        if self.mode == "adversarial":
            return self._adversarial(recs, question)
        return self._out("", [], [])

    @staticmethod
    def _image_name(image) -> str:
        if isinstance(image, (tuple, list)) and len(image) == 2:
            return str(image[0])
        return ""

    def vqa(self, question: str, image, deadline, max_new_tokens: int = 48, longest_edge: int | None = None) -> str:
        self.calls.append("vqa")
        if self.mode != "oracle":
            return ""
        g = self._gold(question)
        if not g:
            return ""
        cited = g.get("expected_citations", [])
        name = self._image_name(image)
        if name:
            return str(g.get("expected_answer", "")) if name in cited else ""
        return str(g.get("expected_answer", "")) if any(c.lower().endswith(_IMAGE_EXT) for c in cited) else ""

    def transcribe(self, image, deadline, max_new_tokens: int = config.TRANSCRIBE_MAX_NEW_TOKENS) -> str:
        self.calls.append("transcribe")
        return self.transcripts.get(self._image_name(image), "")

    def logprob(self, package_text: str, question: str, images, answer: str) -> float:
        self.calls.append("logprob")
        return -1.0

    def yes_no(self, evidence_text: str, question: str, answer: str, images=None, deadline=None) -> float:
        """H6 warrant score. oracle: +5 when ``answer`` is the gold answer of ``question``, -5 otherwise; other modes 0.0."""
        self.calls.append("yes_no")
        if self.mode != "oracle":
            return 0.0
        g = self._gold(question)
        gold = str(g.get("expected_answer", "")) if g else ""
        a = str(answer or "")
        if not gold or not a.strip():
            return -5.0
        same = re.sub(r"[\s\-.·_]", "", a).upper() == re.sub(r"[\s\-.·_]", "", gold).upper()
        return 5.0 if same or textnorm.find_span(a, gold) is not None or textnorm.find_span(gold, a) is not None \
            else -5.0
