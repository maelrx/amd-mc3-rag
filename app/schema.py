"""Shared data contracts for the MC3 pipeline. Stdlib only; importable by every module.

Modules import each other as top-level names (``import schema``) because inside the
container they run as ``python3 /app/app.py`` with ``/app`` on sys.path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Record:
    """One atomic evidence unit. ``rid`` is the global 0-based position in records.jsonl."""

    rid: int
    file: str            # corpus-relative POSIX path, exactly as walked
    loc: str             # parse.py loc (p3, table1.row2, Sheet!R4, row2, p1.L7, L12.const, ocr.L3, vlm.L2, name)
    text: str
    stale: bool = False  # superseded/legacy row or hidden sheet; never a support record unless wants_old
    meta: bool = False   # name record: retrievable, never citable as support
    src: str = "text"    # text | ocr | vlm | name

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Record":
        return cls(rid=int(d["rid"]), file=d["file"], loc=d["loc"], text=d["text"],
                   stale=bool(d.get("stale", False)), meta=bool(d.get("meta", False)),
                   src=d.get("src", "text"))


@dataclass
class FileEntry:
    file: str
    kind: str                     # pdf|docx|xlsx|csv|log|txt|md|py|image|pptx|other
    status: str                   # ok|unreadable|encrypted|binary|empty|error
    withdrawn: bool = False
    revision: str | None = None
    family: str = ""
    superseded_by: str | None = None
    supersedes: list[str] = field(default_factory=list)
    eligible: bool = True         # status ok and not withdrawn and superseded_by is None
    images: list = field(default_factory=list)   # [["file", rel] | ["zip", member] | ["pdfpage", n1]]
    rid_range: list[int] = field(default_factory=lambda: [0, 0])  # [start, end) in records.jsonl
    n_products: int = 0
    draft: bool = False           # "draft" in the file name or title; ineligible when a non-draft sibling exists

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "FileEntry":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Q:
    """Question analysis (retrieve.analyze)."""

    text: str
    ids: list[str] = field(default_factory=list)        # raw IDs found in the question
    rare_ids: list[str] = field(default_factory=list)   # ids with df <= RARE_DF or absent from corpus
    entities: list[str] = field(default_factory=list)   # product codes like TQ-40
    numbers: list[str] = field(default_factory=list)    # numeric conditions outside IDs/years
    unit: str | None = None                             # unit the question asks for ("seconds")
    attr_terms: list[str] = field(default_factory=list) # price/cost, temperature, timeout, version...
    type_hint: set[str] = field(default_factory=set)    # {"log","pdf","image","xlsx","py","docx","csv","txt"}
    wants_old: bool = False                             # question explicitly asks for the withdrawn/old version
    named_kinds: set[str] = field(default_factory=set)  # kinds the question names ("the production log")


@dataclass
class FileHit:
    file: str
    score: float
    rids: list[int] = field(default_factory=list)
    via: dict[int, int] = field(default_factory=dict)   # rid -> via_rid (hop parent record)
    best: float = 0.0                                   # RRF score of the file's best record


@dataclass
class Package:
    rids: list[int]                       # records included, in document order
    text: str                             # the EVIDENCE block sent to the reader
    images: list = field(default_factory=list)   # descriptors (file, desc); pipeline loads pixels
    token_est: int = 0
    files: list[str] = field(default_factory=list)
    via: dict[int, int] = field(default_factory=dict)


@dataclass
class ReaderOut:
    answer: str
    support: list[int] = field(default_factory=list)
    bridge: list[int] = field(default_factory=list)
    raw: str = ""
    n_new_tokens: int = 0
    mean_logprob: float | None = None


@dataclass
class Verdict:
    ok: bool
    reason: str                           # OK|EMPTY|NO_SPAN|INELIGIBLE_ONLY|COND_MISMATCH|ENTITY_MISMATCH|UNIT_MISMATCH|FALSE_PREMISE|CHAIN_OPEN
    answer_span: str | None = None        # verbatim span from the support record
    support_rid: int | None = None
    bridge_rids: list[int] = field(default_factory=list)
    grounding_score: int = 0
    violators: list[int] = field(default_factory=list)   # rids to drop from the retry package
    premise_files: list[str] = field(default_factory=list)
    hard: bool = False                    # True -> no retry (FALSE_PREMISE, ONLY_IN_INELIGIBLE)


@dataclass
class Result:
    answer: str
    citations: list[str]
    confidence: float
    trace: dict = field(default_factory=dict)

    def to_output(self) -> dict:
        """Exactly the three keys the harness grades."""
        return {"answer": str(self.answer), "citations": [str(c) for c in self.citations],
                "confidence": float(self.confidence)}


def refusal(reason: str = "") -> Result:
    return Result("", [], 0.0, {"reason": reason} if reason else {})
