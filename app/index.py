"""Records, files, families, ID index, BM25 and persistence. Stdlib only."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import config
import textnorm
from schema import FileEntry, Record

_PRODUCT_RE = re.compile(r"^[A-Z]{1,4}-?\d{1,4}[A-Z]?$", re.I)
_REV_TOKEN_RE = re.compile(r"(?:^|[_\-\s])(?:r|rev|v)\d+[a-z]?(?=[_\-\s]|$)", re.I)
_STATUS_TOKEN_RE = re.compile(
    r"(?:^|[_\-\s])(?:withdrawn|superseded|obsolete|draft|final|old|legacy)(?=[_\-\s]|$)", re.I)
_DATE_TOKEN_RE = re.compile(r"\d{4}[-_]\d{2}(?:[-_]\d{2})?")
_REV_PARSE_RE = re.compile(r"(\d+)([a-z]?)", re.I)
_VERSION_ID_RE = re.compile(r"\d+(?:\.\d+){1,3}")
_HEADING_RE = re.compile(r"(?:columns\b|#|\[sheet [^\]]*\]\s*columns\b)", re.I)


# ---------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25 over pre-tokenised documents; the doc index is the record rid."""

    def __init__(self, docs: list[list[str]], k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        self.post: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.post.setdefault(term, []).append((i, tf))

    def search(self, q_tokens: list[str], top_n: int) -> list[tuple[int, float]]:
        """Top (doc, score) pairs, best first; query terms are counted once."""
        scores: dict[int, float] = {}
        for term in dict.fromkeys(q_tokens):
            plist = self.post.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in plist:
                norm_len = 1 - self.b + self.b * (self.dl[i] / self.avgdl if self.avgdl else 0.0)
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * norm_len)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]


# ---------------------------------------------------------------- IDs

def _indexable(raw: str) -> bool:
    """IDs worth indexing: code with >=2 digits and a letter, dotted version, quarter, NAME#."""
    if raw.endswith("#") or _VERSION_ID_RE.fullmatch(raw):
        return True
    if re.fullmatch(r"Q[1-4]\s*FY\s*\d{2,4}", raw):
        return True
    return sum(c.isdigit() for c in raw) >= 2 and any(c.isalpha() for c in raw)


def _add_ids(ids: dict[str, dict], records: list[Record]) -> None:
    """Fold records into the ID index (in place)."""
    for r in records:
        for raw in textnorm.ids(r.text):
            if not _indexable(raw):
                continue
            e = ids.setdefault(textnorm.nid(raw), {"raw": [], "rids": [], "files": [], "df": 0})
            if raw not in e["raw"]:
                e["raw"].append(raw)
            if not e["rids"] or e["rids"][-1] != r.rid:
                e["rids"].append(r.rid)
            if r.file not in e["files"]:
                e["files"].append(r.file)
            e["df"] = len(e["files"])


def build_ids(records: list[Record]) -> dict[str, dict]:
    """ID index over every record (name and OCR/VLM included)."""
    ids: dict[str, dict] = {}
    _add_ids(ids, records)
    return ids


# ---------------------------------------------------------------- files / families

def _family(path: str) -> str:
    p = Path(path)
    stem = _DATE_TOKEN_RE.sub(" ", _STATUS_TOKEN_RE.sub(" ", _REV_TOKEN_RE.sub(" ", p.stem)))
    stem = re.sub(r"[_\-\s]+", "_", stem).strip("_").lower()
    return ("" if p.parent == Path(".") else p.parent.as_posix()) + "/" + stem


def _rev_key(rev: str) -> tuple[int, str]:
    m = _REV_PARSE_RE.match(rev)
    return (int(m.group(1)), m.group(2).lower()) if m else (-1, "")


def _apply_families(files: list[FileEntry]) -> None:
    """Fill family/superseded_by/eligible: the newest non-withdrawn revision wins its family."""
    fams: dict[str, list[FileEntry]] = {}
    for f in files:
        f.family = _family(f.file)
        fams.setdefault(f.family, []).append(f)
    for members in fams.values():
        revised = [f for f in members if f.revision]
        live = [f for f in revised if not f.withdrawn]
        if len(revised) >= 2 and live:
            current = max(live, key=lambda f: _rev_key(f.revision))
            for f in revised:
                if f is not current and _rev_key(f.revision) < _rev_key(current.revision):
                    f.superseded_by = current.file
        # a DRAFT loses to a non-draft file of its family; a lone draft stays eligible
        final = next((f for f in members if not f.draft and f.status == "ok" and not f.withdrawn
                      and f.superseded_by is None), None)
        if final is not None:
            for f in members:
                if f.draft and f.superseded_by is None:
                    f.superseded_by = final.file
    for f in files:
        f.eligible = f.status == "ok" and not f.withdrawn and f.superseded_by is None


def _name_text(path: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[/_\-.]", " ", path)).strip()


# ---------------------------------------------------------------- Index

class Index:
    """In-memory view over files, records, ID index and BM25."""

    def __init__(self, files: list[FileEntry], records: list[Record], ids: dict[str, dict] | None = None):
        self.files = files
        self.file_by_path = {f.file: f for f in files}
        self.records = records
        self.ids = ids if ids is not None else build_ids(records)
        self.bm25 = BM25([textnorm.tokens(r.text) for r in records])
        self.by_file: dict[str, list[int]] = {}
        for r in records:
            self.by_file.setdefault(r.file, []).append(r.rid)

    def record(self, rid: int) -> Record:
        return self.records[rid]

    def file_of(self, rid: int) -> FileEntry:
        return self.file_by_path[self.records[rid].file]

    def eligible(self, rid: int) -> bool:
        """Record usable as support: its file is eligible and the record is not stale."""
        return self.file_of(rid).eligible and not self.records[rid].stale

    def rare_ids_in(self, text: str) -> list[str]:
        """IDs of text (as written) that exist in the corpus with df <= RARE_DF."""
        out = []
        for raw in textnorm.ids(text):
            e = self.ids.get(textnorm.nid(raw))
            if e and e["df"] <= config.RARE_DF:
                out.append(raw)
        return out

    def has_id(self, raw: str) -> bool:
        return textnorm.nid(raw) in self.ids

    def files_with_id(self, raw: str) -> list[str]:
        e = self.ids.get(textnorm.nid(raw))
        return list(e["files"]) if e else []

    def _is_heading(self, r: Record) -> bool:
        return r.loc == "header" or r.loc.endswith(".header") or bool(_HEADING_RE.match(r.text))

    def context_rids(self, rid: int, k: int = 3) -> list[int]:
        """Name record, first k records and nearest preceding heading/header (<=5 back) of rid's file."""
        r = self.records[rid]
        rids = self.by_file[r.file]
        out = [x for x in rids if self.records[x].meta][:1]
        out += [x for x in rids if not self.records[x].meta][:k]
        for back in range(1, 6):
            p = rid - back
            if p < 0 or self.records[p].file != r.file:
                break
            if self._is_heading(self.records[p]):
                out.append(p)
                break
        return list(dict.fromkeys(out))


def build(parsed: list[dict], ocr: dict[str, dict] | None = None) -> Index:
    """Index from parse.walk output and optional OCR results {file: {"lines": [...], "pairs": [...]}}."""
    ocr = ocr or {}
    records: list[Record] = []
    files: list[FileEntry] = []

    def add(file: str, loc: str, text: str, src: str, stale: bool = False, meta: bool = False) -> None:
        records.append(Record(len(records), file, loc, text, stale=stale, meta=meta, src=src))

    for d in parsed:
        start = len(records)
        add(d["path"], "name", _name_text(d["path"]), "name", meta=True)
        for r in d["records"]:
            add(d["path"], r["loc"], r["text"], "text", stale=bool(r.get("stale")))
        o = ocr.get(d["path"])
        if o:
            for i, line in enumerate(o.get("lines", []), 1):
                if str(line).strip():
                    add(d["path"], f"ocr.L{i}", str(line), "ocr")
            if o.get("pairs"):
                add(d["path"], "ocr.pairs", "; ".join(str(p) for p in o["pairs"]), "ocr")
        fl = d.get("flags", {})
        n_prod = {textnorm.nid(x) for x in textnorm.ids(d.get("text", "")) if _PRODUCT_RE.match(x)}
        files.append(FileEntry(
            file=d["path"], kind=d["kind"], status=d["status"], withdrawn=bool(fl.get("withdrawn")),
            revision=fl.get("revision"), supersedes=list(fl.get("supersedes", [])),
            images=[list(im) for im in d.get("images", [])], rid_range=[start, len(records)],
            n_products=len(n_prod), draft=bool(fl.get("draft"))))
    _apply_families(files)
    return Index(files, records)


# ---------------------------------------------------------------- persistence

def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _records_jsonl(records: list[Record]) -> str:
    return "".join(json.dumps(r.to_json(), ensure_ascii=True) + "\n" for r in records)


def save(index_dir: Path, idx: Index) -> None:
    """Write files.json, records.jsonl and ids.json atomically."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(index_dir / "files.json", json.dumps([f.to_json() for f in idx.files], ensure_ascii=True))
    _write_atomic(index_dir / "records.jsonl", _records_jsonl(idx.records))
    _write_atomic(index_dir / "ids.json", json.dumps(idx.ids, ensure_ascii=True))


def _load_records(index_dir: Path) -> list[Record]:
    with open(index_dir / "records.jsonl", encoding="utf-8") as fh:
        return [Record.from_json(json.loads(line)) for line in fh if line.strip()]


def load(index_dir: Path) -> Index:
    """Rebuild an Index from disk (BM25 always; ids.json too when it is missing)."""
    index_dir = Path(index_dir)
    files = [FileEntry.from_json(d) for d in json.loads((index_dir / "files.json").read_text("utf-8"))]
    records = _load_records(index_dir)
    ids_path = index_dir / "ids.json"
    ids = json.loads(ids_path.read_text("utf-8")) if ids_path.exists() else None
    return Index(files, records, ids)


def append_records(index_dir: Path, new_records: list[Record]) -> list[int]:
    """Append records (rids reassigned to the end), update ids.json, return the new rids."""
    index_dir = Path(index_dir)
    if not new_records:
        return []
    existing = _load_records(index_dir)
    fresh = [replace(r, rid=len(existing) + i) for i, r in enumerate(new_records)]
    ids_path = index_dir / "ids.json"
    ids = json.loads(ids_path.read_text("utf-8")) if ids_path.exists() else build_ids(existing)
    _add_ids(ids, fresh)
    with open(index_dir / "records.jsonl", "a", encoding="utf-8") as fh:
        fh.write(_records_jsonl(fresh))
        fh.flush()
        os.fsync(fh.fileno())
    _write_atomic(ids_path, json.dumps(ids, ensure_ascii=True))
    return [r.rid for r in fresh]
