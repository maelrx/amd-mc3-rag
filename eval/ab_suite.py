#!/usr/bin/env python3
"""A/B several hypothesis-flag configs against a baseline in ONE process, with one model load and a reader cache.

    python3 eval/ab_suite.py --suites sample,s1,s2 --configs configs.json --reader real --model DIR \\
        --out OUT --cache CACHE [--soft 22] [--limit N] [--group TUNE=sample,s1 --group HOLDOUT=h101]

``--configs`` is a JSON file or an inline JSON list: ``[{"name": "v5", "env": {}}, {"name": "rank",
"env": {"MC3_RANK": "cov_first"}}]``. The FIRST config is the baseline. Env names are mapped onto attributes of the
already imported ``config`` module (restored after each config); unknown names fail fast.

Reader calls (read / vqa / logprob / transcribe) are memoised by ``CachedReader`` across configs and suites, keyed by
method + sha1(prompt, question, image names + pixel hashes, token budgets, reader-relevant config) and persisted as one
JSON file per key under --cache, so a second process of the same Colab session reuses them. Outputs cut by the
deadline are never cached. Cache hits are instant, so per-question timings add the original latency back
(``seconds_est``, ``p50_est``...; the table shows those).

Writes OUT/<config>/<suite>/{query_NN_output.json,report.json} (run_suite format) and OUT/ab_summary.json.
Index dirs live in OUT/_index/<suite>-<flags signature> and are shared by configs with the same index-affecting flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "app", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
import pipeline  # noqa: E402
from budget import Deadline  # noqa: E402
from eval import run_suite  # noqa: E402

CACHED_METHODS = ("read", "vqa", "logprob", "transcribe")
CUT_MARGIN_S = 0.5            # a call that ended within this of its deadline is treated as cut by it
INDEX_FLAGS = ("XLSX_JOIN", "PDF_COLS", "RARE_DF")   # config attrs that change the index build
# config attrs that change what the reader sees/produces without being call arguments
READER_CONFIG_ATTRS = ("IMAGE_LONGEST_EDGE_READ", "IMAGE_LONGEST_EDGE_TRANSCRIBE", "TRANSCRIBE_MAX_NEW_TOKENS",
                       "MAX_NEW_TOKENS", "CHARS_PER_TOKEN")


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------- config application

def _bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _explicit_knobs() -> dict[str, tuple[str, type]]:
    return {"MC3_XLSX_JOIN": ("XLSX_JOIN", bool), "MC3_PDF_COLS": ("PDF_COLS", bool), "MC3_RANK": ("RANK", str),
            "MC3_CHAIN_GUARD": ("CHAIN_GUARD", bool), "MC3_GATE_B": ("GATE_B_MODE", str),
            "MC3_GATE_B_T1": ("GATE_B_THETA1", float), "MC3_GATE_B_T2": ("GATE_B_THETA2", float),
            "MC3_WARRANT": ("WARRANT", str), "MC3_LABEL_STRIP": ("LABEL_STRIP", bool),
            "MC3_DATE_COND": ("DATE_COND", bool)}


def _numeric_knobs() -> dict[str, tuple[str, type]]:
    """Scalar numeric knobs of app/config.py (``X = _f("MC3_X", ..)`` / ``_i``), found in its source."""
    src = Path(config.__file__).read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r'^([A-Z][A-Z0-9_]*)\s*=\s*_([fi])\("(MC3_[A-Z0-9_]+)"', src, re.M):
        out[m.group(3)] = (m.group(1), float if m.group(2) == "f" else int)
    return out


def knob_table() -> dict[str, tuple[str, type]]:
    t = _numeric_knobs()
    t.update(_explicit_knobs())
    return t


def resolve_env(env: dict) -> dict[str, object]:
    """{"MC3_X": "v"} -> {config attr: typed value}. Raises ValueError on an unknown name or bad value."""
    table = knob_table()
    attrs: dict[str, object] = {}
    for name, raw in (env or {}).items():
        if name not in table:
            raise ValueError(f"unknown knob {name!r} (known: {sorted(table)})")
        attr, typ = table[name]
        try:
            val = _bool(raw) if typ is bool else typ(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad value for {name}: {raw!r}") from e
        attrs[attr] = val
        if attr == "GATE_B_MODE":
            attrs["GATE_B"] = (val == "1")
    return attrs


class applied:
    """Context manager: set config attrs (and os.environ) for one config, restore everything on exit."""

    def __init__(self, env: dict):
        self.env = dict(env or {})
        self.attrs = resolve_env(self.env)

    def __enter__(self):
        self._attr_old = {k: getattr(config, k) for k in self.attrs if hasattr(config, k)}
        self._attr_new = [k for k in self.attrs if not hasattr(config, k)]
        self._env_old = {k: os.environ.get(k) for k in self.env}
        for k, v in self.attrs.items():
            setattr(config, k, v)
        for k, v in self.env.items():
            os.environ[k] = str(v)
        return self

    def __exit__(self, *exc):
        for k, v in self._attr_old.items():
            setattr(config, k, v)
        for k in self._attr_new:
            if hasattr(config, k):
                delattr(config, k)
        for k, v in self._env_old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def index_signature() -> str:
    """Current values of the index-affecting flags, e.g. ``XLSX_JOIN=0,PDF_COLS=1,RARE_DF=5``."""
    return ",".join(f"{k}={int(v) if isinstance(v, bool) else v}"
                    for k in INDEX_FLAGS if (v := getattr(config, k, None)) is not None)


def index_dir_name(suite: str) -> str:
    sig = index_signature()
    return f"{suite}-{re.sub(r'[^A-Za-z0-9=,._-]+', '_', sig).replace(',', '_').replace('=', '')}"


# ----------------------------------------------------------------------------- reader cache

class Cache:
    """Disk-backed memo (one JSON file per key) + hit/miss counters. Shared by every CachedReader of a run."""

    def __init__(self, directory: str | Path | None):
        self.dir = Path(directory) if directory else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self.skipped = 0           # computed but not cached (cut by the deadline, error, empty)
        self.saved_s = 0.0         # original latency of the calls served from cache
        self._lock = threading.Lock()

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "skipped": self.skipped,
                "saved_s": round(self.saved_s, 1)}

    def _path(self, key: str) -> Path | None:
        return self.dir / f"{key}.json" if self.dir else None

    def get(self, key: str) -> dict | None:
        with self._lock:
            ent = self.mem.get(key)
        if ent is None and (p := self._path(key)) is not None and p.exists():
            try:
                ent = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                ent = None
            if ent is not None:
                with self._lock:
                    self.mem[key] = ent
        return ent

    def put(self, key: str, ent: dict) -> None:
        with self._lock:
            self.mem[key] = ent
        if (p := self._path(key)) is not None:
            tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
            try:
                tmp.write_text(json.dumps(ent, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, p)
            except OSError:
                pass


def _image_identity(img) -> str:
    """Name (for ``(file, PIL)`` pairs) + mode/size + sha1 of the pixel bytes; anything else by repr."""
    name = None
    if isinstance(img, (tuple, list)) and len(img) == 2:
        name, img = str(img[0]), img[1]
    try:
        h = hashlib.sha1(img.tobytes()).hexdigest()
        return f"{name}|{img.mode}|{img.size[0]}x{img.size[1]}|{h}"
    except Exception:  # noqa: BLE001 - not a PIL image
        return f"{name}|{img!r}"


def _norm_arg(v):
    """JSON-able, stable form of one call argument for hashing."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        if v and all(hasattr(x, "tobytes") or (isinstance(x, (tuple, list)) and len(x) == 2
                                              and hasattr(x[1], "tobytes")) for x in v):
            return ["img:" + _image_identity(x) for x in v]
        if isinstance(v, tuple) and len(v) == 2 and hasattr(v[1], "tobytes"):
            return "img:" + _image_identity(v)
        return [_norm_arg(x) for x in v]
    if hasattr(v, "tobytes"):
        return "img:" + _image_identity(v)
    return repr(v)


class CachedReader:
    """Memoising wrapper around a reader. ``has_vision`` is forced true (pipeline.has_vision opts in by attribute,
    the class name is no longer ``Reader``); every other attribute (force_math, timings, family, ...) is forwarded.
    Methods the inner reader lacks stay missing (``__getattr__`` falls through to it)."""

    has_vision = True

    def __init__(self, inner, cache: Cache, salt: str = ""):
        self._inner = inner
        self._cache = cache
        self._salt = salt

    def __getattr__(self, name: str):
        if name.startswith("__") or name in ("_inner", "_cache", "_salt"):
            raise AttributeError(name)
        attr = getattr(self._inner, name)            # AttributeError if the inner reader lacks it
        if name in CACHED_METHODS and callable(attr):
            return lambda *a, **kw: self._call(name, attr, a, kw)
        return attr

    def __setattr__(self, name, value):
        if name in ("_inner", "_cache", "_salt") or name == "has_vision":
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    def _key(self, name: str, fn, args: tuple, kwargs: dict) -> str | None:
        import inspect
        try:
            ba = inspect.signature(fn).bind(*args, **kwargs)
            ba.apply_defaults()
            params = {k: _norm_arg(v) for k, v in ba.arguments.items() if k != "deadline"}
        except (TypeError, ValueError):
            params = {"args": [_norm_arg(a) for a in args],
                      "kwargs": {k: _norm_arg(v) for k, v in kwargs.items() if k != "deadline"}}
        rc = {k: getattr(config, k, None) for k in READER_CONFIG_ATTRS}
        payload = json.dumps([self._salt, name, params, rc], sort_keys=True, ensure_ascii=False, default=repr)
        return f"{name}-{hashlib.sha1(payload.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _deadline_of(fn, args, kwargs):
        if "deadline" in kwargs:
            return kwargs["deadline"]
        try:
            import inspect
            return inspect.signature(fn).bind(*args, **kwargs).arguments.get("deadline")
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _encode(name: str, value):
        if name == "read":
            from dataclasses import asdict
            return asdict(value)
        return value

    @staticmethod
    def _decode(name: str, value):
        if name == "read":
            from schema import ReaderOut
            return ReaderOut(**value)
        if name == "logprob":
            return float(value)
        return value

    @staticmethod
    def _cacheable(name: str, value) -> bool:
        if name == "read":
            return not str(getattr(value, "raw", "")).startswith("ERROR")
        if name in ("vqa", "transcribe"):
            return bool(str(value).strip())          # "" is what the reader returns on error or cut
        return isinstance(value, float)

    def _call(self, name: str, fn, args: tuple, kwargs: dict):
        cache = self._cache
        key = self._key(name, fn, args, kwargs)
        ent = cache.get(key)
        if ent is not None:
            with cache._lock:
                cache.hits += 1
                cache.saved_s += float(ent.get("elapsed", 0.0))
            return self._decode(name, ent["value"])
        deadline = self._deadline_of(fn, args, kwargs)
        rem0 = deadline.remaining() if hasattr(deadline, "remaining") else None
        t0 = time.perf_counter()
        value = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        cut = rem0 is not None and (rem0 <= 0 or deadline.remaining() < CUT_MARGIN_S)
        with cache._lock:
            cache.misses += 1
            ok = not cut and self._cacheable(name, value)
            if not ok:
                cache.skipped += 1
        if ok:
            cache.put(key, {"method": name, "elapsed": round(elapsed, 3), "value": self._encode(name, value)})
        return value


# ----------------------------------------------------------------------------- suites, configs, groups

def resolve_suites(names: list[str], root: Path = ROOT) -> list[tuple[str, Path, Path]]:
    """Like gpu/colab_gpu.py suite_list(): ``sample`` = sample-corpus + sample-questions.json, else
    eval/suites/<name>/{corpus|mc3-corpus|.} + questions.json."""
    out = []
    for n in names:
        if n == "sample":
            out.append((n, root / "sample-corpus", root / "sample-questions.json"))
            continue
        d = root / "eval" / "suites" / n
        if not (d / "questions.json").exists():
            raise SystemExit(f"suite {n!r}: {d / 'questions.json'} not found")
        corpus = next((d / c for c in ("corpus", "mc3-corpus") if (d / c).is_dir()), d)
        out.append((n, corpus, d / "questions.json"))
    return out


def load_configs(spec: str) -> list[dict]:
    text = spec
    if not spec.lstrip().startswith(("[", "{")):
        text = Path(spec).read_text(encoding="utf-8")
    cfgs = json.loads(text)
    if isinstance(cfgs, dict):
        cfgs = [cfgs]
    names = [c.get("name") for c in cfgs]
    if not cfgs or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", n) for n in names) \
            or len(set(names)) != len(names):
        raise SystemExit(f"configs need unique names matching [A-Za-z0-9._-]+ (got {names})")
    for c in cfgs:
        c.setdefault("env", {})
        resolve_env(c["env"])                       # fail fast on typos
    return cfgs


def parse_groups(items: list[str] | None) -> dict[str, list[str]]:
    groups = {}
    for it in items or []:
        for part in it.split(";"):
            if part.strip():
                name, _, rest = part.partition("=")
                groups[name.strip()] = [s.strip() for s in rest.split(",") if s.strip()]
    return groups


# ----------------------------------------------------------------------------- summary

def _cell(rep: dict) -> dict:
    s = rep.get("seconds", {})
    return {"total": rep["total"], "max": rep["max"],
            "answerable_correct": rep["answerable_correct"], "answerable": rep["answerable"],
            "canaries": rep["canaries"], "false_answer_ids": list(rep["false_answer_ids"]),
            "p50": s.get("p50_est", s.get("p50", 0.0)), "p95": s.get("p95_est", s.get("p95", 0.0)),
            "max_s": s.get("max_est", s.get("max", 0.0))}


def _delta(base: dict, rep: dict) -> dict:
    bp = {r["id"]: r["pts"] for r in base["rows"]}
    cp = {r["id"]: r["pts"] for r in rep["rows"]}
    gained = sorted(i for i, p in cp.items() if p and not bp.get(i))
    lost = sorted(i for i, p in bp.items() if p and not cp.get(i))
    new_fa = sorted(set(rep["false_answer_ids"]) - set(base["false_answer_ids"]))
    return {"gained": gained, "lost": lost, "new_false_answers": new_fa,
            "net_questions": (rep["total"] - base["total"]) // 20}


def summarize(reports: dict[str, dict[str, dict]], groups: dict[str, list[str]] | None = None,
              cache: dict | None = None) -> dict:
    """reports: {config: {suite: report.json dict}}; the first config is the baseline."""
    names = list(reports)
    base = names[0]
    out: dict = {"baseline": base, "configs": {}, "groups": {}}
    for cfg in names:
        entry = {"suites": {}}
        for suite, rep in reports[cfg].items():
            cell = _cell(rep)
            if cfg != base and suite in reports[base]:
                cell["vs_baseline"] = _delta(reports[base][suite], rep)
            entry["suites"][suite] = cell
        out["configs"][cfg] = entry
        for gname, members in (groups or {}).items():
            have = [s for s in members if s in reports[cfg]]
            if not have:
                continue
            g = {"suites": have, "total": sum(reports[cfg][s]["total"] for s in have),
                 "max": sum(reports[cfg][s]["max"] for s in have),
                 "answerable_correct": sum(reports[cfg][s]["answerable_correct"] for s in have),
                 "answerable": sum(reports[cfg][s]["answerable"] for s in have),
                 "false_answers": sum(len(reports[cfg][s]["false_answer_ids"]) for s in have),
                 "canaries": sum(reports[cfg][s]["canaries"] for s in have)}
            if cfg != base:
                ds = [_delta(reports[base][s], reports[cfg][s]) for s in have if s in reports[base]]
                g["vs_baseline"] = {"gained": sum(len(d["gained"]) for d in ds), "lost": sum(len(d["lost"]) for d in ds),
                                    "new_false_answers": sum(len(d["new_false_answers"]) for d in ds),
                                    "net_questions": sum(d["net_questions"] for d in ds)}
            out["groups"].setdefault(gname, {})[cfg] = g
    if cache is not None:
        out["cache"] = cache
    return out


def _ids(ids: list[str]) -> str:
    return f"{len(ids)}" + (f" {[i.replace('query_', 'q') for i in ids]}" if ids else "")


def format_table(summary: dict) -> str:
    L = []
    base = summary["baseline"]
    L.append(f"A/B summary, baseline = {base}")
    for cfg, entry in summary["configs"].items():
        L.append(f"\n== {cfg}" + ("  (baseline)" if cfg == base else ""))
        L.append(f"{'suite':<8}{'score':>10}{'answ':>9}  {'false answers':<28}{'p50':>6}{'p95':>6}{'max':>6}  vs baseline")
        for suite, c in entry["suites"].items():
            fa = f"{len(c['false_answer_ids'])}/{c['canaries']} {c['false_answer_ids']}" if c["false_answer_ids"] \
                else f"0/{c['canaries']}"
            d = c.get("vs_baseline")
            vs = ""
            if d:
                vs = f"net {d['net_questions']:+d}  gained {_ids(d['gained'])} lost {_ids(d['lost'])}"
                if d["new_false_answers"]:
                    vs += f"  NEW FALSE {d['new_false_answers']}"
            L.append(f"{suite:<8}{c['total']:>5}/{c['max']:<4}{c['answerable_correct']:>4}/{c['answerable']:<4}  "
                     f"{fa:<28.28}{c['p50']:>6.1f}{c['p95']:>6.1f}{c['max_s']:>6.1f}  {vs}")
    for gname, per in summary["groups"].items():
        L.append(f"\n== group {gname}")
        for cfg, g in per.items():
            vs = ""
            if "vs_baseline" in g:
                d = g["vs_baseline"]
                vs = f"  net {d['net_questions']:+d} (+{d['gained']} -{d['lost']}) new false answers {d['new_false_answers']}"
            L.append(f"{cfg:<20}{g['total']:>5}/{g['max']:<5} answerable {g['answerable_correct']}/{g['answerable']}"
                     f"  false answers {g['false_answers']}/{g['canaries']}{vs}")
    if "cache" in summary:
        c = summary["cache"]
        L.append(f"\ncache: {c.get('hits', 0)} hits, {c.get('misses', 0)} misses, {c.get('skipped', 0)} not cached, "
                 f"{c.get('saved_s', 0)}s of reader time saved")
    return "\n".join(L)


# ----------------------------------------------------------------------------- runner

def _build_reader(kind: str, data: dict, model: str | None):
    if kind == "real":
        return run_suite.make_reader("real", data, model)
    return run_suite.make_reader(kind, data, None)


def run_ab(suites: list[tuple[str, Path, Path]], configs: list[dict], reader_kind: str, model: str | None,
           out: Path, cache_dir: str | Path | None, soft: float = 22.0, limit: int | None = None,
           groups: dict[str, list[str]] | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(cache_dir)
    real = None
    if reader_kind == "real":
        t0 = time.time()
        real = _build_reader("real", {}, model)
        log(f"[ab] model loaded in {time.time() - t0:.0f}s")
    model_salt = f"real:{Path(model or config.MODEL_DIR).name}" if real is not None else reader_kind
    datas, readers = {}, {}
    for name, _corpus, qpath in suites:
        data = json.loads(Path(qpath).read_text(encoding="utf-8"))
        if limit:
            data = dict(data, queries=data["queries"][:limit])
        datas[name] = data
        inner = real if real is not None else _build_reader(reader_kind, data, None)
        salt = model_salt if real is not None else f"{model_salt}:{name}"   # stub gold differs per suite
        readers[name] = CachedReader(inner, cache, salt)
    indexes: dict[str, tuple] = {}
    reports: dict[str, dict[str, dict]] = {}
    t_start = time.time()
    for cfg in configs:
        cname = cfg["name"]
        reports[cname] = {}
        log(f"[ab] === config {cname} {cfg.get('env', {})}")
        with applied(cfg.get("env", {})):
            for name, corpus, qpath in suites:
                data, rd = datas[name], readers[name]
                c0 = cache.stats()
                ikey = index_dir_name(name)
                if ikey not in indexes:
                    t0 = time.time()
                    idx_dir = out / "_index" / ikey
                    idx = pipeline.index_corpus(corpus, idx_dir, rd, Deadline.after(config.INDEX_BUDGET_S))
                    indexes[ikey] = (idx, time.time() - t0)
                    log(f"[ab] {cname}/{name}: indexed {len(idx.files)} files, {len(idx.records)} records "
                        f"in {indexes[ikey][1]:.1f}s -> {ikey}")
                idx, t_index = indexes[ikey]
                odir = out / cname / name
                n = len(data["queries"])
                done = [0]

                def tick(qid, row, cname=cname, name=name, n=n, done=done):
                    done[0] += 1
                    if done[0] % 5 == 0 or done[0] == n:
                        log(f"[ab] {cname}/{name}: {done[0]}/{n} (last {qid} {row['seconds']:.1f}s {row['reason']})")

                rep = run_suite.run_questions(corpus, qpath, data, odir, rd, idx, t_index, reader_kind, soft,
                                              verbose=False, extra_seconds=lambda: cache.saved_s, on_question=tick)
                reports[cname][name] = rep
                c1 = cache.stats()
                log(f"[ab] {cname}/{name}: {rep['total']}/{rep['max']} answerable "
                    f"{rep['answerable_correct']}/{rep['answerable']} false answers {rep['false_answers']} "
                    f"{rep['false_answer_ids']} | cache +{c1['hits'] - c0['hits']} hits "
                    f"+{c1['misses'] - c0['misses']} misses | elapsed {time.time() - t_start:.0f}s")
    summary = summarize(reports, groups, cache.stats())
    summary["wall_seconds"] = round(time.time() - t_start, 1)
    summary["reader"] = reader_kind
    summary["suites"] = [s[0] for s in suites]
    (out / "ab_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    log(format_table(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suites", required=True, help="comma list: sample, s1, h101, ...")
    ap.add_argument("--configs", required=True, help="JSON file or inline JSON list; first = baseline")
    ap.add_argument("--reader", default="real", choices=["real", "oracle", "lexical", "adversarial"])
    ap.add_argument("--model", default=None, help="model dir for --reader real (default MC3_MODEL)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None, help="reader cache dir (default: not persisted)")
    ap.add_argument("--soft", type=float, default=config.ANSWER_SOFT_S)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--group", action="append", default=[], help="NAME=a,b (repeatable; ';' also separates groups)")
    args = ap.parse_args(argv)
    suites = resolve_suites([s.strip() for s in args.suites.split(",") if s.strip()])
    configs = load_configs(args.configs)
    run_ab(suites, configs, args.reader, args.model, args.out, args.cache, args.soft, args.limit,
           parse_groups(args.group))
    return 0


if __name__ == "__main__":
    sys.exit(main())
