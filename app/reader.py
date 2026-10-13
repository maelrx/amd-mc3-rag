"""Qwen3-VL reader: evidence package + question -> ``ReaderOut`` (spec 5.6, 5.7).

Reader protocol (what ``pipeline``/``server``/``vision`` rely on; ``eval/stub_reader.StubReader``
implements the same surface): ``load()``, ``self_test_sdpa() -> bool``, ``warmup()``,
``read(package_text, question, images, deadline, max_new_tokens) -> ReaderOut``,
``vqa(question, image, deadline, max_new_tokens) -> str``,
``transcribe(image, deadline, max_new_tokens) -> str``,
``logprob(package_text, question, images, answer) -> float``.
Optional (H6, ``warrant.check`` skips readers without it): ``yes_no(evidence_text, question, answer, images, deadline)
-> float | None`` = log P(YES) - log P(NO) of the first reply token.

``import reader`` is safe without torch/transformers/PIL: every heavy import is inside a function.
The model is greedy, batch=1, bf16; every forward/generate runs under ``torch.inference_mode()`` and,
when the SDPA self-test failed, under ``sdpa_kernel(SDPBackend.MATH)``.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import time

import config
from budget import Deadline, make_deadline_criteria, make_tail_string_criteria
from schema import ReaderOut

PREFILL = '{"answer":"'

SYSTEM = (
    "You are an extraction engine. You are given numbered evidence records, optionally with attached images, "
    "and one question. Use ONLY the numbered records and attached images; never use outside knowledge and "
    "never guess. Reply with a single JSON object and nothing else.\n"
    "Format examples (illustrative values only):\n"
    '{"answer":"4.3.2","support":["R12"],"bridge":["R7"]}\n'
    '{"answer":"","support":[],"bridge":[]}'
)

RULES = (
    '1. "answer" is the value only, copied verbatim from one record (number, part number, version, quarter, '
    "code). Keep identifying qualifiers (Q3 FY27, REV-C2). No units, no sentence.\n"
    "2. Every condition in the question (product, variant, quantity, unit, time) must hold in the SAME record "
    "or its header. A record matching only some conditions is not the answer.\n"
    "3. In code, a constant or default literal beats a comment that mentions another value.\n"
    "4. If one record supplies an identifier (ticket, error code, part number) and another supplies the value "
    'for it, list the first in "bridge" and the second in "support".\n'
    '5. If no record contains the value, answer "" with empty lists.'
)

OUTPUT_LINE = 'Output exactly: {"answer":"<value>","support":["R.."],"bridge":["R.."]}'

TRANSCRIBE_PROMPT = (
    "Transcribe this image exactly. Plain text, one item per line: every visible text verbatim; tables as "
    "`col=value; col=value` with header names; labels/nameplates as `FIELD: VALUE`; diagrams and pin maps as "
    "`PIN: SIGNAL` for every box or pin connected to or next to a label. Do not add, summarise or interpret. "
    "Stop when everything is transcribed."
)

_ANSWER_START_RE = re.compile(r'"answer"\s*:\s*"')
_SUPPORT_RE = re.compile(r'"support"\s*:\s*\[(.*?)(?:\]|$)', re.S)
_BRIDGE_RE = re.compile(r'"bridge"\s*:\s*\[(.*?)(?:\]|$)', re.S)
_RID_RE = re.compile(r"R(\d+)", re.I)
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", '"': '"', "\\": "\\", "/": "/"}


def _log(msg: str) -> None:
    print(f"reader: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# pure logic (no torch): parsing and prompt assembly
# ---------------------------------------------------------------------------------------------

def _rids(pattern: re.Pattern, text: str) -> list[int]:
    m = pattern.search(text)
    if not m:
        return []
    seen: list[int] = []
    for x in _RID_RE.findall(m.group(1)):
        v = int(x)
        if v not in seen:
            seen.append(v)
    return seen


def parse_output(raw: str) -> tuple[str, list[int], list[int]]:
    """Tolerant parse of ``{"answer":"..","support":["R..",..],"bridge":[..]}``, also when truncated.

    answer: text after the ``"answer":"`` prefill up to the next unescaped ``"``; with no closing quote
    it runs to the end (or to a ``,"support"`` marker) minus trailing ``}``/``,``/space.
    support/bridge: ints from ``R(\\d+)`` inside the respective brackets; missing -> [].
    """
    raw = raw or ""
    m = _ANSWER_START_RE.search(raw)
    if not m:
        return "", [], []
    i, n = m.end(), len(raw)
    buf: list[str] = []
    closed = False
    while i < n:
        c = raw[i]
        if c == "\\" and i + 1 < n:
            buf.append(_ESCAPES.get(raw[i + 1], raw[i + 1]))
            i += 2
            continue
        if c == '"':
            closed = True
            break
        buf.append(c)
        i += 1
    answer = "".join(buf)
    if not closed:
        cut = re.search(r',\s*"?support', answer)
        if cut:
            answer = answer[:cut.start()]
        answer = answer.rstrip(" \t\r\n,}")
    rest = raw[i + 1:] if closed else ""
    return answer.strip(), _rids(_SUPPORT_RE, rest), _rids(_BRIDGE_RE, rest)


def _split_images(images) -> tuple[list, list[str | None]]:
    """Accept PIL images or ``(file, PIL.Image)`` pairs (as in ``Package.images``)."""
    pil, names = [], []
    for it in images or []:
        if isinstance(it, (tuple, list)) and len(it) == 2:
            names.append(str(it[0]))
            pil.append(it[1])
        else:
            names.append(None)
            pil.append(it)
    return pil, names


def image_note(names: list[str | None]) -> str:
    return " ".join(f"<IMAGE {i} = {n}>" if n else f"<IMAGE {i} attached>" for i, n in enumerate(names, 1))


def build_messages(package_text: str, question: str, images: list) -> list[dict]:
    """System + user (images first, then EVIDENCE/QUESTION/RULES) + assistant prefill (spec 5.7)."""
    pil, names = _split_images(images)
    user_text = ("EVIDENCE\n" + package_text + ("\n" + image_note(names) if pil else "")
                 + f"\n\nQUESTION: {question}\n\nRULES\n" + RULES + "\n" + OUTPUT_LINE)
    content = [{"type": "image", "image": im} for im in pil] + [{"type": "text", "text": user_text}]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
            {"role": "assistant", "content": PREFILL}]


def _size_dict(edge: int) -> dict:
    # Qwen2VLImageProcessor (used by Qwen3-VL) reads size={"shortest_edge": min_pixels, "longest_edge": max_pixels};
    # both are pixel counts, not edge lengths. Checked in 5.5.4 and in the 5.17 sources under research/infra/tf.
    return {"shortest_edge": 65536, "longest_edge": int(edge)}


def _strip_pil(messages: list[dict]) -> list[dict]:
    """Drop PIL payloads so the chat template (which only counts image blocks) never copies pixel data."""
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = [{"type": "image"} if b.get("type") == "image" else b for b in c]
        out.append({"role": m["role"], "content": c})
    return out


# ---------------------------------------------------------------------------------------------
# model families (A/B of alternative VLMs; ``qwen3vl`` is the shipped default and its code path is frozen)
# ---------------------------------------------------------------------------------------------

FAMILIES = ("qwen3vl", "gemma4", "qwen35", "generic")
# Gemma 4 image processors accept only these ``max_soft_tokens`` budgets (image_processing_gemma4_unified.py)
GEMMA_SOFT_TOKENS = (70, 140, 280, 560, 1120)
GEMMA_PX_PER_SOFT_TOKEN = (16 * 3) ** 2     # patch_size 16 x pooling_kernel_size 3 -> 48x48 px per soft token


def detect_family(cfg: dict | None) -> str:
    """``config.json`` dict -> ``qwen3vl | gemma4 | qwen35 | generic``.

    A missing/empty config (or one with neither field) stays ``qwen3vl``: that is the pre-A/B behaviour.
    """
    cfg = cfg or {}
    archs = cfg.get("architectures") or []
    arch = str(archs[0]) if archs else ""
    mt = str(cfg.get("model_type") or "")
    if not arch and not mt:
        return "qwen3vl"
    if mt == "qwen3_vl" or arch == "Qwen3VLForConditionalGeneration":
        return "qwen3vl"
    if mt.startswith("gemma4") or arch.startswith("Gemma4"):
        return "gemma4"
    if mt.startswith("qwen3_5") or arch.startswith("Qwen3_5"):
        return "qwen35"
    return "generic"


def gemma_soft_tokens(edge: int) -> int:
    """Gemma ``max_soft_tokens`` bucket closest (log scale) to the same *pixel* budget the Qwen path gets.

    1_228_800 px (read) -> 560; 2_097_152 px (transcribe) -> 1120 (the largest bucket the processor accepts).
    """
    import math
    want = max(1.0, int(edge) / GEMMA_PX_PER_SOFT_TOKEN)
    return min(GEMMA_SOFT_TOKENS, key=lambda b: abs(math.log(b / want)))


def fold_system(messages: list[dict]) -> list[dict]:
    """For templates without a system role: prepend the system text to the first text block of the first user turn."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_text = messages[0]["content"]
    if isinstance(sys_text, list):
        sys_text = "\n".join(b.get("text", "") for b in sys_text if b.get("type") == "text")
    out, done = [], False
    for m in messages[1:]:
        if not done and m["role"] == "user":
            done = True
            c = m["content"]
            if isinstance(c, str):
                c = sys_text + "\n\n" + c
            else:
                c, hit = [dict(b) for b in c], False
                for b in c:
                    if b.get("type") == "text" and not hit:
                        b["text"], hit = sys_text + "\n\n" + b["text"], True
                if not hit:
                    c.append({"type": "text", "text": sys_text})
            m = {"role": "user", "content": c}
        out.append(m)
    return out


# ---------------------------------------------------------------------------------------------
# warrant check (H6): YES/NO verification of a proposed answer against its evidence
# ---------------------------------------------------------------------------------------------

WARRANT_SYSTEM = (
    "You verify extracted answers against evidence records. Use ONLY the evidence given; never use outside "
    "knowledge. Reply with exactly one word: YES or NO."
)
WARRANT_MIN_REMAINING_S = 3.0
_WARRANT_YES = ("YES", " YES", "Yes", " Yes", "yes", " yes")
_WARRANT_NO = ("NO", " NO", "No", " No", "no", " no")


def build_warrant_messages(evidence_text: str, question: str, answer: str, images=None) -> list[dict]:
    """System + user (images first, then EVIDENCE / QUESTION / PROPOSED ANSWER / the YES-NO ask); no assistant turn."""
    pil, names = _split_images(images)
    user_text = ("EVIDENCE\n" + evidence_text + ("\n" + image_note(names) if pil else "")
                 + f"\n\nQUESTION: {question}\nPROPOSED ANSWER: {answer}\n\n"
                 "Does the evidence explicitly state this value as the answer to exactly this question "
                 "(same entity/product, same attribute, same condition/unit/date)? Reply YES or NO.")
    content = [{"type": "image", "image": im} for im in pil] + [{"type": "text", "text": user_text}]
    return [{"role": "system", "content": WARRANT_SYSTEM}, {"role": "user", "content": content}]


def warrant_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Token ids of the YES / NO answers (first token of each spelling variant), disjoint from each other.

    A variant that splits into several tokens only counts when no single-token variant exists for that side."""
    def ids_of(variants) -> list[int]:
        single, first = [], []
        for v in variants:
            try:
                ids = list(tokenizer.encode(v, add_special_tokens=False))
            except Exception:  # noqa: BLE001 - tokenizer without encode kwargs
                continue
            if len(ids) == 1:
                single.append(int(ids[0]))
            elif ids:
                first.append(int(ids[0]))
        return list(dict.fromkeys(single or first))

    yes, no = ids_of(_WARRANT_YES), ids_of(_WARRANT_NO)
    both = set(yes) & set(no)
    return [i for i in yes if i not in both], [i for i in no if i not in both]


def warrant_margin(yes_lps: list[float], no_lps: list[float]) -> float | None:
    """log P(YES) - log P(NO): max over the spelling variants of each side; None when a side has no token."""
    if not yes_lps or not no_lps:
        return None
    return float(max(yes_lps) - max(no_lps))


# ---------------------------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------------------------

class Reader:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.processor = None
        self.model = None
        self.attn: str | None = None
        self.force_math = False
        self.timings: dict[str, float] = {}
        self._lock = threading.RLock()
        self._logits_kw: str | None | bool = False   # False = not probed yet
        self.family = "qwen3vl"                       # set by load() from config.json; see detect_family()
        self.sdpa_skipped: str | None = None          # reason, when self_test_sdpa() skips on purpose (not a crash)
        self._fold_system = False                     # generic family: template rejected the system role
        self._logged: set[str] = set()

    def _log_once(self, key: str, msg: str) -> None:
        if key not in self._logged:
            self._logged.add(key)
            _log(msg)

    def _read_config(self) -> dict:
        try:
            with open(os.path.join(self.model_dir, "config.json"), encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001 - no config -> shipped Qwen3-VL path
            _log(f"config.json unreadable ({type(e).__name__}: {e}); assuming qwen3vl")
            return {}

    # -- loading -------------------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor

        t0 = time.perf_counter()
        self.family = detect_family(self._read_config())
        if self.family == "qwen35" and os.environ.get("MC3_NO_FLA", "1") != "0":
            # Gated DeltaNet layers: run the pure-torch reference path (what a ROCm box runs), never fla/causal_conv1d
            # or hub kernels. Only effective before transformers' qwen3_5 modeling is first imported.
            os.environ.setdefault("USE_HUB_KERNELS", "NO")
            sys.modules["fla"] = None
            sys.modules["causal_conv1d"] = None
        if self.family == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration as model_cls
        else:
            from transformers import AutoModelForImageTextToText as model_cls
            if self.family == "qwen35" and self._read_config().get("model_type") == "qwen3_5":
                try:
                    from transformers import Qwen3_5ForConditionalGeneration as model_cls  # noqa: F811
                except ImportError:
                    pass
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        try:
            self.attn = "sdpa"
            self.model = model_cls.from_pretrained(
                self.model_dir, dtype=torch.bfloat16, device_map=self.device, attn_implementation="sdpa")
        except Exception as e:  # noqa: BLE001 - sdpa unsupported/broken on this backend
            _log(f"load with sdpa failed ({type(e).__name__}: {e}); retrying with eager")
            self.model = None
            import gc
            gc.collect()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            self.attn = "eager"
            self.model = model_cls.from_pretrained(
                self.model_dir, dtype=torch.bfloat16, device_map=self.device, attn_implementation="eager")
        self.model.eval()
        # the repo's generation_config.json turns sampling on
        gc_ = self.model.generation_config
        gc_.do_sample = False
        gc_.temperature = None
        gc_.top_p = None
        gc_.top_k = None
        gc_.repetition_penalty = 1.0
        self.force_math = False
        self._set_size(config.IMAGE_LONGEST_EDGE_READ)
        self.timings["load_s"] = time.perf_counter() - t0
        _log(f"loaded {self.model_dir} family={self.family} attn={self.attn} in {self.timings['load_s']:.1f}s")

    def _set_size(self, edge: int) -> None:
        ip = self.processor.image_processor
        if self.family in ("qwen3vl", "qwen35"):     # Qwen2VLImageProcessor: size = pixel counts
            ip.size = _size_dict(edge)
        elif hasattr(ip, "max_soft_tokens"):          # Gemma 4: discrete soft-token budget, same pixel budget
            ip.max_soft_tokens = gemma_soft_tokens(edge)
        else:
            self._log_once("noedge", f"family={self.family}: no known image budget knob on "
                                     f"{type(ip).__name__}; using processor defaults")

    @contextlib.contextmanager
    def _image_edge(self, edge: int | None):
        """Temporarily switch the image processor pixel budget; always restore the READ budget."""
        if edge is None:
            yield
            return
        self._set_size(edge)
        try:
            yield
        finally:
            self._set_size(config.IMAGE_LONGEST_EDGE_READ)

    def _ctx(self):
        if self.force_math:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            return sdpa_kernel(SDPBackend.MATH)
        return contextlib.nullcontext()

    # -- SDPA self-test ------------------------------------------------------------------------

    def _vision_embeds(self, pixel_values, grid):  # grid: image_grid_thw (Qwen) / image_position_ids (Gemma 4)
        """Vision-tower output as one flat tensor (image embeds + deepstack features if present)."""
        import torch
        if hasattr(self.model, "get_image_features"):
            out = self.model.get_image_features(pixel_values, grid)
        else:
            visual = getattr(getattr(self.model, "model", None), "visual", None) or self.model.visual
            out = visual(pixel_values.type(visual.dtype), grid_thw=grid, return_dict=True)
        parts = []

        def add(x):
            if x is None:
                return
            if isinstance(x, torch.Tensor):
                parts.append(x.flatten().float())
            else:
                for y in x:
                    add(y)

        if isinstance(out, torch.Tensor):
            add(out)
        elif isinstance(out, (tuple, list)):
            add(out[0])
        else:
            add(getattr(out, "pooler_output", None))
            add(getattr(out, "deepstack_features", None))
        return torch.cat(parts)

    def self_test_sdpa(self) -> bool:
        """Compare vision-tower output under default SDPA vs the MATH backend; cosine < 0.99 or NaN -> force_math.

        Returns True when the test ran. Never raises; on an exception returns False and leaves force_math alone.
        Non-Qwen families without a known vision-tower path skip: returns False with ``sdpa_skipped`` set.
        """
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image, ImageDraw
            from torch.nn.attention import SDPBackend, sdpa_kernel

            img = Image.new("RGB", (448, 448), "white")
            d = ImageDraw.Draw(img)
            d.rectangle([40, 40, 200, 160], outline="black", width=4)
            d.ellipse([240, 60, 400, 220], fill="red", outline="black")
            d.line([20, 300, 430, 380], fill="blue", width=6)
            d.text((60, 90), "REV-C2 B14 SN-4417", fill="black")
            d.text((60, 400), "PIN 7: THERM_ALERT#", fill="black")
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
            if self.family == "qwen3vl":
                text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = self.processor(text=[text], images=[img], return_tensors="pt")
            else:
                if type(self.model).__name__.startswith("Gemma4Unified"):
                    self.sdpa_skipped = "Gemma4Unified is encoder-free (no vision attention)"
                    _log(f"sdpa self-test skipped: {self.sdpa_skipped}")
                    return False
                inp = self._processor_inputs(self._render(msgs, prefill=None), [img])
            pv = inp["pixel_values"].to(self.model.device)
            gkey = next((k for k in ("image_grid_thw", "image_position_ids") if k in inp), None)
            if gkey is None:
                self.sdpa_skipped = f"family={self.family} has no known vision-tower inputs ({sorted(inp.keys())})"
                _log(f"sdpa self-test skipped: {self.sdpa_skipped}")
                return False
            grid = inp[gkey].to(self.model.device)
            with torch.inference_mode():
                a = self._vision_embeds(pv, grid)
                with sdpa_kernel(SDPBackend.MATH):
                    b = self._vision_embeds(pv, grid)
                nan = bool(torch.isnan(a).any() or torch.isnan(b).any())
                cos = float(F.cosine_similarity(a[None], b[None]).item())
            if nan or not cos >= 0.99:
                self.force_math = True
            _log(f"sdpa self-test cos={cos:.5f} nan={nan} force_math={self.force_math}")
            return True
        except Exception as e:  # noqa: BLE001
            _log(f"sdpa self-test failed to run: {type(e).__name__}: {e}")
            return False

    # -- generation plumbing -------------------------------------------------------------------

    def _encode(self, messages: list[dict], images: list, prefilled: bool = True):
        """Chat template -> processor -> tensors on the model device.

        ``prefilled=True`` keeps the final assistant turn open (``continue_final_message=True``: the template
        output ends right after the prefill, no ``<|im_end|>``); otherwise a generation prompt is appended.
        """
        pil, _ = _split_images(images)
        msgs = _strip_pil(messages)
        if self.family == "qwen3vl":                     # shipped path: do not touch
            if prefilled:
                text = self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True)
            else:
                text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=pil or None, return_tensors="pt")
            return inputs.to(self.model.device)
        # other families: render the generation prompt of everything before the final assistant turn
        # (so each template adds its own "thinking off" scaffold) and append the prefill verbatim.
        # continue_final_message is avoided: it round-trips the prefill through the template's
        # content filters (Gemma's strip_thinking/trim, 5.17's tag injection).
        prefill = ""
        if prefilled:
            if msgs[-1]["role"] != "assistant" or not isinstance(msgs[-1]["content"], str):
                raise ValueError("prefilled encode needs a final assistant message with string content")
            prefill, msgs = msgs[-1]["content"], msgs[:-1]
        text = self._render(msgs, prefill=prefill)
        return self._processor_inputs(text, pil).to(self.model.device)

    def _template_kwargs(self) -> dict:
        # Qwen3.5 thinks by default; its template reads enable_thinking and (False) emits an empty <think> block.
        # Gemma 4's template defaults to thinking off and adds its empty thought channel by itself.
        return {"enable_thinking": False} if self.family == "qwen35" else {}

    def _render(self, msgs: list[dict], prefill: str | None = "") -> str:
        """Generation-prompt text for ``msgs`` + ``prefill`` (non-qwen3vl families)."""
        def go(m):
            return self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, **self._template_kwargs())

        if self._fold_system:
            msgs = fold_system(msgs)
        try:
            text = go(msgs)
        except Exception as e:  # noqa: BLE001
            if self.family != "generic" or self._fold_system or not msgs or msgs[0]["role"] != "system":
                raise
            _log(f"chat template rejected the system role ({type(e).__name__}: {e}); folding it into the user turn")
            self._fold_system = True
            text = go(fold_system(msgs))
        return text + (prefill or "")

    def _processor_inputs(self, text: str, pil: list):
        kw = {}
        bos = getattr(getattr(self.processor, "tokenizer", None), "bos_token", None)
        if isinstance(bos, str) and text.startswith(bos):      # template already wrote <bos>: do not add a second
            kw["add_special_tokens"] = False
        return self.processor(text=[text], images=pil or None, return_tensors="pt", **kw)

    def _generate(self, inputs, max_new_tokens: int, deadline, stop_strings=None,
                  repetition_penalty: float | None = None, want_scores: bool = False):
        """Greedy generate; returns (decoded new text, n_new_tokens, mean_logprob or None).

        Timings land in ``self.timings``: ``last_generate_s`` (always) and, for the A/B families,
        ``last_ttft_s`` (prefill + first token; taken at the first stopping-criteria call).
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        t_start = time.perf_counter()
        if self.family != "qwen3vl":
            self.timings.pop("last_ttft_s", None)

            class _FirstToken(StoppingCriteria):
                def __call__(_s, input_ids, scores, **kw):  # noqa: N805
                    self.timings.setdefault("last_ttft_s", time.perf_counter() - t_start)
                    return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        tok = self.processor.tokenizer
        plen = inputs["input_ids"].shape[1]

        def run(use_stop_strings: bool):
            crit = StoppingCriteriaList()
            if self.family != "qwen3vl":
                crit.append(_FirstToken())
            if deadline is not None:
                crit.append(make_deadline_criteria(deadline))
            kw = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, temperature=None,
                      top_p=None, top_k=None, stopping_criteria=crit)
            if stop_strings:
                if use_stop_strings:
                    kw.update(stop_strings=list(stop_strings), tokenizer=tok)
                else:
                    for s in stop_strings:
                        crit.append(make_tail_string_criteria(tok, s))
            if repetition_penalty:
                kw["repetition_penalty"] = repetition_penalty
            if want_scores:
                kw.update(return_dict_in_generate=True, output_scores=True)
            return self.model.generate(**inputs, **kw)

        with torch.inference_mode(), self._ctx():
            try:
                out = run(True)
            except Exception as e:  # noqa: BLE001
                if not stop_strings or "out of memory" in str(e).lower():
                    raise
                _log(f"generate with stop_strings failed ({type(e).__name__}: {e}); using tail-string criteria")
                out = run(False)
            self.timings["last_generate_s"] = time.perf_counter() - t_start
            seqs = out.sequences if want_scores else out
            new = seqs[0, plen:]
            text = tok.decode(new.tolist(), skip_special_tokens=True)
            mean_lp = None
            if want_scores and getattr(out, "scores", None):
                n = min(len(out.scores), new.shape[0])
                if n:
                    sc = torch.stack([s[0] for s in out.scores[:n]], 0).float()
                    lp = torch.log_softmax(sc, dim=-1).gather(1, new[:n, None])
                    mean_lp = float(lp.mean().item())
            return text, int(new.shape[0]), mean_lp

    def _chat_generate(self, messages, images, deadline, max_new_tokens, *, prefilled,
                       stop_strings=None, repetition_penalty=None, want_scores=False, edge=None):
        with self._lock:
            with self._image_edge(edge):
                inputs = self._encode(messages, images, prefilled=prefilled)
            return self._generate(inputs, max_new_tokens, deadline, stop_strings=stop_strings,
                                  repetition_penalty=repetition_penalty, want_scores=want_scores)

    # -- public API ----------------------------------------------------------------------------

    def build_messages(self, package_text: str, question: str, images: list) -> list[dict]:
        return build_messages(package_text, question, images)

    def read(self, package_text: str, question: str, images: list, deadline,
             max_new_tokens: int = config.MAX_NEW_TOKENS) -> ReaderOut:
        raw = PREFILL
        try:
            msgs = self.build_messages(package_text, question, images)
            text, n_new, mean_lp = self._chat_generate(
                msgs, images, deadline, max_new_tokens, prefilled=True, stop_strings=["}"], want_scores=True)
            raw = PREFILL + text
            answer, support, bridge = parse_output(raw)
            return ReaderOut(answer, support, bridge, raw=raw, n_new_tokens=n_new, mean_logprob=mean_lp)
        except Exception as e:  # noqa: BLE001
            _log(f"read failed: {type(e).__name__}: {e}")
            return ReaderOut("", [], [], raw=f"ERROR: {e}")

    def vqa(self, question: str, image, deadline, max_new_tokens: int = 48,
            longest_edge: int | None = None) -> str:
        """Short direct visual question (image self-consistency). ``longest_edge`` overrides the pixel budget."""
        try:
            pil, _ = _split_images([image])
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil[0]},
                {"type": "text", "text": f"{question}\nAnswer with the value only."}]}]
            text, _, _ = self._chat_generate(msgs, pil, deadline, max_new_tokens, prefilled=False,
                                             edge=longest_edge)
            return text.strip()
        except Exception as e:  # noqa: BLE001
            _log(f"vqa failed: {type(e).__name__}: {e}")
            return ""

    def transcribe(self, image, deadline, max_new_tokens: int = config.TRANSCRIBE_MAX_NEW_TOKENS) -> str:
        """Full-image transcription at the larger pixel budget; READ budget restored afterwards."""
        try:
            pil, _ = _split_images([image])
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil[0]},
                {"type": "text", "text": TRANSCRIBE_PROMPT}]}]
            text, _, _ = self._chat_generate(msgs, pil, deadline, max_new_tokens, prefilled=False,
                                             repetition_penalty=1.05,
                                             edge=config.IMAGE_LONGEST_EDGE_TRANSCRIBE)
            return text.strip()
        except Exception as e:  # noqa: BLE001
            _log(f"transcribe failed: {type(e).__name__}: {e}")
            return ""

    def _logits_to_keep_kw(self) -> str | None:
        if self._logits_kw is False:
            import inspect
            params = inspect.signature(self.model.forward).parameters
            self._logits_kw = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in params), None)
            if self._logits_kw is None:
                _log("WARNING: forward has no logits_to_keep; scoring with full logits (memory heavy)")
        return self._logits_kw

    def logprob(self, package_text: str, question: str, images: list, answer: str) -> float:
        """Teacher-forced log P(answer + closing quote | prompt + prefill) (spec 5.6)."""
        import torch

        with self._lock:
            pre_msgs = self.build_messages(package_text, question, images)
            full_msgs = pre_msgs[:-1] + [{"role": "assistant", "content": PREFILL + answer + '"'}]
            pre = self._encode(pre_msgs, images)
            full = self._encode(full_msgs, images)
            pre_ids = pre["input_ids"][0]
            full_ids = full["input_ids"][0]
            m = min(len(pre_ids), len(full_ids))
            neq = (pre_ids[:m] != full_ids[:m]).nonzero()
            k = int(neq[0].item()) if len(neq) else m          # longest common token prefix
            tgt = full_ids[k:]
            if len(tgt) == 0 or k == 0:
                return float("-inf")
            kw = {}
            name = self._logits_to_keep_kw()
            if name:
                kw[name] = len(tgt) + 1
            with torch.inference_mode(), self._ctx():
                out = self.model(**full, use_cache=False, **kw)
                logits = out.logits[0]
                # keep=len(tgt)+1 -> logits cover positions L-len(tgt)-1 .. L-1; the first len(tgt) predict tgt
                if name is None:
                    logits = logits[k - 1:k - 1 + len(tgt)]
                else:
                    logits = logits[:len(tgt)]
                lp = torch.log_softmax(logits.float(), dim=-1).gather(1, tgt[:, None].to(logits.device))
                return float(lp.sum().item())

    def yes_no(self, evidence_text: str, question: str, answer: str, images=None, deadline=None) -> float | None:
        """log P(YES) - log P(NO) for the first token of the assistant reply to the warrant question (H6).

        One forward pass over the prompt (generation start, no prefill), ``logits_to_keep=1``. None on any error,
        when fewer than ``WARRANT_MIN_REMAINING_S`` remain, or when the tokenizer has no YES/NO token."""
        try:
            if deadline is not None and not deadline.require(WARRANT_MIN_REMAINING_S):
                return None
            import torch

            with self._lock:
                yes_ids, no_ids = warrant_token_ids(self.processor.tokenizer)
                if not yes_ids or not no_ids:
                    return None
                inputs = self._encode(build_warrant_messages(evidence_text, question, answer, images),
                                      images or [], prefilled=False)
                kw = {}
                name = self._logits_to_keep_kw()
                if name:
                    kw[name] = 1
                with torch.inference_mode(), self._ctx():
                    out = self.model(**inputs, use_cache=False, **kw)
                    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                    return warrant_margin([float(lp[i]) for i in yes_ids], [float(lp[i]) for i in no_ids])
        except Exception as e:  # noqa: BLE001
            _log(f"yes_no failed: {type(e).__name__}: {e}")
            return None

    def warmup(self) -> None:
        """One ~6k-token read with a 640x320 image, then one logprob() (exercises logits_to_keep)."""
        t0 = time.perf_counter()
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (640, 320), "white")
            d = ImageDraw.Draw(img)
            d.rectangle([20, 20, 300, 140], outline="black", width=3)
            d.text((40, 60), "PART: ORR-FAN-2214-B  REV-C2", fill="black")
            d.text((40, 200), "PIN 7: THERM_ALERT#", fill="black")
            tok = self.processor.tokenizer
            lines: list[str] = []
            i = 0
            while True:      # grow in chunks of 10 records until the evidence is ~6000 tokens
                for _ in range(10):
                    lines.append(f"[R{i}] logs/prod_{i % 7}.log L{i}: 2026-03-{1 + i % 28:02d}T03:14:{i % 60:02d}Z "
                                 f"node=n{i % 13} ticket=ORR-{1800 + i} status=ok temp_c={40 + i % 50} "
                                 f"fixed_in=4.{i % 9}.{i % 5}")
                    i += 1
                pkg = "## logs/prod.log (log)\n" + "\n".join(lines)
                if len(tok(pkg)["input_ids"]) >= 6000 or i > 2000:
                    break
            q = "What is the fixed_in version for ticket ORR-1847?"
            t1 = time.perf_counter()
            out = self.read(pkg, q, [img], Deadline.after(60))
            self.timings["warmup_read_s"] = time.perf_counter() - t1
            _log(f"warmup read {self.timings['warmup_read_s']:.2f}s raw={out.raw[:80]!r}")
            t2 = time.perf_counter()
            lp = self.logprob(pkg, q, [img], out.answer or "4.3.2")
            self.timings["warmup_logprob_s"] = time.perf_counter() - t2
            _log(f"warmup logprob {self.timings['warmup_logprob_s']:.2f}s lp={lp:.2f}")
        except Exception as e:  # noqa: BLE001
            _log(f"warmup failed: {type(e).__name__}: {e}")
        self.timings["warmup_s"] = time.perf_counter() - t0
