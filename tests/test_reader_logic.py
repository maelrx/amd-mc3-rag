import os
from types import SimpleNamespace

import pytest

import config
import reader
from reader import PREFILL, RULES, Reader, build_messages, parse_output


# ---- parsing ---------------------------------------------------------------------------------

def test_parse_full():
    assert parse_output('{"answer":"94","support":["R6"],"bridge":[]}') == ("94", [6], [])


def test_parse_bridge_and_multiple():
    a, s, b = parse_output('{"answer":"4.3.2","support":["R12", "R13"],"bridge":["R40"]}')
    assert (a, s, b) == ("4.3.2", [12, 13], [40])


def test_parse_truncated_after_closed_answer():
    assert parse_output('{"answer":"Q3 FY27","support":["R12"') == ("Q3 FY27", [12], [])


def test_parse_truncated_inside_answer():
    assert parse_output('{"answer":"ORR-FAN-22') == ("ORR-FAN-22", [], [])
    assert parse_output('{"answer":"180}') == ("180", [], [])
    assert parse_output('{"answer":"') == ("", [], [])


def test_parse_empty_refusal():
    assert parse_output('{"answer":"","support":[],"bridge":[]}') == ("", [], [])


def test_parse_garbage():
    assert parse_output("") == ("", [], [])
    assert parse_output("I think the answer is 94") == ("", [], [])
    assert parse_output(None) == ("", [], [])


def test_parse_escapes_and_dedup():
    a, s, b = parse_output('{"answer":"say \\"hi\\"","support":["R1","R1","r2"],"bridge":["R9"]}')
    assert a == 'say "hi"' and s == [1, 2] and b == [9]


def test_parse_missing_lists():
    assert parse_output('{"answer":"B14"}') == ("B14", [], [])
    assert parse_output('{"answer":"B14","bridge":["R3"]}') == ("B14", [], [3])


# ---- build_messages --------------------------------------------------------------------------

def test_build_messages_text_only():
    msgs = build_messages("## f (txt)\n[R1] f L1: x", "What is x?", [])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2]["content"] == PREFILL == '{"answer":"'
    assert reader.SYSTEM in msgs[0]["content"] and '"bridge":["R7"]' in msgs[0]["content"]
    content = msgs[1]["content"]
    assert [b["type"] for b in content] == ["text"]
    t = content[0]["text"]
    assert t.startswith("EVIDENCE\n## f (txt)\n[R1] f L1: x\n\nQUESTION: What is x?\n\nRULES\n")
    assert t.index("EVIDENCE") < t.index("QUESTION:") < t.index("RULES") < t.index("Output exactly:")
    assert RULES in t and t.endswith('Output exactly: {"answer":"<value>","support":["R.."],"bridge":["R.."]}')
    assert "<IMAGE" not in t


def test_build_messages_images_first_with_note():
    im1, im2 = object(), object()
    msgs = build_messages("PKG", "Q?", [("pics/a.png", im1), im2])
    content = msgs[1]["content"]
    assert [b["type"] for b in content] == ["image", "image", "text"]
    assert content[0]["image"] is im1 and content[1]["image"] is im2
    t = content[2]["text"]
    assert "PKG\n<IMAGE 1 = pics/a.png> <IMAGE 2 attached>\n\nQUESTION: Q?" in t
    assert msgs[-1] == {"role": "assistant", "content": PREFILL}


# ---- encode / size handling (mock processor) -------------------------------------------------

class FakeProcessor:
    def __init__(self):
        self.image_processor = SimpleNamespace(size={"shortest_edge": 1, "longest_edge": 2})
        self.calls = []
        self.sizes_seen = []
        self.tokenizer = object()

    def apply_chat_template(self, msgs, **kw):
        self.calls.append((msgs, kw))
        return "TEXT"

    def __call__(self, text, images, return_tensors):
        self.sizes_seen.append(dict(self.image_processor.size))
        return SimpleNamespace(to=lambda dev: ("inputs", text, images, dev))


def make_reader():
    r = Reader("/nonexistent")
    r.processor = FakeProcessor()
    r.model = SimpleNamespace(device="cpu")
    return r


def test_encode_continue_final_message_and_pil_stripped():
    r = make_reader()
    im = object()
    msgs = r.build_messages("PKG", "Q?", [im])
    out = r._encode(msgs, [im])
    assert out == ("inputs", ["TEXT"], [im], "cpu")
    sent, kw = r.processor.calls[0]
    assert kw == {"tokenize": False, "add_generation_prompt": False, "continue_final_message": True}
    assert sent[1]["content"][0] == {"type": "image"}            # no pixel payload in the template call
    assert sent[-1]["content"] == PREFILL
    r._encode([{"role": "user", "content": "hi"}], [], prefilled=False)
    assert r.processor.calls[1][1] == {"tokenize": False, "add_generation_prompt": True}


def test_transcribe_size_override_and_restore():
    r = make_reader()
    seen = {}

    def fake_encode(messages, images, prefilled=True):
        seen["size"] = dict(r.processor.image_processor.size)
        seen["prefilled"] = prefilled
        return "enc"

    def fake_generate(inputs, max_new_tokens, deadline, stop_strings=None, repetition_penalty=None,
                      want_scores=False):
        seen.update(rp=repetition_penalty, n=max_new_tokens, stop=stop_strings)
        return " line1\nline2 ", 5, None

    r._encode, r._generate = fake_encode, fake_generate
    out = r.transcribe(object(), None)
    assert out == "line1\nline2"
    assert seen["size"] == {"shortest_edge": 65536, "longest_edge": config.IMAGE_LONGEST_EDGE_TRANSCRIBE}
    assert seen["rp"] == 1.05 and seen["n"] == config.TRANSCRIBE_MAX_NEW_TOKENS and seen["prefilled"] is False
    assert r.processor.image_processor.size == {"shortest_edge": 65536,
                                                "longest_edge": config.IMAGE_LONGEST_EDGE_READ}


def test_size_restored_when_encode_raises():
    r = make_reader()

    def boom(*a, **k):
        raise RuntimeError("bad image")

    r._encode = boom
    assert r.transcribe(object(), None) == ""                     # swallowed, logged
    assert r.processor.image_processor.size["longest_edge"] == config.IMAGE_LONGEST_EDGE_READ


def test_vqa_edge_override_and_default():
    r = make_reader()
    sizes = []
    r._encode = lambda m, i, prefilled=True: sizes.append(dict(r.processor.image_processor.size)) or "enc"
    r._generate = lambda *a, **k: (" B14\n", 2, None)
    r.processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": config.IMAGE_LONGEST_EDGE_READ}
    assert r.vqa("Which pin?", object(), None) == "B14"
    assert r.vqa("Which pin?", object(), None, longest_edge=500_000) == "B14"
    assert [s["longest_edge"] for s in sizes] == [config.IMAGE_LONGEST_EDGE_READ, 500_000]
    assert r.processor.image_processor.size["longest_edge"] == config.IMAGE_LONGEST_EDGE_READ


def test_read_assembles_readerout_and_swallows_errors():
    r = make_reader()
    r._encode = lambda m, i, prefilled=True: "enc"
    seen = {}

    def gen(inputs, max_new_tokens, deadline, stop_strings=None, repetition_penalty=None, want_scores=False):
        seen.update(stop=stop_strings, scores=want_scores, n=max_new_tokens)
        return '94","support":["R6"],"bridge":[]}', 14, -0.25

    r._generate = gen
    out = r.read("PKG", "Q?", [], None)
    assert (out.answer, out.support, out.bridge) == ("94", [6], [])
    assert out.raw == '{"answer":"94","support":["R6"],"bridge":[]}'
    assert out.n_new_tokens == 14 and out.mean_logprob == -0.25
    assert seen == {"stop": ["}"], "scores": True, "n": config.MAX_NEW_TOKENS}

    def boom(*a, **k):
        raise RuntimeError("HIP error")

    r._generate = boom
    out = r.read("PKG", "Q?", [], None)
    assert out.answer == "" and out.support == [] and out.raw == "ERROR: HIP error"


# ---- model families (A/B of alternative VLMs) -------------------------------------------------

QWEN3VL_CFG = {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl"}
GEMMA4_CFG = {"architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified"}
QWEN35_CFG = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}


def test_detect_family():
    assert reader.detect_family(QWEN3VL_CFG) == "qwen3vl"
    assert reader.detect_family({"model_type": "qwen3_vl"}) == "qwen3vl"
    assert reader.detect_family({"architectures": ["Qwen3VLForConditionalGeneration"]}) == "qwen3vl"
    assert reader.detect_family(GEMMA4_CFG) == "gemma4"
    assert reader.detect_family({"model_type": "gemma4"}) == "gemma4"
    assert reader.detect_family({"architectures": ["Gemma4ForConditionalGeneration"]}) == "gemma4"
    assert reader.detect_family(QWEN35_CFG) == "qwen35"
    assert reader.detect_family({"model_type": "qwen3_5_moe"}) == "qwen35"
    for cfg in ({"architectures": ["Qwen3VLMoeForConditionalGeneration"], "model_type": "qwen3_vl_moe"},
                {"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3"},
                {"architectures": ["LlavaForConditionalGeneration"], "model_type": "llava"}):
        assert reader.detect_family(cfg) == "generic"
    # unreadable / empty config keeps the shipped behaviour
    assert reader.detect_family({}) == "qwen3vl" and reader.detect_family(None) == "qwen3vl"
    assert reader.FAMILIES == ("qwen3vl", "gemma4", "qwen35", "generic")


def test_default_family_and_read_config(tmp_path):
    r = Reader(str(tmp_path))
    assert r.family == "qwen3vl"                                  # before load(): the shipped path
    assert r._read_config() == {}                                 # no config.json -> {} -> qwen3vl
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}')
    assert reader.detect_family(r._read_config()) == "qwen35"


class FamilyProcessor:
    """Mock processor that records template/processor kwargs and has a configurable image budget knob."""

    def __init__(self, ip, bos=None):
        self.image_processor = ip
        self.tokenizer = SimpleNamespace(bos_token=bos)
        self.template_calls, self.proc_calls = [], []
        self.reject_system = False
        self.proc_result = None          # overrides what __call__ returns

    def apply_chat_template(self, msgs, **kw):
        self.template_calls.append((msgs, kw))
        if self.reject_system and msgs and msgs[0]["role"] == "system":
            raise ValueError("System role not supported")
        return (self.tokenizer.bos_token or "") + "|".join(m["role"] for m in msgs) + "~GEN~"

    def __call__(self, **kw):
        self.proc_calls.append(kw)
        if self.proc_result is not None:
            return self.proc_result
        return SimpleNamespace(to=lambda dev: ("inputs", kw["text"], kw["images"], dev))


def family_reader(family, ip=None, bos=None):
    r = Reader("/nonexistent")
    r.family = family
    r.processor = FamilyProcessor(ip if ip is not None else SimpleNamespace(), bos=bos)
    r.model = SimpleNamespace(device="cpu")
    return r


def test_encode_qwen35_enable_thinking_false_and_prefill_appended():
    r = family_reader("qwen35", SimpleNamespace(size={}))
    im = object()
    msgs = r.build_messages("PKG", "Q?", [im])
    out = r._encode(msgs, [im])
    sent, kw = r.processor.template_calls[0]
    assert kw == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert [m["role"] for m in sent] == ["system", "user"]        # final assistant turn is not sent to the template
    assert sent[1]["content"][0] == {"type": "image"}
    assert out == ("inputs", ["system|user~GEN~" + PREFILL], [im], "cpu")
    r._encode([{"role": "user", "content": "hi"}], [], prefilled=False)
    assert r.processor.template_calls[1][1] == kw
    assert r.processor.proc_calls[1] == {"text": ["user~GEN~"], "images": None, "return_tensors": "pt"}


def test_encode_gemma4_no_thinking_kwarg_and_single_bos():
    r = family_reader("gemma4", SimpleNamespace(max_soft_tokens=280), bos="<bos>")
    msgs = r.build_messages("PKG", "Q?", [])
    r._encode(msgs, [])
    assert r.processor.template_calls[0][1] == {"tokenize": False, "add_generation_prompt": True}
    call = r.processor.proc_calls[0]
    assert call["text"] == ["<bos>system|user~GEN~" + PREFILL]
    assert call["add_special_tokens"] is False                    # template already wrote <bos>
    # the system role is kept natively (Gemma 4 template has a system turn)
    assert r.processor.template_calls[0][0][0]["role"] == "system"


def test_encode_generic_folds_system_when_template_rejects_it():
    r = family_reader("generic")
    r.processor.reject_system = True
    msgs = r.build_messages("PKG", "Q?", [object()])
    r._encode(msgs, [object()])
    assert r._fold_system is True
    sent = r.processor.template_calls[-1][0]
    assert [m["role"] for m in sent] == ["user"]
    blocks = sent[0]["content"]
    assert [b["type"] for b in blocks] == ["image", "text"]       # images stay first
    assert blocks[1]["text"].startswith(reader.SYSTEM + "\n\nEVIDENCE\nPKG")
    n = len(r.processor.template_calls)
    r._encode(msgs, [])                                           # remembered: no second failing call
    assert len(r.processor.template_calls) == n + 1
    assert r.processor.proc_calls[-1]["text"] == ["user~GEN~" + PREFILL]


def test_system_error_not_swallowed_for_named_families():
    r = family_reader("qwen35")
    r.processor.reject_system = True
    with pytest.raises(ValueError):
        r._encode(r.build_messages("PKG", "Q?", []), [])


def test_fold_system_variants():
    out = reader.fold_system([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert out == [{"role": "user", "content": "S\n\nU"}]
    assert reader.fold_system([{"role": "user", "content": "U"}]) == [{"role": "user", "content": "U"}]
    out = reader.fold_system([{"role": "system", "content": "S"},
                              {"role": "user", "content": [{"type": "image"}]}])
    assert out[0]["content"] == [{"type": "image"}, {"type": "text", "text": "S"}]


def test_set_size_per_family():
    ip = SimpleNamespace(size=None)
    r = family_reader("qwen35", ip)
    r._set_size(config.IMAGE_LONGEST_EDGE_READ)
    assert ip.size == {"shortest_edge": 65536, "longest_edge": config.IMAGE_LONGEST_EDGE_READ}

    ip = SimpleNamespace(max_soft_tokens=280)
    r = family_reader("gemma4", ip)
    r._set_size(config.IMAGE_LONGEST_EDGE_READ)
    assert ip.max_soft_tokens == 560 and not hasattr(ip, "size")   # same pixel budget as the Qwen read (48x48 px/token)
    r._set_size(config.IMAGE_LONGEST_EDGE_TRANSCRIBE)
    assert ip.max_soft_tokens == 1120                              # largest bucket the Gemma 4 processor accepts
    r._set_size(300_000)
    assert ip.max_soft_tokens == 140

    ip = SimpleNamespace(do_pan_and_scan=False)
    r = family_reader("generic", ip)
    r._set_size(config.IMAGE_LONGEST_EDGE_READ)                    # no known knob: untouched, logged once
    assert vars(ip) == {"do_pan_and_scan": False}


def test_gemma_soft_tokens_buckets():
    assert [reader.gemma_soft_tokens(e) for e in (1_228_800, 2_097_152, 600_000, 100_000, 10, 10**9)] == \
        [560, 1120, 280, 70, 70, 1120]


def test_transcribe_restores_read_budget_on_gemma():
    r = family_reader("gemma4", SimpleNamespace(max_soft_tokens=1120))
    r._encode = lambda m, i, prefilled=True: "enc"
    r._generate = lambda *a, **k: (" x ", 1, None)
    assert r.transcribe(object(), None) == "x"
    assert r.processor.image_processor.max_soft_tokens == reader.gemma_soft_tokens(config.IMAGE_LONGEST_EDGE_READ)


def test_generate_records_ttft_only_for_ab_families():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)

    class FakeModel:
        def generate(self, input_ids=None, stopping_criteria=(), **kw):
            for c in stopping_criteria:
                c(input_ids, None)
            return torch.cat([input_ids, torch.tensor([[1, 2]])], dim=1)

    for family, has_ttft, n_crit in (("gemma4", True, 1), ("qwen3vl", False, 0)):
        r = family_reader(family)
        r.processor.tokenizer = Tok()
        r.model = FakeModel()
        text, n, _ = r._generate({"input_ids": torch.tensor([[0, 0]])}, 4, None)
        assert (text, n) == ("xx", 2)
        assert ("last_ttft_s" in r.timings) is has_ttft and "last_generate_s" in r.timings


def _fake_transformers(monkeypatch, record):
    import sys as _sys
    import types

    class FakeModel:
        def __init__(self, tag):
            self.tag = tag
            self.generation_config = SimpleNamespace(do_sample=True, temperature=1.0, top_p=.9, top_k=64,
                                                     repetition_penalty=1.1)

        def eval(self):
            return self

    def cls(tag):
        return type(tag, (), {"from_pretrained": staticmethod(
            lambda d, **kw: (record.append((tag, kw)), FakeModel(tag))[1])})

    mod = types.ModuleType("transformers")
    mod.AutoProcessor = type("AutoProcessor", (), {"from_pretrained": staticmethod(
        lambda d: SimpleNamespace(image_processor=SimpleNamespace(size=None, max_soft_tokens=280)))})
    mod.Qwen3VLForConditionalGeneration = cls("Qwen3VL")
    mod.Qwen3_5ForConditionalGeneration = cls("Qwen3_5")
    mod.AutoModelForImageTextToText = cls("Auto")
    monkeypatch.setitem(_sys.modules, "transformers", mod)


@pytest.mark.parametrize("cfg,family,model_tag", [
    (QWEN3VL_CFG, "qwen3vl", "Qwen3VL"),
    (QWEN35_CFG, "qwen35", "Qwen3_5"),
    (GEMMA4_CFG, "gemma4", "Auto"),
    ({"model_type": "llava"}, "generic", "Auto"),
])
def test_load_picks_class_and_forces_greedy(monkeypatch, tmp_path, cfg, family, model_tag):
    pytest.importorskip("torch")
    import json as _json
    import sys as _sys
    record = []
    _fake_transformers(monkeypatch, record)
    monkeypatch.delenv("MC3_NO_FLA", raising=False)
    monkeypatch.delenv("USE_HUB_KERNELS", raising=False)
    monkeypatch.delitem(_sys.modules, "fla", raising=False)
    monkeypatch.delitem(_sys.modules, "causal_conv1d", raising=False)
    (tmp_path / "config.json").write_text(_json.dumps(cfg))
    r = Reader(str(tmp_path), "cpu")
    r.load()
    assert r.family == family and r.attn == "sdpa"
    tag, kw = record[0]
    assert tag == model_tag and kw["attn_implementation"] == "sdpa" and kw["device_map"] == "cpu"
    gc_ = r.model.generation_config
    assert (gc_.do_sample, gc_.temperature, gc_.top_p, gc_.top_k, gc_.repetition_penalty) == (False, None, None, None, 1.0)
    no_fla = family == "qwen35"                                     # torch-reference Gated DeltaNet only for qwen3.5
    assert (_sys.modules.get("fla", 0) is None) is no_fla and (_sys.modules.get("causal_conv1d", 0) is None) is no_fla
    assert (os.environ.get("USE_HUB_KERNELS") == "NO") is no_fla
    monkeypatch.delenv("USE_HUB_KERNELS", raising=False)           # setdefault ran outside monkeypatch's tracking
    for k in ("fla", "causal_conv1d"):
        _sys.modules.pop(k, None)


def test_load_qwen35_respects_mc3_no_fla_0(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import json as _json
    import sys as _sys
    _fake_transformers(monkeypatch, [])
    monkeypatch.setenv("MC3_NO_FLA", "0")
    monkeypatch.delitem(_sys.modules, "fla", raising=False)
    (tmp_path / "config.json").write_text(_json.dumps(QWEN35_CFG))
    Reader(str(tmp_path), "cpu").load()
    assert "fla" not in _sys.modules


def test_self_test_skips_gemma_unified():
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    r = family_reader("gemma4", SimpleNamespace(max_soft_tokens=280))
    r.model = type("Gemma4UnifiedForConditionalGeneration", (), {"device": "cpu"})()
    assert r.self_test_sdpa() is False
    assert "encoder-free" in r.sdpa_skipped and r.force_math is False


def test_self_test_skips_unknown_vision_inputs():
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    r = family_reader("generic")
    r.processor.proc_result = {"input_ids": 1, "pixel_values": SimpleNamespace(to=lambda d: 0)}
    assert r.self_test_sdpa() is False
    assert "no known vision-tower inputs" in r.sdpa_skipped


def test_qwen3vl_encode_unchanged_by_family_machinery():
    r = make_reader()                                             # default family = qwen3vl
    assert r.family == "qwen3vl" and r.sdpa_skipped is None
    r._encode(r.build_messages("PKG", "Q?", []), [])
    sent, kw = r.processor.calls[0]
    assert kw == {"tokenize": False, "add_generation_prompt": False, "continue_final_message": True}
    assert sent[-1]["role"] == "assistant"                        # prefill goes through the template as before


# ---- torch-backed logic with tiny fakes ------------------------------------------------------

def test_logprob_alignment_and_logits_to_keep():
    torch = pytest.importorskip("torch")
    V = 6
    r = make_reader()

    # fake "tokenizer": text -> ids; prefill = [1,2,3]; answer tokens appended; closing quote = 5
    def fake_encode(messages, images, prefilled=True):
        content = messages[-1]["content"]
        ids = [0, 4, 1, 2, 3] if content == PREFILL else [0, 4, 1, 2, 3] + [4, 4, 5]
        d = {"input_ids": torch.tensor([ids])}
        return d

    seen = {}

    class FakeModel:
        device = "cpu"

        def forward(self, input_ids, logits_to_keep=0, use_cache=True):  # signature probed by inspect
            raise NotImplementedError

        def __call__(self, input_ids, logits_to_keep=0, use_cache=True):
            seen["keep"] = logits_to_keep
            L = input_ids.shape[1]
            # logits at position p put all mass (logit 10) on token (input_ids[p+1] if exists else 0)
            full = torch.zeros(1, L, V)
            for p in range(L - 1):
                full[0, p, int(input_ids[0, p + 1])] = 10.0
            return SimpleNamespace(logits=full[:, -logits_to_keep:, :])

    r.model = FakeModel()
    r._encode = fake_encode
    lp = r.logprob("PKG", "Q?", [], "44")
    # tgt = [4,4,5] -> keep = 4; each target has prob softmax(10 vs five zeros) = e^10/(e^10+5)
    import math
    expected = 3 * (10.0 - math.log(math.exp(10.0) + 5))
    assert seen["keep"] == 4
    assert lp == pytest.approx(expected, abs=1e-4)


def test_generate_scores_and_stop_string_fallback():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    V = 5

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join("abcd}"[i] for i in ids)

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, input_ids=None, **kw):
            self.calls.append(kw)
            if "stop_strings" in kw:
                raise ValueError("stop strings broken")
            new = torch.tensor([[1, 2, 4]])
            seq = torch.cat([input_ids, new], dim=1)
            scores = tuple(torch.nn.functional.one_hot(new[:, i], V).float() * 20 for i in range(3))
            return SimpleNamespace(sequences=seq, scores=scores)

    r = make_reader()
    r.processor.tokenizer = Tok()
    r.model = FakeModel()
    inputs = {"input_ids": torch.tensor([[0, 0, 0]])}
    text, n, mlp = r._generate(inputs, 8, None, stop_strings=["}"], want_scores=True)
    assert (text, n) == ("bc}", 3)
    assert mlp == pytest.approx(0.0, abs=1e-6)
    assert "stop_strings" in r.model.calls[0] and "stop_strings" not in r.model.calls[1]
    assert r.model.calls[1]["do_sample"] is False and r.model.calls[1]["temperature"] is None
    assert len(r.model.calls[1]["stopping_criteria"]) == 1        # tail-string fallback
