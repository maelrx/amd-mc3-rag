import time

import pytest

from budget import Deadline, Trace, make_deadline_criteria, make_tail_string_criteria


def test_deadline_math():
    d = Deadline.after(10)
    assert 9.0 < d.remaining() <= 10.0
    assert not d.expired()
    assert d.require(5) and not d.require(11)
    s = d.sub(4)
    assert s.ts == pytest.approx(d.ts - 4)
    assert 5.0 < s.remaining() <= 6.0
    m = d.min_with(d.ts - 3)
    assert m.ts == pytest.approx(d.ts - 3)
    assert d.min_with(d.ts + 100).ts == d.ts            # never later
    assert d.min_with(Deadline(d.ts - 1)).ts == pytest.approx(d.ts - 1)
    past = Deadline(time.time() - 1)
    assert past.expired() and past.remaining() < 0 and not past.require(0.1)


def test_trace_stages_and_notes():
    t = Trace()
    with t.stage("a"):
        time.sleep(0.01)
    with pytest.raises(RuntimeError):
        with t.stage("b"):
            raise RuntimeError("x")                      # stage still recorded
    t.note("k", 3)
    assert [n for n, _ in t.stages] == ["a", "b"]
    assert t.stages[0][1] >= 0.01
    d = t.to_dict()
    assert [n for n, _ in d["stages"]] == ["a", "b"] and d["notes"] == {"k": 3}


def test_deadline_criteria_flips_after_expiry():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import StoppingCriteriaList
    dl = Deadline.after(0.15)
    crit = make_deadline_criteria(dl)
    ids = torch.zeros((2, 7), dtype=torch.long)
    out = crit(ids, None)
    assert out.dtype == torch.bool and out.shape == (2,) and not out.any()
    lst = StoppingCriteriaList([crit])
    assert not lst(ids, None).any()
    time.sleep(0.2)
    out = crit(ids, None)
    assert out.dtype == torch.bool and out.shape == (2,) and out.all()
    assert lst(ids, None).all()


class _Tok:
    table = {1: "a", 2: "b", 3: "}", 4: "c"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.table[int(i)] for i in ids)


def test_tail_string_criteria():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    crit = make_tail_string_criteria(_Tok(), "}", window=2)
    ids = torch.tensor([[1, 2, 4, 3], [1, 3, 2, 4], [3, 1, 2, 4]])
    out = crit(ids, None)
    assert out.dtype == torch.bool and out.tolist() == [True, False, False]
