"""Time budget helpers: Deadline, Trace, and generation stopping criteria.

Stdlib only at import time. ``make_deadline_criteria`` / ``make_tail_string_criteria`` import
transformers (and torch) lazily, so ``import budget`` works on a machine without a GPU stack.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class Deadline:
    """Absolute wall-clock deadline (``time.time()`` seconds, same clock as the socket ``deadline_ts``)."""

    def __init__(self, ts: float):
        self.ts = float(ts)

    @classmethod
    def after(cls, seconds: float) -> "Deadline":
        return cls(time.time() + seconds)

    def remaining(self) -> float:
        return self.ts - time.time()

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def require(self, seconds: float) -> bool:
        """True if at least ``seconds`` are left."""
        return self.remaining() >= seconds

    def sub(self, margin: float) -> "Deadline":
        """Same deadline, ``margin`` seconds earlier."""
        return Deadline(self.ts - margin)

    def min_with(self, other_ts) -> "Deadline":
        """Earlier of this deadline and ``other_ts`` (a timestamp or another Deadline)."""
        return Deadline(min(self.ts, float(getattr(other_ts, "ts", other_ts))))

    def __repr__(self) -> str:
        return f"Deadline(remaining={self.remaining():.2f}s)"


class Trace:
    """Per-request stage timings plus free-form notes (goes into ``Result.trace``)."""

    def __init__(self):
        self.stages: list[tuple[str, float]] = []
        self.notes: dict = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages.append((name, time.perf_counter() - t0))

    def note(self, key: str, value) -> None:
        self.notes[key] = value

    def to_dict(self) -> dict:
        return {"stages": [[n, round(s, 4)] for n, s in self.stages], "notes": dict(self.notes)}


def make_deadline_criteria(deadline: Deadline):
    """``transformers.StoppingCriteria`` that stops every row once ``deadline`` has passed.

    transformers 5.x contract (generation/stopping_criteria.py): ``__call__(input_ids, scores, **kwargs)``
    returns a ``torch.BoolTensor`` of shape ``[batch]``; ``StoppingCriteriaList`` ORs them together.
    """
    import torch
    from transformers import StoppingCriteria

    class DeadlineCriteria(StoppingCriteria):
        def __init__(self, dl):
            self.deadline = dl

        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), bool(self.deadline.expired()),
                              dtype=torch.bool, device=input_ids.device)

    return DeadlineCriteria(deadline)


def make_tail_string_criteria(tokenizer, stop: str, window: int = 4):
    """Fallback for ``stop_strings``: decode the last ``window`` generated tokens, stop when ``stop`` shows up.

    Note ``input_ids`` includes the prompt, so for a prompt whose tail already contains ``stop`` this fires
    immediately; callers pass a prompt that does not end with ``stop`` (ours ends with ``{"answer":"``).
    """
    import torch
    from transformers import StoppingCriteria

    class TailStringCriteria(StoppingCriteria):
        def __init__(self, tok, s, w):
            self.tok, self.stop, self.window = tok, s, w

        def __call__(self, input_ids, scores, **kwargs):
            done = []
            for row in input_ids:
                tail = self.tok.decode(row[-self.window:], skip_special_tokens=False)
                done.append(self.stop in tail)
            return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

    return TailStringCriteria(tokenizer, stop, window)
