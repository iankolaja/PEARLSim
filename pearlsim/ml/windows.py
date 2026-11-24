"""Windows over sequences with explicit alignment and direct multi-horizon targets.

A window ending at row i (the decision step) contains rows i-W+1 .. i of the input
columns: the controls u applied in each step and the measurements D taken before it.
For horizon h = 0 .. H:

* a ``step`` target is read at row i+h (e.g. k_eff of step i+h);
* a ``next`` target is read at row i+h+1 (e.g. the measurements D_{i+h+1} taken
  after step i+h, i.e. before the following step);
* the planned controls u_{i+1} .. u_{i+H} are given as ``future_controls``.

Horizon 0 reproduces the corrected one-step alignment of review #3 (window ..i,
targets rho_i and D_{i+1}). ``input_offset=1`` gives the v1 notebook's alignment
(window ..i-1 for the same targets), for reproducing the paper only. Targets that fall past the end of a sequence are
masked out rather than dropped, so every window with a full history is used.

Branch sequences (``Sequence.first_own_row`` f > 0, see ``pearlsim.ml.data``): a target at horizon h counts for
the branch only when step i+h is its own (i+h >= f), for both kinds. A ``step`` target before row f is the parent's
row; a ``next`` target at row f is the measurement taken before the branch's first step acts (the discharge at its
start, from the parent's state after the fork step), equal to the parent's row f, so it is the parent's sample too.
Such targets are masked; a window whose targets are all masked is dropped, while one that reaches the branch's own
steps at some horizon is kept (its future controls are the branch's).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TargetSpec:
    column: str
    kind: str = "step"  # "step": row i+h; "next": row i+h+1

    def __post_init__(self):
        if self.kind not in ("step", "next"):
            raise ValueError(f"target kind must be 'step' or 'next', not {self.kind!r}")


@dataclass
class WindowSet:
    past: np.ndarray              # (N, W, F) input rows i-W+1 .. i
    future_controls: np.ndarray   # (N, H, C) controls of rows i+1 .. i+H (H may be 0)
    targets: np.ndarray           # (N, H+1, T)
    mask: np.ndarray              # (N, H+1, T) bool, target available
    sequence: np.ndarray          # (N,) sequence name
    family: np.ndarray            # (N,) sequence family
    row: np.ndarray               # (N,) decision row i
    input_columns: list = field(default_factory=list)
    control_columns: list = field(default_factory=list)
    target_columns: list = field(default_factory=list)

    def __len__(self):
        return len(self.past)

    def subset(self, selection):
        return WindowSet(self.past[selection], self.future_controls[selection], self.targets[selection],
                         self.mask[selection], self.sequence[selection], self.family[selection],
                         self.row[selection], self.input_columns, self.control_columns, self.target_columns)


def make_windows(sequences, input_columns, targets, window=8, horizon=0, control_columns=None, input_offset=0):
    """WindowSet from sequences (``pearlsim.ml.data.Sequence``), see the module docstring."""
    targets = [t if isinstance(t, TargetSpec) else TargetSpec(*t) if isinstance(t, tuple) else TargetSpec(t)
               for t in targets]
    control_columns = list(control_columns or [])
    past, future, values, masks, names, families, rows = [], [], [], [], [], [], []
    for seq in sequences:
        x = seq.frame[input_columns].to_numpy(dtype=float)
        u = seq.frame[control_columns].to_numpy(dtype=float) if control_columns else np.zeros((len(x), 0))
        y = seq.frame[[t.column for t in targets]].to_numpy(dtype=float)
        n = len(x)
        first_own = getattr(seq, "first_own_row", 0)
        for i in range(window - 1 + input_offset, n):
            if i + horizon < first_own:
                continue  # every target is the parent's sample
            past.append(x[i - window + 1 - input_offset: i + 1 - input_offset])
            fut = np.zeros((horizon, len(control_columns)))
            for h in range(1, horizon + 1):
                if i + h < n:
                    fut[h - 1] = u[i + h]
            future.append(fut)
            value = np.zeros((horizon + 1, len(targets)))
            mask = np.zeros((horizon + 1, len(targets)), dtype=bool)
            for h in range(horizon + 1):
                for j, spec in enumerate(targets):
                    r = i + h + (spec.kind == "next")
                    if i + h >= first_own and r < n and not np.isnan(y[r, j]):
                        value[h, j], mask[h, j] = y[r, j], True
            values.append(value)
            masks.append(mask)
            names.append(seq.name)
            families.append(seq.family)
            rows.append(i)
    count = len(past)
    return WindowSet(np.asarray(past, dtype=np.float32).reshape(count, window, len(input_columns)),
                     np.asarray(future, dtype=np.float32).reshape(count, horizon, len(control_columns)),
                     np.asarray(values, dtype=np.float32).reshape(count, horizon + 1, len(targets)),
                     np.asarray(masks, dtype=bool).reshape(count, horizon + 1, len(targets)),
                     np.asarray(names), np.asarray(families), np.asarray(rows),
                     list(input_columns), control_columns, [t.column for t in targets])
