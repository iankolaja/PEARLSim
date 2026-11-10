"""Load sequences from v1 per-sequence CSVs or from tables gathered by ``pearlsim.gather``.

Columns use the v2 record names (``pearlsim.records.flatten``). v1 columns are
renamed on loading: ``final_analog_keff`` -> ``keff_eos``; v1 has no exact fuel
pebble count, so ``power_per_fuel_pebble`` is estimated as in v1 (see
``features.v1_power_per_fuel_pebble``).

Branch runs (``fork_from``, decision 5a): a branch's sequence is its parent's rows up to the fork step followed
by its own rows, so models see the history that led to the branch. Only rows from ``first_own_row`` on are the
branch's own data: windows and tokens mask every target read from the earlier rows, which already count as the
parent's samples. A branch takes its root run's family, so a run and its branches share a side of every split.
"""

import glob
import os
import re
from dataclasses import dataclass

import pandas as pd

V1_RENAME = {"final_analog_keff": "keff_eos", "final_analog_keff_unc": "keff_eos_unc",
             "cumulative_time": "days"}


def v1_family(name):
    """Sequence family: the name without its circulation-time suffix (``_40s``, ``_60s_75``, ...).

    Sequences of one family share their control history apart from that setting, so
    they must sit on the same side of a split.
    """
    return re.sub(r"_(\d+)s(_\d+)?$", "", name)


@dataclass
class Sequence:
    name: str
    family: str
    frame: pd.DataFrame  # one row per step, in step order
    first_own_row: int = 0  # rows before it are a parent's (branch runs): context, not samples
    parent: str = None

    def __len__(self):
        return len(self.frame)

    def own_frame(self):
        """The rows that are this sequence's own data (all rows unless it is a branch)."""
        return self.frame.iloc[self.first_own_row:]


def attach_parents(sequences):
    """Give branch sequences (``fork_parent`` / ``fork_step`` columns) their parent's rows as a prefix.

    The prefix is the parent's frame (itself with any prefix) up to the fork step; the branch keeps its own rows
    after it, marks where they start (``first_own_row``) and takes the parent's family. Raises ValueError when a
    parent is missing or a fork step is not among the parent's steps.
    """
    by_name = {seq.name: seq for seq in sequences}
    done = set()

    def resolve(seq, chain=()):
        if seq.name in done or "fork_parent" not in seq.frame or seq.frame["fork_parent"].isna().all():
            done.add(seq.name)
            return
        if seq.name in chain:
            raise ValueError(f"branch cycle through {seq.name}")
        parent_name = seq.frame["fork_parent"].dropna().iloc[0]
        fork_step = int(seq.frame["fork_step"].dropna().iloc[0])
        if parent_name not in by_name:
            raise ValueError(f"{seq.name} branches from {parent_name}, which is not in the data")
        parent = by_name[parent_name]
        resolve(parent, chain + (seq.name,))
        steps = parent.frame["step"].to_numpy()
        if fork_step not in steps:
            raise ValueError(f"{seq.name} branches after step {fork_step}, which {parent_name} does not have")
        prefix = parent.frame[steps <= fork_step]
        own = seq.frame[seq.frame["step"] > fork_step]
        seq.frame = pd.concat([prefix, own], ignore_index=True)
        seq.first_own_row, seq.parent, seq.family = len(prefix), parent_name, parent.family
        done.add(seq.name)

    for seq in sequences:
        resolve(seq)
    return sequences


def load_v1_directory(directory, exclude=(), family=v1_family):
    """v1 ``sequence_data/<name>_inputs.csv.gz`` + ``_outputs.csv.gz`` pairs as sequences.

    exclude: regular expressions; sequences whose name matches one are skipped
    (e.g. the v1 forecast perturbation runs, which branch off a shared base).
    """
    from pearlsim.ml.features import v1_power_per_fuel_pebble

    sequences = []
    for path in sorted(glob.glob(os.path.join(directory, "*_inputs.csv.gz"))):
        name = os.path.basename(path)[: -len("_inputs.csv.gz")]
        if any(re.search(pattern, name) for pattern in exclude):
            continue
        inputs = pd.read_csv(path, index_col=0)
        outputs = pd.read_csv(path.replace("_inputs.csv.gz", "_outputs.csv.gz"), index_col=0)
        if len(inputs) != len(outputs):
            raise ValueError(f"{name}: {len(inputs)} input rows but {len(outputs)} output rows")
        frame = pd.concat([inputs.reset_index(drop=True), outputs.reset_index(drop=True)], axis=1)
        frame = frame.rename(columns=V1_RENAME)
        frame["power_per_fuel_pebble"] = v1_power_per_fuel_pebble(frame)
        sequences.append(Sequence(name, family(name), frame))
    return sequences


def load_gathered_table(path, family=None, exclude=()):
    """Sequences from a ``pearlsim.gather`` table (Parquet or CSV), one per ``sequence`` value.

    family: function of the sequence name (default: v1_family), or the name of a column.
    """
    table = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    sequences = []
    for name, frame in table.groupby("sequence", sort=True):
        if any(re.search(pattern, name) for pattern in exclude):
            continue
        frame = frame.sort_values("step").reset_index(drop=True)
        if isinstance(family, str):
            fam = frame[family].iloc[0]
        else:
            fam = (family or v1_family)(name)
        sequences.append(Sequence(name, fam, frame))
    return attach_parents(sequences)
