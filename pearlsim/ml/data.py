"""Load sequences from v1 per-sequence CSVs or from tables gathered by ``pearlsim.gather``.

Columns use the v2 record names (``pearlsim.records.flatten``). v1 columns are
renamed on loading: ``final_analog_keff`` -> ``keff_eos``; v1 has no exact fuel
pebble count, so ``power_per_fuel_pebble`` is estimated as in v1 (see
``features.v1_power_per_fuel_pebble``).
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

    def __len__(self):
        return len(self.frame)


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
    return sequences
