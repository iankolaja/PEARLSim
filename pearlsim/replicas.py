"""n independent Serpent replicas per depletion step, averaged (docs/dataset_v2_design.md section 1b).

Replicas of one step share the zone inventories and differ only in the pebble-to-
material assignment and the Serpent seed. With one replica the v1 file names are
kept; with n > 1 every per-replica file carries an ``_r<r>`` suffix:

    <sim>_<i>_r<r>.serpent, pebble_positions_<i>_r<r>.csv, <sim>_<i>_r<r>.serpent.bumat<k>, ...

Compositions are averaged nuclide by nuclide over the replicas (a nuclide missing
from a replica counts as zero), which to first order is depletion with the
shuffle-averaged flux.
"""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Nuclides whose replica spread is reported for every material
SPREAD_NUCLIDES = ("92235<lib>", "94239<lib>", "541350")
# Nuclides below this fraction of a material's total density are left out of its max spread
SPREAD_MIN_FRACTION = 1e-6


def stem(simulation_name, iteration, replica=None):
    """Deck name of one step (replica None: single-replica v1 naming)."""
    suffix = "" if replica is None else f"_r{replica}"
    return f"{simulation_name}_{iteration}{suffix}.serpent"


def pebble_file(iteration, replica=None):
    suffix = "" if replica is None else f"_r{replica}"
    return f"pebble_positions_{iteration}{suffix}.csv"


def replica_ids(num_replicas):
    """Replica indices used in file names: [None] for one replica, else 0 .. n-1."""
    return [None] if num_replicas == 1 else list(range(num_replicas))


def read_bumat(file_name):
    """Material compositions of a Serpent .bumat file as {material: {nuclide: atom density}}.

    Serpent's particle renaming (``<mat>pp<k>r1``) is undone, and library suffixes
    are replaced by ``<lib>`` so the material can be rewritten with any library.
    """
    materials = {}
    current = None
    with open(file_name) as f:
        for line in f:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "mat":
                current = materials.setdefault(fields[1].split("pp")[0], {})
            elif current is not None:
                za = fields[0].split(".")
                nuclide = za[0] + "<lib>" if len(za) > 1 else za[0]
                current[nuclide] = float(fields[1])
    return materials


def average_compositions(replica_materials):
    """Nuclide-wise mean of several {material: {nuclide: density}} sets, plus a spread summary.

    All replicas must hold the same materials (inventories don't depend on the
    shuffle); a nuclide absent from one replica counts as zero there.
    """
    names = list(replica_materials[0])
    for r, materials in enumerate(replica_materials[1:], start=1):
        if set(materials) != set(names):
            missing = sorted(set(names) ^ set(materials))
            raise ValueError(f"replica {r} has a different material set (differs in {missing[:5]} ...)")
    n = len(replica_materials)
    averaged, spread = {}, {}
    for name in names:
        nuclides = list(replica_materials[0][name])
        seen = set(nuclides)
        for materials in replica_materials[1:]:
            for nuclide in materials[name]:
                if nuclide not in seen:
                    nuclides.append(nuclide)
                    seen.add(nuclide)
        values = {nuc: [m[name].get(nuc, 0.0) for m in replica_materials] for nuc in nuclides}
        averaged[name] = {nuc: sum(v) / n for nuc, v in values.items()}
        if n > 1:
            spread[name] = _material_spread(values, averaged[name])
    return averaged, spread


def _material_spread(values, mean):
    """Relative sample standard deviation over replicas, for key nuclides and the largest one."""
    total = sum(mean.values())
    summary = {}
    worst, worst_nuclide = 0.0, None
    for nuclide, v in values.items():
        m = mean[nuclide]
        if m <= 0:
            continue
        rel = float(np.std(v, ddof=1) / m)
        if nuclide in SPREAD_NUCLIDES:
            summary[nuclide] = rel
        if m >= SPREAD_MIN_FRACTION * total and rel > worst:
            worst, worst_nuclide = rel, nuclide
    summary["max"] = worst
    summary["max_nuclide"] = worst_nuclide
    return summary


def write_spread(file_name, spread):
    with open(file_name, "w") as f:
        json.dump(spread, f)
    if spread:
        name = max(spread, key=lambda k: spread[k]["max"])
        logger.info("Replica spread: largest relative std %.3g (%s, %s); written to %s",
                    spread[name]["max"], name, spread[name]["max_nuclide"], file_name)
