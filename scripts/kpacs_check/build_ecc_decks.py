"""Load Kairos's published gFHR equilibrium compositions (ECC) into PEARLSim's Serpent model.

    python scripts/kpacs_check/build_ecc_decks.py ECC_JSON OUT_DIR

Separates the Serpent model and nuclear data from depletion: the v1 PEARLSim equilibrium
(gFHR_benchmark_1975, pass mode, 8 passes) has k ~1.033-1.045, while KPACS reports
1.01230 +/- 0.00068 and Serpent with the KPACS compositions 1.01443 +/- 0.00023 (320 materials,
random bed) and 1.01472 +/- 0.00019 (40 zone-averaged materials, hexagonal lattice), both
ENDF/B-VII.1 with graphite S(a,b) off (Satvat et al. 2021, §4.1, §4.4 and Table 10).

ECC_JSON: {"Z<a>R<r>": {"ZZAAA": atom/b/cm, ...}} parsed from Kairos's "Equilibrium Core
Concentrations" PDF (40 kernel compositions, 88-89 nuclides; docs/literature).

The core is all fuel (250,131 pebble positions of gFHR_points.dat; benchmark 250,190) on the
benchmark zoning: 4 equal-width radial zones (30/60/90/120 cm) and 10 equal axial zones of the
309.47 cm bed (Satvat 2021 §4.1). Zone (a, r) pebbles carry the ECC kernel composition of that
zone; coatings, matrix and pebble graphite are PEARLSim's. Transport only: the dep card and burn
flags are removed. Nuclear data stays the v1 ENDF/B-VII.0 library (docs/nuclear_data.md).

Variants (one directory each), in steps toward the benchmark's conditions:
  A_pearlsim       PEARLSim v1 model: graphite S(a,b) on; fuel-pebble components at 959 K;
                   the 10 control-rod channels in the reflector (rods withdrawn, FLiBe below)
  B_tsl_off        A without graphite S(a,b) (free-gas carbon), as in the Kairos k values
  C_tsl_off_900K   B with coatings, matrix and pebble graphite at 900 K (benchmark: all at 900 K,
                   kernels 959 K; Satvat 2021 p. 4)
  D_benchmark      C with the rod channels filled with reflector graphite (the benchmark has none)
  E_benchmark_li6  D with Flibe Li-6 at the specified 50 ppm of lithium (99.995 at.% Li-7; Satvat 2021
                   Table 2). The v1 template has 0.00005 Li-6 per 2 Li, i.e. 25 ppm (diagnostic only)
  F_pearlsim_li6   A with the same Li-6 correction
Serpent cards follow docs/serpent/cards_reference.md (mat tmp/moder, therm, cell, set pop).
"""

import argparse
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from pearlsim.core import Core  # noqa: E402
from pearlsim.material import Material  # noqa: E402

BED_BOTTOM, BED_HEIGHT = 60.0, 309.47
RADIAL_BOUNDS = (30.0, 60.0, 90.0, 120.0)
SETTINGS = {
    "pop": "50000 100 30",
    "seed": "20261016",
    "acelib": '"/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7u.xsdata"',
    "declib": '"/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7.dec"',
    "nfylib": '"/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7.nfy"',
    "opti": "2", "gcu": "-1", "memfrac": "0.99", "repro": "0", "shbuf": "0 0", "outp": "100000",
    "nbuf": "1000", "ures": "1", "power": "2.8E+08",
}
VARIANTS = ("A_pearlsim", "B_tsl_off", "C_tsl_off_900K", "D_benchmark", "E_benchmark_li6", "F_pearlsim_li6")


def write_zones(path):
    divisions = " ".join(f"{BED_BOTTOM + BED_HEIGHT * k / 10:.3f}" for k in range(1, 10))
    with open(path, "w") as f:
        for r in RADIAL_BOUNDS:
            f.write(f"radial_channel\nz_divisions {divisions}\npoint 0 {r}\npoint 429 {r}\nend\n\n")


def build_base(ecc, out):
    case = os.path.join(out, "case")
    os.makedirs(case, exist_ok=True)
    write_zones(os.path.join(case, "zones.inp"))
    core = Core("ecc")
    core.define_zones(os.path.join(case, "zones.inp"), os.path.join(REPO, "gFHR_points.dat"))
    core.fresh_pebbles["fuel20"] = Material("fuel20", os.path.join(REPO, "materials", "fuel20.dat"))
    core.fresh_pebbles["graphpeb"] = Material("graphpeb", os.path.join(REPO, "materials", "graphpeb.dat"))
    for key, name in (("SiC", "SiC_gFHR"), ("PyC", "PyC_gFHR"), ("buffer", "buffer_gFHR"), ("matrix", "matrix_gFHR")):
        core.static_fuel_materials[key] = Material(key, os.path.join(REPO, "materials", f"{name}.dat"))
    with open(os.path.join(REPO, "gFHR_template.serpent")) as f:
        core.core_geometry = f.read()
    rng = np.random.default_rng(int(SETTINGS["seed"]))
    fresh = core.fresh_pebbles["fuel20"]
    for r, channel in enumerate(core.zones, start=1):
        for a, zone in enumerate(channel, start=1):
            zone.initialize({"fuel20": 1.0}, rng=rng)
            core.initialize_materials(zone.inventory)
            (name,) = zone.inventory
            material = core.materials[name]
            material.concentrations = {f"{za}<lib>": value for za, value in ecc[f"Z{a}R{r}"].items()}
            material.temperature = fresh.temperature
    core.rng = rng
    cwd = os.getcwd()
    os.chdir(case)
    try:
        deck_name = core.generate_input(SETTINGS, 6.525, 0, cr_position=429.469)  # rods fully withdrawn
    finally:
        os.chdir(cwd)
    with open(os.path.join(case, deck_name)) as f:
        deck = f.read()
    pebbles = os.path.join(case, f"pebble_positions_{core.iteration}.csv")
    return deck, pebbles


def transport_only(deck):
    deck, n = re.subn(r"^dep daystep[^\n]*\n?", "", deck, flags=re.M)
    if n != 1:
        raise ValueError("expected one dep card")
    return re.sub(r" burn 1\b", "", deck)


def li6_50ppm(deck):
    """Flibe lithium at 99.995 at.% Li-7: 2 Li per formula unit = 1.9999 Li-7 + 0.0001 Li-6."""
    deck, n7 = re.subn(r"^(\s*3007\.09c\s+)1\.999950\s*$", r"\g<1>1.999900", deck, flags=re.M)
    deck, n6 = re.subn(r"^(\s*3006\.09c\s+)0\.000050\s*$", r"\g<1>0.000100", deck, flags=re.M)
    if n7 != 2 or n6 != 2:  # FLiBe and FLiBeInC
        raise ValueError(f"expected 2 Flibe materials, changed {n7} Li-7 and {n6} Li-6 lines")
    return deck


def variant(deck, name):
    if name == "F_pearlsim_li6":
        return li6_50ppm(deck)
    if name == "E_benchmark_li6":
        return li6_50ppm(variant(deck, "D_benchmark"))
    if name == "A_pearlsim":
        return deck
    deck = re.sub(r"\s+moder\s+grph\s+6000", "", deck)
    deck, n = re.subn(r"^therm grph [^\n]*\n", "", deck, flags=re.M)
    if n != 1 or "moder" in deck:
        raise ValueError("graphite S(a,b) not fully removed")
    if name == "B_tsl_off":
        return deck
    lines = []
    for line in deck.splitlines(keepends=True):
        # fuel-pebble components other than the kernel (coatings, matrix, pebble graphite) to 900 K
        if line.startswith("mat ") and re.match(r"mat fuel20_R\d+Z\d+G\d+_(SiC|PyC|buffer|matrix) ", line):
            line = re.sub(r"tmp 959(\.0)?\b", "tmp 900", line)
        lines.append(line)
    deck = "".join(lines)
    if name == "C_tsl_off_900K":
        return deck
    deck, n = re.subn(r"^(cell cr\d+(?:_flibe)?\s+0\s+)(B4C|FLiBe)(\s)", r"\1reflector\3", deck, flags=re.M)
    if n != 20:
        raise ValueError(f"expected 20 rod-channel cells, found {n}")
    return deck


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ecc_json")
    parser.add_argument("out")
    args = parser.parse_args(argv)
    out = os.path.abspath(args.out)
    with open(args.ecc_json) as f:
        ecc = json.load(f)
    deck, pebbles = build_base(ecc, out)
    deck = transport_only(deck)
    triso_ids = sorted(set(re.findall(r'"trisos/triso(\d+)"', deck)), key=int)
    trisos = os.path.join(out, "trisos")
    os.makedirs(trisos, exist_ok=True)
    with open(os.path.join(REPO, "gFHR_triso.dat")) as f:
        triso_lines = [line.split()[:4] for line in f if line.strip()]
    for i in triso_ids:
        with open(os.path.join(trisos, f"triso{i}"), "w") as f:
            f.writelines(f"{x} {y} {z} {r} p{i}\n" for x, y, z, r in triso_lines)
    pebble_name = os.path.basename(pebbles)
    for name in VARIANTS:
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.serpent"), "w") as f:
            f.write(variant(deck, name))
        for link, target in (("trisos", trisos), (pebble_name, pebbles)):
            if not os.path.lexists(os.path.join(d, link)):
                os.symlink(target, os.path.join(d, link))
    print(f"{len(triso_ids)} fuel materials; variants {', '.join(VARIANTS)} in {out}")


if __name__ == "__main__":
    sys.exit(main())
