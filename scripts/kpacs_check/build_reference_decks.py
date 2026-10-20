"""M2 reference-state checks on a PEARLSim equilibrium core: nuclear-data switch and power defect.

    python scripts/kpacs_check/build_reference_decks.py RUN_DECK STEP OUT_DIR [--pop "10000 765 15"]

Loads the core at the end of STEP as build_coef_decks.py does (same pebble placement and Serpent seed in
every case) and writes transport-only decks:
  d7_v1         the run's own rendering: ENDF/B-VII.0 (v1 library), v1 temperatures (959 K fuel, 900 K
                elsewhere), one shared 900 K graphite S(a,b), graphite S(a,b) on SiC, Li-6 50 ppm, core 1.41
  d8_uncoupled  v2 data at the same temperatures (thermal.uncoupled_conditions): ENDF/B-VIII.0 Lib80x,
                isotopic carbon, grph30 per component, c-sic/si-sic on SiC (docs/nuclear_data.md §7)
  d8_hzp        v2 data, hot zero power: thermal.core_conditions(0) (everything at the 823.15 K inlet)
  d8_hfp        v2 data, hot full power: thermal.core_conditions(280 MW) with this core's fuel pebbles
d8_uncoupled - d7_v1 is the data-switch delta-k at a reference state (nuclear_data.md §7, due in M2);
d8_hfp - d8_hzp is the power defect, to compare with the first-order estimate of about -660 pcm
(dataset_v2_design.md §1a). The compositions are the v1-data equilibrium (fine for these differences).
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_coef_decks import load_core, write_case  # noqa: E402

from pearlsim.nuclear_data import ENDF8_LIB80X  # noqa: E402
from pearlsim.thermal import core_conditions, uncoupled_conditions  # noqa: E402

ENDF8_DIR = "/global/home/groups/co_nuclear/serpent/xsdata/endf8"
ENDF8_SETTINGS = {
    "acelib": '"/global/scratch/users/ikolaja/PEARLSim_dev/xsdata/endf8_pearlsim.xsdata"',
    "declib": f'"{ENDF8_DIR}/endf8.dec"',
    "nfylib": f'"{ENDF8_DIR}/endf8.nfy"',
}


def _endf_float(text):
    text = text.strip()
    for i in range(1, len(text)):
        if text[i] in "+-" and text[i - 1] not in "eE":
            return float(text[:i] + "e" + text[i:])
    return float(text or 0)


def decay_zais(path=f"{ENDF8_DIR}/endf8.dec"):
    """ZAI (10*ZA + isomer state) of every material in an ENDF-6 decay file (first MF1/MT451 record)."""
    with open(path) as f:
        lines = f.read().split("\n")
    zais, current = set(), None
    for i, line in enumerate(lines):
        if len(line) < 75:
            continue
        mat = line[66:70]
        if line[70:75] == " 1451" and mat != current:
            zais.add(int(round(_endf_float(line[0:11]))) * 10 + int(lines[i + 1][33:44]))
        current = mat
    return zais


def drop_missing_decay_nuclides(text, zais):
    """Remove decay-only (ZAI, no library suffix) composition lines absent from the decay library.

    Needed only when v1 (ENDF/B-VII.0) compositions are rendered with v2 data: e.g. Gd-153m (641531)
    is in the VII.0 decay file but not in endf8.dec. Such nuclides carry no cross sections, so
    transport is unaffected. Returns (text, {zai: summed density removed}).
    """
    removed = {}

    def keep(match):
        zai, value = match.group(1), float(match.group(2))
        if int(zai) in zais:
            return match.group(0)
        removed[zai] = removed.get(zai, 0.0) + abs(value)
        return ""

    text = re.sub(r"^\s+(\d{4,7})\s+([-\d.eE+]+)\s*\n", keep, text, flags=re.M)
    return text, removed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("deck")
    parser.add_argument("step", type=int)
    parser.add_argument("out")
    parser.add_argument("--pop", default="10000 765 15")
    args = parser.parse_args(argv)
    sim = load_core(args.deck, args.step, args.out, args.pop)
    write_case(sim, "d7_v1", sim.core.conditions)
    sim.core.nuclear_data = ENDF8_LIB80X
    settings = {k: v for k, v in sim.serpent_settings.items() if k != "pdatadir"}
    settings.update(ENDF8_SETTINGS)
    fuel, nominal = sim.fuel_pebbles_in_core(), len(sim.core.pebble_locations)
    write_case(sim, "d8_uncoupled", uncoupled_conditions(), settings)
    write_case(sim, "d8_hzp", core_conditions(0.0, fuel, nominal_pebbles=nominal), settings)
    write_case(sim, "d8_hfp", core_conditions(float(sim.power), fuel, nominal_pebbles=nominal), settings)
    clean_v2_decks(("d8_uncoupled", "d8_hzp", "d8_hfp"))


def clean_v2_decks(names):
    zais = decay_zais()
    for name in names:
        with open(f"{name}.serpent") as f:
            text, removed = drop_missing_decay_nuclides(f.read(), zais)
        with open(f"{name}.serpent", "w") as f:
            f.write(text)
        print(name, "removed decay-only nuclides absent from endf8.dec:", removed)


if __name__ == "__main__":
    main()
