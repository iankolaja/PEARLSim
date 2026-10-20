"""M2 pilot noise study: transport-only decks of one fixed core state (dataset_v2_design.md §2).

    python scripts/pilot/build_pilot_decks.py RUN_DECK STEP OUT_DIR [--data endf8|endf7] [--prefix eq]

Loads the core at the end of STEP (see scripts/kpacs_check/build_coef_decks.py: OUT_DIR is a sibling of
the run directory) and writes 5 pebble shuffles x 2 histories x 2 Serpent seeds = 20 decks named
<prefix>_sh<i>_h<1|4>_ms<j>.serpent. h1 is v1's production population (24000 per cycle, 20 active of 30);
h4 has 4x the histories as 4x the active cycles (80 of 90) at the same population, so memory does not
grow. Each shuffle has its own pebble file. --data endf8 renders v2 data at hot full power
(thermal.core_conditions); endf7 keeps the run's own rendering.

The spread over shuffles at fixed Serpent seed vs over Serpent seeds at fixed shuffle separates the
reshuffle and Monte Carlo parts of the per-step noise in k and the meshes.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kpacs_check"))

from build_coef_decks import load_core, write_case  # noqa: E402
from build_reference_decks import ENDF8_SETTINGS, clean_v2_decks  # noqa: E402

from pearlsim.nuclear_data import ENDF8_LIB80X  # noqa: E402
from pearlsim.thermal import core_conditions  # noqa: E402

SHUFFLE_SEEDS = (11, 12, 13, 14, 15)
SERPENT_SEEDS = ("1001", "1002")
HISTORIES = {"h1": "24000 30 10", "h4": "24000 90 10"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("deck")
    parser.add_argument("step", type=int)
    parser.add_argument("out")
    parser.add_argument("--data", choices=("endf8", "endf7"), default="endf8")
    parser.add_argument("--prefix", default="eq")
    args = parser.parse_args(argv)
    sim = load_core(args.deck, args.step, args.out, HISTORIES["h1"])
    settings = dict(sim.serpent_settings)
    conditions = sim.core.conditions
    if args.data == "endf8":
        sim.core.nuclear_data = ENDF8_LIB80X
        settings = {k: v for k, v in settings.items() if k != "pdatadir"}
        settings.update(ENDF8_SETTINGS)
        conditions = core_conditions(float(sim.power), sim.fuel_pebbles_in_core(),
                                     nominal_pebbles=len(sim.core.pebble_locations))
    names = []
    for i, shuffle in enumerate(SHUFFLE_SEEDS):
        for h, pop in HISTORIES.items():
            for j, seed in enumerate(SERPENT_SEEDS):
                name = f"{args.prefix}_sh{i}_{h}_ms{j}"
                write_case(sim, name, conditions, {**settings, "pop": pop, "seed": seed}, shuffle_seed=shuffle,
                           pebble_file=f"pebbles_{args.prefix}_sh{i}.csv")
                names.append(name)
    if args.data == "endf8":
        clean_v2_decks(names)


if __name__ == "__main__":
    main()
