"""M3: rod worth vs fuel density on fresh, uniformly loaded cores (v2 data), transport only.

    python scripts/rod_worth/build_fresh_rod_decks.py SETUP_DECK OUT_DIR --densities 0.1 0.25 0.5 0.75 1.0
        [--power 84e6] [--pop "24000 90 10"] [--depths 369.47 215 150 60]

SETUP_DECK supplies the core set-up lines (materials, zones, geometry, Serpent settings; its initialize and
insertion lines are ignored), e.g. runs/v2_runin_pilot/gFHR_v2_runin_pilot.inp. Each density d gets
OUT_DIR/d<d>/ with a fresh core initialised to fuel fraction d everywhere (graphite pebbles otherwise) and
decks rod_z<depth>_ms0.serpent sharing one pebble file and Serpent seed (common random numbers).
The pilot (dataset_v2_design.md §1c) showed W_tot follows the fuel density where there is fuel and moves
< ~170 pcm with burnup, so fresh cores map W_tot(density); the 0.5 and 1.0 cores compare with burned states.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kpacs_check"))

from build_coef_decks import write_case  # noqa: E402
from build_reference_decks import v2_settings  # noqa: E402

from pearlsim.nuclear_data import ENDF8_LIB80X  # noqa: E402
from pearlsim.simulation import Simulation  # noqa: E402
from pearlsim.thermal import core_conditions  # noqa: E402

LINKS = ("gFHR_template.serpent", "gFHR_triso.dat", "gFHR_points.dat", "materials")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("deck")
    parser.add_argument("out")
    parser.add_argument("--densities", nargs="+", type=float, required=True)
    parser.add_argument("--power", type=float, default=84e6)
    parser.add_argument("--pop", default="24000 90 10")
    parser.add_argument("--depths", nargs="+", type=float, default=[369.47, 215.0, 150.0, 60.0])
    parser.add_argument("--seed", default="1001")
    args = parser.parse_args(argv)
    deck_path, out = os.path.abspath(args.deck), os.path.abspath(args.out)
    with open(deck_path) as f:
        lines = [line for line in f
                 if line.split("#", 1)[0].split()[:1] not in (["initialize"], ["insertion"])]
    os.makedirs(out, exist_ok=True)
    deck_dir = os.path.dirname(deck_path)
    for name in LINKS + tuple(sorted(f for f in os.listdir(deck_dir) if f.endswith("_zones_exact.inp"))):
        link = os.path.join(out, name)
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(os.path.join(deck_dir, name)), link)
    for density in args.densities:
        state = os.path.join(out, f"d{density:g}")
        os.makedirs(state, exist_ok=True)
        os.chdir(state)
        sim = Simulation("gFHR_equilibrium")
        for line in lines:
            sim.read_input_line(line)
        sim.core.rng = np.random.default_rng(7)
        sim.read_input_line(f"initialize 0 graphpeb {1 - density:g} fuel20 {density:g}")
        sim.core.nuclear_data = ENDF8_LIB80X
        settings = {**v2_settings(sim.serpent_settings), "pop": args.pop, "seed": args.seed,
                    "power": f"{args.power:g}"}
        fuel = sim.fuel_pebbles_in_core()
        conditions = core_conditions(args.power, fuel, nominal_pebbles=len(sim.core.pebble_locations))
        print(f"density {density:g}: {fuel} fuel pebbles, kernel {conditions.kernel:.1f} K")
        for depth in args.depths:
            sim.rod_insertion = depth
            write_case(sim, f"rod_z{depth:g}_ms0", conditions, settings, shuffle_seed=21,
                       pebble_file="pebbles_rods.csv")
        with open("state.txt", "w") as f:
            f.write(f"setup_deck {deck_path}\nfresh_core 1\nfuel_fraction {fuel / len(sim.core.pebble_locations):.4f}\n"
                    f"fuel_density {density:g}\npower_W {args.power:g}\nfuel_pebbles {fuel}\n")


if __name__ == "__main__":
    main()
