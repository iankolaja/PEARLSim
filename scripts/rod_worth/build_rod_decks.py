"""M3 pilot: rod-worth curves at fixed core states with common random numbers (dataset_v2_design.md §1c).

    python scripts/rod_worth/build_rod_decks.py RUN_DECK STEP OUT_DIR --power W [--seeds 1001 ...]
        [--pop "24000 30 10"]

Loads the core at the end of STEP (scripts/kpacs_check/build_coef_decks.py: OUT_DIR is a sibling of the run
directory) and writes transport-only v2-data decks (thermal.core_conditions at --power W; opti 1, default
nbuf, shared fuel coating/matrix materials) rod_z<depth>_ms<j>.serpent for each rod depth and Serpent seed.
All depths share one pebble file and, per j, one Serpent seed, so worth differences have correlated noise.

Rod position = z of the B4C tip (gFHR_template.serpent: B4C from the plane to 429.47 cm, FLiBe below);
the pebble bed spans 60-369.47 cm, so 369.47 is fully withdrawn and 60 fully inserted.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kpacs_check"))

from build_coef_decks import load_core, write_case  # noqa: E402
from build_reference_decks import clean_v2_decks, v2_settings  # noqa: E402

from pearlsim.nuclear_data import ENDF8_LIB80X  # noqa: E402
from pearlsim.thermal import core_conditions  # noqa: E402

DEPTHS = (369.47, 320.0, 280.0, 240.0, 215.0, 190.0, 150.0, 110.0, 60.0)
SHUFFLE_SEED = 21


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("deck")
    parser.add_argument("step", type=int)
    parser.add_argument("out")
    parser.add_argument("--power", type=float, required=True, help="thermal power (W) for the temperatures")
    parser.add_argument("--seeds", nargs="+", default=["1001"])
    parser.add_argument("--pop", default="24000 30 10")
    args = parser.parse_args(argv)
    deck = os.path.abspath(args.deck)  # load_core changes into OUT_DIR
    sim = load_core(deck, args.step, args.out, args.pop)
    sim.core.nuclear_data = ENDF8_LIB80X
    settings = v2_settings(sim.serpent_settings)
    settings["power"] = f"{args.power:g}"
    conditions = core_conditions(args.power, sim.fuel_pebbles_in_core(), nominal_pebbles=len(sim.core.pebble_locations))
    fuel_fraction = sim.fuel_pebbles_in_core() / sum(z.num_pebbles for ch in sim.core.zones for z in ch)
    print(f"step {args.step}: fuel fraction {fuel_fraction:.3f}, kernel {conditions.kernel:.1f} K")
    names = []
    for j, seed in enumerate(args.seeds):
        for depth in DEPTHS:
            sim.rod_insertion = depth
            name = f"rod_z{depth:g}_ms{j}"
            write_case(sim, name, conditions, {**settings, "seed": seed}, shuffle_seed=SHUFFLE_SEED,
                       pebble_file="pebbles_rods.csv")
            names.append(name)
    clean_v2_decks(names)
    with open("state.txt", "w") as f:
        f.write(f"run_deck {deck}\nstep {args.step}\npower_W {args.power:g}\n"
                f"fuel_fraction {fuel_fraction:.4f}\nfuel_pebbles {sim.fuel_pebbles_in_core()}\n")


if __name__ == "__main__":
    main()
