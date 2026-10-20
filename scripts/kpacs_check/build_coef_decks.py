"""Reactivity-coefficient decks for a PEARLSim equilibrium core (gFHR benchmark Table 5 comparison).

    python scripts/kpacs_check/build_coef_decks.py RUN_DECK STEP OUT_DIR [--pop "50000 75 15"]

RUN_DECK: the run's PEARLSim deck (e.g. runs/v2_kpacs_eq/gFHR_v2_kpacs_eq.inp); its setup lines
(materials, zones, Serpent settings, benchmark corrections) are replayed, then the core is loaded at the
end of STEP (zone map, pebble positions and .bumat1 of that step in the run directory). OUT_DIR must be
a sibling of the run directory (deck paths resolve against its parent, as in a PEARLSim run).

Each case is a transport-only deck (dep card and burn flags removed) with identical pebble placement and
Serpent seed. Coefficient = (rho_high - rho_low) / (x_high - x_low).

Benchmark method (Satvat et al. 2021, §4.1, Table 5): +/-100 K on kernels (FTC), pebble graphite
(MTC), coolant with its density 2413 - 0.488 T kg/m3 (CTC) and reflector (RTC); void: coolant density
+/-2.5%. Here the weak coefficients use wider perturbations so that a ~3 h job resolves them
(statistical error per k ~80 pcm): FTC +/-100 K as in the benchmark; CTC +/-200 K; MTC (pebble core,
shell and fuel-zone matrix) and RTC +/-300 K; void +/-10% density. Nominal: the run's temperatures
(kernels, coatings, matrix 959 K; everything else 900 K). Graphite S(a,b) is interpolated per component
in every case (the run shares one 900 K interpolation), so differences reflect only the perturbation.
Library IDs follow each material's temperature (base library at or below it: .06c below 900 K).
"""

import argparse
import os
import shutil
import sys
from dataclasses import replace

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_ecc_decks import transport_only  # noqa: E402

from pearlsim.nuclear_data import validate_deck  # noqa: E402
from pearlsim.simulation import Simulation  # noqa: E402

SEED = "20261019"


def salt_density(temperature):
    """FLiBe density in g/cm3 (Satvat 2021 Table 2: 2413.0 - 0.488 T[K] kg/m3)."""
    return (2413.0 - 0.488 * temperature) / 1000.0


def cases(nominal):
    rho0 = salt_density(nominal.salt)
    return {
        "fuel_m100": replace(nominal, kernel=nominal.kernel - 100),
        "fuel_p100": replace(nominal, kernel=nominal.kernel + 100),
        "coolant_m200": replace(nominal, salt=nominal.salt - 200, salt_density=salt_density(nominal.salt - 200)),
        "coolant_p200": replace(nominal, salt=nominal.salt + 200, salt_density=salt_density(nominal.salt + 200)),
        "moderator_m300": replace(nominal, pebble_graphite=nominal.pebble_graphite - 300, matrix=nominal.matrix - 300),
        "moderator_p300": replace(nominal, pebble_graphite=nominal.pebble_graphite + 300, matrix=nominal.matrix + 300),
        "reflector_m300": replace(nominal, reflector=nominal.reflector - 300),
        "reflector_p300": replace(nominal, reflector=nominal.reflector + 300),
        "void_m10": replace(nominal, salt_density=rho0 * 1.10),  # 10% less void
        "void_p10": replace(nominal, salt_density=rho0 * 0.90),
        # coolant diagnostics: temperature alone (density fixed), and the benchmark's +/-100 K with density
        "coolant_tonly_m200": replace(nominal, salt=nominal.salt - 200, salt_density=rho0),
        "coolant_tonly_p200": replace(nominal, salt=nominal.salt + 200, salt_density=rho0),
        "coolant_m100": replace(nominal, salt=nominal.salt - 100, salt_density=salt_density(nominal.salt - 100)),
        "coolant_p100": replace(nominal, salt=nominal.salt + 100, salt_density=salt_density(nominal.salt + 100)),
        # density alone at the coolant_m200/p200 densities (salt at 900 K), and that pair at higher statistics
        "coolant_donly_m200": replace(nominal, salt_density=salt_density(nominal.salt - 200)),
        "coolant_donly_p200": replace(nominal, salt_density=salt_density(nominal.salt + 200)),
        "coolant_m200_hs": replace(nominal, salt=nominal.salt - 200, salt_density=salt_density(nominal.salt - 200)),
        "coolant_p200_hs": replace(nominal, salt=nominal.salt + 200, salt_density=salt_density(nominal.salt + 200)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("deck")
    parser.add_argument("step", type=int)
    parser.add_argument("out")
    parser.add_argument("--pop", default="50000 75 15")
    parser.add_argument("--cases", nargs="*", help="only these cases (default: all)")
    args = parser.parse_args(argv)
    deck = os.path.abspath(args.deck)
    run_dir = os.path.join(os.path.dirname(deck), os.path.splitext(os.path.basename(deck))[0])
    out = os.path.abspath(args.out)
    if os.path.dirname(out) != os.path.dirname(run_dir):
        raise SystemExit("OUT_DIR must be a sibling of the run directory")
    os.makedirs(out, exist_ok=True)
    sim = Simulation("gFHR_equilibrium")
    stem = f"{sim.simulation_name}_{args.step}.serpent"
    for name in (f"zone_map{args.step}.json", f"{stem}.bumat1"):
        link = os.path.join(out, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(run_dir, name), link)
    # a copy, not a link: generate_input rewrites this file with the cases' own shuffle
    pebbles = os.path.join(out, f"pebble_positions_{args.step}.csv")
    if os.path.islink(pebbles):
        os.remove(pebbles)
    shutil.copyfile(os.path.join(run_dir, f"pebble_positions_{args.step}.csv"), pebbles)
    os.chdir(out)
    with open(deck) as f:
        for line in f:
            keyword = line.split("#", 1)[0].split()[:1]
            if keyword and keyword[0] in ("initialize", "insertion"):
                continue
            sim.read_input_line(line)
    sim.serpent_settings["pop"] = args.pop
    sim.serpent_settings["seed"] = SEED
    sim.load_from_step(args.step, 1)
    sim.update_conditions()
    nominal = replace(sim.core.conditions, shared_graphite_sab=None)
    with open("cases.tsv", "a") as table:
        for name, conditions in cases(nominal).items():
            if args.cases and name not in args.cases:
                continue
            sim.core.conditions = conditions
            generated = sim.core.generate_input(dict(sim.serpent_settings), sim.burnup_time_step, 0,
                                                cr_position=sim.rod_insertion, rng=np.random.default_rng(int(SEED)))
            with open(generated) as f:
                text = transport_only(f.read())
            problems = validate_deck(text, sim.core.nuclear_data)
            if problems:
                raise SystemExit(f"{name}: {problems[:5]}")
            with open(f"{name}.serpent", "w") as f:
                f.write(text)
            os.remove(generated)
            table.write(f"{name}\t{conditions}\n")
            print(name, conditions)


if __name__ == "__main__":
    main()
