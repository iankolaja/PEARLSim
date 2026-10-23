"""Real-Serpent integration tests (marker ``serpent``; batch jobs only: ``sbatch slurm/test_serpent.sub``).

They run the real executable on the full gFHR core (gFHR_points.dat, 50% fresh fuel) at a small population,
because the synthetic test core has no pebbles inside the template's bed. Each Serpent run takes a few
minutes and ~20 GB. They cover what fake Serpent cannot: that the rendered decks are accepted and run.
Found this way in M2 (2026-10-20/22): S(a,b) names shared by isotopes, `therm` on Lib80x tables, per-fuel
component materials exceeding memory, undefined graphite universes after a restart.

Environment: PEARLSIM_SERPENT (default sss2_HxF_dev), SLURM_CPUS_PER_TASK (threads, default 8).
"""

import json
import os
import re

import numpy as np
import pytest

from pearlsim.nuclear_data import ENDF8_LIB80X, missing_library_entries, validate_deck
from pearlsim.simulation import Simulation

pytestmark = pytest.mark.serpent

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENDF7 = "/global/home/groups/co_nuclear/serpent/xsdata/endfb7"
ENDF8 = "/global/home/groups/co_nuclear/serpent/xsdata/endf8"
ENDF8_XSDATA = "/global/scratch/users/ikolaja/PEARLSim_dev/xsdata/endf8_pearlsim.xsdata"
POP = "set pop 2000 20 5"
PRINTM = "set printm 1 1.0E-41"  # writes the .bumat files PEARLSim reads

COMMON = [
    "set_seed 11",
    'core_geometry "gFHR_template.serpent"',
    "triso_file 400 gFHR_triso.dat",
    "fresh fuel20 materials/fuel20.dat graphpeb materials/graphpeb.dat",
    "static_fuel_materials SiC materials/SiC_gFHR.dat PyC materials/PyC_gFHR.dat buffer materials/buffer_gFHR.dat "
    "matrix materials/matrix_gFHR.dat",
    "define_zones gFHR_zones.inp gFHR_points.dat",
]
V1_SETTINGS = [f'set acelib "{ENDF7}/sss_endfb7u.xsdata"', f'set declib "{ENDF7}/sss_endfb7.dec"',
               f'set nfylib "{ENDF7}/sss_endfb7.nfy"', "set opti 2", "set gcu -1", "set memfrac 0.99", POP, PRINTM]
V2_SETTINGS = ["set_nuclear_data endf8_lib80x", "set_thermal_coupling on", f'set acelib "{ENDF8_XSDATA}"',
               f'set declib "{ENDF8}/endf8.dec"', f'set nfylib "{ENDF8}/endf8.nfy"', "set opti 1", "set gcu -1",
               "set memfrac 0.99", POP, PRINTM]
STEP = ["initialize 0 graphpeb 0.5 fuel20 0.5", "set_power 84e6",
        "insertion 1 off 1 graphpeb 0.5 reinsert 1.0 fuel20 1.0"]


@pytest.fixture
def case(tmp_path):
    for name in ("gFHR_template.serpent", "gFHR_triso.dat", "gFHR_points.dat", "gFHR_zones.inp", "materials"):
        os.symlink(os.path.join(REPO, name), tmp_path / name)
    return tmp_path


def simulation():
    sim = Simulation("serpent_test")
    sim.serpent_command = os.environ.get("PEARLSIM_SERPENT", "sss2_HxF_dev")
    sim.cpu_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    return sim


def run_deck(case, lines):
    (case / "case.inp").write_text("\n".join(lines) + "\n")
    sim = simulation()
    sim.read_input_file("case.inp", directory_name="run")
    return sim, case / "run"


def first_keff(res_path):
    with open(res_path) as f:
        line = next(line for line in f if line.startswith("ANA_KEFF"))
    values = line.split("=", 1)[1].split("[", 1)[1].split()
    return float(values[0]), float(values[1])


def test_v1_depletion_step(case):
    sim, run = run_deck(case, COMMON + V1_SETTINGS + STEP)
    assert (run / "serpent_test_1.serpent.bumat1").exists()
    record = json.loads((run / "serpent_test_record_1.json").read_text())
    assert record["nuclear_data"] == "endf7_v1"
    assert 0.9 < record["keff"]["bos"]["mean"] < 1.6


def test_v2_depletion_step_and_restart(case):
    """v2 data with thermal coupling runs; a restart from the step (graphite pebbles present) runs too."""
    sim, run = run_deck(case, COMMON + V2_SETTINGS + STEP)
    deck = (run / "serpent_test_1.serpent").read_text()
    assert validate_deck(deck, ENDF8_LIB80X) == []
    assert missing_library_entries(deck, ENDF8_XSDATA) == []
    assert len(re.findall(r"^mat fuelshared_", deck, re.M)) == 4 and "_R1Z1G1_SiC" not in deck
    record = json.loads((run / "serpent_test_record_1.json").read_text())
    assert record["nuclear_data"] == "endf8_lib80x" and record["thermal_coupling"] is True
    assert 0.9 < record["keff"]["bos"]["mean"] < 1.6
    restart = simulation()
    os.chdir(run)
    for line in COMMON + V2_SETTINGS + ["set_power 84e6"]:  # a burnup deck needs its power normalisation
        restart.read_input_line(line)
    restart.load_from_step(1, 1)
    restart.core.iteration = 2
    restart.update_conditions()
    name = restart.core.generate_input(restart.serpent_settings_for_step(2), restart.burnup_time_step, 0,
                                       cr_position=restart.rod_insertion)
    restart.run_serpent(name)
    assert os.path.exists(f"{name}.bumat1")


def test_rods_in_reduce_k_with_common_random_numbers(case):
    sim, run = run_deck(case, COMMON + V1_SETTINGS + STEP[:2])  # initialized core, no step run; cwd is run/
    sim.core.iteration = 1
    keff = {}
    for depth in (369.47, 60.0):
        name = sim.core.generate_input({**sim.serpent_settings, "seed": "4242", "pop": "10000 30 10"},
                                       sim.burnup_time_step, 0, cr_position=depth, rng=np.random.default_rng(5))
        text = re.sub(r"^dep daystep[^\n]*\n", "", open(name).read(), flags=re.M).replace(" burn 1", "")
        deck = f"rods_{depth:g}.serpent"
        with open(deck, "w") as f:
            f.write(text)
        sim.run_serpent(deck)
        keff[depth] = first_keff(f"{deck}_res.m")
    (k_out, e_out), (k_in, e_in) = keff[369.47], keff[60.0]
    assert k_in < k_out - 3 * (e_out * k_out + e_in * k_in)
