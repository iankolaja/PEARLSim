"""Characterization tests for pearlsim.simulation.Simulation: deck parsing and an end-to-end run.

The end-to-end tests drive the real v1 code path (keyword deck -> Core.insert ->
generate_input -> "Serpent" -> update_from_bumat) with the fake Serpent executable.
"""

import json
import os
import pickle

import pytest
from helpers.synthetic import setup_lines, write_core

from pearlsim.simulation import Simulation

pytestmark = pytest.mark.characterization

STEP_DAYS = 6.525


def deck_lines(num_steps):
    return ["set_seed 7", "set_averaging_mode burnup", *setup_lines(), "initialize 0 graphpeb 1",
            "set_seed 7",  # initialize_pebbles consumes np.random even for 0 pebbles
            "set_discharge_threshold 18.63", "set_threshold_rel_std 0.02288",
            f"set_burnup_time_step {STEP_DAYS}", "set_power 1.0E+06", "set_rod_depth 300",
            "set pop 1000 10 5",
            f"insertion {num_steps} off 1 graphpeb 0.5 reinsert 1.0 fuel20 1.0"]


def run_case(tmp_path, fake_serpent_command, monkeypatch, num_steps=5, mode="ok"):
    write_core(tmp_path)
    (tmp_path / "case.inp").write_text("\n".join(deck_lines(num_steps)) + "\n")
    monkeypatch.setenv("FAKE_SERPENT_MODE", mode)
    monkeypatch.setenv("FAKE_SERPENT_LOG", str(tmp_path / "serpent_log.jsonl"))
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 2
    sim.read_input_file("case.inp", directory_name="run")
    return sim, tmp_path / "run"


@pytest.mark.e2e
def test_insertion_sequence_end_to_end(tmp_path, fake_serpent_command, monkeypatch, golden):
    sim, run = run_case(tmp_path, fake_serpent_command, monkeypatch)
    assert sim.core.iteration == 6
    assert sim.days == pytest.approx(5 * STEP_DAYS)
    for step in range(1, 6):
        for name in (f"synthetic_{step}.serpent", f"synthetic_{step}.serpent.bumat1", f"zone_map{step}.json",
                     f"synthetic_operating_{step}.json", f"synthetic_discharge_inventory{step}.json",
                     f"core_materials_{step}.json.gz", f"pebble_positions_{step}.csv"):
            assert (run / name).exists(), name
    log = [json.loads(line) for line in (tmp_path / "serpent_log.jsonl").read_text().splitlines()]
    assert [entry["argv"] for entry in log] == [[f"synthetic_{s}.serpent", "-omp", "2"] for s in range(1, 6)]
    assert all(entry["daysteps"] == 1 for entry in log)
    # step 5 reinserts the burnup-averaged step-4 discharge at the bottom of the core
    zone_map = json.loads((run / "zone_map5.json").read_text())
    assert any(k.startswith("fuel20_R1Z1G") and not k.endswith("G1") for k in zone_map["zoneR1Z1"])
    operating = json.loads((run / "synthetic_operating_5.json").read_text())
    assert operating == {"power": "1.0E+06", "burnup_step": STEP_DAYS, "graphite_insertion_fraction": 0.5,
                         "control_rod_position": 300.0, "fima_discharge_threshold": 18.63,
                         "fima_discharge_rel_std": 0.02288}
    golden("e2e/zone_map5.json", (run / "zone_map5.json").read_text())
    golden("e2e/synthetic_5.serpent", (run / "synthetic_5.serpent").read_text())
    golden("e2e/synthetic_discharge_inventory4.json", (run / "synthetic_discharge_inventory4.json").read_text())


@pytest.mark.e2e
def test_zone_occupancy_conserved_end_to_end(tmp_path, fake_serpent_command, monkeypatch):
    sim, _ = run_case(tmp_path, fake_serpent_command, monkeypatch, num_steps=2)
    assert sim.core.iteration == 3
    assert [[sum(z.inventory.values()) for z in ch] for ch in sim.core.zones] == [[18, 18, 18], [18, 18, 18]]


@pytest.mark.e2e
def test_serpent_failure_surfaces_as_missing_bumat(tmp_path, fake_serpent_command, monkeypatch):
    """v1 ignores the Serpent exit code; the failure appears later as a missing .bumat (review #17)."""
    with pytest.raises(FileNotFoundError):
        run_case(tmp_path, fake_serpent_command, monkeypatch, num_steps=1, mode="fail")


@pytest.mark.e2e
@pytest.mark.xfail(strict=True, reason="review #17: Serpent exit status is not checked")
def test_serpent_failure_raises_a_serpent_error(tmp_path, fake_serpent_command, monkeypatch):
    with pytest.raises(RuntimeError):
        run_case(tmp_path, fake_serpent_command, monkeypatch, num_steps=1, mode="fail")


# ---------------------------------------------------------------- deck parser


@pytest.fixture
def sim():
    return Simulation("synthetic")


def test_set_options_are_stored_with_leading_space(sim):
    sim.read_input_line("set pop 24000 30 10\n")
    assert sim.serpent_settings["pop"] == " 24000 30 10"


def test_unknown_keywords_are_ignored(sim):
    before = dict(sim.__dict__)
    sim.read_input_line("averaging_mode burnup\n")  # used by gFHR_benchmark.inp; a no-op in v1
    assert sim.__dict__.keys() == before.keys()


def test_comments_are_stripped(sim):
    sim.read_input_line("set_rod_depth 250 # comment\n")
    assert sim.rod_insertion == 250.0


@pytest.mark.xfail(strict=True, reason="review #19: 'set X clear' never matches because of a leading space")
def test_set_clear_removes_option(sim):
    sim.read_input_line("set pop clear\n")
    assert "pop" not in sim.serpent_settings


@pytest.mark.xfail(strict=True, reason="review #20: power is stored as a string")
def test_power_is_numeric(sim):
    sim.read_input_line("set_power 1.0E+06\n")
    assert isinstance(sim.power, float)


@pytest.mark.xfail(strict=True, reason="review #22: tokens are split on single spaces only")
def test_double_space_separator(sim):
    sim.read_input_line("set_rod_depth  300\n")
    assert sim.rod_insertion == 300.0


@pytest.mark.xfail(strict=True, reason="review #18: restart passes a path string to pickle.load")
def test_restart_loads_pickled_core(sim, tmp_path):
    os.makedirs(tmp_path / "run")
    with open(tmp_path / "core.pkl", "wb") as f:
        pickle.dump(sim.core, f)
    os.chdir(tmp_path / "run")
    original = sim.core
    sim.read_input_line("restart core.pkl")
    assert sim.core is not original
