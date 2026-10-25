"""Characterization tests for pearlsim.simulation.Simulation: deck parsing and an end-to-end run.

The end-to-end tests drive the real v1 code path (keyword deck -> Core.insert ->
generate_input -> "Serpent" -> update_from_bumat) with the fake Serpent executable.
"""

import json
import os
import pickle
import re

import pytest
from helpers.synthetic import setup_lines, write_core

from pearlsim.nuclear_data import CoreConditions
from pearlsim.simulation import DeckError, SerpentError, Simulation

pytestmark = pytest.mark.characterization

STEP_DAYS = 6.525


def deck_lines(num_steps):
    return ["set_seed 7", "set_averaging_mode burnup", *setup_lines(), "initialize 0 graphpeb 1",
            "set_seed 7",  # initialize_pebbles consumes np.random even for 0 pebbles
            "set_discharge_threshold 18.63", "set_threshold_rel_std 0.02288",
            f"set_burnup_time_step {STEP_DAYS}", "set_power 1.0E+06", "set_rod_depth 300",
            "set pop 1000 10 5", "set printm 1 1.0E-41",
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
    serpent_seed = operating.pop("serpent_seed")
    assert operating == {"power": 1.0e6, "burnup_step": STEP_DAYS, "graphite_insertion_fraction": 0.5,
                         "control_rod_position": 300.0, "fima_discharge_threshold": 18.63,
                         "fima_discharge_rel_std": 0.02288, "base_seed": 7, "replicas": 1,
                         "nuclear_data": "endf7_v1", "thermal_coupling": False, "flibe_li6_ppm": 25.0,
                         "pebble_core_density": 1.74, "temperatures": None}
    assert f"set seed {serpent_seed}\n" in (run / "synthetic_5.serpent").read_text()
    seeds = {json.loads((run / f"synthetic_operating_{s}.json").read_text())["serpent_seed"] for s in range(1, 6)}
    assert len(seeds) == 5  # a different Serpent seed every step
    golden("e2e/zone_map5.json", (run / "zone_map5.json").read_text())
    golden("e2e/synthetic_5.serpent", (run / "synthetic_5.serpent").read_text())
    golden("e2e/synthetic_discharge_inventory4.json", (run / "synthetic_discharge_inventory4.json").read_text())


@pytest.mark.e2e
def test_zone_occupancy_conserved_end_to_end(tmp_path, fake_serpent_command, monkeypatch):
    sim, _ = run_case(tmp_path, fake_serpent_command, monkeypatch, num_steps=2)
    assert sim.core.iteration == 3
    assert [[sum(z.inventory.values()) for z in ch] for ch in sim.core.zones] == [[18, 18, 18], [18, 18, 18]]


@pytest.mark.e2e
@pytest.mark.parametrize("mode, message", [("fail", "exited with status 1"), ("no_output", "without writing")])
def test_serpent_failure_raises_serpent_error(tmp_path, fake_serpent_command, monkeypatch, mode, message):
    """Review #17 (fixed): a failed or silent Serpent run stops the simulation with SerpentError."""
    with pytest.raises(SerpentError, match=message):
        run_case(tmp_path, fake_serpent_command, monkeypatch, num_steps=1, mode=mode)


# ---------------------------------------------------------------- deck parser


@pytest.fixture
def sim():
    return Simulation("synthetic")


def test_benchmark_correction_keywords(sim):
    sim.read_input_line("set_flibe_li6_ppm 50\n")
    sim.update_conditions()  # v1 library, no coupling: v1 temperatures with the Li-6 fix
    assert sim.core.conditions == CoreConditions(salt_li6_ppm=50.0)
    sim.read_input_line("set_pebble_core_density 1.41\n")
    sim.update_conditions()
    assert sim.core.conditions == CoreConditions(salt_li6_ppm=50.0, pebble_core_density=1.41)
    with pytest.raises(DeckError):
        sim.read_input_line("set_flibe_li6_ppm -1\n")
    with pytest.raises(DeckError):
        sim.read_input_line("set_pebble_core_density 0\n")


def test_set_options_are_stored_as_normalised_text(sim):
    sim.read_input_line("set pop   24000 30\t10 # comment\n")
    assert sim.serpent_settings["pop"] == "24000 30 10"


def test_unknown_keywords_are_ignored(sim):
    before = dict(sim.__dict__)
    sim.read_input_line("averaging_mode burnup\n")  # used by gFHR_benchmark.inp; a no-op in v1
    assert sim.__dict__.keys() == before.keys()


def test_comments_are_stripped(sim):
    sim.read_input_line("set_rod_depth 250 # comment\n")
    assert sim.rod_insertion == 250.0


def test_set_clear_removes_option(sim):
    """Review #19 (fixed)."""
    sim.read_input_line("set pop clear\n")
    assert "pop" not in sim.serpent_settings


def test_power_is_numeric(sim):
    """Review #20 (fixed): power is a float; the Serpent setting keeps the text as written."""
    sim.read_input_line("set_power 1.0E+06\n")
    assert sim.power == 1.0e6
    assert sim.serpent_settings["power"] == "1.0E+06"


def test_double_space_separator(sim):
    """Review #22 (fixed)."""
    sim.read_input_line("set_rod_depth  300\n")
    assert sim.rod_insertion == 300.0


def test_restart_loads_pickled_core(sim, tmp_path):
    """Review #18 (fixed)."""
    os.makedirs(tmp_path / "run")
    with open(tmp_path / "core.pkl", "wb") as f:
        pickle.dump(sim.core, f)
    os.chdir(tmp_path / "run")
    original = sim.core
    sim.read_input_line("restart core.pkl")
    assert sim.core is not original


def test_malformed_line_raises_deck_error(sim):
    with pytest.raises(DeckError, match="set_rod_depth"):
        sim.read_input_line("set_rod_depth deep\n")


def test_step_rng_depends_only_on_seed_and_iteration(sim):
    sim.read_input_line("set_seed 7")
    first = sim.step_rng(12).integers(0, 1 << 30, size=4)
    sim.step_rng(3).integers(0, 1 << 30, size=100)  # unrelated draws in between
    assert (sim.step_rng(12).integers(0, 1 << 30, size=4) == first).all()
    assert (sim.step_rng(13).integers(0, 1 << 30, size=4) != first).any()
    assert (sim.step_rng(12, replica=1).integers(0, 1 << 30, size=4) != first).any()


def test_serpent_seed_per_step_unless_set_explicitly(sim):
    assert "seed" not in sim.serpent_settings_for_step(1)  # no set_seed: Serpent's own default
    sim.read_input_line("set_seed 7")
    assert sim.serpent_settings_for_step(1)["seed"] != sim.serpent_settings_for_step(2)["seed"]
    assert sim.serpent_settings_for_step(1) == sim.serpent_settings_for_step(1)
    sim.read_input_line("set seed 42")
    assert sim.serpent_settings_for_step(1)["seed"] == sim.serpent_settings_for_step(2)["seed"] == "42"


@pytest.mark.e2e
def test_endf8_thermal_coupling_end_to_end(tmp_path, fake_serpent_command, monkeypatch):
    from pearlsim.nuclear_data import ENDF8_LIB80X, validate_deck

    write_core(tmp_path)
    lines = deck_lines(2)
    lines.insert(lines.index("set_power 1.0E+06"), "set_nuclear_data endf8_lib80x")
    lines.insert(lines.index("set_power 1.0E+06"), "set_thermal_coupling on")
    (tmp_path / "case.inp").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("FAKE_SERPENT_LOG", str(tmp_path / "serpent_log.jsonl"))
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 2
    sim.read_input_file("case.inp", directory_name="run")
    run = tmp_path / "run"
    for step in (1, 2):
        deck = (run / f"synthetic_{step}.serpent").read_text()
        assert validate_deck(deck, ENDF8_LIB80X) == []
        operating = json.loads((run / f"synthetic_operating_{step}.json").read_text())
        assert operating["nuclear_data"] == "endf8_lib80x" and operating["thermal_coupling"] is True
        assert operating["flibe_li6_ppm"] == 50.0
        assert operating["pebble_core_density"] == 1.41
        assert re.search(r"^mat pebgraph\s+-1\.4100\s", deck, re.M)
        assert re.search(r"^\s+3006\.\d\dc\s+0\.00010000$", deck, re.M)
        temps = operating["temperatures"]
        # 1 MW of 280 MW: everything a little above the 823.15 K inlet; fuel hottest
        assert 823.15 < temps["salt"] < temps["matrix"] < temps["kernel"] < 900
        assert temps["structure"] == pytest.approx(823.15)


@pytest.mark.e2e
def test_load_from_step_restores_graphite_pebbles(tmp_path, fake_serpent_command, monkeypatch):
    """Graphite pebbles are not in the .bumat; a restart must still define their materials (M2 pilot,
    2026-10-22: "Universe ugraphpeb_R1Z1G1 is not defined")."""
    _, run = run_case(tmp_path, fake_serpent_command, monkeypatch, num_steps=2)
    monkeypatch.chdir(run)
    sim = Simulation("synthetic")
    for line in setup_lines():
        sim.read_input_line(line)
    sim.load_from_step(2, 1)
    names = {name for channel in sim.core.zones for zone in channel for name, n in zone.inventory.items() if n}
    assert any(name.startswith("graphpeb") for name in names)
    assert names <= set(sim.core.materials)
    deck = open(sim.core.generate_input({"pop": "100 10 5"}, STEP_DAYS, 0, cr_position=300)).read()
    assert all(f"cell {name}_c u{name} {name} " in deck for name in names if name.startswith("graphpeb"))


def test_depletion_step_without_printm_is_refused_before_serpent_runs(tmp_path, fake_serpent_command, monkeypatch):
    """Without 'set printm 1' Serpent writes no .bumat; found by the real-Serpent tests (2026-10-23)."""
    write_core(tmp_path)
    lines = [line for line in deck_lines(1) if not line.startswith("set printm")]
    (tmp_path / "case.inp").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("FAKE_SERPENT_LOG", str(tmp_path / "serpent_log.jsonl"))
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    with pytest.raises(DeckError, match="printm"):
        sim.read_input_file("case.inp", directory_name="run")
    assert not (tmp_path / "serpent_log.jsonl").exists()



def _run_with(tmp_path, fake_serpent_command, monkeypatch, extra, num_steps=4, name="run"):
    write_core(tmp_path)
    lines = deck_lines(num_steps)
    at = next(i for i, line in enumerate(lines) if line.startswith("insertion"))
    (tmp_path / "case.inp").write_text("\n".join(lines[:at] + extra + lines[at:]) + "\n")
    monkeypatch.setenv("FAKE_SERPENT_LOG", str(tmp_path / "serpent_log.jsonl"))
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 2
    sim.read_input_file("case.inp", directory_name=name)
    return sim, tmp_path / name


@pytest.mark.e2e
def test_k_target_feedback_sets_the_graphite_fraction(tmp_path, fake_serpent_command, monkeypatch):
    _, run = _run_with(tmp_path, fake_serpent_command, monkeypatch, ["set_k_target 1.0 gain 0.5 max_step 0.05"])
    g = [json.loads((run / f"synthetic_operating_{s}.json").read_text())["graphite_insertion_fraction"]
         for s in range(1, 5)]
    k = [json.loads((run / f"synthetic_record_{s}.json").read_text())["keff"]["eos"]["mean"] for s in range(1, 5)]
    assert g[0] == 0.5  # no previous step: the deck's fraction
    for i in range(1, 4):
        step = max(-0.05, min(0.05, 0.5 * (k[i - 1] - 1.0)))
        assert g[i] == pytest.approx(min(1.0, max(0.0, g[i - 1] + step)))


@pytest.mark.e2e
def test_k_target_dither_is_reproducible(tmp_path, fake_serpent_command, monkeypatch):
    extra = ["set_k_target 1.0 gain 0.0 dither 0.02"]
    _, first = _run_with(tmp_path, fake_serpent_command, monkeypatch, extra, num_steps=3, name="a")
    os.chdir(tmp_path)
    _, second = _run_with(tmp_path, fake_serpent_command, monkeypatch, extra, num_steps=3, name="b")
    ga = [json.loads((first / f"synthetic_operating_{s}.json").read_text())["graphite_insertion_fraction"]
          for s in range(1, 4)]
    gb = [json.loads((second / f"synthetic_operating_{s}.json").read_text())["graphite_insertion_fraction"]
          for s in range(1, 4)]
    assert ga == gb and len(set(ga)) == 3  # dithered, and the same on a rerun


@pytest.mark.e2e
def test_power_per_pebble_cap_lowers_power(tmp_path, fake_serpent_command, monkeypatch):
    from pearlsim.thermal import NOMINAL_PEBBLES, NOMINAL_POWER
    sim, run = _run_with(tmp_path, fake_serpent_command, monkeypatch, ["set_power_per_pebble_cap 1.0"], num_steps=1)
    operating = json.loads((run / "synthetic_operating_1.json").read_text())
    limit = NOMINAL_POWER / NOMINAL_PEBBLES * sim.fuel_pebbles_in_core()
    assert operating["power"] == pytest.approx(min(1.0e6, limit))
    assert f"set power {min(1.0e6, limit):.6g}\n" in (run / "synthetic_1.serpent").read_text()


@pytest.mark.e2e
def test_missing_triso_files_are_written_on_demand(tmp_path, fake_serpent_command, monkeypatch):
    """A deck with more fuel materials than 'triso_file N' copies (family A2 stopped on triso401)."""
    write_core(tmp_path)
    lines = [line.replace("triso_file 4 ", "triso_file 1 ") for line in deck_lines(2)]
    assert any(line.startswith("triso_file 1 ") for line in lines)
    (tmp_path / "case.inp").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("FAKE_SERPENT_LOG", str(tmp_path / "serpent_log.jsonl"))
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 2
    sim.read_input_file("case.inp", directory_name="run")
    run = tmp_path / "run"
    used = {int(i) for s in (1, 2) for i in re.findall(r'"trisos/triso(\d+)"', (run / f"synthetic_{s}.serpent").read_text())}
    assert max(used) > 1 and all((run / "trisos" / f"triso{i}").exists() for i in used)
    assert (run / "trisos" / f"triso{max(used)}").read_text().split("\n")[0].endswith(f"p{max(used)}")
