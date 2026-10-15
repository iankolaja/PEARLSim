"""Tests of n replicas per step: .bumat averaging, concurrent runs, failure and resume."""

import json
import os

import numpy as np
import pytest
from helpers.synthetic import setup_lines, write_core

from pearlsim import replicas
from pearlsim.simulation import SerpentError, Simulation

STEP_DAYS = 6.525

# ------------------------------------------------------------------ averaging


def write(path, text):
    path.write_text(text)
    return str(path)


BUMAT_A = """% replica A
mat  fuel20_R1Z1G1pp1r1  0.0 fix 09c 9.59E+02 vol 1.0
            92235.09c  2.0E-02
            92238.09c  4.0E-03
               541350  1.0E-09

mat  fuel20_R1Z2G1pp2r1  0.0 fix 09c 9.59E+02 vol 1.0
            92235.09c  1.0E-02
"""
BUMAT_B = """% replica B
mat  fuel20_R1Z1G1pp1r1  0.0 fix 09c 9.59E+02 vol 1.0
            92235.09c  4.0E-02
            92238.09c  4.0E-03
               551370  2.0E-09

mat  fuel20_R1Z2G1pp2r1  0.0 fix 09c 9.59E+02 vol 1.0
            92235.09c  3.0E-02
"""


def test_read_bumat_undoes_particle_renaming_and_library(tmp_path):
    materials = replicas.read_bumat(write(tmp_path / "a.bumat1", BUMAT_A))
    assert materials == {"fuel20_R1Z1G1": {"92235<lib>": 2.0e-2, "92238<lib>": 4.0e-3, "541350": 1.0e-9},
                         "fuel20_R1Z2G1": {"92235<lib>": 1.0e-2}}


def test_average_is_exact_with_union_of_nuclides(tmp_path):
    a = replicas.read_bumat(write(tmp_path / "a", BUMAT_A))
    b = replicas.read_bumat(write(tmp_path / "b", BUMAT_B))
    averaged, spread = replicas.average_compositions([a, b])
    assert averaged["fuel20_R1Z1G1"] == {"92235<lib>": 3.0e-2, "92238<lib>": 4.0e-3,
                                         "541350": 0.5e-9, "551370": 1.0e-9}
    assert list(averaged["fuel20_R1Z1G1"]) == ["92235<lib>", "92238<lib>", "541350", "551370"]
    assert averaged["fuel20_R1Z2G1"] == {"92235<lib>": 2.0e-2}
    # sample std of (0.02, 0.04) = 0.01 sqrt(2); relative to the mean 0.03
    assert spread["fuel20_R1Z1G1"]["92235<lib>"] == pytest.approx(0.01 * np.sqrt(2) / 0.03)
    assert spread["fuel20_R1Z1G1"]["541350"] == pytest.approx(np.sqrt(2))
    assert spread["fuel20_R1Z2G1"]["max_nuclide"] == "92235<lib>"


def test_single_replica_average_is_the_identity(tmp_path):
    a = replicas.read_bumat(write(tmp_path / "a", BUMAT_A))
    averaged, spread = replicas.average_compositions([a])
    assert averaged == a and spread == {}


def test_different_material_sets_are_rejected(tmp_path):
    a = replicas.read_bumat(write(tmp_path / "a", BUMAT_A))
    b = replicas.read_bumat(write(tmp_path / "b", BUMAT_B.split("\n\n")[0]))
    with pytest.raises(ValueError, match="fuel20_R1Z2G1"):
        replicas.average_compositions([a, b])


def test_file_names():
    assert replicas.stem("gFHR", 12) == "gFHR_12.serpent"
    assert replicas.stem("gFHR", 12, 1) == "gFHR_12_r1.serpent"
    assert replicas.pebble_file(12, 0) == "pebble_positions_12_r0.csv"
    assert replicas.replica_ids(1) == [None] and replicas.replica_ids(3) == [0, 1, 2]


def test_set_replicas_keyword():
    sim = Simulation("synthetic")
    sim.read_input_line("set_replicas 3 sequential")
    assert (sim.replicas, sim.replica_mode) == (3, "sequential")
    sim.read_input_line("set_replicas 2")
    assert (sim.replicas, sim.replica_mode) == (2, "concurrent")


# ------------------------------------------------------------------ end to end


def deck(num_steps, num_replicas, serpent_flag=1):
    lines = ["set_seed 7", "set_averaging_mode burnup", *setup_lines(), "initialize 0 graphpeb 1",
             "set_seed 7", "set_discharge_threshold 18.63", "set_threshold_rel_std 0.02288",
             f"set_burnup_time_step {STEP_DAYS}", "set_power 1.0E+06", "set_rod_depth 300",
             "set pop 1000 10 5"]
    if num_replicas:
        lines.append(f"set_replicas {num_replicas}")
    lines.append(f"insertion {num_steps} off {serpent_flag} graphpeb 0.5 reinsert 1.0 fuel20 1.0")
    return "\n".join(lines) + "\n"


def run(tmp_path, fake_serpent_command, monkeypatch, num_steps, num_replicas, name="run", serpent_flag=1,
        log="serpent_log.jsonl"):
    write_core(tmp_path)
    (tmp_path / f"{name}.inp").write_text(deck(num_steps, num_replicas, serpent_flag))
    monkeypatch.setenv("FAKE_SERPENT_LOG", str(tmp_path / log))
    os.chdir(tmp_path)
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 2
    sim.read_input_file(f"{name}.inp", directory_name=name)
    return sim, tmp_path / name


def read_log(tmp_path, log="serpent_log.jsonl"):
    return [json.loads(line) for line in (tmp_path / log).read_text().splitlines()]


@pytest.mark.e2e
def test_one_replica_keyword_reproduces_default(tmp_path, fake_serpent_command, monkeypatch):
    run(tmp_path, fake_serpent_command, monkeypatch, 3, None, name="default")
    run(tmp_path, fake_serpent_command, monkeypatch, 3, 1, name="one")
    for step in (1, 2, 3):
        for name in (f"synthetic_{step}.serpent", f"zone_map{step}.json", f"pebble_positions_{step}.csv"):
            assert (tmp_path / "one" / name).read_text() == (tmp_path / "default" / name).read_text()


@pytest.mark.e2e
def test_two_replicas_end_to_end(tmp_path, fake_serpent_command, monkeypatch):
    monkeypatch.setenv("FAKE_SERPENT_SLEEP", "0.5")
    single, single_dir = run(tmp_path, fake_serpent_command, monkeypatch, 3, None, name="single",
                             log="single.jsonl")
    monkeypatch.setenv("FAKE_SERPENT_SEED_NOISE", "1")
    sim, run_dir = run(tmp_path, fake_serpent_command, monkeypatch, 3, 2)

    log = read_log(tmp_path)
    assert [e["argv"] for e in log[:2]] in ([[f"synthetic_1_r{r}.serpent", "-omp", "1"] for r in (0, 1)],
                                            [[f"synthetic_1_r{r}.serpent", "-omp", "1"] for r in (1, 0)])
    assert max(e["start"] for e in log[:2]) < min(e["end"] for e in log[:2])  # ran concurrently
    assert len(log) == 6

    # replica 0 reproduces the single-replica step 1 deck; replica 1 is a different shuffle and seed
    r0 = (run_dir / "synthetic_1_r0.serpent").read_text()
    r1 = (run_dir / "synthetic_1_r1.serpent").read_text()
    assert r0 == (single_dir / "synthetic_1.serpent").read_text().replace(
        "pebble_positions_1.csv", "pebble_positions_1_r0.csv")
    assert (run_dir / "pebble_positions_1_r0.csv").read_text() != (run_dir / "pebble_positions_1_r1.csv").read_text()
    operating = json.loads((run_dir / "synthetic_operating_1.json").read_text())
    assert operating["replicas"] == 2 and len(set(operating["serpent_seed"])) == 2
    assert f"set seed {operating['serpent_seed'][1]}\n" in r1

    # the core holds the replica mean of the last step's compositions
    last = [replicas.read_bumat(run_dir / f"synthetic_3_r{r}.serpent.bumat1") for r in (0, 1)]
    for name, conc in last[0].items():
        for nuclide, value in conc.items():
            mean = (value + last[1][name].get(nuclide, 0.0)) / 2
            assert sim.core.materials[name].concentrations[nuclide] == mean
    assert last[0] != last[1]
    spread = json.loads((run_dir / "synthetic_replica_spread_3.json").read_text())
    assert spread and all(s["92235<lib>"] > 0 for s in spread.values() if "92235<lib>" in s)
    assert sim.core.iteration == single.core.iteration == 4


@pytest.mark.e2e
def test_sequential_replicas_use_all_cores(tmp_path, fake_serpent_command, monkeypatch):
    write_core(tmp_path)
    text = deck(1, 2).replace("set_replicas 2", "set_replicas 2 sequential")
    (tmp_path / "seq.inp").write_text(text)
    monkeypatch.setenv("FAKE_SERPENT_LOG", str(tmp_path / "serpent_log.jsonl"))
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 2
    sim.read_input_file("seq.inp", directory_name="seq")
    assert [e["argv"] for e in read_log(tmp_path)] == [[f"synthetic_1_r{r}.serpent", "-omp", "2"] for r in (0, 1)]


@pytest.mark.e2e
def test_failed_replica_fails_step_and_rerun_resumes(tmp_path, fake_serpent_command, monkeypatch):
    monkeypatch.setenv("FAKE_SERPENT_SEED_NOISE", "1")
    reference, ref_dir = run(tmp_path, fake_serpent_command, monkeypatch, 3, 2, name="reference",
                             serpent_flag=2, log="reference.jsonl")

    monkeypatch.setenv("FAKE_SERPENT_FAIL_DECKS", "synthetic_2_r1")
    with pytest.raises(SerpentError, match=r"synthetic_2_r1\.serpent -omp 1 exited with status 1"):
        run(tmp_path, fake_serpent_command, monkeypatch, 3, 2, serpent_flag=2)
    run_dir = tmp_path / "run"
    assert (run_dir / "synthetic_2_r0.serpent.bumat1").exists()
    assert not (run_dir / "synthetic_2_r1.serpent.bumat1").exists()
    assert not (run_dir / "synthetic_3_r0.serpent").exists()

    # the requeued job replays the deck: step 1 is reused, step 2 reruns only replica 1
    monkeypatch.delenv("FAKE_SERPENT_FAIL_DECKS")
    resumed, _ = run(tmp_path, fake_serpent_command, monkeypatch, 3, 2, serpent_flag=2, log="resume.jsonl")
    assert [e["argv"][0] for e in read_log(tmp_path, "resume.jsonl")] == [
        "synthetic_2_r1.serpent", "synthetic_3_r0.serpent", "synthetic_3_r1.serpent"]
    for name in ("synthetic_2_r1.serpent", "synthetic_3_r0.serpent", "zone_map3.json", "pebble_positions_3_r1.csv"):
        assert (run_dir / name).read_text() == (ref_dir / name).read_text(), name
    for name, material in reference.core.materials.items():
        assert resumed.core.materials[name].concentrations == material.concentrations, name


@pytest.mark.e2e
def test_load_from_step_averages_replicas(tmp_path, fake_serpent_command, monkeypatch):
    monkeypatch.setenv("FAKE_SERPENT_SEED_NOISE", "1")
    sim, run_dir = run(tmp_path, fake_serpent_command, monkeypatch, 2, 2)
    loader = Simulation("synthetic")
    loader.core = sim.core
    loader.replicas = 2
    os.chdir(run_dir)
    expected = {name: dict(m.concentrations) for name, m in sim.core.materials.items()}
    loader.load_from_step(2, 1)
    for name, conc in expected.items():
        assert loader.core.materials[name].concentrations == conc


# ------------------------------------------------------------------ results


def test_collect_results_averages_replicas(equilibrium_dir, tmp_path):
    import re
    import shutil

    from pearlsim.results_processing import collect_results_from_step

    run, step = "gFHR_equilibrium", 100
    for name in os.listdir(equilibrium_dir):
        if name.endswith(".json") and name != "expected.json":
            shutil.copyfile(os.path.join(equilibrium_dir, name), tmp_path / name)
    for r in (0, 1):
        for suffix in ("_res.m", "_det0.m", "_det1.m"):
            shutil.copyfile(os.path.join(equilibrium_dir, f"{run}_{step}.serpent{suffix}"),
                            tmp_path / f"{run}_{step}_r{r}.serpent{suffix}")
    res1 = tmp_path / f"{run}_{step}_r1.serpent_res.m"
    res1.write_text(re.sub(r"(ANA_KEFF .*?\[\s+)1\.22721E\+00", r"\g<1>1.22821E+00", res1.read_text()))

    _, _, single, single_meshes, _ = collect_results_from_step(str(equilibrium_dir), step, run_name=run,
                                                               include_simulation_data=False)
    _, _, outputs, meshes, sim_data = collect_results_from_step(str(tmp_path), step, run_name=run, num_replicas=2)
    assert outputs["keff_rep0"] == single["final_analog_keff"] == 1.22721
    assert outputs["keff_rep1"] == 1.22821
    assert outputs["final_analog_keff"] == pytest.approx(1.22771)
    assert outputs["keff_rep_std"] == pytest.approx(0.001 / np.sqrt(2))
    absolute = np.hypot(1.22721 * 0.00207, 1.22821 * 0.00207) / 2
    assert outputs["final_analog_keff_unc"] == pytest.approx(absolute / 1.22771)
    # identical meshes: same mean, zero replica spread, uncertainty of the mean smaller by sqrt(2)
    power = meshes["subzone_power"]
    assert np.array_equal(power["data"], single_meshes["subzone_power"]["data"])
    assert not power["replica_std"].any()
    assert np.allclose(power["unc"], single_meshes["subzone_power"]["unc"] / np.sqrt(2))
    assert outputs["powerR1Z1"] == single["powerR1Z1"]
    assert "replica_std" in sim_data["coarse_7group_flux"]
