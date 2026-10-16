"""Tests of per-step records and gathering, on the v1 equilibrium fixture and on fake-Serpent runs."""

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from helpers.synthetic import setup_lines, write_core

from pearlsim import gather, records
from pearlsim.simulation import Simulation

RUN, STEP = "gFHR_equilibrium", 100


@pytest.fixture
def expected(equilibrium_dir):
    with open(os.path.join(equilibrium_dir, "expected.json")) as f:
        return json.load(f)


def test_v1_record_from_output_files(equilibrium_dir, expected):
    record = records.build_record(str(equilibrium_dir), RUN, STEP)
    assert record["replicas"] == 1
    assert record["keff"]["eos"]["mean"] == expected["ana_keff"][-1]
    assert record["keff"]["bos"]["mean"] == expected["ana_keff"][0]
    assert record["keff"]["eos"]["unc"] == expected["ana_keff_unc"][-1]
    assert record["keff"]["eos"]["rho_pcm"] == pytest.approx((1 - 1 / expected["ana_keff"][-1]) * 1e5)
    assert record["controls"]["power"] == 280e6  # v1 stored " 280e6"
    assert record["serpent"][0]["VERSION"] == "Serpent pre-2.2.1"
    assert record["serpent"][0]["XS_DATA_FILE_PATH"].endswith("endfb7/sss_endfb7u.xsdata")
    assert record["seeds"]["serpent_seed_in_output"] == [42]
    with open(os.path.join(equilibrium_dir, f"zone_map{STEP}.json")) as f:
        zone_map = json.load(f)
    fuel = sum(n for zone in zone_map.values() for key, n in zone.items() if "fuel" in key)
    assert record["inventory"]["fuel_pebbles"] == fuel
    assert record["inventory"]["power_per_fuel_pebble"] == pytest.approx(280e6 / fuel)
    assert record["discharge"]["avg_FIMA"] == pytest.approx(expected["lstm_input_features"]["avg_FIMA"])
    mesh = np.asarray(record["meshes"]["subzone_power"]["data"])[:, :, 0]
    assert mesh[0, 0] == expected["power_mesh"][0][0][0]
    # Serpent's mesh is in W integrated per cell; the reader divides by cell volume (W/cm3)
    assert record["power"]["mesh_power_W"] == pytest.approx(279448182.0, rel=1e-8)  # raw det1 sum
    assert record["power"]["mesh_power_over_set_power"] == pytest.approx(0.998029, abs=1e-6)
    zone = np.asarray(record["power"]["zone_power_fraction"])
    assert zone.shape == (4, 10) and zone.sum() == pytest.approx(1.0)
    assert zone[1, 3] == pytest.approx(mesh[2:4, 6:8].sum() / mesh.sum())
    assert json.loads(json.dumps(record)) == record  # serialisable as is


def test_smoothed_burnup_on_v1_fixture(equilibrium_dir):
    record = records.build_record(str(equilibrium_dir), RUN, STEP)
    smooth = record["discharge_smooth"]
    with open(os.path.join(equilibrium_dir, f"{RUN}_discharge_inventory{STEP}.json")) as f:
        discharge = json.load(f)
    fuel = sum(v["count"] for k, v in discharge.items() if "graph" not in k)
    assert smooth["discharge_fuel_pebbles"] == fuel
    fracs = [smooth[f"burnup_smooth_bin{i}_frac"] for i in range(1, 10)]
    assert sum(fracs) == pytest.approx(1.0)
    assert smooth["burnup_smooth_p10"] <= smooth["burnup_smooth_p50"] <= smooth["burnup_smooth_p90"]
    # rel_std = 0 gives v1's hard bins, normalised
    hard = records.smoothed_discharge_burnup(discharge, 0.0)
    for i in range(1, 10):
        assert hard[f"burnup_smooth_bin{i}_frac"] == pytest.approx(record["discharge"][f"burnup_bin{i}_count"] / fuel)
    row = records.flatten(record, meshes=False)
    assert row["burnup_smooth_bin1_frac"] == smooth["burnup_smooth_bin1_frac"]


def test_smoothed_burnup_is_continuous_across_bin_edges():
    edge = records.SMOOTH_BURNUP_EDGES[4]  # 10 % FIMA
    below = {"discharge_fuel20_R1Z10G1": {"count": 1000, "FIMA": edge - 1e-3},
             "discharge_graphpeb_R1Z10G1": {"count": 500, "FIMA": 0.0}}
    above = {"discharge_fuel20_R1Z10G1": {"count": 1000, "FIMA": edge + 1e-3}}
    jumps = {}
    for rel_std in (0.0, 0.02576):
        a = records.smoothed_discharge_burnup(below, rel_std)
        b = records.smoothed_discharge_burnup(above, rel_std)
        assert a["discharge_fuel_pebbles"] == 1000  # graphite excluded
        jumps[rel_std] = max(abs(a[k] - b[k]) for k in a if k.startswith("burnup_smooth_bin"))
    assert jumps[0.0] == pytest.approx(1.0)  # hard bins: the whole group changes bin
    # smoothed: Gaussian density at the edge times the shift, pdf(0) * 2e-3 / (0.02576 * 10)
    assert jumps[0.02576] == pytest.approx(0.3989 * 2e-3 / 0.2576, rel=0.01)


def test_smoothed_burnup_without_discharged_fuel():
    out = records.smoothed_discharge_burnup({"discharge_graphpeb_R1Z1G1": {"count": 10, "FIMA": 0.0}}, 0.02576)
    assert out["discharge_fuel_pebbles"] == 0
    assert all(v == 0.0 for v in out.values())


def test_power_summary_on_known_mesh():
    power = np.ones((8, 20))
    power[0, 0] = 5.0
    summary = records.power_summary(power)
    assert summary["mesh_power_W"] == pytest.approx(164.0 * 87500.581065579)
    assert summary["peaking_mesh"] == pytest.approx(5.0 / (164 / 160))
    assert summary["peaking_radial"] == pytest.approx(24.0 / (164 / 8))
    assert summary["peaking_axial"] == pytest.approx(12.0 / (164 / 20))
    assert summary["zone_power_fraction"][0][0] == pytest.approx(8.0 / 164)


def test_inventory_summary_counts_by_channel():
    zone_map = {"zoneR1Z1": {"fuel20_R1Z1G1": 30, "graphpeb_R1Z1G1": 10},
                "zoneR1Z2": {"fuel20_R1Z2G2": 40},
                "zoneR2Z1": {"graphpeb_R2Z1G1": 20}}
    summary = records.inventory_summary(zone_map, held={"held_avgreinsert_fuel20_G2": 3})
    assert summary == {"fuel_pebbles": 70, "graphite_pebbles": 30, "fuel_fraction": 0.7,
                       "channel_fuel_fraction": [70 / 80, 0.0], "held_pebbles": 3}


def test_gather_v1_directory(equilibrium_dir, tmp_path):
    run_dir = tmp_path / "gFHR_equilibrium_v1"
    shutil.copytree(equilibrium_dir, run_dir)
    (run_dir / f"{RUN}_operating_{STEP + 1}.json").write_text("{}")  # bookkeeping-only step: no results
    out = str(tmp_path / "table.csv.gz")
    gather.main([str(run_dir), "--out", out])
    table = pd.read_csv(out)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["sequence"] == "gFHR_equilibrium_v1" and row["step"] == STEP
    assert row["days"] == pytest.approx(6.525)
    assert row["final_analog_keff"] == row["keff_eos"]
    assert "fluxR8Z20E3" in table.columns and "powerR8Z20" in table.columns
    meta = json.loads((tmp_path / "table.csv.gz.meta.json").read_text())
    assert meta["sources"][0]["skipped_without_results"] == [STEP + 1]
    assert not any(name.startswith(f"{RUN}_record_") for name in os.listdir(run_dir))  # read-only by default


@pytest.mark.e2e
def test_records_written_and_gathered_with_replicas(tmp_path, fake_serpent_command, monkeypatch):
    write_core(tmp_path)
    lines = ["set_seed 7", "set_averaging_mode burnup", *setup_lines(), "initialize 0 graphpeb 1", "set_seed 7",
             "set_dataset_version v2-test", "set_discharge_threshold 18.63", "set_threshold_rel_std 0.02288",
             "set_burnup_time_step 6.525", "set_power 1.0E+06", "set_rod_depth 300", "set pop 1000 10 5",
             "set_replicas 2", "insertion 2 off 1 graphpeb 0.5 reinsert 1.0 fuel20 1.0"]
    (tmp_path / "case.inp").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("FAKE_SERPENT_SEED_NOISE", "1")
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 2
    sim.read_input_file("case.inp", directory_name="run")
    run_dir = tmp_path / "run"

    record = json.loads((run_dir / "synthetic_record_2.json").read_text())
    assert record["replicas"] == 2 and record["days"] == pytest.approx(2 * 6.525)
    assert len(record["keff"]["eos"]["replicas"]) == 2 and record["keff"]["eos"]["replica_std"] == 0.0
    assert record["provenance"]["dataset_version"] == "v2-test"
    assert record["provenance"]["serpent_settings"]["pop"] == "1000 10 5"
    assert len(record["provenance"]["git_commit"]) == 40
    assert record["replica_spread"]["max_rel_std"] > 0
    assert record["inventory"]["fuel_pebbles"] == sim.fuel_pebbles_in_core() > 0
    assert record["controls"]["power"] == 1.0e6

    out = str(tmp_path / "table.parquet")
    table = gather.gather([str(run_dir)], out)
    assert list(table["step"]) == [1, 2]
    assert list(table["days"]) == pytest.approx([6.525, 13.05])
    assert {"keff_eos_rep0", "keff_eos_rep1", "keff_eos_rep_std", "replica_max_rel_std", "fluxR1Z1E1"} <= set(
        table.columns)
    assert len(pd.read_parquet(out)) == 2


def test_archive_policy_classification(tmp_path, repo_root):
    import importlib.util

    spec = importlib.util.spec_from_file_location("pack_run", os.path.join(repo_root, "scripts", "pack_run.py"))
    pack_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack_run)
    names = ["sim_record_10.json", "sim_operating_10.json", "zone_map10.json", "sim_held_inventory10.json",
             "sim_10_r0.serpent", "sim_10_r0.serpent_res.m", "sim_10_r0.serpent_det1.m",
             "sim_10_r0.serpent.bumat0", "sim_10_r0.serpent.bumat1", "sim_10_r0.serpent_dep.m",
             "pebble_positions_10_r0.csv", "pebble_positions_11_r0.csv", "pebble_positions_12.csv",
             "core_materials_11.json.gz", "sim_12.serpent.out", "sim_12.serpent.bumat1", "notes.txt"]
    for name in names:
        (tmp_path / name).write_text("x")
    kept = {name for name, (_, keep) in pack_run.classify(str(tmp_path)).items() if keep}
    assert kept == {"sim_record_10.json", "sim_operating_10.json", "zone_map10.json", "sim_held_inventory10.json",
                    "sim_10_r0.serpent_res.m", "sim_10_r0.serpent_det1.m", "sim_10_r0.serpent.bumat1",
                    "pebble_positions_10_r0.csv", "pebble_positions_12.csv",  # checkpoint step and last step
                    "sim_12.serpent.bumat1", "notes.txt"}
    assert pack_run.classify(str(tmp_path))["notes.txt"] == ("unknown", True)
