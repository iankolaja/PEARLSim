"""Characterization tests for the v1 Serpent output readers, on a real full-core step.

Expected values in fixtures/equilibrium_step100/expected.json were computed from
the untrimmed source files (see tests/fixtures/README.md).
"""

import json
import os
import shutil

import numpy as np
import pytest

from pearlsim.core import Core
from pearlsim.group_definitions import ENERGY_GRID_3, HEIGHT_GRID_20, RADIUS_GRID_8
from pearlsim.ml_utilities import read_det_file
from pearlsim.results_processing import collect_results_from_step, extract_lstm_input_features, read_res_file

pytestmark = pytest.mark.characterization

RUN, STEP = "gFHR_equilibrium", 100


@pytest.fixture
def expected(equilibrium_dir):
    with open(os.path.join(equilibrium_dir, "expected.json")) as f:
        return json.load(f)


def test_read_res_file_all_burnup_points(equilibrium_dir, expected):
    values, unc = read_res_file(os.path.join(equilibrium_dir, f"{RUN}_{STEP}.serpent_res.m"), "ANA_KEFF",
                                burnup_step="all")
    assert values == expected["ana_keff"]
    assert unc == expected["ana_keff_unc"]


def test_read_res_file_single_burnup_point_is_one_based(equilibrium_dir, expected):
    path = os.path.join(equilibrium_dir, f"{RUN}_{STEP}.serpent_res.m")
    assert read_res_file(path, "ANA_KEFF", burnup_step=1) == (expected["ana_keff"][0], expected["ana_keff_unc"][0])
    assert read_res_file(path, "ANA_KEFF", burnup_step=2) == (expected["ana_keff"][1], expected["ana_keff_unc"][1])


def test_read_det_file_meshes_normalised_by_volume(equilibrium_dir, expected):
    meshes = read_det_file(
        os.path.join(equilibrium_dir, f"{RUN}_{STEP}.serpent_det1.m"),
        meshes_to_read=["subzone_3group_flux", "subzone_power"],
        mesh_grids_rze=[(RADIUS_GRID_8, HEIGHT_GRID_20, ENERGY_GRID_3), (RADIUS_GRID_8, HEIGHT_GRID_20, [0.0])],
    )
    flux = meshes["subzone_3group_flux"]["data"]
    power = meshes["subzone_power"]["data"]
    assert flux.shape == (8, 20, 3) and power.shape == (8, 20, 1)
    np.testing.assert_array_equal(flux, np.array(expected["flux_mesh"]))
    np.testing.assert_array_equal(power, np.array(expected["power_mesh"]))


def test_extract_lstm_input_features(equilibrium_dir, expected):
    features = extract_lstm_input_features(equilibrium_dir, STEP, run_name=RUN, merge=True)
    got = {k: (v if isinstance(v, str) else float(v)) for k, v in features.items()}
    assert got == pytest.approx(expected["lstm_input_features"])
    assert got["power"] == " 280e6"  # v1 stores power as a string (review #20)


def test_collect_results_from_step_flattens_meshes(equilibrium_dir, expected):
    inputs, averages, outputs, meshes, _ = collect_results_from_step(equilibrium_dir, STEP, run_name=RUN,
                                                                     include_simulation_data=False)
    assert outputs["final_analog_keff"] == expected["ana_keff"][-1]
    assert sum(k.startswith("flux") for k in outputs) == 8 * 20 * 3
    assert sum(k.startswith("power") for k in outputs) == 8 * 20
    assert outputs["powerR1Z1"] == expected["power_mesh"][0][0][0]
    assert float(averages["avg_FIMA"]) == pytest.approx(expected["lstm_input_features"]["avg_FIMA"])


def test_update_from_bumat_strips_particle_suffix(equilibrium_dir, expected, tmp_path):
    shutil.copyfile(os.path.join(equilibrium_dir, f"{RUN}_{STEP}.serpent.bumat1"),
                    tmp_path / f"{RUN}_{STEP}.serpent.bumat1")
    core = Core(RUN)
    core.update_from_bumat(0, iteration=STEP, step=1)
    got = {name: [len(m.concentrations), sum(m.concentrations.values())] for name, m in core.materials.items()}
    assert list(got) == list(expected["bumat1_first_materials"])
    for name, (n_nuclides, total) in expected["bumat1_first_materials"].items():
        assert got[name][0] == n_nuclides
        assert got[name][1] == pytest.approx(total, rel=1e-12)
    material = core.materials["fuel20_R1Z1G1"]
    assert "92235<lib>" in material.concentrations  # library suffix replaced by a placeholder
    assert "20060" in material.concentrations  # decay-only nuclides keep their ZAI


@pytest.mark.xfail(strict=True, reason="review #35: all-graphite discharge raises ZeroDivisionError")
def test_extract_lstm_input_features_with_graphite_only_discharge(equilibrium_dir, tmp_path):
    shutil.copyfile(os.path.join(equilibrium_dir, f"{RUN}_operating_{STEP}.json"),
                    tmp_path / f"{RUN}_operating_{STEP}.json")
    (tmp_path / f"{RUN}_discharge_inventory{STEP}.json").write_text(json.dumps(
        {"discharge_graphpeb_R1Z10G1": {"count": 500, "FIMA": 100.0, "FIMA_last_pass": 0.0}}))
    (tmp_path / f"{RUN}_discard_inventory{STEP}.json").write_text("{}")
    extract_lstm_input_features(str(tmp_path), STEP, run_name=RUN, merge=True)
