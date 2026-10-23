"""Tests of the ML data pipeline: alignment, splits, features and metrics (no torch)."""

import numpy as np
import pandas as pd
import pytest

from pearlsim.ml import features, metrics, splits
from pearlsim.ml.data import Sequence, load_gathered_table, v1_family
from pearlsim.ml.windows import TargetSpec, make_windows


def ramp_sequence(name="seq_40s", n=12, family=None):
    """Every column encodes its row number, so alignment can be read off the values."""
    rows = np.arange(n, dtype=float)
    frame = pd.DataFrame({"u": 100 + rows, "d": 200 + rows, "k": 300 + rows})
    return Sequence(name, family or v1_family(name), frame)


def test_one_step_alignment_window_ends_at_decision_row():
    ws = make_windows([ramp_sequence()], ["u", "d"], [TargetSpec("k"), TargetSpec("d", "next")], window=4)
    assert len(ws) == 12 - 3
    first = ws.past[0]
    assert first[:, 0].tolist() == [100, 101, 102, 103]  # rows 0..3
    assert ws.row[0] == 3
    assert ws.targets[0, 0, 0] == 303  # k of the decision step i = 3
    assert ws.targets[0, 0, 1] == 204  # measurements before the next step, row i + 1
    last = len(ws) - 1
    assert ws.row[last] == 11 and ws.mask[last, 0].tolist() == [True, False]  # no row 12


def test_paper_alignment_is_offset_by_one_row():
    ws = make_windows([ramp_sequence()], ["u"], ["k"], window=4, input_offset=1)
    assert ws.past[0, :, 0].tolist() == [100, 101, 102, 103] and ws.row[0] == 4
    assert ws.targets[0, 0, 0] == 304  # target row i = 4, last input row i - 1 = 3


def test_multi_horizon_targets_and_future_controls():
    ws = make_windows([ramp_sequence(n=10)], ["u", "d"], ["k", ("d", "next")], window=3, horizon=2,
                      control_columns=["u"])
    i = ws.row[0]
    assert i == 2
    assert ws.future_controls[0, :, 0].tolist() == [103, 104]  # u_{i+1}, u_{i+2}
    assert ws.targets[0, :, 0].tolist() == [302, 303, 304]  # k_{i}, k_{i+1}, k_{i+2}
    assert ws.targets[0, :, 1].tolist() == [203, 204, 205]  # d_{i+1}, d_{i+2}, d_{i+3}
    end = np.flatnonzero(ws.row == 8)[0]
    assert ws.mask[end, :, 0].tolist() == [True, True, False]
    assert ws.future_controls[end, 1, 0] == 0.0  # past the end of the sequence


def test_family_names():
    assert v1_family("gFHR_ascension_graphite_1_40s") == "gFHR_ascension_graphite_1"
    assert v1_family("gFHR_runin_graphite_4_60s_75") == "gFHR_runin_graphite_4"
    assert v1_family("gFHR_sequence7") == "gFHR_sequence7"


def test_splits_keep_families_apart():
    families = [f"fam{j}" for j in range(11)]
    test = splits.choose_test_families(families, 0.2, seed=1)
    assert test == splits.choose_test_families(families, 0.2, seed=1) and len(test) == 2
    rest = [f for f in families if f not in test]
    folds = splits.grouped_folds(rest, k=3, seed=1)
    assert sorted(f for _, val in folds for f in val) == sorted(rest)
    for fit, val in folds:
        assert not set(fit) & set(val) and not (set(fit) | set(val)) & set(test)
    window_families = np.array(["fam0", "fam0", "fam1", "fam2"])
    a, b = splits.masks(window_families, ["fam0"], ["fam1", "fam2"])
    assert a.tolist() == [True, True, False, False] and b.tolist() == [False, False, True, True]
    with pytest.raises(ValueError, match="fam1"):
        splits.masks(window_families, ["fam0", "fam1"], ["fam1"])


def test_v1_power_per_pebble_uses_full_history():
    frame = pd.DataFrame({"power": [100.0] * 4, "graphite_insertion_fraction": [0.0, 1.0, 0.0, 0.0]})
    value = features.v1_power_per_fuel_pebble(frame, num_pebbles=1, residence_steps=2)
    # fuel fraction = mean of (previous, current) insertion fractions, row -1 taken as row 0
    assert value[0] == pytest.approx(100.0) and value[1] == pytest.approx(200.0)


def test_noise_model_matches_paper_sigma():
    model = features.NoiseModel()
    assert model.sigma("burnup_bin3_count") == pytest.approx(0.05 / np.sqrt(2 / np.pi))
    assert model.sigma("R2_avg_FIMA_last_pass") == pytest.approx(0.10 / np.sqrt(2 / np.pi))
    assert model.sigma("power") == 0.0
    values = np.ones((200_000, 2))
    noisy = model.apply(values, ["avg_FIMA", "power"], np.random.default_rng(0))
    assert np.mean(np.abs(noisy[:, 0] - 1)) == pytest.approx(0.025, rel=0.02)  # MAPE recovered
    assert (noisy[:, 1] == 1).all()


def test_mesh_pca_scores_are_standardised():
    rng = np.random.default_rng(0)
    frames = [pd.DataFrame(rng.normal(size=(50, 6)) * [10, 5, 1, 1, 1, 1],
                           columns=[f"powerR1Z{z}" for z in range(1, 7)])]
    pca = features.MeshPCA("power", 3).fit(frames)
    scored = pca.add_scores(frames[0])
    assert scored[pca.names].std(ddof=0).to_numpy() == pytest.approx([1, 1, 1])
    full = pca.reconstruct(scored[pca.names].to_numpy())
    assert full.shape == (50, 6)


def test_reactivity_metrics_by_band():
    k_true = np.array([1.0, 1.01, 0.9])
    k_pred = np.array([1.001, 1.01, 0.9])
    m = metrics.reactivity_metrics(k_true, k_pred, bands_pcm=(500, 2000), noise_pcm=50.0)
    expected = (1 - 1 / 1.001) * 1e5  # ~99.9 pcm on the critical sample
    assert m["abs_rho_lt_500"]["n"] == 1 and m["abs_rho_lt_500"]["rmse"] == pytest.approx(expected)
    assert m["abs_rho_lt_2000"]["n"] == 2
    assert m["abs_rho_lt_500"]["excess_rmse"] == pytest.approx(np.sqrt(expected ** 2 - 50 ** 2))
    assert m["outside_largest_band"]["n"] == 1 and m["outside_largest_band"]["rmse"] == 0.0


def test_near_critical_weights():
    w = features.near_critical_weights(np.array([0.0, 0.01, 0.1]), band_pcm=5000, outside_weight=0.2)
    assert w.tolist() == [1.0, 1.0, 0.2]


def test_gathered_table_groups_sequences(tmp_path):
    table = pd.DataFrame({"sequence": ["b_40s", "a", "a"], "step": [1, 2, 1], "keff_eos": [1.0, 1.1, 1.2]})
    path = tmp_path / "t.csv"
    table.to_csv(path, index=False)
    seqs = load_gathered_table(str(path))
    assert [s.name for s in seqs] == ["a", "b_40s"] and [s.family for s in seqs] == ["a", "b"]
    assert seqs[0].frame["step"].tolist() == [1, 2]


def test_ridge_experiment_end_to_end(tmp_path):
    """Corrected pipeline on a small synthetic data set with the ridge baseline."""
    from pearlsim.ml import experiment

    rng = np.random.default_rng(0)
    rows = []
    for s in range(10):
        n = 30
        u = rng.uniform(0.5, 1.0, n)
        d = np.cumsum(rng.normal(0, 0.01, n)) + 1
        k = 1.0 + 0.01 * (u - 0.75) + 0.005 * (d - 1)
        for i in range(n):
            rows.append({"sequence": f"fam{s % 5}_{40 + 20 * (s // 5)}s", "step": i, "power": u[i],
                         "avg_FIMA": d[i], "keff_eos": k[i]})
    path = tmp_path / "table.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    cfg = experiment.merge(experiment.DEFAULTS, {
        "name": "synthetic", "data": {"source": "table", "path": str(path), "exclude": []},
        "inputs": {"controls": ["power"], "dependents": ["avg_FIMA"], "computed": []},
        "targets": {"meshes": {}}, "window": 3, "split": {"test_families": ["fam0"], "folds": 4},
        "noise": {"train": "none", "eval_draws": 0}})
    results = experiment.run(cfg, str(tmp_path / "out"), log=lambda *_: None)
    assert results["families"]["test"] == ["fam0"]
    assert not set(results["families"]["test"]) & set(results["families"]["fit"] + results["families"]["validation"])
    assert results["metrics"]["test"]["reactivity"]["all"]["rmse"] < 20  # pcm; the relation is linear
    assert (tmp_path / "out" / "results.json").exists()


def test_config_merge_replaces_data_mappings():
    from pearlsim.ml import experiment

    cfg = experiment.merge(experiment.DEFAULTS, {"targets": {"meshes": {}}, "split": {"seed": 3}})
    assert cfg["targets"]["meshes"] == {} and cfg["targets"]["reactivity"] == "k"
    assert cfg["split"]["seed"] == 3 and cfg["split"]["folds"] == 5
    assert experiment.DEFAULTS["targets"]["meshes"] == {"power": 5, "flux": 5}  # defaults untouched


def _controls(g, z):
    return pd.DataFrame({"graphite_insertion_fraction": g, "control_rod_position": z})


def test_rod_worth_feature_full_core_follows_s_curve():
    from pearlsim.ml.features import rod_worth_feature
    inserted, total = rod_worth_feature(_controls([0.0] * 4, [369.47, 60.0, 214.735, 1000.0]))
    assert total == pytest.approx([3600.0] * 4)
    assert inserted == pytest.approx([0.0, 3600.0, 1800.0, 0.0])  # out, in, half-way (S(0.5) = 0.5), above bed


def test_rod_worth_feature_tracks_fuel_density_and_fill_front():
    from pearlsim.ml.features import rod_worth_feature
    g = [0.5] * 12
    inserted, total = rod_worth_feature(_controls(g, [200.0] * 12), initial_front=0.0)
    assert total == pytest.approx([2400.0] * 12)
    # one zone per step from an all-graphite start: rods at 200 cm see no fuel until the front passes them
    front_top = 60.0 + 309.47 * np.minimum(1.0, np.arange(1, 13) / 10)
    assert np.all(inserted[front_top <= 200.0] == 0) and np.all(inserted[front_top > 200.0] > 0)
    assert np.all(np.diff(inserted) >= 0)


def test_rod_worth_feature_is_causal():
    from pearlsim.ml.features import rod_worth_feature
    rng = np.random.default_rng(3)
    frame = _controls(rng.uniform(0, 1, 30), rng.uniform(60, 369.47, 30))
    full, _ = rod_worth_feature(frame, initial_front=0.0)
    head, _ = rod_worth_feature(frame.iloc[:17], initial_front=0.0)
    assert full[:17] == pytest.approx(head)
