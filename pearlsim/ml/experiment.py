"""Run one ML experiment from a YAML config and write its results.

    python -m pearlsim.ml.experiment CONFIG.yaml [--out DIR]

Writes ``DIR/results.json`` (config, split families, metrics, training history,
code version) and ``DIR/predictions.npz``. See ``experiments/*.yaml`` for examples;
every key and its default is in ``DEFAULTS`` below.

Corrected evaluation (default): family-grouped splits with the test families fixed
in the config, validation on one grouped fold of the remaining families, window
ending at the decision row (review #1-#3), reactivity errors in pcm by |rho| band
(review #4), noisy-input evaluation over several draws (review #11).

``split.mode: paper_random_windows`` with ``alignment: paper`` reproduces the
paper's setup (random 70/15/15 split of overlapping windows, window ..i-1) to check
the PyTorch port against the published numbers; it is not a valid generalisation
estimate.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

from pearlsim.ml import data as ml_data
from pearlsim.ml import features, metrics, splits
from pearlsim.ml.windows import TargetSpec, make_windows

DEFAULTS = {
    "name": "experiment",
    "data": {"source": "v1_directory", "path": "sequence_data", "exclude": ["^gFHR_forecast_(?!base$)"]},
    "inputs": {"controls": features.CONTROL_FEATURES, "dependents": features.DEPENDENT_FEATURES,
               "computed": ["power_per_fuel_pebble"]},
    "targets": {"reactivity": "k", "meshes": {"power": 5, "flux": 5}, "next_dependents": True},
    "window": 8,
    "horizon": 0,
    "alignment": "corrected",  # or "paper" (window ..i-1)
    "split": {"mode": "families", "test_families": None, "test_fraction": 0.2, "seed": 0,
              "folds": 5, "validation_fold": 0, "paper_fractions": [0.7, 0.15, 0.15]},
    "weighting": {"band_pcm": None, "outside_weight": 1.0},
    "noise": {"train": "per_epoch", "eval_draws": 5, "mape": None, "seed": 0},
    "model": {"type": "ridge", "params": {}, "train": {}},
}

# Paper Table 2 best hidden sizes per target (one network per target)
PAPER_SIZES = {
    "reactivity": [256, 128], "power_pc1": [64], "power_pc2": [256, 128], "power_pc3": [32],
    "power_pc4": [256, 128], "power_pc5": [256, 128], "flux_pc1": [256], "flux_pc2": [256],
    "flux_pc3": [64, 4], "flux_pc4": [256, 128], "flux_pc5": [128, 64], "avg_FIMA": [256],
    "R1_avg_FIMA_last_pass": [64], "R2_avg_FIMA_last_pass": [256], "R3_avg_FIMA_last_pass": [64],
    "R4_avg_FIMA_last_pass": [256], "num_discarded": [32], "burnup_bin1_count": [32],
    "burnup_bin2_count": [128, 64], "burnup_bin3_count": [128, 8], "burnup_bin4_count": [64],
    "burnup_bin5_count": [256, 128], "burnup_bin6_count": [128, 64], "burnup_bin7_count": [64, 4],
    "burnup_bin8_count": [256, 128], "burnup_bin9_count": [16, 8],
}


# Mappings whose keys are data (mesh names, column patterns, model arguments): an override replaces them
REPLACED_MAPPINGS = {"meshes", "mape", "params", "sizes"}


def merge(defaults, override):
    """Config ``override`` on top of ``defaults``; nested sections merge, data mappings are replaced."""
    out = copy.deepcopy(defaults)
    for key, value in (override or {}).items():
        nested = isinstance(value, dict) and isinstance(out.get(key), dict) and key not in REPLACED_MAPPINGS
        out[key] = merge(out[key], value) if nested else copy.deepcopy(value)
    return out


def load_config(path):
    import yaml

    with open(path) as f:
        return merge(DEFAULTS, yaml.safe_load(f))


def load_sequences(cfg):
    d = cfg["data"]
    if d["source"] == "v1_directory":
        return ml_data.load_v1_directory(d["path"], exclude=d["exclude"])
    if d["source"] == "table":
        return ml_data.load_gathered_table(d["path"], family=d.get("family"), exclude=d["exclude"])
    raise ValueError(f"unknown data source {d['source']!r}")


def split_families(cfg, sequences):
    s = cfg["split"]
    families = sorted({seq.family for seq in sequences})
    test = s["test_families"] or splits.choose_test_families(families, s["test_fraction"], s["seed"])
    unknown = set(test) - set(families)
    if unknown:
        raise ValueError(f"test families not in the data: {sorted(unknown)}")
    rest = [f for f in families if f not in test]
    fit, val = splits.grouped_folds(rest, s["folds"], s["seed"])[s["validation_fold"]]
    return fit, val, list(test)


def target_specs(cfg, mesh_names):
    t = cfg["targets"]
    specs = []
    if t["reactivity"] == "k":
        specs.append(TargetSpec("keff_eos"))
    elif t["reactivity"] == "rho":
        specs.append(TargetSpec("rho_eos"))
    specs += [TargetSpec(name) for name in mesh_names]
    if t["next_dependents"]:
        specs += [TargetSpec(c, "next") for c in cfg["inputs"]["dependents"]]
    return specs


def k_of(values, cfg):
    return values if cfg["targets"]["reactivity"] == "k" else 1.0 / (1.0 - values)


def build_model(cfg, n_features, n_controls, n_targets):
    from pearlsim.ml.models import SequenceRegressor

    m = cfg["model"]
    params = dict(m["params"])
    return SequenceRegressor(n_features, n_controls, n_targets, horizon=cfg["horizon"], encoder=m["type"],
                             window=cfg["window"], **params)


def evaluate(cfg, windows, prediction, noise_pcm=None):
    """Metrics for one split: reactivity (pcm) and every other target in its units."""
    result = {"windows": len(windows)}
    columns = windows.target_columns
    if columns and columns[0] in ("keff_eos", "rho_eos"):
        k_true, k_pred = k_of(windows.targets[:, :, 0], cfg), k_of(prediction[:, :, 0], cfg)
        available = windows.mask[:, 0, 0]
        result["reactivity"] = metrics.reactivity_metrics(
            k_true[available, 0], k_pred[available, 0],
            noise_pcm=None if noise_pcm is None else noise_pcm[available])
        if windows.targets.shape[1] > 1:
            result["horizon_curve"] = metrics.horizon_curve(k_true, k_pred, windows.mask[:, :, 0])
    result["targets"] = {}
    for j, column in enumerate(columns):
        available = windows.mask[:, 0, j]
        result["targets"][column] = metrics.regression_metrics(windows.targets[available, 0, j],
                                                               prediction[available, 0, j])
    return result


def noise_floor(cfg, sequences, windows):
    """Per-window 1-sigma reactivity noise (pcm) of the reference k: replica spread (v2) or the v1 floor."""
    lookup = {}
    for seq in sequences:
        if "keff_eos_rep_std" in seq.frame and seq.frame["keff_eos_rep_std"].notna().any():
            k = seq.frame["keff_eos"].to_numpy(dtype=float)
            n = seq.frame["replicas"].to_numpy(dtype=float)
            sigma = seq.frame["keff_eos_rep_std"].to_numpy(dtype=float) / np.sqrt(n) / k ** 2 * 1e5
            lookup.update({(seq.name, i): sigma[i] for i in range(len(seq))})
    if not lookup:
        return np.full(len(windows), metrics.V1_NOISE_FLOOR_PCM)
    return np.array([lookup.get((s, r), np.nan) for s, r in zip(windows.sequence, windows.row)])


def run(cfg, out_dir, log=print):
    start = time.time()
    sequences = load_sequences(cfg)
    if cfg["targets"]["reactivity"] == "rho":
        for seq in sequences:
            seq.frame["rho_eos"] = features.reactivity(seq.frame["keff_eos"])
    fit_fams, val_fams, test_fams = split_families(cfg, sequences)
    paper_split = cfg["split"]["mode"] == "paper_random_windows"

    # mesh PCA on training sequences only (all non-test sequences for the paper split)
    pca_source = [s for s in sequences if s.family in (fit_fams + val_fams if paper_split else fit_fams)]
    mesh_names = []
    for prefix, n in cfg["targets"]["meshes"].items():
        pca = features.MeshPCA(prefix, n).fit([s.frame for s in pca_source])
        for seq in sequences:
            seq.frame = pca.add_scores(seq.frame)
        mesh_names += pca.names

    inputs = cfg["inputs"]["controls"] + cfg["inputs"]["dependents"] + cfg["inputs"]["computed"]
    specs = target_specs(cfg, mesh_names)
    windows = make_windows(sequences, inputs, specs, window=cfg["window"], horizon=cfg["horizon"],
                           control_columns=cfg["inputs"]["controls"] + cfg["inputs"]["computed"],
                           input_offset=1 if cfg["alignment"] == "paper" else 0)
    if paper_split:
        rest = ~np.isin(windows.family, test_fams)
        order = np.random.default_rng(cfg["split"]["seed"]).permutation(np.flatnonzero(rest))
        a, b, _ = cfg["split"]["paper_fractions"]
        cut1, cut2 = int(a * len(order)), int((a + b) * len(order))
        fit_mask, val_mask, test_mask = (np.isin(np.arange(len(windows)), part) for part in
                                         (order[:cut1], order[cut1:cut2], order[cut2:]))
        held_out_mask = ~rest
    else:
        fit_mask, val_mask, test_mask = splits.masks(windows.family, fit_fams, val_fams, test_fams)
        held_out_mask = None
    fit_w, val_w, test_w = windows.subset(fit_mask), windows.subset(val_mask), windows.subset(test_mask)
    log(f"{len(sequences)} sequences; windows fit {len(fit_w)}, validation {len(val_w)}, test {len(test_w)}")

    weights = None
    if cfg["weighting"]["band_pcm"] is not None:
        rho = features.reactivity(k_of(fit_w.targets[:, 0, 0], cfg))
        weights = features.near_critical_weights(rho, cfg["weighting"]["band_pcm"], cfg["weighting"]["outside_weight"])
        keep = weights > 0
        fit_w, weights = fit_w.subset(keep), weights[keep]

    noise = features.NoiseModel(cfg["noise"]["mape"]) if cfg["noise"]["train"] != "none" else None
    noise_rng = np.random.default_rng(cfg["noise"]["seed"])
    if noise is not None and cfg["noise"]["train"] == "fixed":  # v1: one draw on the training inputs
        fit_w.past = noise.apply(fit_w.past, fit_w.input_columns, noise_rng).astype(np.float32)

    kind = cfg["model"]["type"]
    history = None
    if kind in ("ridge", "hgb"):
        from pearlsim.ml.baselines import FlatBaseline

        model = FlatBaseline(kind, **cfg["model"]["params"]).fit(fit_w, weights)

        def predict_fn(ws, past=None):
            if past is not None:
                ws = copy.copy(ws)
                ws.past = past.astype(np.float32)
            return model.predict(ws)
    else:
        from pearlsim.ml import train

        tcfg = train.TrainConfig(**cfg["model"]["train"])
        per_epoch = noise if cfg["noise"]["train"] == "per_epoch" else None
        if kind == "paper_lstm":
            from pearlsim.ml.models import paper_lstm

            networks, history = [], {}
            for j, column in enumerate(fit_w.target_columns):
                key = "reactivity" if column in ("keff_eos", "rho_eos") else column
                sizes = cfg["model"]["params"].get("sizes", {}).get(key, PAPER_SIZES.get(key, [64]))
                one = lambda ws, j=j: _single_target(ws, j)  # noqa: E731
                net = paper_lstm(fit_w.past.shape[-1], sizes=sizes,
                                 dropout=cfg["model"]["params"].get("dropout", 0.1))
                net, scalers, hist = train.fit(net, one(fit_w), one(val_w), tcfg, weights, per_epoch, log=None)
                networks.append((net, scalers))
                history[column] = hist
                log(f"{column}: {len(hist)} epochs, best val {min(h['val_loss'] for h in hist):.4g}")

            def predict_fn(ws, past=None):
                return np.concatenate([train.predict(net, sc, _single_target(ws, j), past)
                                       for j, (net, sc) in enumerate(networks)], axis=2)
        else:
            net = build_model(cfg, fit_w.past.shape[-1], fit_w.future_controls.shape[-1], len(specs))
            net, scalers, history = train.fit(net, fit_w, val_w, tcfg, weights, per_epoch, log=log)

            def predict_fn(ws, past=None):
                return train.predict(net, scalers, ws, past)

    noise_all = noise_floor(cfg, sequences, windows)
    results = {"name": cfg["name"], "config": cfg,
               "families": {"fit": fit_fams, "validation": val_fams, "test": test_fams},
               "windows": {"fit": int(fit_mask.sum()), "validation": int(val_mask.sum()), "test": int(test_mask.sum())},
               "metrics": {}}
    predictions = {}
    for split_name, mask in (("validation", val_mask), ("test", test_mask), ("held_out", held_out_mask)):
        if mask is None or not mask.any():
            continue
        ws = windows.subset(mask)
        prediction = predict_fn(ws)
        results["metrics"][split_name] = evaluate(cfg, ws, prediction, noise_all[mask])
        predictions[f"{split_name}_prediction"] = prediction
        predictions[f"{split_name}_targets"] = ws.targets
        predictions[f"{split_name}_sequence"] = ws.sequence
        predictions[f"{split_name}_row"] = ws.row
        draws = cfg["noise"]["eval_draws"] if noise is not None else 0
        if draws:
            noisy = [evaluate(cfg, ws, predict_fn(ws, noise.apply(ws.past, ws.input_columns, noise_rng)),
                              noise_all[mask]) for _ in range(draws)]
            results["metrics"][f"{split_name}_noisy_inputs"] = _mean_metrics(noisy)
    from pearlsim.records import code_version

    results.update({"history": history, "seconds": time.time() - start, "code": code_version(),
                    "target_columns": windows.target_columns, "input_columns": windows.input_columns})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=_json_default)
    np.savez_compressed(os.path.join(out_dir, "predictions.npz"), **predictions)
    return results


def _single_target(windows, j):
    ws = copy.copy(windows)
    ws.targets, ws.mask = windows.targets[:, :, j:j + 1], windows.mask[:, :, j:j + 1]
    ws.target_columns = [windows.target_columns[j]]
    return ws


def _mean_metrics(results):
    """Element-wise mean of numeric leaves over several metric dicts of the same shape."""
    first = results[0]
    if isinstance(first, dict):
        return {k: _mean_metrics([r[k] for r in results]) for k in first}
    if isinstance(first, list):
        return [_mean_metrics([r[i] for r in results]) for i in range(len(first))]
    if isinstance(first, (int, float)) and not isinstance(first, bool) and all(r is not None for r in results):
        return float(np.mean(results))
    return first


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config")
    parser.add_argument("--out", default=None, help="output directory (default: ml_runs/<name>)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out = args.out or os.path.join("ml_runs", cfg["name"])
    results = run(cfg, out)
    test = results["metrics"].get("test", {}).get("reactivity", {})
    near = test.get("abs_rho_lt_2000", {})
    print(f"{cfg['name']}: test near-critical (|rho|<2000 pcm) RMSE {near.get('rmse')} pcm over "
          f"{near.get('n')} windows; results in {out}")


if __name__ == "__main__":
    sys.exit(main())
