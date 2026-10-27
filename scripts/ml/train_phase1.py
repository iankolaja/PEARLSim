"""Phase-1 forecaster on a sequence set: next-step pre-training on observables, then rollout evaluation.

    python scripts/ml/train_phase1.py OUT_DIR [--data PEARLSim/sequence_data] [--epochs 60] [--device cpu|cuda]

Data and splits as in ``pearlsim.ml.experiment`` (v1 directory by default: the development set; family-grouped
split with the test families held out, validation on one grouped fold). Observables: the paper's discharge
measurements (``features.DEPENDENT_FEATURES``); controls: ``features.CONTROL_FEATURES``. The variance floor is
the paper's measurement noise (``features.NoiseModel``), converted to scaled units.

Writes OUT_DIR/results.json (split, config, history, test metrics) and OUT_DIR/model.pt. Test metrics:
teacher-forced NLL; rollout error by horizon (mean |error| / column std, averaged over observables) from every
start with >= ``--context`` rows of history, against persistence (D_{t+h} = D_t).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pearlsim.ml import experiment, features  # noqa: E402
from pearlsim.ml.forecaster import ForecasterConfig, evaluate_nll, fit, make_tokens, rollout  # noqa: E402


def rollout_errors(model, scaler, sequences, obs, ctrl, context, horizon, stride):
    """Mean |error| / std per horizon 0..horizon for the model and for persistence."""
    model_err = [[] for _ in range(horizon + 1)]
    persist_err = [[] for _ in range(horizon + 1)]
    scale = scaler.std[: len(obs)]
    for seq in sequences:
        d, u = seq.frame[obs].to_numpy(np.float32), seq.frame[ctrl].to_numpy(np.float32)
        for t0 in range(context, len(d) - horizon - 1, stride):
            pred = rollout(model, scaler, d[t0 - context:t0], u[t0 - context:t0], u[t0:t0 + horizon])
            truth = d[t0:t0 + horizon + 1]
            for h in range(horizon + 1):
                model_err[h].append(np.mean(np.abs(pred[h] - truth[h]) / scale))
                persist_err[h].append(np.mean(np.abs(d[t0 - 1] - truth[h]) / scale))
    return ([float(np.mean(e)) if e else None for e in model_err],
            [float(np.mean(e)) if e else None for e in persist_err])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    parser.add_argument("--data", default="PEARLSim/sequence_data")
    parser.add_argument("--config", help="experiment YAML whose split (fixed test families) to use, e.g. "
                                         "PEARLSim/experiments/v1_lstm.yaml, so results compare with its baselines")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    override = {"data": {"path": args.data}}
    if args.config:
        override["split"] = experiment.load_config(args.config)["split"]
    cfg = experiment.merge(experiment.DEFAULTS, override)
    sequences = experiment.load_sequences(cfg)
    fit_fams, val_fams, test_fams = experiment.split_families(cfg, sequences)
    pick = lambda fams: [s for s in sequences if s.family in fams]  # noqa: E731
    obs, ctrl = list(features.DEPENDENT_FEATURES), list(features.CONTROL_FEATURES)
    train_tokens = make_tokens(pick(fit_fams), obs, ctrl)
    val_tokens, test_tokens = make_tokens(pick(val_fams), obs, ctrl), make_tokens(pick(test_fams), obs, ctrl)

    config = ForecasterConfig(chunk=args.chunk, epochs=args.epochs, layers=args.layers, d_model=args.d_model,
                              device=args.device, seed=args.seed)
    from pearlsim.ml.forecaster import TokenScaler
    scaler = TokenScaler.fit(train_tokens)  # targets: the change D_{i+1} - D_i
    y = np.concatenate(train_tokens.y)
    sigma = features.NoiseModel().sigmas(obs) * np.abs(y).mean(axis=0)  # multiplicative noise at typical size
    noise_var = (sigma / scaler.y_std) ** 2  # the measured D_{i+1} carries it; in the target's scaled units

    start = time.time()
    model, scaler, history = fit(train_tokens, config, validation=val_tokens, noise_var=noise_var,
                                 log=lambda e: print(e, flush=True), scaler=scaler)
    import torch
    floor = torch.as_tensor(noise_var, dtype=torch.float32, device=config.device)
    test_nll = evaluate_nll(model, scaler, test_tokens, config, floor)
    model.cpu()
    model_err, persist_err = rollout_errors(model, scaler, pick(test_fams), obs, ctrl, args.context, args.horizon,
                                            args.stride)
    results = {"data": args.data, "split": {"fit": fit_fams, "validation": val_fams, "test": test_fams},
               "config": asdict(config), "observables": obs, "controls": ctrl, "history": history,
               "test_nll": test_nll, "rollout_mae_over_std": {"model": model_err, "persistence": persist_err},
               "context": args.context, "train_seconds": time.time() - start,
               "n_tokens": {"fit": int(sum(len(x) for x in train_tokens.x)),
                            "validation": int(sum(len(x) for x in val_tokens.x)),
                            "test": int(sum(len(x) for x in test_tokens.x))}}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    torch.save({"state_dict": model.state_dict(), "mean": scaler.mean, "std": scaler.std, "scaler": asdict(scaler),
                "obs": obs, "ctrl": ctrl, "config": asdict(config)}, os.path.join(args.out, "model.pt"))
    print("test NLL", test_nll)
    for h in (0, 1, 5, 10, args.horizon):
        if h < len(model_err):
            print(f"horizon {h}: model {model_err[h]:.3f}  persistence {persist_err[h]:.3f}")


if __name__ == "__main__":
    main()
