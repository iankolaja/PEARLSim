"""Phase-2 forecaster: reactivity head on the trunk, comparing joint / probe / finetune (ROADMAP item 9).

    python scripts/ml/train_phase2.py OUT_DIR --phase1 PHASE1_DIR/model.pt [--modes joint probe finetune]
        [--data PEARLSim/sequence_data] [--epochs 300] [--device cpu]

Same data, split, observables, controls and noise floor as train_phase1.py. Step target: the step's reactivity
rho = (k_eos - 1) / k_eos in pcm (``keff_eos``). ``probe`` and ``finetune`` start from the phase-1 weights; the
scaler is refitted on the same training families and checked against the phase-1 one.

Writes OUT_DIR/<mode>/results.json and model.pt. Test metrics: teacher-forced rho error (MAE, RMSE in pcm, and
MAE where |rho| < 1000 pcm), and rollout rho MAE by horizon (horizon 0 = decision step) from every start with
>= ``--context`` rows of history, feeding back predicted observables under the true future controls.
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
from pearlsim.ml.forecaster import ForecasterConfig, TokenScaler, fit, make_tokens, rollout  # noqa: E402


def with_reactivity(sequences):
    for seq in sequences:
        seq.frame["rho_pcm"] = 1e5 * features.reactivity(seq.frame["keff_eos"].to_numpy(dtype=float))
    return sequences


def teacher_forced_rho(model, scaler, tokens, chunk):
    import torch

    errors, truths = [], []
    model.eval()
    with torch.no_grad():
        for b, xs in enumerate(tokens.x):
            for start in range(0, len(xs), chunk):
                part = slice(start, start + chunk)
                out = model(torch.as_tensor(scaler.x(xs[part])[None]))
                pred = scaler.unscale_z(out["step_mean"][0].numpy())[:, 0]
                truth = tokens.z[b][part][:, 0]
                errors.append(pred - truth)
                truths.append(truth)
    e, t = np.concatenate(errors), np.concatenate(truths)
    near = np.abs(t) < 1000
    return {"mae_pcm": float(np.mean(np.abs(e))), "rmse_pcm": float(np.sqrt(np.mean(e ** 2))),
            "mae_near_critical_pcm": float(np.mean(np.abs(e[near]))) if near.any() else None,
            "n": int(len(e)), "n_near_critical": int(near.sum())}


def rollout_rho(model, scaler, sequences, obs, ctrl, context, horizon, stride):
    errors = [[] for _ in range(horizon + 1)]
    for seq in sequences:
        d, u = seq.frame[obs].to_numpy(np.float32), seq.frame[ctrl].to_numpy(np.float32)
        rho = seq.frame["rho_pcm"].to_numpy()
        for t0 in range(context, len(d) - horizon - 1, stride):
            _, steps = rollout(model, scaler, d[t0 - context:t0], u[t0 - context:t0], u[t0:t0 + horizon],
                               with_steps=True)
            for h in range(horizon + 1):
                errors[h].append(abs(steps[h, 0] - rho[t0 - 1 + h]))
    return [float(np.mean(e)) if e else None for e in errors]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--modes", nargs="+", default=["joint", "probe", "finetune"])
    parser.add_argument("--data", default="PEARLSim/sequence_data")
    parser.add_argument("--config", help="experiment YAML whose split (fixed test families) to use, e.g. "
                                         "PEARLSim/experiments/v1_lstm.yaml, so results compare with its baselines")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--label-noise-pcm", type=float, default=265.0,
                        help="global reactivity label noise (1 sigma, pcm) used as the head's variance floor; "
                             "v1: 265; v2: from the replica spread")
    args = parser.parse_args(argv)
    import torch

    phase1 = torch.load(args.phase1, weights_only=False)
    p1 = phase1["config"]
    override = {"data": {"path": args.data}}
    if args.config:
        override["split"] = experiment.load_config(args.config)["split"]
    cfg = experiment.merge(experiment.DEFAULTS, override)
    sequences = with_reactivity(experiment.load_sequences(cfg))
    fit_fams, val_fams, test_fams = experiment.split_families(cfg, sequences)
    pick = lambda fams: [s for s in sequences if s.family in fams]  # noqa: E731
    obs, ctrl = phase1["obs"], phase1["ctrl"]
    train = make_tokens(pick(fit_fams), obs, ctrl, ["rho_pcm"])
    val, test = make_tokens(pick(val_fams), obs, ctrl, ["rho_pcm"]), make_tokens(pick(test_fams), obs, ctrl, ["rho_pcm"])
    scaler = TokenScaler.fit(train)
    if not (np.allclose(scaler.mean, phase1["mean"]) and np.allclose(scaler.std, phase1["std"])):
        raise SystemExit("refitted input scaling differs from phase 1: different data or split")
    y = np.concatenate(train.y)
    noise_var = (features.NoiseModel().sigmas(obs) * np.abs(y).mean(axis=0) / scaler.y_std) ** 2
    for mode in args.modes:
        out = os.path.join(args.out, mode)
        os.makedirs(out, exist_ok=True)
        config = ForecasterConfig(chunk=p1["chunk"], batch_size=p1["batch_size"], epochs=args.epochs,
                                  d_model=p1["d_model"], heads=p1["heads"], layers=p1["layers"],
                                  dropout=p1["dropout"], device=args.device, mode=mode,
                                  lr=p1["lr"] if mode != "finetune" else p1["lr"] / 3)
        start = time.time()
        step_noise_var = np.array([(args.label_noise_pcm / scaler.z_std[0]) ** 2], dtype=np.float32)
        model, scaler_m, history = fit(train, config, validation=val, noise_var=noise_var, scaler=scaler,
                                       step_noise_var=step_noise_var,
                                       init=None if mode == "joint" else phase1["state_dict"],
                                       log=lambda e, m=mode: print(m, e, flush=True))
        model.cpu()
        results = {"mode": mode, "config": asdict(config), "split": {"test": test_fams},
                   "history": history, "train_seconds": time.time() - start,
                   "test_teacher_forced": teacher_forced_rho(model, scaler_m, test, config.chunk),
                   "test_rollout_rho_mae_pcm": rollout_rho(model, scaler_m, pick(test_fams), obs, ctrl,
                                                           args.context, args.horizon, args.stride)}
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump(results, f, indent=1)
        torch.save({"state_dict": model.state_dict(), "scaler": asdict(scaler_m), "obs": obs, "ctrl": ctrl,
                    "step": ["rho_pcm"], "config": asdict(config)}, os.path.join(out, "model.pt"))
        r = results["test_rollout_rho_mae_pcm"]
        print(mode, "teacher-forced", results["test_teacher_forced"], "rollout MAE pcm h0/5/10/20:",
              [round(r[h]) for h in (0, 5, 10, args.horizon) if h < len(r)], flush=True)


if __name__ == "__main__":
    main()
