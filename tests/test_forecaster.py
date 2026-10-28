"""Phase-1 forecaster: tokens, causality, learning a known linear system, rollout (torch, CPU, seconds)."""

import numpy as np
import pandas as pd
import pytest

from pearlsim.ml.data import Sequence
from pearlsim.ml.forecaster import TokenScaler, make_tokens

OBS, CTRL = ["d1", "d2"], ["u"]


def linear_sequences(n=40, length=60, seed=0):
    """D_{i+1} = A D_i + B u_i + small noise; u a random piecewise-constant schedule."""
    rng = np.random.default_rng(seed)
    a = np.array([[0.9, 0.05], [0.0, 0.8]])
    b = np.array([0.3, -0.2])
    sequences = []
    for s in range(n):
        u = np.repeat(rng.uniform(-1, 1, length // 6 + 1), 6)[:length]
        d = np.zeros((length, 2))
        d[0] = rng.normal(0, 1, 2)
        for i in range(length - 1):
            d[i + 1] = a @ d[i] + b * u[i] + rng.normal(0, 0.01, 2)
        # a "step" quantity of row i (like reactivity): depends on D_i and on the control u_i of that step
        rho = 2.0 * d[:, 0] - 1.0 * d[:, 1] + 1.5 * u
        frame = pd.DataFrame({"d1": d[:, 0], "d2": d[:, 1], "u": u, "rho": rho})
        sequences.append(Sequence(f"lin{s}", f"lin{s}", frame))
    return sequences, a, b


def test_tokens_pair_each_row_with_the_next_measurements():
    sequences, _, _ = linear_sequences(n=1, length=5)
    tokens = make_tokens(sequences, OBS, CTRL)
    frame = sequences[0].frame
    assert tokens.x[0].shape == (4, 3) and tokens.y[0].shape == (4, 2)
    assert np.allclose(tokens.x[0][2], frame.loc[2, ["d1", "d2", "u"]].to_numpy())
    assert np.allclose(tokens.y[0][2], frame.loc[3, ["d1", "d2"]].to_numpy())
    for delta in (True, False):
        scaler = TokenScaler.fit(tokens, delta=delta)
        scaled = scaler.y(tokens.y[0], tokens.x[0])
        assert np.allclose(scaler.unscale_y(scaled, tokens.x[0]), tokens.y[0], atol=1e-5)


@pytest.mark.torch
def test_model_is_causal():
    import torch

    from pearlsim.ml.forecaster import build_model
    torch.manual_seed(0)
    model = build_model(3, 2, n_step=1, d_model=16, heads=2, layers=2, max_len=32, dropout=0.0).eval()
    x = torch.randn(1, 20, 3)
    changed = x.clone()
    changed[0, 12:] += 5.0
    with torch.no_grad():
        a, b = model(x), model(changed)
    for key in ("obs_mean", "step_mean"):
        assert torch.allclose(a[key][0, :12], b[key][0, :12], atol=1e-5)
        assert not torch.allclose(a[key][0, 12:], b[key][0, 12:])


@pytest.mark.torch
def test_learns_a_linear_system_and_rolls_out():
    from pearlsim.ml.forecaster import ForecasterConfig, fit, rollout
    train, a, b = linear_sequences(n=60, seed=1)
    test, _, _ = linear_sequences(n=5, seed=2)
    cfg = ForecasterConfig(chunk=32, batch_size=16, epochs=25, d_model=32, heads=2, layers=2, dropout=0.0, lr=3e-3)
    tokens = make_tokens(train, OBS, CTRL)
    model, scaler, history = fit(tokens, cfg, validation=make_tokens(test, OBS, CTRL), log=lambda e: None)
    assert history[-1]["validation_obs_nll"] < history[0]["validation_obs_nll"]
    frame = test[0].frame
    d, u = frame[OBS].to_numpy(), frame[CTRL].to_numpy()
    pred = rollout(model, scaler, d[:20], u[:20], u[20:30])
    truth = np.zeros((11, 2))
    truth[0] = a @ d[19] + b * u[19, 0]
    for h in range(10):
        truth[h + 1] = a @ truth[h] + b * u[20 + h, 0]
    scale = d.std(axis=0)
    assert np.abs(pred[0] - truth[0]).max() < 0.2 * scale.max()   # one step
    assert np.abs(pred - truth).mean() < 0.3 * scale.max()          # ten-step rollout stays on track


@pytest.mark.torch
def test_phase2_modes_learn_the_step_target():
    import torch

    from pearlsim.ml.forecaster import ForecasterConfig, evaluate, fit, rollout
    train, _, _ = linear_sequences(n=60, seed=1)
    test, _, _ = linear_sequences(n=5, seed=2)
    small = dict(chunk=32, batch_size=16, d_model=32, heads=2, layers=2, dropout=0.0, lr=3e-3)
    base, _, _ = fit(make_tokens(train, OBS, CTRL), ForecasterConfig(epochs=10, **small), log=lambda e: None)
    tokens, held = make_tokens(train, OBS, CTRL, ["rho"]), make_tokens(test, OBS, CTRL, ["rho"])
    init = base.state_dict()
    results = {}
    for mode in ("joint", "probe", "finetune"):
        model, scaler, _ = fit(tokens, ForecasterConfig(epochs=15, mode=mode, **small), log=lambda e: None,
                               init=None if mode == "joint" else init)
        results[mode] = evaluate(model, scaler, held, ForecasterConfig(**small))["step_nll"]
        if mode == "probe":  # the phase-1 trunk is unchanged
            assert all(torch.equal(p, init[name]) for name, p in model.state_dict().items() if name in init)
    assert all(v < 0.0 for v in results.values()), results  # well below a unit-variance guess (NLL ~1.42)
    frame = test[0].frame
    d, u = frame[OBS].to_numpy(), frame[CTRL].to_numpy()
    obs, steps = rollout(model, scaler, d[:20], u[:20], u[20:25], with_steps=True)
    assert obs.shape == (6, 2) and steps.shape == (6, 1)
    assert abs(steps[0, 0] - frame["rho"].iloc[19]) < 0.3 * frame["rho"].std()  # horizon 0: the decision step
    # a global label-noise floor bounds the step NLL from below: 0.5 log(var) >= 0.5 log(floor)
    floor = np.array([0.25], dtype=np.float32)
    floored = evaluate(model, scaler, held, ForecasterConfig(**small), step_noise_var=torch.as_tensor(floor))
    assert floored["step_nll"] >= 0.5 * np.log(floor[0]) - 1e-6
    with pytest.raises(ValueError, match="phase-1 state"):
        fit(tokens, ForecasterConfig(epochs=1, mode="probe", **small), log=lambda e: None, init=None)

