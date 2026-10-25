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
        frame = pd.DataFrame({"d1": d[:, 0], "d2": d[:, 1], "u": u})
        sequences.append(Sequence(f"lin{s}", f"lin{s}", frame))
    return sequences, a, b


def test_tokens_pair_each_row_with_the_next_measurements():
    sequences, _, _ = linear_sequences(n=1, length=5)
    tokens = make_tokens(sequences, OBS, CTRL)
    frame = sequences[0].frame
    assert tokens.x[0].shape == (4, 3) and tokens.y[0].shape == (4, 2)
    assert np.allclose(tokens.x[0][2], frame.loc[2, ["d1", "d2", "u"]].to_numpy())
    assert np.allclose(tokens.y[0][2], frame.loc[3, ["d1", "d2"]].to_numpy())
    scaler = TokenScaler.fit(tokens)
    assert np.allclose(scaler.unscale_y(scaler.y(tokens.y[0])), tokens.y[0], atol=1e-6)


@pytest.mark.torch
def test_model_is_causal():
    import torch

    from pearlsim.ml.forecaster import build_model
    torch.manual_seed(0)
    model = build_model(3, 2, d_model=16, heads=2, layers=2, max_len=32, dropout=0.0).eval()
    x = torch.randn(1, 20, 3)
    changed = x.clone()
    changed[0, 12:] += 5.0
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(changed)
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-5) and not torch.allclose(a[0, 12:], b[0, 12:])


@pytest.mark.torch
def test_learns_a_linear_system_and_rolls_out():
    from pearlsim.ml.forecaster import ForecasterConfig, fit, rollout
    train, a, b = linear_sequences(n=60, seed=1)
    test, _, _ = linear_sequences(n=5, seed=2)
    cfg = ForecasterConfig(chunk=32, batch_size=16, epochs=25, d_model=32, heads=2, layers=2, dropout=0.0, lr=3e-3)
    tokens = make_tokens(train, OBS, CTRL)
    model, scaler, history = fit(tokens, cfg, validation=make_tokens(test, OBS, CTRL), log=lambda e: None)
    assert history[-1]["validation_nll"] < history[0]["validation_nll"]
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
