"""CPU smoke tests of the PyTorch models and training loop (run with ``pytest -m torch``)."""

import numpy as np
import pytest

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def torch():
    import torch

    torch.set_num_threads(1)
    return torch


@pytest.mark.parametrize("encoder, params", [("lstm", {"sizes": (8, 4), "keras_init": True}),
                                             ("gru", {"sizes": (8,)}),
                                             ("tcn", {"channels": 8, "levels": 2}),
                                             ("transformer", {"d_model": 8, "heads": 2, "layers": 1})])
def test_encoders_shapes_and_causality(torch, encoder, params):
    from pearlsim.ml.models import SequenceRegressor

    model = SequenceRegressor(5, 2, 3, horizon=2, encoder=encoder, window=6, **params).eval()
    past, future = torch.randn(4, 6, 5), torch.randn(4, 2, 2)
    out = model(past, future)
    assert out.shape == (4, 3, 3)
    changed = future.clone()
    changed[:, 1] += 10.0  # control of step i+2 cannot affect horizons 0 and 1
    out2 = model(past, changed)
    assert torch.allclose(out[:, :2], out2[:, :2]) and not torch.allclose(out[:, 2], out2[:, 2])


def test_keras_init_sets_forget_bias(torch):
    from pearlsim.ml.models import paper_lstm

    lstm = paper_lstm(5, sizes=(4,)).encoder.layers[0]
    assert torch.all(lstm.bias_ih_l0[4:8] == 1) and torch.all(lstm.bias_ih_l0[:4] == 0)


def test_training_learns_a_linear_relation(torch):
    from pearlsim.ml import train
    from pearlsim.ml.models import SequenceRegressor
    from pearlsim.ml.windows import WindowSet

    rng = np.random.default_rng(0)

    def windows(n):
        past = rng.normal(size=(n, 4, 3)).astype(np.float32)
        y = (past[:, -1, 0] * 2 + past[:, -2, 1]).reshape(n, 1, 1).astype(np.float32)
        return WindowSet(past, np.zeros((n, 0, 0), np.float32), y, np.ones((n, 1, 1), bool),
                         np.array(["s"] * n), np.array(["f"] * n), np.arange(n), ["a", "b", "c"], [], ["y"])

    fit, val = windows(512), windows(128)
    model = SequenceRegressor(3, 0, 1, encoder="gru", sizes=(16,))
    config = train.TrainConfig(learning_rate=0.01, batch_size=64, max_epochs=60, patience=10, device="cpu")
    model, scalers, history = train.fit(model, fit, val, config, log=None)
    prediction = train.predict(model, scalers, val)
    assert prediction.shape == (128, 1, 1)
    assert history[-1]["val_loss"] < 0.2 * history[0]["val_loss"]
    assert np.sqrt(np.mean((prediction - val.targets) ** 2)) < 0.5 * val.targets.std()
