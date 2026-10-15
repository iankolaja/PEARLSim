"""Training loop for ``pearlsim.ml.models`` (PyTorch).

Defaults follow the paper: Adam, initial learning rate 0.01 decayed by 0.9 every
10,000 optimiser steps, batch 1024, at most 500 epochs, early stopping on the
validation loss after at least 10 epochs with patience 25, best weights restored.

The loss is a masked, sample-weighted MSE on standardised targets (statistics from
the training windows). Measurement noise (``features.NoiseModel``) can be redrawn
every epoch on the input columns it covers, instead of v1's single fixed draw.
"""

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from pearlsim.ml.features import Standardizer


@dataclass
class TrainConfig:
    learning_rate: float = 0.01
    decay_rate: float = 0.9
    decay_steps: int = 10_000
    batch_size: int = 1024
    max_epochs: int = 500
    min_epochs: int = 10
    patience: int = 25
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "auto"
    threads: int | None = None


class Scalers:
    """Standardisation of past inputs, future controls and targets, fitted on training windows."""

    def __init__(self, windows):
        self.past = Standardizer().fit(windows.past)
        n_controls = windows.future_controls.shape[-1]
        self.controls = (Standardizer().fit(windows.future_controls.reshape(-1, n_controls))
                         if windows.future_controls.size else None)
        t = windows.targets.shape[-1]
        flat_y, flat_m = windows.targets.reshape(-1, t), windows.mask.reshape(-1, t)
        self.target_mean = np.array([flat_y[flat_m[:, j], j].mean() for j in range(t)])
        std = np.array([flat_y[flat_m[:, j], j].std() for j in range(t)])
        self.target_std = np.where(std > 0, std, 1.0)

    def inputs(self, past, controls):
        past = self.past.transform(past)
        controls = self.controls.transform(controls) if self.controls is not None else controls
        return past.astype(np.float32), np.asarray(controls, dtype=np.float32)

    def targets(self, y):
        return ((y - self.target_mean) / self.target_std).astype(np.float32)

    def unscale(self, y):
        return y * self.target_std + self.target_mean


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _batches(n, size, rng):
    order = rng.permutation(n)
    for start in range(0, n, size):
        yield order[start: start + size]


def _loss(prediction, target, mask, weight):
    squared = (prediction - target) ** 2 * mask
    per_sample = squared.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)
    return (per_sample * weight).sum() / weight.sum().clamp(min=1e-12)


def fit(model, train, validation, config=TrainConfig(), weights=None, noise=None, log=print):
    """Train ``model`` on WindowSet ``train``; returns (model with best weights, scalers, history)."""
    if config.threads:
        torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = _device(config.device)
    model.to(device)
    scalers = Scalers(train)
    weights = np.ones(len(train)) if weights is None else np.asarray(weights, dtype=float)

    def tensors(windows, past=None):
        p, c = scalers.inputs(windows.past if past is None else past, windows.future_controls)
        return (torch.from_numpy(p).to(device), torch.from_numpy(c).to(device),
                torch.from_numpy(scalers.targets(windows.targets)).to(device),
                torch.from_numpy(windows.mask.astype(np.float32)).to(device))

    val_p, val_c, val_y, val_m = tensors(validation)
    val_w = torch.ones(len(validation), device=device)
    train_p, train_c, train_y, train_m = tensors(train)
    train_w = torch.from_numpy(weights.astype(np.float32)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: config.decay_rate ** (step / config.decay_steps))
    best, best_state, since_best = float("inf"), None, 0
    history = []
    for epoch in range(config.max_epochs):
        start = time.time()
        if noise is not None:
            noisy = noise.apply(train.past, train.input_columns, rng)
            train_p = torch.from_numpy(scalers.inputs(noisy, train.future_controls)[0]).to(device)
        model.train()
        total = 0.0
        for idx in _batches(len(train), config.batch_size, rng):
            idx = torch.from_numpy(idx).to(device)
            optimizer.zero_grad()
            loss = _loss(model(train_p[idx], train_c[idx]), train_y[idx], train_m[idx], train_w[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = _loss(model(val_p, val_c), val_y, val_m, val_w).item()
        history.append({"epoch": epoch + 1, "train_loss": total / len(train), "val_loss": val_loss,
                        "seconds": time.time() - start})
        if val_loss < best:
            best, best_state, since_best = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
        if log and (epoch % 10 == 0 or since_best == 0):
            log(f"epoch {epoch + 1}: train {total / len(train):.4g} val {val_loss:.4g}")
        if epoch + 1 >= config.min_epochs and since_best >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, scalers, history


def predict(model, scalers, windows, past=None, batch_size=8192):
    """Predictions in target units, shape (N, H+1, T); ``past`` overrides the window inputs (noise draws)."""
    device = next(model.parameters()).device
    p, c = scalers.inputs(windows.past if past is None else past, windows.future_controls)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(p), batch_size):
            out.append(model(torch.from_numpy(p[start: start + batch_size]).to(device),
                             torch.from_numpy(c[start: start + batch_size]).to(device)).cpu().numpy())
    return scalers.unscale(np.concatenate(out) if out else np.zeros((0,) + windows.targets.shape[1:]))
