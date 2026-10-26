"""Causal transformer forecaster, phase 1: next-step pre-training on observables (ROADMAP item 9).

Tokens: one per step. Row i of a sequence holds the measurements D_i taken before step i and the controls u_i
applied in step i (``pearlsim.ml.windows`` alignment), so token i = [D_i, u_i] and its target is D_{i+1}.
Only operator observables enter (no k, no meshes); phase 2 adds reactivity and mesh heads on the same trunk.

The decoder predicts a mean and a log-variance per observable at every position (teacher forcing: one forward
pass trains all positions). Loss: Gaussian negative log-likelihood, with the known measurement noise as a floor on
the predicted variance (noise-aware). Rollout feeds the predicted means back with planned future controls.
Torch is imported lazily so ``import pearlsim.ml`` stays light.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TokenSet:
    """Per-sequence token arrays (scaled): x (T-1, n_obs + n_ctrl), y (T-1, n_obs)."""

    x: list
    y: list
    names: list
    obs_columns: list
    control_columns: list


def make_tokens(sequences, obs_columns, control_columns):
    """Unscaled tokens: x_i = [D_i, u_i], y_i = D_{i+1}, for i = 0 .. T-2 of each sequence."""
    xs, ys, names = [], [], []
    for seq in sequences:
        obs = seq.frame[list(obs_columns)].to_numpy(dtype=np.float32)
        ctrl = seq.frame[list(control_columns)].to_numpy(dtype=np.float32)
        if len(obs) < 2:
            continue
        xs.append(np.concatenate([obs[:-1], ctrl[:-1]], axis=1))
        ys.append(obs[1:])
        names.append(seq.name)
    return TokenSet(xs, ys, names, list(obs_columns), list(control_columns))


@dataclass
class TokenScaler:
    """Scaling fitted on training tokens. Inputs: mean/std per column. Targets: the change D_{i+1} - D_i
    (``delta``, default), scaled by its own mean/std, so "no change" is the easy prediction; or D_{i+1} with the
    observables' input scale."""

    mean: np.ndarray
    std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    n_obs: int
    delta: bool = True

    @classmethod
    def fit(cls, tokens, delta=True):
        x = np.concatenate(tokens.x)
        mean, std = x.mean(axis=0), x.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        n_obs = len(tokens.obs_columns)
        if delta:
            change = np.concatenate([y - xs[:, :n_obs] for xs, y in zip(tokens.x, tokens.y, strict=True)])
            y_mean, y_std = change.mean(axis=0), change.std(axis=0)
            y_std = np.where(y_std > 0, y_std, 1.0)
        else:
            y_mean, y_std = mean[:n_obs], std[:n_obs]
        f32 = np.float32
        return cls(mean.astype(f32), std.astype(f32), y_mean.astype(f32), y_std.astype(f32), n_obs, delta)

    def x(self, x):
        return (x - self.mean) / self.std

    def y(self, y, x):
        """Scaled target for next observables y given the (unscaled) tokens x."""
        base = x[..., : self.n_obs] if self.delta else 0.0
        return (y - base - self.y_mean) / self.y_std

    def unscale_y(self, prediction, x):
        """Next observables from a scaled prediction and the (unscaled) tokens x."""
        base = x[..., : self.n_obs] if self.delta else 0.0
        return base + prediction * self.y_std + self.y_mean


def build_model(n_in, n_obs, d_model=64, heads=4, layers=3, max_len=512, dropout=0.1):
    import torch
    from torch import nn

    class CausalForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(n_in, d_model)
            self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
                                               batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.mean = nn.Linear(d_model, n_obs)
            self.log_var = nn.Linear(d_model, n_obs)
            self.max_len = max_len

        def hidden(self, x, padding=None):
            length = x.shape[1]
            if length > self.max_len:
                raise ValueError(f"sequence chunk of {length} tokens exceeds max_len {self.max_len}")
            causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
            return self.trunk(self.embed(x) + self.position[:, :length], mask=causal,
                              src_key_padding_mask=padding)

        def forward(self, x, padding=None):
            h = self.hidden(x, padding)
            return self.mean(h), self.log_var(h).clamp(-12.0, 8.0)

    return CausalForecaster()


def gaussian_nll(mean, log_var, target, mask, noise_var=None):
    """Mean Gaussian NLL over unmasked entries; noise_var (per observable, scaled units) floors the variance."""
    import torch

    var = log_var.exp()
    if noise_var is not None:
        var = var + noise_var
    nll = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)
    return (nll * mask).sum() / mask.sum().clamp(min=1)


@dataclass
class ForecasterConfig:
    chunk: int = 64
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 64
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1
    seed: int = 0
    device: str = "cpu"


def _chunks(tokens, scaler, length, rng):
    """One random chunk per sequence (shorter sequences padded), scaled."""
    n_in, n_obs = tokens.x[0].shape[1], tokens.y[0].shape[1]
    x = np.zeros((len(tokens.x), length, n_in), np.float32)
    y = np.zeros((len(tokens.x), length, n_obs), np.float32)
    valid = np.zeros((len(tokens.x), length), bool)
    for b, (xs, ys) in enumerate(zip(tokens.x, tokens.y, strict=True)):
        start = rng.integers(0, max(1, len(xs) - length + 1))
        part = slice(start, start + length)
        n = len(xs[part])
        x[b, :n], y[b, :n], valid[b, :n] = scaler.x(xs[part]), scaler.y(ys[part], xs[part]), True
    return x, y, valid


def fit(tokens, config=ForecasterConfig(), validation=None, noise_var=None, log=print, scaler=None):
    """Train a phase-1 forecaster on TokenSet ``tokens``; returns (model, scaler, history).

    noise_var: per-observable variance floor in the scaler's target units (``scaler.y_std``).
    """
    import torch

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    scaler = scaler or TokenScaler.fit(tokens)
    n_in, n_obs = tokens.x[0].shape[1], tokens.y[0].shape[1]
    model = build_model(n_in, n_obs, config.d_model, config.heads, config.layers, max(config.chunk, 8),
                        config.dropout).to(config.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    floor = None if noise_var is None else torch.as_tensor(noise_var, dtype=torch.float32, device=config.device)
    steps_per_epoch = max(1, sum(len(x) for x in tokens.x) // (config.chunk * config.batch_size))
    history = []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            pick = rng.integers(0, len(tokens.x), config.batch_size)
            sub = TokenSet([tokens.x[i] for i in pick], [tokens.y[i] for i in pick], [], tokens.obs_columns,
                           tokens.control_columns)
            x, y, valid = (torch.as_tensor(a, device=config.device) for a in _chunks(sub, scaler, config.chunk, rng))
            mean, log_var = model(x, padding=~valid)
            mask = valid.unsqueeze(-1).expand_as(y).float()
            loss = gaussian_nll(mean, log_var, y, mask, floor)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(loss.item())
        entry = {"epoch": epoch, "train_nll": float(np.mean(losses))}
        if validation is not None:
            entry["validation_nll"] = evaluate_nll(model, scaler, validation, config, floor)
        history.append(entry)
        log(entry)
    return model, scaler, history


def evaluate_nll(model, scaler, tokens, config=ForecasterConfig(), noise_var=None):
    """Teacher-forced NLL over every token of ``tokens`` (chunked from the start of each sequence)."""
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xs, ys in zip(tokens.x, tokens.y, strict=True):
            for start in range(0, len(xs), config.chunk):
                part = slice(start, start + config.chunk)
                x = torch.as_tensor(scaler.x(xs[part])[None], device=config.device)
                y = torch.as_tensor(scaler.y(ys[part], xs[part])[None], device=config.device)
                mean, log_var = model(x)
                mask = torch.ones_like(y)
                total += gaussian_nll(mean, log_var, y, mask, noise_var).item() * y.numel()
                count += y.numel()
    return total / max(1, count)


def rollout(model, scaler, context_obs, context_controls, future_controls, max_context=None):
    """Autoregressive forecast of the observables (unscaled means).

    context_obs (T0, n_obs) and context_controls (T0, n_ctrl): the known rows D_0..D_{T0-1}, u_0..u_{T0-1};
    future_controls (H, n_ctrl): u_{T0} .. u_{T0+H-1}. Returns (H+1, n_obs): D_{T0} .. D_{T0+H}, each step fed
    back as the next token's observables. ``max_context`` (default: the model's max_len) truncates history.
    """
    import torch

    model.eval()
    limit = max_context or model.max_len
    obs = [np.asarray(o, np.float32) for o in context_obs]
    ctrl = [np.asarray(u, np.float32) for u in context_controls] + [np.asarray(u, np.float32) for u in future_controls]
    out = []
    with torch.no_grad():
        for _ in range(len(future_controls) + 1):
            t = len(obs)
            tokens = np.concatenate([np.stack(obs), np.stack(ctrl[:t])], axis=1)[-limit:]
            mean, _ = model(torch.as_tensor(scaler.x(tokens)[None]))
            nxt = scaler.unscale_y(mean[0, -1].cpu().numpy(), tokens[-1])
            out.append(nxt)
            obs.append(nxt.astype(np.float32))
    return np.stack(out)
