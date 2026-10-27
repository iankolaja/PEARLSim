"""Causal transformer forecaster (ROADMAP item 9).

Tokens: one per step. Row i of a sequence holds the measurements D_i taken before step i and the controls u_i
applied in step i (``pearlsim.ml.windows`` alignment), so token i = [D_i, u_i]. Every position predicts:
  - phase 1 (pre-training, observables only): the next measurements D_{i+1};
  - phase 2 (added heads on the same trunk): "step" quantities of step i, e.g. its reactivity, the windows'
    ``step`` targets; the next-step loss stays as an auxiliary.
Phase-2 modes: ``joint`` (all from scratch), ``probe`` (phase-1 trunk frozen, new heads only), ``finetune``
(start from phase 1, train everything).

Each head predicts a mean and a log-variance (Gaussian NLL); the known measurement noise floors the
observables' variance (noise-aware). Observable targets default to the change D_{i+1} - D_i. Rollout feeds the
predicted means back with the planned controls. Torch is imported lazily so ``import pearlsim.ml`` stays light.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TokenSet:
    """Per-sequence arrays (unscaled): x (T-1, n_obs + n_ctrl), y next observables (T-1, n_obs),
    z step quantities (T-1, n_step) or None."""

    x: list
    y: list
    names: list
    obs_columns: list
    control_columns: list
    z: list = None
    step_columns: list = field(default_factory=list)

    def subset(self, index):
        z = None if self.z is None else [self.z[i] for i in index]
        return TokenSet([self.x[i] for i in index], [self.y[i] for i in index], [self.names[i] for i in index],
                        self.obs_columns, self.control_columns, z, self.step_columns)


def make_tokens(sequences, obs_columns, control_columns, step_columns=()):
    """Tokens x_i = [D_i, u_i] with targets y_i = D_{i+1} and z_i = step columns of row i, i = 0 .. T-2."""
    xs, ys, zs, names = [], [], [], []
    for seq in sequences:
        obs = seq.frame[list(obs_columns)].to_numpy(dtype=np.float32)
        ctrl = seq.frame[list(control_columns)].to_numpy(dtype=np.float32)
        if len(obs) < 2:
            continue
        xs.append(np.concatenate([obs[:-1], ctrl[:-1]], axis=1))
        ys.append(obs[1:])
        if step_columns:
            zs.append(seq.frame[list(step_columns)].to_numpy(dtype=np.float32)[:-1])
        names.append(seq.name)
    return TokenSet(xs, ys, names, list(obs_columns), list(control_columns), zs if step_columns else None,
                    list(step_columns))


@dataclass
class TokenScaler:
    """Scaling fitted on training tokens. Inputs: mean/std per column. Next-observable targets: the change
    D_{i+1} - D_i (``delta``, default) with its own mean/std, so "no change" is the easy prediction, or D_{i+1}
    with the observables' input scale. Step targets: their own mean/std."""

    mean: np.ndarray
    std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    n_obs: int
    delta: bool = True
    z_mean: np.ndarray = None
    z_std: np.ndarray = None

    @classmethod
    def fit(cls, tokens, delta=True):
        x = np.concatenate(tokens.x)
        mean, std = x.mean(axis=0), _safe(x.std(axis=0))
        n_obs = len(tokens.obs_columns)
        if delta:
            change = np.concatenate([y - xs[:, :n_obs] for xs, y in zip(tokens.x, tokens.y, strict=True)])
            y_mean, y_std = change.mean(axis=0), _safe(change.std(axis=0))
        else:
            y_mean, y_std = mean[:n_obs], std[:n_obs]
        z_mean = z_std = None
        if tokens.z is not None:
            z = np.concatenate(tokens.z)
            z_mean, z_std = z.mean(axis=0).astype(np.float32), _safe(z.std(axis=0)).astype(np.float32)
        f32 = np.float32
        return cls(mean.astype(f32), std.astype(f32), y_mean.astype(f32), y_std.astype(f32), n_obs, delta,
                   z_mean, z_std)

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

    def z(self, z):
        return (z - self.z_mean) / self.z_std

    def unscale_z(self, z):
        return z * self.z_std + self.z_mean


def _safe(std):
    return np.where(std > 0, std, 1.0)


def build_model(n_in, n_obs, n_step=0, d_model=64, heads=4, layers=3, max_len=512, dropout=0.1):
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
            self.step_head = nn.Linear(d_model, 2 * n_step) if n_step else None
            self.max_len, self.n_step = max_len, n_step

        def hidden(self, x, padding=None):
            length = x.shape[1]
            if length > self.max_len:
                raise ValueError(f"sequence chunk of {length} tokens exceeds max_len {self.max_len}")
            causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
            return self.trunk(self.embed(x) + self.position[:, :length], mask=causal,
                              src_key_padding_mask=padding)

        def forward(self, x, padding=None):
            h = self.hidden(x, padding)
            out = {"obs_mean": self.mean(h), "obs_log_var": self.log_var(h).clamp(-12.0, 8.0)}
            if self.step_head is not None:
                step = self.step_head(h)
                out["step_mean"] = step[..., : self.n_step]
                out["step_log_var"] = step[..., self.n_step:].clamp(-12.0, 8.0)
            return out

        def phase1_parameters(self):
            return [p for name, p in self.named_parameters() if not name.startswith("step_head")]

    return CausalForecaster()


def gaussian_nll(mean, log_var, target, mask, noise_var=None):
    """Mean Gaussian NLL over unmasked entries; noise_var (per column, scaled units) floors the variance."""
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
    mode: str = "pretrain"     # pretrain (phase 1) | joint | probe | finetune (phase 2)
    aux_weight: float = 0.3    # phase 2: weight of the next-observable loss next to the step loss
    keep_best: bool = True     # with validation data, return the weights of the best validation epoch

    def __post_init__(self):
        if self.mode not in ("pretrain", "joint", "probe", "finetune"):
            raise ValueError(f"unknown mode {self.mode!r}")


def _chunks(tokens, scaler, length, rng):
    """One random chunk per sequence (shorter sequences padded), scaled."""
    n_in, n_obs = tokens.x[0].shape[1], tokens.y[0].shape[1]
    n_step = 0 if tokens.z is None else tokens.z[0].shape[1]
    x = np.zeros((len(tokens.x), length, n_in), np.float32)
    y = np.zeros((len(tokens.x), length, n_obs), np.float32)
    z = np.zeros((len(tokens.x), length, n_step), np.float32)
    valid = np.zeros((len(tokens.x), length), bool)
    for b, xs in enumerate(tokens.x):
        start = rng.integers(0, max(1, len(xs) - length + 1))
        part = slice(start, start + length)
        n = len(xs[part])
        x[b, :n], y[b, :n], valid[b, :n] = scaler.x(xs[part]), scaler.y(tokens.y[b][part], xs[part]), True
        if n_step:
            z[b, :n] = scaler.z(tokens.z[b][part])
    return x, y, z, valid


def _loss(model_out, y, z, valid, config, floor):
    import torch

    mask = valid.unsqueeze(-1).float()
    obs = gaussian_nll(model_out["obs_mean"], model_out["obs_log_var"], y, mask.expand_as(y), floor)
    if config.mode == "pretrain":
        return obs, {"obs_nll": obs.item()}
    zero = torch.zeros((), device=y.device)
    step = gaussian_nll(model_out["step_mean"], model_out["step_log_var"], z, mask.expand_as(z)) if z.numel() else zero
    total = step + (0.0 if config.mode == "probe" else config.aux_weight) * obs
    return total, {"obs_nll": obs.item(), "step_nll": step.item()}


def fit(tokens, config=ForecasterConfig(), validation=None, noise_var=None, log=print, scaler=None, init=None):
    """Train a forecaster on TokenSet ``tokens``; returns (model, scaler, history).

    noise_var: per-observable variance floor in the scaler's target units (``scaler.y_std``).
    init: a phase-1 state dict (``probe``/``finetune``): the step head starts fresh, the rest is loaded.
    """
    import torch

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    scaler = scaler or TokenScaler.fit(tokens)
    n_in, n_obs = tokens.x[0].shape[1], tokens.y[0].shape[1]
    n_step = 0 if config.mode == "pretrain" or tokens.z is None else tokens.z[0].shape[1]
    if config.mode != "pretrain" and not n_step:
        raise ValueError(f"mode {config.mode!r} needs step targets (make_tokens(..., step_columns=...))")
    model = build_model(n_in, n_obs, n_step, config.d_model, config.heads, config.layers, max(config.chunk, 8),
                        config.dropout).to(config.device)
    if config.mode in ("probe", "finetune"):
        if init is None:
            raise ValueError(f"mode {config.mode!r} needs the phase-1 state dict (init=...)")
        missing, unexpected = model.load_state_dict(init, strict=False)
        if unexpected or any(not name.startswith("step_head") for name in missing):
            raise ValueError(f"phase-1 state does not match: missing {missing}, unexpected {unexpected}")
    if config.mode == "probe":
        for p in model.phase1_parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    floor = None if noise_var is None else torch.as_tensor(noise_var, dtype=torch.float32, device=config.device)
    steps_per_epoch = max(1, sum(len(x) for x in tokens.x) // (config.chunk * config.batch_size))
    history = []
    best_score, best_state = np.inf, None
    for epoch in range(config.epochs):
        model.train()
        parts = []
        for _ in range(steps_per_epoch):
            sub = tokens.subset(rng.integers(0, len(tokens.x), config.batch_size))
            x, y, z, valid = (torch.as_tensor(a, device=config.device) for a in _chunks(sub, scaler, config.chunk, rng))
            loss, detail = _loss(model(x, padding=~valid), y, z, valid, config, floor)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            parts.append(detail)
        entry = {"epoch": epoch, **{f"train_{k}": float(np.mean([p[k] for p in parts])) for k in parts[0]}}
        if validation is not None:
            entry.update({f"validation_{k}": v for k, v in evaluate(model, scaler, validation, config, floor).items()})
            score = entry["validation_obs_nll" if config.mode == "pretrain" else "validation_step_nll"]
            if config.keep_best and score < best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                entry["best"] = True
        history.append(entry)
        log(entry)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, history


def evaluate_nll(model, scaler, tokens, config=ForecasterConfig(), noise_var=None):
    """Teacher-forced NLL of the next observables over every token (kept for phase-1 callers)."""
    return evaluate(model, scaler, tokens, config, noise_var)["obs_nll"]


def evaluate(model, scaler, tokens, config=ForecasterConfig(), noise_var=None):
    """Teacher-forced NLL per head over every token, chunked from the start of each sequence."""
    import torch

    model.eval()
    sums, counts = {}, {}
    with torch.no_grad():
        for b, xs in enumerate(tokens.x):
            for start in range(0, len(xs), config.chunk):
                part = slice(start, start + config.chunk)
                x = torch.as_tensor(scaler.x(xs[part])[None], device=config.device)
                y = torch.as_tensor(scaler.y(tokens.y[b][part], xs[part])[None], device=config.device)
                out = model(x)
                pieces = {"obs_nll": (out["obs_mean"], out["obs_log_var"], y, noise_var)}
                if "step_mean" in out and tokens.z is not None:
                    z = torch.as_tensor(scaler.z(tokens.z[b][part])[None], device=config.device)
                    pieces["step_nll"] = (out["step_mean"], out["step_log_var"], z, None)
                for key, (mean, log_var, target, floor) in pieces.items():
                    value = gaussian_nll(mean, log_var, target, torch.ones_like(target), floor).item()
                    sums[key] = sums.get(key, 0.0) + value * target.numel()
                    counts[key] = counts.get(key, 0) + target.numel()
    return {key: sums[key] / counts[key] for key in sums}


def rollout(model, scaler, context_obs, context_controls, future_controls, max_context=None, with_steps=False):
    """Autoregressive forecast (unscaled means).

    context_obs (T0, n_obs) and context_controls (T0, n_ctrl): the known rows D_0..D_{T0-1}, u_0..u_{T0-1};
    future_controls (H, n_ctrl): u_{T0} .. u_{T0+H-1}. Returns the observables D_{T0} .. D_{T0+H} as (H+1, n_obs),
    each fed back as the next token's measurements; with ``with_steps`` also the step quantities of steps
    T0-1 .. T0+H-1 as (H+1, n_step) (horizon 0 is the decision step T0-1, as in ``pearlsim.ml.windows``).
    ``max_context`` (default: the model's max_len) truncates history.
    """
    import torch

    model.eval()
    limit = max_context or model.max_len
    obs = [np.asarray(o, np.float32) for o in context_obs]
    ctrl = [np.asarray(u, np.float32) for u in context_controls] + [np.asarray(u, np.float32) for u in future_controls]
    out_obs, out_step = [], []
    with torch.no_grad():
        for _ in range(len(future_controls) + 1):
            t = len(obs)
            tokens = np.concatenate([np.stack(obs), np.stack(ctrl[:t])], axis=1)[-limit:]
            out = model(torch.as_tensor(scaler.x(tokens)[None]))
            nxt = scaler.unscale_y(out["obs_mean"][0, -1].cpu().numpy(), tokens[-1])
            out_obs.append(nxt)
            if with_steps:
                out_step.append(scaler.unscale_z(out["step_mean"][0, -1].cpu().numpy()))
            obs.append(nxt.astype(np.float32))
    if with_steps:
        return np.stack(out_obs), np.stack(out_step)
    return np.stack(out_obs)
