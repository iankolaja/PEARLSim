"""PyTorch sequence models: an encoder over the past window and a direct multi-horizon head.

    past (B, W, F) --encoder--> summary (B, D)
    for h = 0..H: [summary, planned controls u_{i+1..i+h} (later rows zeroed), one-hot(h)] --MLP--> (T,)

Encoders: ``lstm`` and ``gru`` (stacked, sizes per layer, last hidden state), ``tcn``
(causal dilated convolutions with residuals, last time step) and ``transformer``
(causal self-attention with learned positions, last token).

``lstm`` with ``keras_init=True`` mimics the initialisation of the paper's Keras
LSTM (Glorot-uniform input weights, orthogonal recurrent weights, forget-gate bias 1).
Keras's recurrent dropout has no cuDNN counterpart in PyTorch; ``dropout`` here
acts between layers and on the encoder output.
"""

import torch
from torch import nn


class RecurrentEncoder(nn.Module):
    def __init__(self, n_features, sizes=(64,), cell="lstm", dropout=0.0, keras_init=False):
        super().__init__()
        layer = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.layers = nn.ModuleList()
        width = n_features
        for size in sizes:
            self.layers.append(layer(width, size, batch_first=True))
            width = size
        self.dropout = nn.Dropout(dropout)
        self.out_features = width
        if keras_init:
            for rnn in self.layers:
                for name, p in rnn.named_parameters():
                    if "weight_ih" in name:
                        for gate in p.data.chunk(4 if cell == "lstm" else 3):
                            nn.init.xavier_uniform_(gate)
                    elif "weight_hh" in name:
                        for gate in p.data.chunk(4 if cell == "lstm" else 3):
                            nn.init.orthogonal_(gate)
                    elif "bias" in name:
                        nn.init.zeros_(p)
                        if cell == "lstm" and "bias_ih" in name:
                            p.data[rnn.hidden_size: 2 * rnn.hidden_size] = 1.0  # forget gate (i, f, g, o)

    def forward(self, x):
        for j, rnn in enumerate(self.layers):
            x, _ = rnn(x)
            if j < len(self.layers) - 1:
                x = self.dropout(x)
        return self.dropout(x[:, -1])


class TCNEncoder(nn.Module):
    def __init__(self, n_features, channels=64, levels=3, kernel_size=3, dropout=0.0):
        super().__init__()
        self.inp = nn.Conv1d(n_features, channels, 1)
        self.blocks = nn.ModuleList()
        self.pads = []
        for level in range(levels):
            dilation = 2 ** level
            self.pads.append((kernel_size - 1) * dilation)
            self.blocks.append(nn.Sequential(nn.Conv1d(channels, channels, kernel_size, dilation=dilation),
                                             nn.ReLU(), nn.Dropout(dropout)))
        self.out_features = channels

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))  # (B, C, W)
        for pad, block in zip(self.pads, self.blocks):
            h = h + block(nn.functional.pad(h, (pad, 0)))  # left padding keeps it causal
        return h[:, :, -1]


class TransformerEncoder(nn.Module):
    def __init__(self, n_features, d_model=64, heads=4, layers=2, window=8, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.out_features = d_model

    def forward(self, x):
        w = x.shape[1]
        causal = torch.triu(torch.full((w, w), float("-inf"), device=x.device), diagonal=1)
        h = self.encoder(self.embed(x) + self.position[:, -w:], mask=causal)
        return h[:, -1]


class SequenceRegressor(nn.Module):
    def __init__(self, n_features, n_controls, n_targets, horizon=0, encoder="lstm", head_hidden=(),
                 window=8, **encoder_params):
        super().__init__()
        if encoder in ("lstm", "gru"):
            self.encoder = RecurrentEncoder(n_features, cell=encoder, **encoder_params)
        elif encoder == "tcn":
            self.encoder = TCNEncoder(n_features, **encoder_params)
        elif encoder == "transformer":
            self.encoder = TransformerEncoder(n_features, window=window, **encoder_params)
        else:
            raise ValueError(f"unknown encoder {encoder!r}")
        self.horizon, self.n_targets, self.n_controls = horizon, n_targets, n_controls
        width = self.encoder.out_features + horizon * n_controls + (horizon + 1 if horizon else 0)
        layers = []
        for size in head_hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, n_targets))
        self.head = nn.Sequential(*layers)

    def forward(self, past, future_controls):
        summary = self.encoder(past)
        if self.horizon == 0:
            return self.head(summary).unsqueeze(1)
        outputs = []
        batch = past.shape[0]
        for h in range(self.horizon + 1):
            controls = future_controls.clone()
            controls[:, h:] = 0.0  # only the controls of steps i+1 .. i+h affect step i+h
            one_hot = torch.zeros(batch, self.horizon + 1, device=past.device)
            one_hot[:, h] = 1.0
            outputs.append(self.head(torch.cat([summary, controls.reshape(batch, -1), one_hot], dim=1)))
        return torch.stack(outputs, dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def paper_lstm(n_features, n_controls=0, sizes=(256, 128), dropout=0.1):
    """The paper's per-target network: stacked LSTM layers and one linear output (horizon 0, one target)."""
    return SequenceRegressor(n_features, n_controls, 1, horizon=0, encoder="lstm", sizes=tuple(sizes),
                             dropout=dropout, keras_init=True)
