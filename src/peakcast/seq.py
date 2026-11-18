"""Sequence models: an LSTM and a small Transformer encoder.

Both read the last 48 hours (192 blocks) of demand and calendar, and forecast
the next four blocks at once. 48 hours keeps the same block yesterday inside
the window for every horizon. The calendar of the four target blocks (their
tariff band and time of day) is added after the encoder, so a ramp at a band
change, like Light to Medium at 09:00, can be anticipated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import calendar

WINDOW, HORIZON = 192, 4
STEP_FEATS = ["blk_sin", "blk_cos", "band", "off_day", "weekend"]


def arrays(d: pd.DataFrame, mu: float, sd: float):
    """Per-step inputs (n, 6) and per-block target calendar (n, 4)."""
    cal = calendar(d.index, d["load_type"])
    z = ((d["kw"] - mu) / sd).to_numpy(np.float32)
    step = np.column_stack([z, cal[STEP_FEATS].to_numpy(np.float32)])
    step[:, 3] /= 2.0  # band 0..2 -> 0..1
    tgt = cal[["blk_sin", "blk_cos", "band", "off_day"]].to_numpy(np.float32)
    tgt[:, 2] /= 2.0
    return step, tgt, z


def samples(step, tgt, z, origins: np.ndarray):
    """origins: positions t (block ends). Input rows t-191..t; targets t+1..t+4."""
    idx = origins[:, None] + np.arange(-WINDOW + 1, 1)[None, :]
    x = step[idx]
    fut = origins[:, None] + np.arange(1, HORIZON + 1)[None, :]
    c = tgt[fut].reshape(len(origins), -1)
    y = z[fut]
    return torch.from_numpy(x), torch.from_numpy(c), torch.from_numpy(y)


class LSTMForecaster(nn.Module):
    def __init__(self, n_in=6, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 4 * HORIZON, 64), nn.ReLU(), nn.Linear(64, HORIZON))

    def forward(self, x, c):
        _, (h, _) = self.rnn(x)
        return self.head(torch.cat([h[-1], c], dim=1))


class TransformerForecaster(nn.Module):
    def __init__(self, n_in=6, d=32, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(2 * d + 4 * HORIZON, 64), nn.ReLU(), nn.Linear(64, HORIZON))

    def forward(self, x, c):
        e = self.enc(self.inp(x) + self.pos)
        summary = torch.cat([e[:, -1], e.mean(dim=1)], dim=1)  # latest step and whole window
        return self.head(torch.cat([summary, c], dim=1))


def fit(model, train, val, seed=0, epochs=30, batch=256, patience=4, lr=2e-3):
    """Adam on MSE; stops when validation MSE has not improved for `patience` epochs."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad, history = float("inf"), None, 0, []
    x, c, y = train
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(x), generator=g).split(batch):
            opt.zero_grad()
            loss = ((model(x[idx], c[idx]) - y[idx]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate_mse(model, val)
        history.append(v)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), history


@torch.no_grad()
def predict(model, data, batch=1024):
    model.eval()
    x, c, _ = data
    return torch.cat([model(x[i : i + batch], c[i : i + batch]) for i in range(0, len(x), batch)]).numpy()


def evaluate_mse(model, data):
    return float(((predict(model, data) - data[2].numpy()) ** 2).mean())
