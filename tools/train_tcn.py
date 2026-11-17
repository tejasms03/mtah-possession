"""Train the network sequence model (src/possession/tcn.py) in PyTorch, offline, on outside
matches only, and save numpy weights. Not part of the pipeline (needs torch, which the project's
uv env does not have).

Input: per-match datasets <ds_dir>/<match>_ds.npz with, for each half k, X<k> (N, 12) features
from tcn.features() on the out-of-fold first-stage outputs, y<k> labels (0 H, 1 A, 2 D, -1 none),
t<k> turnover target (0 none, 1 H->A, 2 A->H within +-5 frames, -1 none).

Usage: python tools/train_tcn.py DS_DIR OUT.npz TRAIN_IDS (comma separated) [STEPS] [SEED]
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, CENTER, BATCH, C = 512, 2048, 16, 64
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SWAP = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10, 11]
SIGN = torch.tensor([-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)


class TCN(nn.Module):
    def __init__(self, n_in=12, c=C, dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_in, c, 1)
        self.convs = nn.ModuleList([nn.Conv1d(c, c, 3, dilation=d, padding=d) for d in DILATIONS])
        self.outs = nn.ModuleList([nn.Conv1d(c, c, 1) for _ in DILATIONS])
        self.drop = nn.Dropout(dropout)
        self.em = nn.Conv1d(c, 3, 1)
        self.turn = nn.Conv1d(c, 3, 1)

    def forward(self, x):                      # x (B, n_in, T)
        h = self.inp(x)
        for conv, out in zip(self.convs, self.outs):
            h = h + out(self.drop(F.relu(conv(F.relu(h)))))
        a = F.relu(h)
        return self.em(a), self.turn(a)


def load(ds_dir, ids):
    halves = []
    for m in ids:
        z = np.load(f"{ds_dir}/{m}_ds.npz")
        for k in range(2):
            X = np.zeros((len(z[f"X{k}"]) + 2 * PAD, z[f"X{k}"].shape[1]), np.float32)
            X[PAD:-PAD] = z[f"X{k}"]
            halves.append((X, z[f"y{k}"].astype(np.int64), z[f"t{k}"].astype(np.int64)))
    return halves


def batch(halves, rng):
    xs, ys, ts = [], [], []
    w = np.array([len(h[1]) for h in halves], float)
    for _ in range(BATCH):
        X, y, t = halves[rng.choice(len(halves), p=w / w.sum())]
        s = rng.integers(0, len(y) - CENTER)
        x = X[s : s + CENTER + 2 * PAD]           # padded index s .. covers frames s-PAD .. s+CENTER+PAD
        yy, tt = y[s : s + CENTER].copy(), t[s : s + CENTER].copy()
        if rng.random() < 0.5:                    # H <-> A swap
            x = x[:, SWAP] * SIGN.numpy()
            yy = np.where(yy == 0, 1, np.where(yy == 1, 0, yy))
            tt = np.where(tt == 1, 2, np.where(tt == 2, 1, tt))
        xs.append(x.T); ys.append(yy); ts.append(tt)
    return (torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys)), torch.tensor(np.stack(ts)))


def export(net, path):
    sd = {k: v.detach().cpu().double().numpy() for k, v in net.state_dict().items()}
    W = {"in_w": sd["inp.weight"][:, :, 0], "in_b": sd["inp.bias"],
         "em_w": sd["em.weight"][:, :, 0], "em_b": sd["em.bias"],
         "to_w": sd["turn.weight"][:, :, 0], "to_b": sd["turn.bias"], "dilations": np.array(DILATIONS)}
    for k in range(len(DILATIONS)):
        W[f"c{k}_w"], W[f"c{k}_b"] = sd[f"convs.{k}.weight"], sd[f"convs.{k}.bias"]
        W[f"o{k}_w"], W[f"o{k}_b"] = sd[f"outs.{k}.weight"][:, :, 0], sd[f"outs.{k}.bias"]
    np.savez(path, **{k: (v.astype(np.float32) if k != "dilations" else v) for k, v in W.items()})


def main():
    ds_dir, out, ids = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    halves = load(ds_dir, ids)
    net = TCN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    t0, run = time.time(), []
    for step in range(steps):
        x, y, t = (v.to(dev) for v in batch(halves, rng))
        em, to = net(x)
        em, to = em[:, :, PAD : PAD + CENTER], to[:, :, PAD : PAD + CENTER]
        loss = F.cross_entropy(em, y, ignore_index=-1) + F.cross_entropy(to, t, ignore_index=-1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"step {step + 1} loss {np.mean(run[-100:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval()
    export(net.cpu(), out)


if __name__ == "__main__":
    main()
