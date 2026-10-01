"""
Optional: reconstruct the raw P3/P4 *waveforms* (not just band power) from the
two frontal channels with a 1-D convolutional network - the literal
"4 channels from a 2-channel EEG".

Each subject's signals are z-scored per channel (per-user calibration). Input =
2 s of F3/F4 (or Fp1/Fp2), output = the same 2 s of P3/P4. Evaluated on held-out
subjects (one 80/20 subject split) with per-channel correlation, compared with
a linear FIR least-squares baseline.

Usage:  python3 train_waveform.py [--frontal F3F4|Fp1Fp2] [--epochs 30]
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SETS = {"F3F4": ["F3", "F4"], "Fp1Fp2": ["Fp1", "Fp2"]}


def load(frontal, win=500, max_subjects=None):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "prepared", "sub-*.npz")))[:max_subjects]
    Xs, Ys, G = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        ch = list(z["channels"])
        eeg = z["eeg"]
        eeg = (eeg - eeg.mean(1, keepdims=True)) / (eeg.std(1, keepdims=True) + 1e-6)
        eeg = np.clip(eeg, -8, 8)
        n = eeg.shape[1] // win
        seg = eeg[:, : n * win].reshape(eeg.shape[0], n, win).transpose(1, 0, 2)
        Xs.append(seg[:, [ch.index(c) for c in SETS[frontal]]])
        Ys.append(seg[:, [ch.index(c) for c in ("P3", "P4")]])
        G.append(np.full(n, int(os.path.basename(f)[4:7])))
    return np.concatenate(Xs).astype(np.float32), np.concatenate(Ys).astype(np.float32), np.concatenate(G)


def corr(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1) + 1e-9)


def fir_baseline(Xtr, Ytr, Xte, taps=25):
    """Linear least-squares FIR filter (each output = sum of lagged inputs)."""
    def design(X):
        pad = taps // 2
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad)))
        cols = [Xp[:, c, i:i + X.shape[2]] for c in range(X.shape[1]) for i in range(taps)]
        return np.stack(cols, -1).reshape(-1, len(cols))
    idx = np.random.default_rng(0).choice(len(Xtr), size=min(len(Xtr), 4000), replace=False)
    A = design(Xtr[idx]); B = Ytr[idx].transpose(0, 2, 1).reshape(-1, Ytr.shape[1])
    W = np.linalg.lstsq(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ B, rcond=None)[0]
    return (design(Xte) @ W).reshape(len(Xte), Xte.shape[2], -1).transpose(0, 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontal", default="F3F4", choices=list(SETS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    X, Y, G = load(args.frontal, max_subjects=args.max_subjects)
    subs = np.unique(G)
    rng = np.random.default_rng(0)
    test_subs = rng.choice(subs, size=max(1, len(subs) // 5), replace=False)
    te = np.isin(G, test_subs)
    val_subs = rng.choice(np.setdiff1d(subs, test_subs), size=max(1, len(subs) // 10), replace=False)
    va = np.isin(G, val_subs)
    tr = ~te & ~va
    print(f"{len(subs)} subjects: train {tr.sum()} / val {va.sum()} / test {te.sum()} windows")

    def block(ci, co, d):
        return nn.Sequential(nn.Conv1d(ci, co, 7, padding=3 * d, dilation=d), nn.BatchNorm1d(co), nn.GELU())
    net = nn.Sequential(block(2, 32, 1), block(32, 64, 2), block(64, 64, 4), block(64, 64, 8),
                        block(64, 32, 1), nn.Conv1d(32, 2, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt, Yt = torch.tensor(X[tr]), torch.tensor(Y[tr])
    Xv, Yv = torch.tensor(X[va]), torch.tensor(Y[va])
    best, state = np.inf, None
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            loss = nn.functional.mse_loss(net(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([nn.functional.mse_loss(net(Xv[i:i + 512]), Yv[i:i + 512]).item()
                                for i in range(0, len(Xv), 512)]))
        print(f"  epoch {ep + 1}: val MSE {vl:.4f}", flush=True)
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        P = np.concatenate([net(torch.tensor(X[te][i:i + 512])).numpy() for i in range(0, te.sum(), 512)])
    F = fir_baseline(X[tr], Y[tr], X[te])
    res = {"frontal": args.frontal, "n_subjects": int(len(subs)),
           "cnn_corr": dict(zip(["P3", "P4"], corr(P, Y[te]).mean(0).round(4).tolist())),
           "fir_corr": dict(zip(["P3", "P4"], corr(F, Y[te]).mean(0).round(4).tolist())),
           "copy_nearest_frontal_corr": dict(zip(["P3", "P4"], corr(X[te], Y[te]).mean(0).round(4).tolist()))}
    print(json.dumps(res, indent=2))
    out = os.path.join(ROOT, "results", f"waveform_{args.frontal}")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=2)
    torch.save(net.state_dict(), os.path.join(out, "cnn.pt"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i = int(np.argsort(corr(P, Y[te]).mean(1))[len(P) // 2])   # a median example, not cherry-picked
    t = np.arange(X.shape[2]) / 250
    fig, axs = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True, facecolor="#fcfcfb")
    for c, ax in enumerate(axs):
        ax.plot(t, Y[te][i, c], color="#52514e", lw=1.2, label="real")
        ax.plot(t, P[i, c], color="#eb6834", lw=1.2, label="CNN reconstruction")
        ax.set_ylabel(["P3", "P4"][c] + " (z)"); ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(frameon=False, loc="upper right", ncol=2)
    axs[0].set_title("Held-out subject, median-quality 2 s window", loc="left")
    axs[1].set_xlabel("seconds")
    fig.tight_layout(); fig.savefig(os.path.join(out, "example_reconstruction.png"), dpi=150)


if __name__ == "__main__":
    main()
