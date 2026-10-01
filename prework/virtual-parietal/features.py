"""Window-level spectral features shared by the training scripts."""
import glob
import os

import numpy as np
from scipy.signal import welch

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
FRONTAL_SETS = {"F3F4": ["F3", "F4"], "Fp1Fp2": ["Fp1", "Fp2"], "both": ["Fp1", "Fp2", "F3", "F4"]}
PARIETAL = ["P3", "P4"]


def channel_features(win, sfreq):
    """win: (n_ch, n_samples) -> dict of (n_ch,) arrays.

    Absolute log band power for all 5 bands, plus 'auxiliary' descriptors
    (relative power, Hjorth mobility/complexity, spectral entropy, log variance).
    """
    f, pxx = welch(win, fs=sfreq, nperseg=int(sfreq), axis=-1)
    total_mask = (f >= 1) & (f < 45)
    total = pxx[:, total_mask].sum(-1) + 1e-12
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp = pxx[:, m].sum(-1) + 1e-12
        out[f"logpow_{name}"] = np.log10(bp)
        out[f"relpow_{name}"] = bp / total
    d1 = np.diff(win, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = win.var(-1) + 1e-12, d1.var(-1) + 1e-12, d2.var(-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    out["hjorth_mobility"] = mob
    out["hjorth_complexity"] = np.sqrt(v2 / v1) / mob
    p = pxx[:, total_mask] / total[:, None]
    out["spectral_entropy"] = -(p * np.log(p + 1e-12)).sum(-1) / np.log(p.shape[1])
    out["log_var"] = np.log10(v0)
    return out


def load_subject(path, win_s=4.0, step_s=2.0):
    z = np.load(path, allow_pickle=True)
    eeg, lengths, sfreq = z["eeg"], z["lengths"], float(z["sfreq"])
    chans = list(z["channels"])
    w, s = int(win_s * sfreq), int(step_s * sfreq)
    rows, trial_idx = [], []
    start = 0
    for t, L in enumerate(lengths):
        seg = eeg[:, start:start + L]
        start += L
        for a in range(0, L - w + 1, s):
            rows.append(channel_features(seg[:, a:a + w], sfreq))
            trial_idx.append(t)
    keys = list(rows[0].keys())
    feats = {k: np.stack([r[k] for r in rows]) for k in keys}  # each (n_win, n_ch)
    return dict(feats=feats, keys=keys, chans=chans, trial=np.array(trial_idx),
                arousal=z["arousal"], valence=z["valence"], video=z["video"])


def load_subject_cached(path):
    """Per-subject feature cache so long extractions can be resumed in chunks."""
    cdir = os.path.join(os.path.dirname(path), "_featcache")
    os.makedirs(cdir, exist_ok=True)
    cp = os.path.join(cdir, os.path.basename(path))
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        keys = list(z["keys"])
        return dict(feats={k: z["f_" + k] for k in keys}, keys=keys, chans=list(z["chans"]),
                    trial=z["trial"], arousal=z["arousal"], valence=z["valence"], video=z["video"])
    d = load_subject(path)
    np.savez(cp + ".tmp.npz", keys=np.array(d["keys"]), chans=np.array(d["chans"]), trial=d["trial"],
             arousal=d["arousal"], valence=d["valence"], video=d["video"],
             **{"f_" + k: v for k, v in d["feats"].items()})
    os.replace(cp + ".tmp.npz", cp)
    return d


def build_dataset(prepared_dir, frontal="F3F4", cache=True):
    """Returns dict with window-level X (frontal), Y (parietal log powers), groups, trial ids, ratings."""
    cache_path = os.path.join(prepared_dir, f"_features_{frontal}.npz")
    if cache and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    files = sorted(glob.glob(os.path.join(prepared_dir, "sub-[0-9][0-9][0-9].npz")))
    if not files:
        raise SystemExit(f"No prepared subjects in {prepared_dir}; run prepare.py first")
    Xs, Ys, G, T, A, V, VID = [], [], [], [], [], [], []
    xnames = ynames = None
    for i, fpath in enumerate(files):
        sub = int(os.path.basename(fpath)[4:7])
        d = load_subject_cached(fpath)
        ci = {c: j for j, c in enumerate(d["chans"])}
        fch = [ci[c] for c in FRONTAL_SETS[frontal]]
        pch = [ci[c] for c in PARIETAL]
        X = np.concatenate([d["feats"][k][:, fch] for k in d["keys"]], axis=1)
        bandkeys = [f"logpow_{b}" for b in BANDS]
        Y = np.concatenate([d["feats"][k][:, pch] for k in bandkeys], axis=1)
        if xnames is None:
            xnames = [f"{c}:{k}" for k in d["keys"] for c in FRONTAL_SETS[frontal]]
            ynames = [f"{c}:{k}" for k in bandkeys for c in PARIETAL]
        Xs.append(X); Ys.append(Y)
        G.append(np.full(len(X), sub)); T.append(d["trial"] + sub * 1000)
        A.append(d["arousal"][d["trial"]]); V.append(d["valence"][d["trial"]]); VID.append(d["video"][d["trial"]])
        print(f"  features {os.path.basename(fpath)}: {len(X)} windows", flush=True)
    out = dict(X=np.concatenate(Xs).astype(np.float32), Y=np.concatenate(Ys).astype(np.float32),
               groups=np.concatenate(G), trial=np.concatenate(T), arousal=np.concatenate(A),
               valence=np.concatenate(V), video=np.concatenate(VID),
               xnames=np.array(xnames), ynames=np.array(ynames))
    if cache:
        np.savez(cache_path, **out)
    return out


def zscore_within(arr, groups):
    """Z-score columns separately inside each subject (simulates per-user calibration)."""
    out = np.empty_like(arr, dtype=np.float32)
    for g in np.unique(groups):
        m = groups == g
        mu = arr[m].mean(0)
        sd = arr[m].std(0) + 1e-6
        out[m] = (arr[m] - mu) / sd
    return out
