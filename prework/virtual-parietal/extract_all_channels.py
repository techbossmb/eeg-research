"""Channel scan, step 1: per-video log band power for all 30 scalp channels.
Output: data/allch/sub-XXX.npz  (28 trials x 30 channels x 5 bands) + ratings/labels. Resumable."""
import glob, os, numpy as np, pandas as pd, mne
from scipy.signal import welch
mne.set_log_level("ERROR")
BANDS = [(1,4),(4,8),(8,13),(13,30),(30,45)]
out_dir = "data/allch"; os.makedirs(out_dir, exist_ok=True)
for bdf in sorted(glob.glob("data/bids/sub-*/eeg/*_eeg.bdf")):
    if os.path.getsize(bdf) < 1_000_000: continue
    sub = os.path.basename(bdf).split("_")[0]; out = f"{out_dir}/{sub}.npz"
    if os.path.exists(out): continue
    raw0 = mne.io.read_raw_bdf(bdf, preload=False)
    chans = [c for c in raw0.ch_names if c not in ("A1", "A2")][:30]
    parts = []
    for k in range(0, len(chans), 6):          # small channel groups keep memory low (3 GB VM)
        r = raw0.copy().pick(chans[k:k + 6]).load_data(); r.resample(250.); r.filter(1., 45.)
        parts.append(r.get_data().astype(np.float32)); del r
    x = np.concatenate(parts); sf = 250.
    ev = pd.read_csv(bdf.replace("_eeg.bdf", "_events.tsv"), sep="\t", encoding="utf-8-sig", na_values="n/a")
    ev = ev[ev.video_index.notna() & ev.Arousal.notna() & (ev.duration > 5)]
    F, keep = [], []
    for i, r in ev.iterrows():
        a, b = int(r.onset * sf), int((r.onset + r.duration) * sf)
        if a < 0 or b > x.shape[1]: continue
        f, p = welch(x[:, a:b], fs=sf, nperseg=int(4 * sf), axis=-1)
        F.append(np.stack([np.log10(p[:, (f >= lo) & (f < hi)].sum(-1) + 1e-30) for lo, hi in BANDS], 1)); keep.append(i)
    ev = ev.loc[keep]
    np.savez(out + ".tmp.npz", feats=np.array(F, np.float32), channels=np.array(chans),
             arousal=ev.Arousal.values, valence=ev.Valence.values, emotion=ev.emotion_label.astype(str).values,
             binary=ev.binary_label.astype(str).values, video=ev.video_index.values)
    os.replace(out + ".tmp.npz", out); print(sub, len(F), "trials", len(chans), "ch", flush=True)
