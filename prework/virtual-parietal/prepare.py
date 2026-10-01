"""
Step 1: turn FACED BDF recordings into small per-subject .npz files.

For each subject that has a downloaded BDF:
  * keep only Fp1, Fp2, F3, F4, P3, P4
  * band-pass 1-45 Hz, 50 Hz notch, resample to 250 Hz
  * cut out every video-watching trial and keep its self-reported
    Arousal / Valence ratings (0-7 scale) from the BIDS events.tsv

Output: data/prepared/sub-XXX.npz  (~5-10 MB each)

Usage:
    python3 prepare.py                 # every subject with a full BDF
    python3 prepare.py --delete-bdf    # also delete each BDF once converted
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "P3", "P4"]
SFREQ = 250.0
ROOT = os.path.dirname(os.path.abspath(__file__))


def trials_from_events(tsv):
    ev = pd.read_csv(tsv, sep="\t", encoding="utf-8-sig", na_values="n/a")
    ev = ev[ev["video_index"].notna() & ev["Arousal"].notna() & (ev["duration"] > 5)]
    return ev[["onset", "duration", "video_index", "Arousal", "Valence", "emotion_label"]].reset_index(drop=True)


def convert(bdf, out_path):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR")
    missing = [c for c in CHANNELS if c not in raw.ch_names]
    if missing:
        raise RuntimeError(f"missing channels {missing}")
    raw.pick(CHANNELS)
    raw.load_data()
    raw.notch_filter(50.0, verbose="ERROR")
    raw.filter(1.0, 45.0, verbose="ERROR")
    raw.resample(SFREQ, verbose="ERROR")
    data = raw.get_data()
    # FACED mixes V and uV headers; after filtering (no DC offset) real EEG has a
    # std of ~1e-6..1e-4 V, so anything that small is in volts -> convert to uV
    if np.median(data.std(1)) < 1e-2:
        data = data * 1e6
    data = data.astype(np.float32)

    tsv = bdf.replace("_eeg.bdf", "_events.tsv")
    tr = trials_from_events(tsv)
    segs, keep = [], []
    for i, r in tr.iterrows():
        a = int(round(r.onset * SFREQ))
        b = int(round((r.onset + r.duration) * SFREQ))
        if a < 0 or b > data.shape[1] or b - a < 5 * SFREQ:
            continue
        segs.append(data[:, a:b])
        keep.append(i)
    if not segs:
        raise RuntimeError("no usable trials")
    tr = tr.loc[keep].reset_index(drop=True)
    lengths = np.array([s.shape[1] for s in segs])
    np.savez_compressed(
        out_path,
        eeg=np.concatenate(segs, axis=1),          # (6, total_samples) microvolts
        lengths=lengths,                            # samples per trial
        arousal=tr["Arousal"].to_numpy(np.float32),
        valence=tr["Valence"].to_numpy(np.float32),
        video=tr["video_index"].to_numpy(np.int16),
        emotion=tr["emotion_label"].astype(str).to_numpy(),
        channels=np.array(CHANNELS),
        sfreq=SFREQ,
    )
    return len(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-bdf", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "data", "prepared")
    os.makedirs(out_dir, exist_ok=True)
    bdfs = sorted(glob.glob(os.path.join(ROOT, "data", "bids", "sub-*", "eeg", "*_eeg.bdf")))
    bdfs = [b for b in bdfs if os.path.getsize(b) > 1_000_000]
    print(f"{len(bdfs)} downloaded BDF files")
    for bdf in bdfs:
        sub = os.path.basename(bdf).split("_")[0]
        out = os.path.join(out_dir, f"{sub}.npz")
        if os.path.exists(out):
            print(f"{sub}: already prepared")
        else:
            try:
                n = convert(bdf, out + ".tmp.npz")
                os.replace(out + ".tmp.npz", out)
                print(f"{sub}: {n} trials", flush=True)
            except Exception as e:  # keep going on bad files
                print(f"{sub}: FAILED ({e})", file=sys.stderr, flush=True)
                continue
        if args.delete_bdf:
            try:
                os.remove(bdf)
            except OSError as e:
                print(f"{sub}: could not delete BDF ({e})", file=sys.stderr)


if __name__ == "__main__":
    main()
