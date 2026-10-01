"""
Step 2: train and evaluate.

Stage A  "virtual parietal channels"
    Frontal features (F3/F4 or Fp1/Fp2: log band power in 5 bands + auxiliary
    descriptors) -> P3/P4 log band power (5 bands x 2 channels).
    Models: Ridge regression (baseline) and a neural network (MLP, scikit-learn).
    Long runs are checkpointed: re-running the same command resumes.

Stage B  "does the virtual parietal signal carry arousal?"
    Trial-level arousal rating predicted from different inputs:
      valence_rating        self-reported valence only (the literal "arousal from valence" test)
      valence_decoded       valence decoded from frontal EEG -> arousal
      frontal               frontal EEG features
      virtual_parietal      Stage-A predicted P3/P4 only
      frontal+virtual       both
      true_parietal         real P3/P4 (upper bound for what a 4-channel device would give)
      frontal+true_parietal real 4-channel montage (upper bound)

All evaluation is leave-subjects-out (5-fold GroupKFold): no subject is ever in
both train and test, so results measure generalisation to a new wearer.

Usage:
    python3 train.py                    # F3/F4 inputs
    python3 train.py --frontal Fp1Fp2
    python3 train.py --no-subject-norm  # skip per-subject z-scoring
"""
import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from features import build_dataset, zscore_within

ROOT = os.path.dirname(os.path.abspath(__file__))
ALPHAS = np.logspace(-2, 4, 13)


# --------------------------------------------------------------------------- NN
def train_mlp(Xtr, Ytr, gtr, hidden=(256, 128), seed=0):
    """Fully-connected neural net (scikit-learn MLP, Adam, early stopping)."""
    from sklearn.neural_network import MLPRegressor
    xs, ys = StandardScaler().fit(Xtr), StandardScaler().fit(Ytr)
    net = MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-2, batch_size=256,
                       learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                       validation_fraction=0.15, n_iter_no_change=10, random_state=seed)
    yt = ys.transform(Ytr)
    net.fit(xs.transform(Xtr), yt.ravel() if yt.shape[1] == 1 else yt)

    def predict(X):
        p = net.predict(xs.transform(X))
        return ys.inverse_transform(p.reshape(len(X), -1))
    return predict


def fit_predict(kind, Xtr, Ytr, gtr, Xte):
    if kind == "ridge":
        sc = StandardScaler().fit(Xtr)
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), Ytr)
        return m.predict(sc.transform(Xte))
    return train_mlp(Xtr, Ytr, gtr)(Xte)


# ------------------------------------------------------------------- metrics
def per_subject_r(y, yhat, groups):
    rs = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() > 3 and np.std(y[m]) > 0 and np.std(yhat[m]) > 0:
            rs.append(pearsonr(y[m], yhat[m])[0])
    return float(np.mean(rs)), float(np.std(rs) / np.sqrt(len(rs)))


def median_split_auc(y, yhat, groups):
    """High vs low arousal, split at each subject's own median."""
    lab = np.zeros(len(y), int)
    for g in np.unique(groups):
        m = groups == g
        lab[m] = (y[m] > np.median(y[m])).astype(int)
    return float(roc_auc_score(lab, yhat))


def trial_mean(arr, trial):
    ut, inv = np.unique(trial, return_inverse=True)
    out = np.zeros((len(ut), arr.shape[1]), np.float64)
    np.add.at(out, inv, arr)
    return out / np.bincount(inv)[:, None], ut, inv


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontal", default="F3F4", choices=["F3F4", "Fp1Fp2", "both"])
    ap.add_argument("--no-subject-norm", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", default="ridge,mlp")
    args = ap.parse_args()
    models = args.models.split(",")
    tag = args.frontal + ("_raw" if args.no_subject_norm else "")
    out_dir = os.path.join(ROOT, "results", tag)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    print(f"Loading features ({args.frontal})...", flush=True)
    D = build_dataset(os.path.join(ROOT, "data", "prepared"), args.frontal)
    X, Y, groups, trial = D["X"], D["Y"], D["groups"], D["trial"]
    ynames = [str(n) for n in D["ynames"]]
    if not args.no_subject_norm:
        X, Y = zscore_within(X, groups), zscore_within(Y, groups)
    n_sub = len(np.unique(groups))
    print(f"{n_sub} subjects, {len(X)} windows, {X.shape[1]} frontal features -> {Y.shape[1]} parietal targets")
    folds = list(GroupKFold(n_splits=min(args.folds, n_sub)).split(X, Y, groups))

    # ---------------- Stage A: frontal -> parietal (window level) --------------
    res = {"config": vars(args), "n_subjects": n_sub, "n_windows": int(len(X)), "stageA": {}, "stageB": {}}
    Yhat = {m: np.zeros_like(Y) for m in models}
    ck_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    for k, (tr, te) in enumerate(folds):
        for m in models:
            ck = os.path.join(ck_dir, f"stageA_{m}_fold{k}.npy")
            if os.path.exists(ck):
                Yhat[m][te] = np.load(ck)
                continue
            Yhat[m][te] = fit_predict(m, X[tr], Y[tr], groups[tr], X[te])
            np.save(ck, Yhat[m][te])
            print(f"  Stage A {m} fold {k + 1}/{len(folds)} done ({time.time() - t0:.0f}s)", flush=True)
    for m in models:
        r2 = {n: float(r2_score(Y[:, j], Yhat[m][:, j])) for j, n in enumerate(ynames)}
        rsub = {n: per_subject_r(Y[:, j], Yhat[m][:, j], groups)[0] for j, n in enumerate(ynames)}
        res["stageA"][m] = {"r2": r2, "mean_within_subject_r": rsub,
                            "mean_r2": float(np.mean(list(r2.values())))}
        print(f"  [{m}] mean R^2 over P3/P4 bands = {res['stageA'][m]['mean_r2']:.3f}")
    best_A = max(models, key=lambda m: res["stageA"][m]["mean_r2"])
    res["stageA"]["best_model"] = best_A

    # ---------------- Stage B: arousal (trial level) ---------------------------
    Xt_, ut, _ = trial_mean(X, trial)
    Yt_, _, _ = trial_mean(Y, trial)
    Vt_, _, _ = trial_mean(Yhat[best_A], trial)      # out-of-fold virtual parietal
    first = np.array([np.where(trial == t)[0][0] for t in ut])
    g_t, aro, val = groups[first], D["arousal"][first], D["valence"][first]
    aro_z = zscore_within(aro[:, None], g_t)[:, 0]

    # decoded valence: frontal EEG -> valence rating, out-of-fold
    val_dec = np.zeros(len(ut))
    tfolds = list(GroupKFold(n_splits=len(folds)).split(Xt_, aro, g_t))
    for tr, te in tfolds:
        val_dec[te] = fit_predict("ridge", Xt_[tr], val[tr][:, None], g_t[tr], Xt_[te]).ravel()
    r_vdec = per_subject_r(val, val_dec, g_t)
    res["stageB"]["valence_decoding_within_subject_r"] = r_vdec[0]
    print(f"  valence decoded from frontal EEG: within-subject r = {r_vdec[0]:.3f}")

    vfeat = lambda v: np.c_[v, (v - 3.5) ** 2]      # arousal is U-shaped in valence
    sets = {
        "valence_rating": vfeat(val),
        "valence_decoded": vfeat(val_dec),
        "frontal": Xt_,
        "virtual_parietal": Vt_,
        "frontal+virtual": np.c_[Xt_, Vt_],
        "true_parietal": Yt_,
        "frontal+true_parietal": np.c_[Xt_, Yt_],
    }
    for name, F in sets.items():
        for m in models:
            if m == "mlp" and F.shape[1] <= 2:
                continue
            ck = os.path.join(ck_dir, f"stageB_{name}_{m}.npy")
            if os.path.exists(ck):
                pred = np.load(ck)
            else:
                pred = np.zeros(len(ut))
                for tr, te in tfolds:
                    pred[te] = fit_predict(m, F[tr], aro_z[tr][:, None], g_t[tr], F[te]).ravel()
                np.save(ck, pred)
            r, se = per_subject_r(aro, pred, g_t)
            auc = median_split_auc(aro, pred, g_t)
            res["stageB"][f"{name}|{m}"] = {"within_subject_r": r, "se": se, "auc": auc}
            print(f"  arousal <- {name:22s} [{m:5s}] r={r:+.3f}±{se:.3f}  AUC={auc:.3f}", flush=True)

    res["elapsed_s"] = time.time() - t0
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    make_plots(res, ynames, models, out_dir)
    print(f"Saved results to {out_dir} ({res['elapsed_s']:.0f}s)")


def make_plots(res, ynames, models, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"ridge": "#2a78d6", "mlp": "#eb6834"}
    label = {"ridge": "Ridge (linear)", "mlp": "Neural net (MLP)"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})

    # Stage A
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yy = np.arange(len(ynames))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        v = [res["stageA"][m]["r2"][n] for n in ynames]
        ax.barh(yy + (i - (len(models) - 1) / 2) * h, v, height=h * 0.9, color=colors[m], label=label[m])
    ax.set_yticks(yy, [n.replace(":logpow_", " ") for n in ynames])
    ax.invert_yaxis()
    ax.axvline(0, color=ink2, lw=0.8)
    ax.grid(axis="x", color=grid, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("R² on held-out subjects")
    ax.set_title("Stage A: predicting P3/P4 band power from frontal channels", color=ink, loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "stageA_virtual_parietal.png"), dpi=150); plt.close(fig)

    # Stage B
    keys = [k for k in res["stageB"] if "|" in k]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(keys) + 1.2))
    vals = [res["stageB"][k]["within_subject_r"] for k in keys]
    ses = [res["stageB"][k]["se"] for k in keys]
    cols = [colors[k.split("|")[1]] for k in keys]
    yy = np.arange(len(keys))
    ax.barh(yy, vals, xerr=ses, color=cols, height=0.7, error_kw={"ecolor": ink2, "lw": 0.8})
    ax.set_yticks(yy, [k.split("|")[0] for k in keys])
    ax.invert_yaxis()
    ax.axvline(0, color=ink2, lw=0.8)
    ax.grid(axis="x", color=grid, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Within-subject correlation with arousal rating (held-out subjects, ±SE)")
    ax.set_title("Stage B: predicting arousal", color=ink, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[m], label=label[m]) for m in models], frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "stageB_arousal.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
