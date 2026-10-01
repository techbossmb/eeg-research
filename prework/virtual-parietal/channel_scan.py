"""Channel scan, step 2 (needs data/allch from extract_all_channels.py).

A) Arousal high/low (per-subject median split) from every single channel and every
   electrode pair, ridge regression, 5-fold leave-subjects-out.
B) Emotion classification from F3/F4 only (and all channels for reference):
   positive/negative/neutral and the 9 FACED categories, logistic regression.
Writes results/channel_scan.json. Resumable (skips finished parts).
"""
import glob, itertools, json, os, warnings
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
OUT = "results/channel_scan.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

files = sorted(glob.glob("data/allch/sub-*.npz"))
F, A, V, E, B, G = [], [], [], [], [], []
for f in files:
    z = np.load(f, allow_pickle=True)
    x = z["feats"]  # trials x ch x 5
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)          # per-subject calibration
    F.append(x); A.append(z["arousal"]); V.append(z["valence"]); E.append(z["emotion"]); B.append(z["binary"])
    G.append(np.full(len(x), int(os.path.basename(f)[4:7])))
    chans = list(z["channels"])
F, A, V, E, B, G = map(np.concatenate, (F, A, V, E, B, G))
folds = list(GroupKFold(5).split(F, A, G))
def zs(a):
    out = np.empty(len(a))
    for s_ in np.unique(G):
        m = G == s_; out[m] = (a[m] - a[m].mean()) / (a[m].std() + 1e-6)
    return out


def arousal_acc(X, a=None):
    a = A if a is None else a
    y = zs(a); p = np.zeros(len(X))
    for tr, te in folds:
        p[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], y[tr]).predict(X[te])
    return float(np.mean([((a[G == s] > np.median(a[G == s])) == (p[G == s] > np.median(p[G == s]))).mean()
                          for s in np.unique(G)]) * 100)


def pair_X(i, j):
    return np.c_[F[:, i, :], F[:, j, :], F[:, j, :] - F[:, i, :]]


if "single" not in res:
    res["single"] = {c: round(arousal_acc(F[:, i, :]), 1) for i, c in enumerate(chans)}
    json.dump(res, open(OUT, "w"), indent=1); print("single done", flush=True)
if "pairs" not in res:
    res["pairs"] = {}
for i, j in itertools.combinations(range(len(chans)), 2):
    k = f"{chans[i]}-{chans[j]}"
    if k in res["pairs"]:
        continue
    res["pairs"][k] = round(arousal_acc(pair_X(i, j)), 1)
    if len(res["pairs"]) % 25 == 0:
        json.dump(res, open(OUT, "w"), indent=1)
json.dump(res, open(OUT, "w"), indent=1)
if "all_channels" not in res:
    res["all_channels"] = round(arousal_acc(F.reshape(len(F), -1)), 1)
    json.dump(res, open(OUT, "w"), indent=1)


N_PERM = 10
res.setdefault("null", {"best_single": [], "best_pair": [], "all_channels": []})
rng = np.random.default_rng(len(res["null"]["best_pair"]))
while len(res["null"]["best_pair"]) < N_PERM:
    a = A.copy()
    for s_ in np.unique(G):
        m = np.where(G == s_)[0]; a[m] = A[rng.permutation(m)]
    res["null"]["best_single"].append(max(arousal_acc(F[:, i, :], a) for i in range(len(chans))))
    res["null"]["best_pair"].append(max(arousal_acc(pair_X(i, j), a) for i, j in itertools.combinations(range(len(chans)), 2)))
    res["null"]["all_channels"].append(arousal_acc(F.reshape(len(F), -1), a))
    json.dump(res, open(OUT, "w"), indent=1); print("perm", len(res["null"]["best_pair"]), flush=True)


def classify(X, y):
    p = np.empty(len(y), dtype=object)
    for tr, te in folds:
        p[te] = LogisticRegression(max_iter=2000, C=0.1).fit(X[tr], y[tr]).predict(X[te])
    return round(float((p == y).mean() * 100), 1), round(float(balanced_accuracy_score(y, p.astype(str)) * 100), 1)


if "emotion" not in res:
    ci = [chans.index("F3"), chans.index("F4")]
    Xf = np.c_[F[:, ci[0], :], F[:, ci[1], :], F[:, ci[1], :] - F[:, ci[0], :]]
    Xa = F.reshape(len(F), -1)
    em = {}
    for name, y, chance in [("pos_neg_neutral", B, 100 / 3), ("nine_emotions", E, 100 / 9)]:
        em[name] = {"chance_%": round(chance, 1), "F3F4": classify(Xf, y), "all_30_channels": classify(Xa, y)}
    res["emotion"] = em
    json.dump(res, open(OUT, "w"), indent=1)

s = sorted(res["single"].items(), key=lambda kv: -kv[1])
p = sorted(res["pairs"].items(), key=lambda kv: -kv[1])
print("top single:", s[:5], "\nP3", res["single"].get("P3"), "P4", res["single"].get("P4"), "F3", res["single"].get("F3"), "F4", res["single"].get("F4"))
print("top pairs:", p[:5], "\nF3-F4", res["pairs"].get("F3-F4"), "P3-P4", res["pairs"].get("P3-P4"),
      "rank P3-P4:", [k for k, _ in p].index("P3-P4") + 1 if "P3-P4" in res["pairs"] else None, "of", len(p))
print("null (shuffled labels) best single / best pair / all-ch, mean & max:", {k: (round(np.mean(v),1), round(max(v),1)) for k, v in res["null"].items()})
print("all channels:", res["all_channels"]); print("emotion:", json.dumps(res["emotion"]))
