"""RQ1 ablation: does adding auxiliary features help predict P3/P4 band power?
  A) band log-powers only (10 per pair)
  B) A + auxiliary descriptors (relative power, Hjorth, entropy, log variance) = the main model (28)
  C) B + explicit ratios/asymmetries (beta/alpha, theta/beta, theta/alpha per channel; right-left per band)
Usage: python3 ablation_aux.py [ridge|mlp] [F3F4|Fp1Fp2]"""
import sys, json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from features import build_dataset, zscore_within
from train import fit_predict
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
model = sys.argv[1] if len(sys.argv) > 1 else "ridge"; fr = sys.argv[2] if len(sys.argv) > 2 else "F3F4"
L, R = ("F3", "F4") if fr == "F3F4" else ("Fp1", "Fp2")
D = build_dataset("data/prepared", fr); g = D["groups"]; xn = list(D["xnames"]); X0 = D["X"]
col = lambda n: X0[:, xn.index(n)]
bands = ["delta", "theta", "alpha", "beta", "gamma"]
A = np.stack([col(f"{c}:logpow_{b}") for c in (L, R) for b in bands], 1)
extra = [col(f"{c}:logpow_{x}") - col(f"{c}:logpow_{y}") for c in (L, R) for x, y in [("beta","alpha"),("theta","beta"),("theta","alpha")]]
extra += [col(f"{R}:logpow_{b}") - col(f"{L}:logpow_{b}") for b in bands]
sets = {"A_bandpower": A, "B_plus_aux(main)": X0, "C_plus_ratios": np.c_[X0, np.stack(extra, 1)]}
Y = zscore_within(D["Y"], g); folds = list(GroupKFold(5).split(Y, Y, g))
out_path = f"results/ablation_aux_{fr}.json"
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
for name, Xs in sets.items():
    key = f"{name}|{model}"
    if key in res: continue
    X = zscore_within(Xs, g); P = np.zeros_like(Y)
    for tr, te in folds: P[te] = fit_predict(model, X[tr], Y[tr], g[tr], X[te])
    r = np.mean([np.mean([pearsonr(Y[g==s, j], P[g==s, j])[0] for s in np.unique(g)]) for j in range(Y.shape[1])])
    r2 = np.mean([r2_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])])
    res[key] = {"n_features": int(X.shape[1]), "within_subject_r": round(float(r), 3), "r2": round(float(r2), 3)}
    json.dump(res, open(out_path, "w"), indent=1); print(key, res[key], flush=True)
print(json.dumps(res))
