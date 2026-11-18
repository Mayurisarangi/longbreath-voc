import numpy as np, json, glob, os
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, roc_curve
rng = np.random.default_rng(1); TIMES = np.array([0, 6, 12, 18, 24.])
ORDER = ["PLS-DA", "SVM", "LMM", "FPCA", "LSTM", "DLSTM", "GRU-D", "TT", "HTAN-small", "HTAN"]
def load(name):
    z = np.load(f"results/{name}.npz"); info = json.load(open(f"results/{name}_info.json"))
    return z["y"], z["tnod"], {k.replace("_", "-"): z[k] for k in z.files if k not in ("y", "tnod")}, info
def metrics(y, p, tn, boot_idx=None):
    if boot_idx is not None: y, p, tn = y[boot_idx], p[boot_idx], tn[boot_idx]
    pf = p[:, 4]; out = {}
    out["AUC"] = roc_auc_score(y, pf, multi_class="ovr", average="macro")
    m = y != 2; s = pf[m, 1] / (pf[m, 1] + pf[m, 0] + 1e-12); out["AUC_PMPvsHC"] = roc_auc_score(y[m] == 1, s)
    pred = pf.argmax(1); cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    out["Sens"] = np.mean([cm[i, i] / cm[i].sum() for i in range(3)])
    spec = []
    for i in range(3):
        tn_ = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]; fp = cm[:, i].sum() - cm[i, i]; spec.append(tn_ / (tn_ + fp))
    out["Spec"] = np.mean(spec); out["F1"] = f1_score(y, pred, average="macro")
    # lead time: first prefix with P(PMP)>0.8
    pmp = y == 1; flag = np.full(len(y), np.nan)
    for L in range(2, 6):
        pl = p[:, L - 1, 1]; first = np.isnan(flag) & (pl > 0.8); flag[first] = TIMES[L - 1]
    fl = ~np.isnan(flag)
    out["pct_flagged_PMP"] = fl[pmp].mean() * 100
    lead = np.clip(tn - flag, 0, None); out["Lead"] = np.nanmean(lead[pmp & fl]) if (pmp & fl).any() else np.nan
    out["pct_flagged_HC"] = fl[y == 0].mean() * 100
    return out, flag
def with_ci(y, p, tn, B=1000):
    pt, flag = metrics(y, p, tn); keys = list(pt); bs = {k: [] for k in keys}
    cls = [np.where(y == c)[0] for c in range(3)]
    for b in range(B):
        idx = np.concatenate([rng.choice(c, len(c)) for c in cls])
        try:
            m, _ = metrics(y, p, tn, idx)
            for k in keys: bs[k].append(m[k])
        except Exception: pass
    return {k: (pt[k], np.nanpercentile(bs[k], 2.5), np.nanpercentile(bs[k], 97.5)) for k in keys}, flag
def run(name):
    y, tn, P, info = load(name); res = {}; flags = {}
    for k, p in P.items():
        if k in ("PLS-DA", "SVM"):
            pp = np.full_like(p, np.nan); pp[:, 4] = p[:, 4]; p = pp
            m, fl = with_ci_cs(y, p, tn)
        else: m, fl = with_ci(y, p, tn)
        m["params"] = info.get(k, {}).get("params"); res[k] = m; flags[k] = fl
    return res, flags
def with_ci_cs(y, p, tn, B=1000):
    pt, _ = metrics_cs(y, p); bs = {k: [] for k in pt}; cls = [np.where(y == c)[0] for c in range(3)]
    for b in range(B):
        idx = np.concatenate([rng.choice(c, len(c)) for c in cls]); m, _ = metrics_cs(y[idx], p[idx])
        for k in pt: bs[k].append(m[k])
    return {k: (pt[k], np.percentile(bs[k], 2.5), np.percentile(bs[k], 97.5)) for k in pt}, np.full(len(y), np.nan)
def metrics_cs(y, p):
    p2 = p.copy(); p3 = np.repeat(p2[:, 4:5], 5, 1); m, _ = metrics(y, p3, np.full(len(y), np.nan)); 
    for k in ("Lead", "pct_flagged_PMP", "pct_flagged_HC"): m[k] = np.nan
    return m, None
if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        r, _ = run(name); json.dump(r, open(f"results/{name}_metrics.json", "w"), default=float, indent=1)
        print(name)
        for k in ORDER + ["HTAN-noIIPE", "HTAN-noBiLSTM", "HTAN-noCross"]:
            if k in r: print(f"{k:14s} AUC {r[k]['AUC'][0]:.3f} [{r[k]['AUC'][1]:.3f},{r[k]['AUC'][2]:.3f}] PMPvsHC {r[k]['AUC_PMPvsHC'][0]:.3f} sens {r[k]['Sens'][0]:.3f} spec {r[k]['Spec'][0]:.3f} F1 {r[k]['F1'][0]:.3f} lead {r[k]['Lead'][0]:.1f} flagged {r[k]['pct_flagged_PMP'][0]:.0f}% HCfl {r[k]['pct_flagged_HC'][0]:.1f}%")
