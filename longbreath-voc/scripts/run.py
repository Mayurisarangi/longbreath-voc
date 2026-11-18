import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)  # outputs go to <repo>/results and <repo>/figs
import argparse, json, time, numpy as np, torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import sim, prep, models
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="ideal"); ap.add_argument("--effect", type=float, default=1.0)
ap.add_argument("--models", default="PLS-DA,SVM,LMM,FPCA,LSTM,DLSTM,GRU-D,TT,HTAN-small,HTAN")
ap.add_argument("--impute", default="ffill"); ap.add_argument("--global_conf", action="store_true")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--out", required=True)
a = ap.parse_args()
names = a.models.split(","); d = sim.simulate(a.scenario, a.seed, a.effect)
V, M, y, smoke, pack = d["V"], d["M"], d["y"], d["smoke"], d["pack"]; N = len(y)
P = {n: np.full((N, 5, 3), np.nan) for n in names}; info = {n: {} for n in names}
skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
cf_global = prep.fit_confounder(V, smoke, pack, np.arange(N)) if a.global_conf else None
for f, (tra, te) in enumerate(skf.split(np.zeros(N), y)):
    t0 = time.time()
    tr, va = train_test_split(tra, test_size=0.15, stratify=y[tra], random_state=a.seed)
    cf = cf_global if a.global_conf else prep.fit_confounder(V, smoke, pack, tra)
    Vc = prep.apply_confounder(V, cf)
    need_lmm = ("LMM" in names) or a.impute == "lmm"
    lmm = prep.fit_lmm(Vc, tra) if need_lmm else None
    # ---- cross-sectional: last observed visit
    last = np.array([[Vc[i, np.where(M[i])[0][-1], j] for j in range(10)] for i in range(N)])
    for n in names:
        if n == "PLS-DA":
            sc = StandardScaler().fit(last[tra]); pl = PLSRegression(5).fit(sc.transform(last[tra]), np.eye(3)[y[tra]])
            s = np.clip(pl.predict(sc.transform(last[te])), 1e-3, None); P[n][te, 4] = s / s.sum(1, keepdims=True)
        if n == "SVM":
            Xs = StandardScaler().fit(last[tra]); Z = Xs.transform(last); rel = mutual_info_classif(Z[tra], y[tra], random_state=0)
            C = np.abs(np.corrcoef(Z[tra].T)); sel = [int(np.argmax(rel))]
            while len(sel) < 6:
                sc_ = [(rel[j] - C[j, sel].mean()) if j not in sel else -9 for j in range(10)]; sel.append(int(np.argmax(sc_)))
            sv = SVC(kernel="rbf", probability=True, random_state=0).fit(Z[tra][:, sel], y[tra]); P[n][te, 4] = sv.predict_proba(Z[te][:, sel])
        if n == "LMM":      # two-stage statistical baseline: LMM random intercept+slope -> multinomial logistic regression
            for L in range(1, 6):
                Mu = M.copy(); Mu[:, L:] = False; Fz = prep.lmm_features(Vc, Mu, lmm)
                sc = StandardScaler().fit(Fz[tra]); lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Fz[tra]), y[tra])
                P[n][te, L - 1] = lr.predict_proba(sc.transform(Fz[te]))
        if n == "FPCA":
            Vi = prep.impute(Vc, M, "linear")
            for L in range(2, 6):
                Vp = Vi.copy(); Vl = Vc.copy(); Vl[:, L:] = np.nan; Vp = prep.impute(Vl, M, "linear")[:, :L]   # prefix curves only
                Fz = []
                for j in range(10):
                    pc = PCA(min(2, L)).fit(Vp[tra, :, j]); s_ = pc.transform(Vp[:, :, j]); Fz.append(np.pad(s_, ((0, 0), (0, 2 - s_.shape[1]))))
                Fz = np.concatenate(Fz, 1); sc = StandardScaler().fit(Fz[tra]); lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Fz[tra]), y[tra])
                P[n][te, L - 1] = lr.predict_proba(sc.transform(Fz[te]))
    # ---- sequence models
    deep = [n for n in names if n not in ("PLS-DA", "SVM", "LMM", "FPCA")]
    if deep:
        mode = a.impute
        B = {L: prep.build(Vc, M, L, mode, lmm) for L in range(1, 6)}
        def sub(b, idx):
            o = {}
            for k, v in b.items(): o[k] = v if k == "use_obs" else v[idx]
            return o
        fin = B[5]["x"].shape[2]
        for n in deep:
            trb = [sub(B[L], tr) for L in range(2, 6)]; vab = sub(B[5], va); teb = [sub(B[L], te) for L in range(1, 6)]
            outs, m, npar, ep = models.train_predict(n, fin, trb, y[tr], vab, y[va], teb, seed=a.seed + f, epochs=a.epochs)
            for L in range(5): P[n][te, L] = outs[L]
            info[n]["params"] = npar; info[n].setdefault("epochs", []).append(ep)
            if n == "HTAN" and f == 0 and a.scenario == "ideal" and a.impute == "ffill" and not a.global_conf and a.effect == 1.0:
                torch.save(m.state_dict(), "results/htan_fold0.pt")
    print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)
np.savez(f"results/{a.out}.npz", y=y, tnod=d["tnod"], **{n.replace("-", "_"): P[n] for n in names})
json.dump(info, open(f"results/{a.out}_info.json", "w"))
