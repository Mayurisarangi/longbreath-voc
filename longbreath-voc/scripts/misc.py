import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)  # outputs go to <repo>/results and <repo>/figs
import numpy as np, json, sim, prep, analyze as A
out = {}
# imputation attenuation diagnostics on complete-data ground truth (realistic benchmark, PMP subjects with >=1 missing visit)
for sc, eff in (("ideal", 1.0), ("real", 0.5)):
    d = sim.simulate(sc, 0, eff); V, Vf, M, y = d["V"], d["Vfull"], d["M"], d["y"]
    cf = prep.fit_confounder(V, d["smoke"], d["pack"], np.arange(len(y))); Vc = prep.apply_confounder(V, cf); Vcf = prep.apply_confounder(Vf, cf)
    lmm = prep.fit_lmm(Vc, np.arange(len(y)))
    rows = {}
    sel = (y == 1) & (~M).any(1)
    true_cum = (Vcf[:, -1] - Vcf[:, 0])               # 24-month cumulative change, standardised units
    for mode in ("ffill", "linear", "lmm"):
        Xi = prep.impute(Vc, M, mode, lmm); est = Xi[:, -1] - Xi[:, 0]
        dd = np.diff(Xi, axis=1); miss_after = (~M[:, 1:])
        zero_frac = np.mean(np.abs(dd[miss_after[:, :, None].repeat(10, 2)]) < 1e-9) * 100
        err = np.sqrt(np.mean((Xi - Vcf)[~M[:, :, None].repeat(10, 2)] ** 2))
        ratio = np.mean(np.abs(est[sel])) / np.mean(np.abs(true_cum[sel])) * 100
        # error of imputed (missing) entries only
        rows[mode] = dict(rmse_missing=float(err), pct_zero_delta_at_missing=float(zero_frac), cum_drift_recovered_pct=float(ratio))
    out[sc] = rows
json.dump(out, open("results/imputation_diag.json", "w"), indent=1); print(json.dumps(out, indent=1))
