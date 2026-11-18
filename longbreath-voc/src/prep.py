"""Pre-processing: confounder correction (fit on TRAIN only by default), imputation, feature construction, LMM utilities."""
import numpy as np, warnings
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from sim import TIMES
warnings.filterwarnings("ignore")


def fit_confounder(V, smoke, pack, tr):
    """Per-feature LMM: y ~ smoke(former,current) + pack-years + month + (1|subject), fitted on subjects `tr` only.
    Returns smoking/pack coefficients and the standardisation constants (train only)."""
    N, T, J = V.shape
    coefs, mean, sd = [], np.zeros(J), np.zeros(J)
    X = np.column_stack([np.ones(N), smoke == 1, smoke == 2, pack / 10.0]).astype(float)
    for j in range(J):
        rows = [(i, k) for i in tr for k in range(T) if not np.isnan(V[i, k, j])]
        ii = np.array([r[0] for r in rows]); kk = np.array([r[1] for r in rows])
        exog = np.column_stack([X[ii], TIMES[kk] / 12.0]); endog = V[ii, kk, j]
        try:
            r = MixedLM(endog, exog, groups=ii).fit(reml=True, method="lbfgs")
            b = np.asarray(r.fe_params)
        except Exception:
            b = np.linalg.lstsq(exog, endog, rcond=None)[0]
        coefs.append(b[1:4])
        adj = V[:, :, j] - (X[:, 1:] @ b[1:4])[:, None]
        mean[j] = np.nanmean(adj[tr]); sd[j] = np.nanstd(adj[tr])
    return dict(coefs=np.array(coefs), X=X, mean=mean, sd=sd)


def apply_confounder(V, cf):
    adj = V - np.einsum("nc,jc->nj", cf["X"][:, 1:], cf["coefs"])[:, None, :]
    return (adj - cf["mean"]) / cf["sd"]


# ---------------- LMM (random intercept + random slope) for the two-stage statistical baseline -----------
def fit_lmm(Vc, tr):
    """One LMM per feature on training subjects: y = b0 + b1*t + u0 + u1*t + e. Returns params for BLUP computation."""
    N, T, J = Vc.shape; out = []
    for j in range(J):
        rows = [(i, k) for i in tr for k in range(T) if not np.isnan(Vc[i, k, j])]
        ii = np.array([r[0] for r in rows]); kk = np.array([r[1] for r in rows]); t = TIMES[kk] / 12.0
        exog = np.column_stack([np.ones(len(t)), t]); endog = Vc[ii, kk, j]
        try:
            r = MixedLM(endog, exog, groups=ii, exog_re=exog).fit(reml=True, method="lbfgs")
            fe = np.asarray(r.fe_params); G = np.asarray(r.cov_re); s2 = float(r.scale)
            if not np.all(np.isfinite(G)): raise ValueError
        except Exception:
            fe = np.linalg.lstsq(exog, endog, rcond=None)[0]; G = np.eye(2) * .3; s2 = .3
        out.append((fe, G + 1e-6 * np.eye(2), max(s2, 1e-6)))
    return out


def blups(Vc, M_use, lmm):
    """Empirical-Bayes random effects (u0,u1) per subject/feature from the observed visits in M_use (N,T)."""
    N, T, J = Vc.shape; U = np.zeros((N, J, 2)); tt = TIMES / 12.0
    for j, (fe, G, s2) in enumerate(lmm):
        for i in range(N):
            o = M_use[i] & ~np.isnan(Vc[i, :, j])
            if not o.any(): continue
            Z = np.column_stack([np.ones(o.sum()), tt[o]])
            r = Vc[i, o, j] - Z @ fe
            U[i, j] = G @ Z.T @ np.linalg.solve(Z @ G @ Z.T + s2 * np.eye(o.sum()), r)
    return U


def lmm_features(Vc, M_use, lmm):
    U = blups(Vc, M_use, lmm); fe_slope = np.array([p[0][1] for p in lmm])
    return np.concatenate([U[:, :, 0], U[:, :, 1] + fe_slope], 1)     # intercept effects, subject slopes


# ---------------- imputation ----------------
def impute(Vc, M, mode, lmm=None):
    """Vc: (N,T,J) standardised, NaN where missing. Returns filled array."""
    X = Vc.copy(); N, T, J = X.shape
    if mode == "ffill":
        for k in range(1, T):
            m = np.isnan(X[:, k]); X[:, k][m] = X[:, k - 1][m]
    elif mode == "linear":
        for i in range(N):
            for j in range(J):
                o = ~np.isnan(X[i, :, j])
                X[i, :, j] = np.interp(TIMES, TIMES[o], X[i, o, j])
    elif mode == "lmm":     # Gaussian-process style: random-effects BLUP prediction of the missing visits
        tt = TIMES / 12.0
        for j, (fe, G, s2) in enumerate(lmm):
            for i in range(N):
                o = ~np.isnan(Vc[i, :, j])
                if o.all(): continue
                Z = np.column_stack([np.ones(o.sum()), tt[o]]); r = Vc[i, o, j] - Z @ fe
                u = G @ Z.T @ np.linalg.solve(Z @ G @ Z.T + s2 * np.eye(o.sum()), r)
                pred = fe[0] + fe[1] * tt + u[0] + u[1] * tt
                X[i, ~o, j] = pred[~o]
    elif mode == "mask":
        pass
    return X


def build(Vc, M, L, mode, lmm=None):
    """Construct model inputs using only the first L visits.  Returns dict of arrays (padded to T=5)."""
    N, T, J = Vc.shape
    V = Vc.copy(); V[:, L:] = np.nan; Mu = M.copy(); Mu[:, L:] = False
    V0 = V[:, :1].copy()                                    # first visit (always observed)
    valid = np.zeros((N, T), bool); valid[:, :L] = True
    if mode == "mask":
        Z = (V - V0); obs = ~np.isnan(Z[:, :, 0]) & valid
        Zf = np.nan_to_num(Z); d = np.zeros_like(Zf)
        last = np.zeros((N, J)); lt = np.zeros(N); gap = np.zeros((N, T))
        for k in range(1, T):
            ok = obs[:, k]
            d[:, k] = np.where(ok[:, None], Zf[:, k] - last, 0)
            gap[:, k] = (TIMES[k] - lt) / 12
            last = np.where(ok[:, None], Zf[:, k], last); lt = np.where(ok, TIMES[k], lt)
        x = np.concatenate([Zf, d, Zf, obs[:, :, None].astype(float), gap[:, :, None]], 2)
        use_obs = True
    else:
        Vi = impute(V, Mu, mode, lmm); Vi[:, L:] = 0 if False else Vi[:, L:]
        Z = Vi - V0; Z = np.nan_to_num(Z); Z[:, L:] = 0
        d = np.zeros_like(Z); d[:, 1:] = Z[:, 1:] - Z[:, :-1]; c = np.cumsum(d, 1)
        x = np.concatenate([Z, d, c], 2); obs = valid.copy(); use_obs = False
    # raw (un-imputed) inputs for GRU-D
    Vr = V.copy(); m = ~np.isnan(Vr); Vr0 = np.nan_to_num(Vr)
    delta = np.zeros((N, T, J)); lt = np.zeros((N, J))
    xl = np.zeros((N, T, J)); cur = np.zeros((N, J))
    for k in range(T):
        if k > 0: delta[:, k] = TIMES[k] - lt
        cur = np.where(m[:, k], Vr0[:, k], cur); xl[:, k] = cur
        lt = np.where(m[:, k], TIMES[k], lt)
    return dict(x=x.astype(np.float32), t=np.tile(TIMES, (N, 1)).astype(np.float32), valid=valid, obs=obs, use_obs=use_obs,
                v=Vr0.astype(np.float32), m=m.astype(np.float32), delta=(delta / 12).astype(np.float32), xl=xl.astype(np.float32),
                lens=np.full(N, L))
