"""LongTraj-VOC simulator (idealised + realistic scenarios).
Parameters (mu, SD, PMP drift %/6mo) are taken from Table 1 of the manuscript."""
import numpy as np

VOCS = [  # name, HC mean (ppb_v), HC between-subject SD, PMP drift %/6mo
    ("2-Butanone", 1.2, 0.4, 18.3), ("Isoprene", 96.0, 22.0, -8.1),
    ("Hexanal", 2.1, 0.7, 14.6), ("1-Propanol", 0.8, 0.3, 11.2),
    ("3-Hydroxy-2-butanone", 0.5, 0.2, 22.7), ("Styrene", 0.3, 0.15, 9.4),
    ("Ethyl benzene", 0.4, 0.2, 7.8), ("Pentanal", 1.1, 0.5, 13.1),
    ("Acetone", 477.0, 145.0, 4.2), ("Nonanal", 1.9, 0.6, 16.5)]
NAMES = [v[0] for v in VOCS]
MU = np.array([v[1] for v in VOCS]); SD = np.array([v[2] for v in VOCS])
PCT = np.array([v[3] for v in VOCS])
TIMES = np.array([0., 6., 12., 18., 24.])
# smoking effect per feature (in units of HC SD for a current smoker)
SMK = np.array([.5, 0, .1, .1, 0, .6, .5, .1, .4, .1])
AMBIENT = [0, 5, 6]   # 2-butanone, styrene, ethylbenzene: strong ambient sources


def simulate(scenario="ideal", seed=0, effect=1.0, miss=0.15, groups=(480, 360, 360)):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], groups)            # 0 HC, 1 PMP, 2 MAL
    N, T, J = len(y), len(TIMES), len(MU)
    real = scenario == "real"
    smoke = rng.choice([0, 1, 2], N, p=[.2, .4, .4]); smoke[y == 2] = rng.choice([0, 1, 2], (y == 2).sum(), p=[.1, .35, .55])
    pack = np.clip(rng.normal(38, 12, N), 20, 80) * (smoke > 0)
    rate = np.exp(rng.normal(0, 0.2 if not real else 0.3, N))   # subject drift-rate heterogeneity
    beta = (MU * PCT / 100 / 6)[None, :] * effect                # per month, PMP
    # effective "drift exposure" u(t) per subject, in months-equivalents
    t = TIMES[None, :].repeat(N, 0)
    onset = np.zeros(N); u = t.copy()
    mult = np.where(y == 2, 2.0, np.where(y == 1, 1.0, 0.0))
    benign = np.zeros(N, bool)
    if real:
        onset = np.where(y == 1, rng.uniform(0, 18, N), np.where(y == 2, rng.uniform(0, 6, N), 0.))
        u = np.clip(t - onset[:, None], 0, None)
        plat = (y == 1) & (rng.random(N) < .30)                   # stochastic plateauing
        up = rng.uniform(6, 12, N)
        u = np.where(plat[:, None], np.minimum(u, up[:, None]), u)
        benign = (y == 0) & (rng.random(N) < .20)                 # benign dysplasia, spontaneous regression
        b_on = rng.uniform(0, 12, N); b_pk = rng.uniform(6, 15, N)
        ub = np.clip(t - b_on[:, None], 0, None)
        ub = np.where(ub > b_pk[:, None], np.clip(2 * b_pk[:, None] - ub, 0, None), ub)  # rise then fall back
        u = np.where(benign[:, None], ub, u)
        mult = np.where(benign, 0.6, mult)
    drift = (mult * rate)[:, None, None] * u[:, :, None] * beta[:, None, :]       # (N,T,J)
    if real:
        sw, sb = 0.6, 1.0
    else:
        sw, sb = 0.25, 1.0
    eta = rng.normal(0, sb, (N, 1, J)) * SD
    eps = rng.normal(0, sw, (N, T, J)) * SD
    smk = (SMK * SD)[None, None, :] * ((smoke == 2) * 1.0 + (smoke == 1) * 0.3 + pack / 40 * 0.3)[:, None, None]
    V = MU + drift + eta + eps + smk
    if real:
        # physiological stochasticity
        V[:, :, 1] *= np.exp(rng.normal(0, .25, (N, T)))      # isoprene: cardiac output / ventilation
        V[:, :, 8] *= np.exp(rng.normal(0, .35, (N, T)))      # acetone: fasting / glucose
        # ambient room-air contamination (additive, per visit, not blank-corrected)
        amb = np.exp(rng.normal(0, .8, (N, T))) * 0.6
        for j in AMBIENT: V[:, :, j] += amb * MU[j] * np.exp(rng.normal(0, .3, (N, T)))
        # analytical batch drift (shared by all subjects at the same calendar visit)
        V *= (1 + rng.normal(0, .05, (1, T, J)))
        V *= np.exp(rng.normal(0, .15, (N, T, J)))             # GC-IMS multiplicative measurement noise
    V = np.clip(V, 0.02 * MU, None)
    Vfull = V.copy()
    # missingness: 15% per visit, visit 0 always observed, at most 2 missing visits
    M = rng.random((N, T)) > miss; M[:, 0] = True
    for i in np.where((~M).sum(1) > 2)[0]:
        idx = np.where(~M[i])[0]; M[i, rng.choice(idx, len(idx) - 2, replace=False)] = True
    V = np.where(M[:, :, None], V, np.nan)
    tnod = np.where(y == 1, onset + 24.0 / rate, np.nan)       # LDCT-visible nodule time (months): drift reaches
    return dict(V=V, Vfull=Vfull, M=M, y=y, smoke=smoke, pack=pack, rate=rate, tnod=tnod, t=TIMES, benign=benign)
