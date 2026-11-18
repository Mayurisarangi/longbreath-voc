"""Final large-font figures used in the manuscript (Figures 1-6 and 8). Needs results/*.npz and results/*_metrics.json
(produced by run.py + analyze.py).  Figure 7 is produced by fig7L.py."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)  # outputs go to <repo>/results and <repo>/figs

import numpy as np, json, os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve
import analyze as A, sim
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13, "axes.linewidth": 1.2, "lines.linewidth": 2.4})
col = {"PLS-DA": "#9aa0a6", "SVM": "#5b6470", "LMM": "#2a6fdb", "FPCA": "#5aa9e6", "LSTM": "#c98a14", "DLSTM": "#e0a030", "GRU-D": "#8e5bd0", "TT": "#1b8a7a", "HTAN-small": "#e07a9a", "HTAN": "#d62839"}
LAB = {"HTAN-small": "HTAN-s"}
def load_m(n): return json.load(open(f"results/{n}_metrics.json"))
scen = [("ideal", "Idealised"), ("real", "Realistic")]
def box(ax, x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=2))
    nt = title.count("\n") + 1
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=14.5, fontweight="bold", color="#222", linespacing=1.15)
    for i, l in enumerate(lines): ax.text(x + w / 2, y + h - 0.03 - nt * 0.055 - 0.035 - i * 0.052, l, ha="center", va="top", fontsize=13, color="#333")
def arrow(ax, a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, lw=2.2, color="#444"))
# ---- Fig 1
fig, ax = plt.subplots(figsize=(10, 8)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, .02, .68, .34, .30, "Kaggle Survey Dataset", ["n = 309 subjects", "15 clinical features", "age, smoking, gender", "binary cancer label"], "#dbe9ff", "#4a7bc8")
box(ax, .02, .34, .34, .30, "Shaffie et al. (2022)", ["n = 504 subjects", "27 VOC features", "2-butanone ranked #1", "malignant / benign"], "#dbe9ff", "#4a7bc8")
box(ax, .42, .42, .26, .40, "Published GC-MS\ndistributions", ["LuCID 2025", "Setlhare et al. 2025", "ERS review", "mean, SD per VOC"], "#ebe3ff", "#7a4bc8")
box(ax, .73, .42, .25, .40, "Data integration\n& drift extraction", ["risk stratification", "VOC mean/SD", "drift per 6 months"], "#ebe3ff", "#7a4bc8")
box(ax, .30, .02, .40, .24, "LongTraj-VOC benchmark", ["N = 1,200, T = 5 visits", "HC 480 / PMP 360 / MAL 360", "~12% missing visits"], "#ffd9cc", "#c8603a")
arrow(ax, (.36, .80), (.42, .68)); arrow(ax, (.36, .48), (.42, .58)); arrow(ax, (.68, .62), (.73, .62)); arrow(ax, (.855, .42), (.62, .26))
plt.tight_layout(); plt.savefig("figs/L_fig1.png", dpi=170); plt.close()
# ---- Fig 2
sel = [0, 1, 2, 4, 9, 8]; cols = {0: "#1b8a7a", 1: "#c98a14", 2: "#e0563a"}; lab = {0: "Healthy control", 1: "Pre-malignant progressor", 2: "Malignant"}
fig, ax = plt.subplots(4, 3, figsize=(11, 15))
for s, (sc, eff, ttl) in enumerate([("ideal", 1.0, "Idealised"), ("real", 0.5, "Realistic")]):
    d = sim.simulate(sc, 0, eff); V, y = d["Vfull"], d["y"]
    for c, j in enumerate(sel):
        a = ax[s * 2 + c // 3, c % 3]
        for g in range(3):
            m = V[y == g, :, j].mean(0); sd = V[y == g, :, j].std(0)
            a.plot(sim.TIMES, m, "-o", ms=6, color=cols[g], label=lab[g]); a.fill_between(sim.TIMES, m - sd, m + sd, color=cols[g], alpha=.15)
        a.set_title(f"{ttl}: {sim.NAMES[j]}", fontsize=15, fontweight="bold"); a.set_xticks(sim.TIMES)
        if c % 3 == 0: a.set_ylabel("ppb$_v$")
        if s * 2 + c // 3 == 3: a.set_xlabel("Time (months)")
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=14, frameon=False)
plt.tight_layout(rect=[0, .03, 1, 1]); plt.savefig("figs/L_fig2.png", dpi=150); plt.close()
# ---- Fig 3
fig, ax = plt.subplots(figsize=(11, 8)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
def blk(x, y, w, h, t, fc, ec, fs=14): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=2)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold")
ax.text(.5, .97, "Input: serial VOC vectors x(t₁)…x(t_K), 30 features each", ha="center", fontsize=15, fontweight="bold")
for i, t in enumerate(["t=0m", "t=6m", "t=12m", "t=18m", "t=24m"]): blk(.03 + i * .125, .84, .11, .08, t, "#c7d6f2", "#4a6fb5", 13)
blk(.03, .68, .6, .10, "Irregular Interval Positional\nEncoding (IIPE)", "#9fb7e6", "#4a6fb5")
blk(.03, .50, .6, .12, "Bidirectional LSTM encoder (2 layers)\nhidden states H", "#9fb7e6", "#4a6fb5")
blk(.03, .32, .6, .10, "Linear projection to model dimension", "#c7d6f2", "#4a6fb5")
blk(.70, .62, .27, .12, "[DRIFT] token\n(learned query)", "#f0c040", "#a07800")
blk(.70, .40, .27, .16, "Cross-attention\ntransformer decoder\n(4 heads, 2 layers)", "#d9c2f0", "#7a4bc8", 13)
blk(.70, .22, .27, .12, "MLP head\n(dropout 0.3)", "#9ad08a", "#3a7a2a")
blk(.70, .04, .27, .12, "HC / PMP / MAL\n(3-class softmax)", "#ffd9cc", "#c8603a", 13)
for a_, b_ in [((.33, .84), (.33, .78)), ((.33, .68), (.33, .62)), ((.33, .50), (.33, .42)), ((.835, .62), (.835, .56)), ((.835, .40), (.835, .34)), ((.835, .22), (.835, .16))]: arrow(ax, a_, b_)
arrow(ax, (.63, .37), (.70, .46))
ax.text(.03, .22, "Full HTAN: ~2.2 M parameters\nHTAN-small: ~40 k parameters", fontsize=13.5, va="top")
plt.tight_layout(); plt.savefig("figs/L_fig3.png", dpi=170); plt.close()
# ---- Fig 4 ROC
fig, ax = plt.subplots(1, 2, figsize=(13, 6.4)); rng = np.random.default_rng(3); grid = np.linspace(0, 1, 101)
for a, (s, t) in zip(ax, scen):
    y, tn, P, info = A.load(s); m = y != 2
    for k in A.ORDER:
        p = P[k][:, 4]; sc = p[m, 1] / (p[m, 1] + p[m, 0] + 1e-12); yy = (y[m] == 1)
        fpr, tpr, _ = roc_curve(yy, sc); a.plot(fpr, tpr, color=col[k], lw=3.4 if k in ("HTAN", "LMM") else 1.6, label=LAB.get(k, k), alpha=1 if k in ("HTAN", "LMM") else .8)
        if k in ("HTAN", "LMM"):
            bt = []
            for _ in range(300):
                idx = rng.integers(0, len(yy), len(yy))
                if yy[idx].all() or (~yy[idx]).all(): continue
                f_, t_, _ = roc_curve(yy[idx], sc[idx]); bt.append(np.interp(grid, f_, t_))
            lo, hi = np.percentile(bt, [2.5, 97.5], 0); a.fill_between(grid, lo, hi, color=col[k], alpha=.2)
    a.plot([0, 1], [0, 1], "k:", lw=1.2); a.set_title(t + " (PMP vs HC)"); a.set_xlabel("1 − specificity"); a.set_ylabel("Sensitivity")
ax[1].legend(loc="lower right", ncol=2, fontsize=11.5); plt.tight_layout(); plt.savefig("figs/L_fig4.png", dpi=160); plt.close()
# ---- Fig 5
fig, ax = plt.subplots(2, 1, figsize=(12, 11)); names = [LAB.get(k, k) for k in A.ORDER]
for a, (s, t) in zip(ax, scen):
    r = load_m(s); mets = [("AUC", "AUC"), ("Sens", "Sensitivity"), ("Spec", "Specificity"), ("F1", "F1-score")]; w = .2
    for i, (k, nm) in enumerate(mets):
        v = np.array([r[x][k][0] for x in A.ORDER]); lo = np.array([r[x][k][1] for x in A.ORDER]); hi = np.array([r[x][k][2] for x in A.ORDER])
        a.bar(np.arange(len(v)) + (i - 1.5) * w, v, w, yerr=[v - lo, hi - v], capsize=3, label=nm, color=["#1f3a5f", "#2a6fdb", "#5aa9e6", "#a9c8ee"][i], error_kw=dict(lw=1.2))
    a.set_xticks(range(len(A.ORDER))); a.set_xticklabels(names, rotation=30, ha="right"); a.set_ylim(0.4, 1.12); a.set_title(t + " scenario (95% bootstrap CI)"); a.legend(ncol=4, loc="upper center", fontsize=12)
plt.tight_layout(); plt.savefig("figs/L_fig5.png", dpi=150); plt.close()
# ---- Fig 6
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
for a, (s, t) in zip(ax, scen):
    y, tn, P, info = A.load(s); _, fl = A.metrics(y, P["HTAN"], tn); _, fl2 = A.metrics(y, P["LMM"], tn); pm = y == 1; any_ = False
    for nm, f_, c in (("HTAN", fl, col["HTAN"]), ("LMM", fl2, col["LMM"])):
        ld = np.clip(tn - f_, 0, None)[pm & ~np.isnan(f_)]
        if len(ld): any_ = True; a.hist(ld, bins=np.arange(0, 60, 3), alpha=.6, color=c, label=f"{nm}: {100*np.mean(~np.isnan(f_[pm])):.0f}% flagged, mean {ld.mean():.1f} mo")
    if not any_: a.text(.5, .5, "No PMP subject reached\nposterior > 0.8 (any model)", transform=a.transAxes, ha="center", fontsize=15)
    a.set_title(t); a.set_xlabel("Lead time (months)"); a.set_ylabel("PMP subjects")
    if any_: a.legend(fontsize=12)
plt.tight_layout(); plt.savefig("figs/L_fig6.png", dpi=160); plt.close()
# ---- Fig 8 effect-size sweep
fig, a = plt.subplots(figsize=(9, 6.2)); effs = [1.0, 0.5, 0.35, 0.25, 0.15]
for k in ("LMM", "FPCA", "HTAN-small"):
    v = np.array([load_m("real" if e == 0.5 else f"sweep_{e}")[k]["AUC"] for e in effs]); a.plot(effs, v[:, 0], "-o", ms=9, color=col[k], label=k); a.fill_between(effs, v[:, 1], v[:, 2], color=col[k], alpha=.17)
a.axhspan(.54, .58, color="grey", alpha=.3); a.text(.95, .56, "LuCID validation AUC 0.54–0.58", fontsize=13, va="center")
a.set_xlabel("Drift-effect multiplier (1 = Table 1 magnitudes)"); a.set_ylabel("Macro AUC (3-class)"); a.invert_xaxis(); a.legend(fontsize=13); a.set_title("Realistic scenario: effect-size sweep")
plt.tight_layout(); plt.savefig("figs/L_fig8.png", dpi=160); plt.close()
