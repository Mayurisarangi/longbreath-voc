import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)  # outputs go to <repo>/results and <repo>/figs
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size":15,"axes.labelsize":15,"xtick.labelsize":14,"ytick.labelsize":14,"axes.titlesize":15})
from sklearn.model_selection import StratifiedKFold, train_test_split
import sim, prep, models
d = sim.simulate("ideal", 0, 1.0); V, M, y = d["V"], d["M"], d["y"]; N = len(y)
skf = StratifiedKFold(5, shuffle=True, random_state=0); tra, te = next(iter(skf.split(np.zeros(N), y)))
cf = prep.fit_confounder(V, d["smoke"], d["pack"], tra); Vc = prep.apply_confounder(V, cf)
B = prep.build(Vc, M, 5, "ffill"); m = models.make("HTAN", B["x"].shape[2]); m.load_state_dict(torch.load("results/htan_fold0.pt")); m.eval()
full = np.where(M.all(1))[0]; te_full = [i for i in te if i in set(full)]
pmp = [i for i in te_full if y[i] == 1][:2]; hc = [i for i in te_full if y[i] == 0][:1]
idx = np.array(pmp + hc); b = models.tens(B, idx); w = m.attention(b).numpy(); w = w / w.sum(1, keepdims=True)
allw = m.attention(models.tens(B, np.array(te_full))).numpy(); allw = allw / allw.sum(1, keepdims=True); yt = y[te_full]
gm = np.stack([allw[yt == g].mean(0) for g in (0, 1, 2)]); w6 = np.vstack([w, gm]); w_ind = w; w = w6
labels = ["PMP subject A", "PMP subject B", "HC subject C", f"HC mean (n={int((yt==0).sum())})", f"PMP mean (n={int((yt==1).sum())})", f"MAL mean (n={int((yt==2).sum())})"]
fig, ax = plt.subplots(figsize=(9.5, 6.2)); im = ax.imshow(w, cmap="YlOrRd", vmin=0, vmax=max(.4, w.max()), aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels([f"t={int(t)}m" for t in sim.TIMES]); ax.set_yticks(range(6)); ax.set_yticklabels(labels)
for i in range(6):
    for j in range(5): ax.text(j, i, f"{w[i,j]:.2f}", ha="center", va="center", fontsize=14)
ax.set_xlabel("Time point"); plt.colorbar(im, label="Attention weight"); plt.tight_layout(); plt.savefig("figs/L_fig7a.png", dpi=170); plt.close()
# 7b: bubble plot on 2-butanone trajectory of PMP subject A (ppb_v)
i = idx[0]; fig, ax = plt.subplots(figsize=(9, 5.4)); y_ = d["Vfull"][i, :, 0]
ax.plot(sim.TIMES, y_, "-", color="#c98a14"); ax.scatter(sim.TIMES, y_, s=w[0] * 4500, color="#d62839", alpha=.55, edgecolor="k")
ax.set_xlabel("Time (months)"); ax.set_ylabel("2-Butanone (ppb$_v$)"); ax.set_xticks(sim.TIMES); ax.set_title("PMP subject A: bubble area = attention weight", fontsize=15)
plt.tight_layout(); plt.savefig("figs/L_fig7b.png", dpi=170); plt.close(); print(w.round(2))
