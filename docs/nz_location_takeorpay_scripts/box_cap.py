# zero-sum style boxplots (TK world), leader chooses location + common capacity level; store cost = qfix + c*b
import csv, glob, re, collections, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/"
DS = ["0.05", "0.1", "0.2", "0.3"]; q90 = lambda v: np.quantile(v, 0.9)
LV = {200: "topbb200", 225: "topbb225", 250: "topbb250", 275: "topbb275", 300: "topbb300", 483: "top"}
MOD = [("N", "Nominal", "#0072B2", None), ("L1", "Belief-only, location only (delta = 0.1)", "#E69F00", "\\\\"), ("B1", "Belief-only (delta = 0.1)", "#CC79A7", "..."), ("BS", "Belief-only (calibrated delta)", "#009E73", "///"), ("OR", "Tail oracle", "#7F7F7F", "xxx")]
SC = [(450.0, 1.0, "c/v = 0.2"), (150.0, 2.0, "c/v = 0.4"), (0.0, 2.5, "c/v = 0.5")]
def rd(path): return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else None
def load(key, nf):
    g, s = key; C = []
    for b, cfg in LV.items():
        T = rd(O + "tk/tk_%s_g%d_s%d.csv" % (cfg, g, s)); lg = O + "tk/tk_%s_g%d_s%d.log" % (cfg, g, s)
        if T is None or (cfg != "top" and not (os.path.exists(lg) and "TKD_DONE" in open(lg, encoding="utf-8", errors="replace").read())): return None
        D = rd(O + "tkd/tkd_%s_g%d_s%d.csv" % (cfg, g, s)); J = collections.defaultdict(list)
        for r in (D or []):
            if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        if not J and cfg == "top":
            for r in (rd(O + "tkd/tkd_topnf%d_g%d_s%d.csv" % (nf, g, s)) or []):
                if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        for r in T:
            if r["x"] not in J: return None
            C.append((b, r["x"].count(";") + 1, float(r["VN"]), np.maximum.accumulate([float(r["adv" + d]) for d in DS]), np.array(J[r["x"]])))
    nd = len(C[0][4]); C.append((0, 0, 0.0, np.zeros(4), np.zeros(nd)))   # open nothing
    return C
def vbox(ax, x, d, col, hatch, w=0.2, cw=0.1):
    q05, q25, med, q75, q95 = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
    ax.plot([x, x], [q05, q25], color=col, lw=1.5, ls="--"); ax.plot([x, x], [q75, q95], color=col, lw=1.5, ls="--")
    ax.plot([x - cw, x + cw], [q05, q05], color=col, lw=2); ax.plot([x - cw, x + cw], [q95, q95], color=col, lw=2)
    fc = matplotlib.colors.to_rgba(col, 0.15) if hatch is None else "none"
    ax.add_patch(Rectangle((x - w, q25), 2 * w, q75 - q25, facecolor=fc, edgecolor=col, lw=2, hatch=hatch))
    ax.plot([x - w, x + w], [med, med], color=col, lw=3); ax.plot([x], [d.mean()], marker="D", color=col, ms=6)
def legend(ax, mods):
    ax.legend(handles=[Patch(facecolor=(matplotlib.colors.to_rgba(col, 0.15) if h is None else "none"), edgecolor=col, lw=2, hatch=h, label=ml) for _, ml, col, h in mods], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)
def style(ax):
    ax.grid(axis="y", color="0.85"); ax.set_axisbelow(True); ax.axhline(0, color="0.5", lw=0.8)

def decide(C, qf, c):
    b = np.array([t[0] for t in C]); n_ = np.array([t[1] for t in C]); add = (qf - 500 + c * b) * n_
    v = np.array([t[2] for t in C]) + add; A = np.array([t[3] for t in C]); M = np.array([t[4] for t in C]) + add[:, None]
    tied = np.where(v <= v.min() + 1e-6)[0]; n = int(tied[np.argmin([q90(M[i]) for i in tied])]); sc = abs(v[n]) / 100
    bq = q90(M[n]) - v[n]; k = next((i for i in range(4) if A[n, i] >= bq - 1e-9), 3); same = np.where(b == b[n])[0]
    dec = {"N": n, "L1": int(same[np.argmin((v + A[:, 1])[same])]), "B1": int(np.argmin(v + A[:, 1])), "BS": int(np.argmin(v + A[:, k])), "OR": int(np.argmin(np.quantile(M, 0.9, axis=1)))}
    return {m: (M[i] - M[n].mean()) / sc for m, i in dec.items()}
for nf in [5, 20]:
    L = {(g, s): load((g, s), nf) for g in range(1, 16) for s in range(1, 3)}; L = {k: v for k, v in L.items() if v is not None}; print("N_F", nf, len(L))
    sp = 3.6; off = [-1.0, -0.5, 0.0, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(3.0 * len(SC) + 3.4, 4.8))
    for gi, (qf, c, lab) in enumerate(SC):
        R = {k: decide(C, qf, c) for k, C in L.items()}
        for mi, (m, ml, col, h) in enumerate(MOD):
            d = np.concatenate([r[m] for r in R.values()]); vbox(ax, gi * sp + off[mi], d, col, h)
            print("  pooled %s %-42s mean %+6.2f q75 %+6.2f q90 %+6.2f q95 %+6.2f" % (lab, ml, d.mean(), np.quantile(d, .75), q90(d), np.quantile(d, .95)))
    ax.set_xticks([i * sp for i in range(len(SC))]); ax.set_xticklabels(["Capacity cost, " + l for _, _, l in SC], fontsize=10); style(ax)
    ax.set_ylabel("Leader cost over retailer beliefs (% of cost)"); legend(ax, MOD)
    fig.tight_layout(); fig.savefig(OUT + "tk_cost_boxplot_cap_nf%d.png" % nf, dpi=200); plt.close(fig)
