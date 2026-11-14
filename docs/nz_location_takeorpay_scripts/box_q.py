# zero-sum style boxplots (TK world) for the fixed-cost sweep: candidates = union of top-8 per opening cap, V' = V + (q-500)|x|
import csv, glob, re, collections, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/"
DS = ["0.05", "0.1", "0.2", "0.3"]; q90 = lambda v: np.quantile(v, 0.9); CFGS = ["top", "topr2", "topr1"]
MOD = [("N", "Nominal", "#0072B2", None), ("B1", "Belief-only (delta = 0.1)", "#CC79A7", "..."), ("BS", "Belief-only (calibrated delta)", "#009E73", "///"), ("OR", "Tail oracle", "#7F7F7F", "xxx")]
QS = [500, 750, 1000]
def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else None
def load(key, nf):
    g, s = key; C = collections.OrderedDict(); chk = 0
    for cfg in CFGS:
        T = rd(O + "tk/tk_%s_g%d_s%d.csv" % (cfg, g, s))
        if T is None: return None
        D = rd(O + "tkd/tkd_%s_g%d_s%d.csv" % (cfg, g, s)); J = collections.defaultdict(list)
        if D is not None:
            for r in D:
                if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        if not J and cfg == "top":
            D = rd(O + "tkd/tkd_topnf%d_g%d_s%d.csv" % (nf, g, s))
            if D is None: return None
            for r in D:
                if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        for r in T:
            x = r["x"]
            if x not in J: return None
            a = np.maximum.accumulate([float(r["adv" + d]) for d in DS]); j = np.array(J[x])
            if x in C: chk = max(chk, float(np.abs(C[x][2] - j).max())); continue
            C[x] = (float(r["VN"]), a, j, x.count(";") + 1)
    return C, chk
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

def decide(C, Q):
    xs = list(C); n_ = np.array([C[x][3] for x in xs]); v = np.array([C[x][0] for x in xs]) + (Q - 500) * n_
    A = np.array([C[x][1] for x in xs]); M = np.array([C[x][2] for x in xs]) + (Q - 500) * n_[:, None]
    tied = np.where(v <= v.min() + 1e-6)[0]; n = int(tied[np.argmin([q90(M[i]) for i in tied])]); sc = abs(v[n]) / 100
    bq = q90(M[n]) - v[n]; k = next((i for i in range(4) if A[n, i] >= bq - 1e-9), 3)
    dec = {"N": n, "B1": int(np.argmin(v + A[:, 1])), "BS": int(np.argmin(v + A[:, k])), "OR": int(np.argmin(np.quantile(M, 0.9, axis=1)))}
    return {m: (M[i] - M[n].mean()) / sc for m, i in dec.items()}
for nf in [5, 20]:
    L = {(g, s): load((g, s), nf) for g in range(1, 16) for s in range(1, 6)}; L = {k: v[0] for k, v in L.items() if v is not None}
    ALL = {Q: {k: decide(C, Q) for k, C in L.items()} for Q in QS}; print("N_F", nf, len(L))
    sp = 3.0; off = [-0.75, -0.25, 0.25, 0.75]
    fig, ax = plt.subplots(figsize=(2.6 * len(QS) + 2.6, 4.8))
    for gi, Q in enumerate(QS):
        for mi, (m, ml, col, h) in enumerate(MOD):
            d = np.concatenate([r[m] for r in ALL[Q].values()]); vbox(ax, gi * sp + off[mi], d, col, h)
            print("  pooled q=%d %-30s mean %+6.2f q75 %+6.2f q90 %+6.2f q95 %+6.2f" % (Q, ml, d.mean(), np.quantile(d, .75), q90(d), np.quantile(d, .95)))
    ax.set_xticks([i * sp for i in range(len(QS))]); ax.set_xticklabels(["Opening cost %d" % Q for Q in QS], fontsize=10); style(ax)
    ax.set_ylabel("Leader cost over retailer beliefs (% of cost)"); legend(ax, MOD)
    fig.tight_layout(); fig.savefig(OUT + "tk_cost_boxplot_q750_nf%d.png" % nf, dpi=200); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(2.3 * len(QS) * 2 + 3.2, 4.8), sharey=True); off3 = [-0.5, 0.0, 0.5]
    for ax, (sl, st) in zip(axs, [("Mean over retailer beliefs", np.mean), ("q90 over retailer beliefs", q90)]):
        for gi, Q in enumerate(QS):
            for mi, (m, ml, col, h) in enumerate(MOD[1:]):
                d = np.array([st(r[m]) - st(r["N"]) for r in ALL[Q].values()]); vbox(ax, gi * sp + off3[mi], d, col, h)
                print("  paired %-27s q=%d %-30s mean %+5.2f median %+5.2f better %2.0f%% worse %2.0f%% same %2.0f%%" % (sl, Q, ml, d.mean(), np.median(d), 100 * (d < -0.01).mean(), 100 * (d > 0.01).mean(), 100 * (abs(d) <= 0.01).mean()))
        ax.set_xticks([i * sp for i in range(len(QS))]); ax.set_xticklabels(["Opening cost %d" % Q for Q in QS], fontsize=9); ax.set_title(sl, fontsize=11); style(ax)
    axs[0].set_ylabel("Difference to Nominal per run (% of cost)"); legend(axs[1], MOD[1:])
    fig.tight_layout(); fig.savefig(OUT + "tk_paired_boxplot_q750_nf%d.png" % nf, dpi=200); plt.close(fig)
