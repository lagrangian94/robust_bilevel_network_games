# zero-sum style boxplots for the location data-world runs (leader estimates q-hat)
import csv, glob, re, collections, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
G = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/oos/true/"
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/"
CFG = [("base", "Manuscript formulation"), ("top", "Take-or-pay, f/p = 0.25"), ("topfr50", "Take-or-pay, f/p = 0.5")]
MOD = [("N", "Nominal", "#0072B2", None), ("D", "Data-DR", "#E69F00", "///"), ("C1", "Belief-only", "#CC79A7", "..."), ("C", "Coupled", "#009E73", "|||")]
def load(cfg):
    R = {}
    for f in sorted(glob.glob(G + "ot_dw%s_g*_s*.csv" % cfg)):
        m = re.search(r"_g(\d+)_s(\d+)\.csv", f); J = collections.defaultdict(list)
        for r in csv.DictReader(open(f, encoding="utf-8")): J[r["model"]].append(float(r["J"]))
        if len(J["N"]) < 10: continue
        R[(int(m.group(1)), int(m.group(2)))] = {k: np.array(v) for k, v in J.items()}
    return R
def vbox(ax, x, d, col, hatch, w=0.2, cw=0.1):
    q05, q25, med, q75, q95 = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
    ax.plot([x, x], [q05, q25], color=col, lw=1.5, ls="--"); ax.plot([x, x], [q75, q95], color=col, lw=1.5, ls="--")
    ax.plot([x - cw, x + cw], [q05, q05], color=col, lw=2); ax.plot([x - cw, x + cw], [q95, q95], color=col, lw=2)
    fc = matplotlib.colors.to_rgba(col, 0.15) if hatch is None else "none"
    ax.add_patch(Rectangle((x - w, q25), 2 * w, q75 - q25, facecolor=fc, edgecolor=col, lw=2, hatch=hatch))
    ax.plot([x - w, x + w], [med, med], color=col, lw=3); ax.plot([x], [d.mean()], marker="D", color=col, ms=6)
    return q95, d.mean()
ALL = {c: load(c) for c, _ in CFG}
# ---- figure 1: pooled out-of-sample cost, % of the Nominal decision's mean cost in the same run
fig, ax = plt.subplots(figsize=(11, 4.6)); sp = 3.0; off = [-0.75, -0.25, 0.25, 0.75]
for gi, (c, lab) in enumerate(CFG):
    R = ALL[c]
    for mi, (m, ml, col, h) in enumerate(MOD):
        d = np.concatenate([100 * (J[m] - J["N"].mean()) / abs(J["N"].mean()) for J in R.values()])
        q95, mu = vbox(ax, gi * sp + off[mi], d, col, h)
        print("%-8s %-11s n=%d mean %+.2f q90 %+.2f q95 %+.2f" % (c, ml, len(d), mu, np.quantile(d, 0.9), q95))
ax.set_xticks([i * sp for i in range(len(CFG))]); ax.set_xticklabels([l for _, l in CFG], fontsize=11)
ax.set_ylabel("Out-of-sample cost (% of Nominal mean)"); ax.grid(axis="y", color="0.85"); ax.set_axisbelow(True); ax.axhline(0, color="0.5", lw=0.8)
ax.legend(handles=[Patch(facecolor=(matplotlib.colors.to_rgba(col, 0.15) if h is None else "none"), edgecolor=col, lw=2, hatch=h, label=ml) for _, ml, col, h in MOD], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "dw_cost_boxplot.png", dpi=200); plt.close(fig)
# ---- figure 2: per-run paired differences (75 runs each), mean and q90 over retailer belief draws
PAIRS = [("C1", "N", "Belief-only - Nominal", "#CC79A7", "..."), ("D", "N", "Data-DR - Nominal", "#E69F00", "///"), ("C", "N", "Coupled - Nominal", "#009E73", "|||"), ("C", "D", "Coupled - Data-DR", "#56B4E9", "xxx")]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, (sl, st) in zip(axs, [("Mean over retailer beliefs", np.mean), ("q90 over retailer beliefs", lambda v: np.quantile(v, 0.9))]):
    for gi, (c, lab) in enumerate(CFG):
        R = ALL[c]
        for pi, (a, b, pl, col, h) in enumerate(PAIRS):
            d = np.array([100 * (st(J[a]) - st(J[b])) / abs(J["N"].mean()) for J in R.values()])
            vbox(ax, gi * sp + off[pi], d, col, h)
            print("%-28s %-8s %-22s mean %+.2f median %+.2f share<0 %.0f%% share>0 %.0f%% zero %.0f%%" % (sl, c, pl, d.mean(), np.median(d), 100 * (d < -0.01).mean(), 100 * (d > 0.01).mean(), 100 * (abs(d) <= 0.01).mean()))
    ax.set_xticks([i * sp for i in range(len(CFG))]); ax.set_xticklabels([l for _, l in CFG], fontsize=10)
    ax.set_title(sl, fontsize=11); ax.grid(axis="y", color="0.85"); ax.set_axisbelow(True); ax.axhline(0, color="0.3", lw=1)
axs[0].set_ylabel("Cost difference per run (% of Nominal mean)")
axs[1].legend(handles=[Patch(facecolor="none", edgecolor=col, lw=2, hatch=h, label=pl) for _, _, pl, col, h in PAIRS], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "dw_paired_boxplot.png", dpi=200); plt.close(fig)
