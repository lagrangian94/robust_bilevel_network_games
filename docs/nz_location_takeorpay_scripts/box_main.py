# box_main.py = box_eps.py with the configuration (BE_CFGS), seeds (BE_NS), file prefix (BE_PRE), y range (BE_YLIM) as options
# boxplots with the follower radius on the horizontal axis (TK world, take-or-pay, sites + per-site capacity tiers, 75 runs)
import csv, collections, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/"
DS = ["0.05", "0.1", "0.2", "0.3"]; q90 = lambda v: np.quantile(v, 0.9); rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
Q0, C_ = float(os.environ.get("BE_Q0", "0")), float(os.environ.get("BE_C", "2.5")); TAGF = os.environ.get("BE_TAG", "cv05")
BLUE, GREEN, GREY = "#0072B2", "#009E73", "#7F7F7F"
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()]); return np.mean(list(d.values())), m.std(ddof=1) / np.sqrt(len(m)), int((m < -0.01).sum()), int((m > 0.01).sum())
def run1(g, s, nf):
    T = []; J = collections.defaultdict(list)
    for cf in os.environ.get("BE_CFGS", "toptier,toptier2").split(","):
        T += rd(O + "tk/tk_%s_g%d_s%d.csv" % (cf, g, s))
        for r in rd(O + "tkd/tkd_%s_g%d_s%d.csv" % (cf, g, s)):
            if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
    def cst(x): pr = [p.split(":") for p in x.strip("{}").split(";")]; return Q0 * len(pr) + C_ * sum(float(b) for _, b in pr)
    cost = np.array([cst(r["x"]) for r in T] + [0.0]); nd = len(J[T[0]["x"]]); v = np.array([float(r["V0"]) for r in T] + [0.0]) + cost
    A = np.array([np.maximum.accumulate([float(r["adv" + d]) for d in DS]) for r in T] + [np.zeros(4)]); M = np.array([J[r["x"]] for r in T] + [np.zeros(nd)]) + cost[:, None]
    tied = np.where(v <= v.min() + 1e-6)[0]; n = int(tied[np.argmin([q90(M[i]) for i in tied])]); sc = abs(v[n]) / 100
    out = {"0": (M[n] - M[n].mean()) / sc, "CL": (M.min(axis=0) - M[n].mean()) / sc}
    for k, d in enumerate(DS): out[d] = (M[int(np.argmin(v + A[:, k]))] - M[n].mean()) / sc
    return out
def vbox(ax, x, d, col, hatch, w=0.28, cw=0.14):
    q05, q25, med, q75, q95 = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
    ax.plot([x, x], [q05, q25], color=col, lw=1.5, ls="--"); ax.plot([x, x], [q75, q95], color=col, lw=1.5, ls="--")
    ax.plot([x - cw, x + cw], [q05, q05], color=col, lw=2); ax.plot([x - cw, x + cw], [q95, q95], color=col, lw=2)
    fc = matplotlib.colors.to_rgba(col, 0.15) if hatch is None else "none"
    ax.add_patch(Rectangle((x - w, q25), 2 * w, q75 - q25, facecolor=fc, edgecolor=col, lw=2, hatch=hatch))
    ax.plot([x - w, x + w], [med, med], color=col, lw=3); ax.plot([x], [d.mean()], marker="D", color=col, ms=6)
def style(ax): ax.grid(axis="y", color="0.85"); ax.set_axisbelow(True); ax.axhline(0, color="0.5", lw=0.8)
COLS = [("0", "0\n(Nominal)", BLUE, None)] + [(d, d, GREEN, "///") for d in DS] + [("CL", "belief\nknown", GREY, "xxx")]
HAND = [Patch(facecolor=matplotlib.colors.to_rgba(BLUE, 0.15), edgecolor=BLUE, lw=2, label="Nominal"), Patch(facecolor="none", edgecolor=GREEN, lw=2, hatch="///", label="Belief-only"), Patch(facecolor="none", edgecolor=GREY, lw=2, hatch="xxx", label="Retailer belief known")]
KEYS = [(g, s) for g in range(1, 16) for s in range(1, int(os.environ.get("BE_NS", "5")) + 1)]; ALL = {nf: {k: run1(k[0], k[1], nf) for k in KEYS} for nf in (5, 20)}
fig, axs = plt.subplots(1, 2, figsize=(14.5, 4.8), sharey=True)
for ax, nf in zip(axs, (5, 20)):
    for i, (key, lab, col, h) in enumerate(COLS):
        d = np.concatenate([r[key] for r in ALL[nf].values()]); vbox(ax, i, d, col, h)
        print("cost N_F %2d %-6s mean %+6.2f q05 %+6.2f q25 %+6.2f med %+6.2f q75 %+6.2f q90 %+6.2f q95 %+6.2f" % (nf, key, d.mean(), *np.quantile(d, [.05, .25, .5, .75, .9, .95])))
    ax.axvline(4.5, color="0.7", lw=0.8, ls=":"); ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[1] for c in COLS], fontsize=10); ax.set_xlabel("Radius of the retailer-belief ambiguity set"); ax.set_title("Retailer belief from %d samples" % nf, fontsize=11); style(ax)
axs[0].set_ylabel("Leader cost over retailer beliefs (% of cost)"); YL = os.environ.get("BE_YLIM", ""); YL and axs[0].set_ylim(*[float(v) for v in YL.split(",")]); axs[1].legend(handles=HAND, loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + os.environ.get("BE_PRE", "tk_eps") + "_cost_boxplot_%s.png" % TAGF, dpi=200); plt.close(fig)
fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.2), sharey="row")
for ri, nf in enumerate((5, 20)):
    for ci, (sl, st) in enumerate([("Mean over retailer beliefs", np.mean), ("Upper tail (q90) over retailer beliefs", q90)]):
        ax = axs[ri, ci]
        for i, (key, lab, col, h) in enumerate(COLS[1:]):
            dd = {k: st(r[key]) - st(r["0"]) for k, r in ALL[nf].items()}; d = np.array(list(dd.values())); vbox(ax, i, d, col, h)
            print("pair N_F %2d %-6s %-38s mean %+5.2f +- %.2f [geo %d/%d] median %+5.2f better %2.0f%% worse %2.0f%% same %2.0f%%" % ((nf, key, sl) + cl(dd) + (np.median(d), 100 * (d < -0.01).mean(), 100 * (d > 0.01).mean(), 100 * (abs(d) <= 0.01).mean())))
        ax.axvline(3.5, color="0.7", lw=0.8, ls=":"); ax.set_xticks(range(len(COLS) - 1)); ax.set_xticklabels([c[1] for c in COLS[1:]], fontsize=10); ax.set_title("%s, %d samples" % (sl, nf), fontsize=11); style(ax)
        if ri == 1: ax.set_xlabel("Radius of the retailer-belief ambiguity set")
    axs[ri, 0].set_ylabel("Difference to Nominal per run (% of cost)")
axs[0, 1].legend(handles=HAND[1:], loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + os.environ.get("BE_PRE", "tk_eps") + "_paired_boxplot_%s.png" % TAGF, dpi=200); plt.close(fig)
