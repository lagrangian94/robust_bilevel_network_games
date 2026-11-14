# zero-sum style boxplots, leader knows the true distribution (TK). per-draw costs from geo/tkd, decisions from geo/tk summaries.
import csv, glob, re, collections, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/"
CFG = [("base", "Manuscript"), ("top", "Take-or-pay"), ("topkd100", "Take-or-pay, kappa = 100"), ("topkd200", "Take-or-pay, kappa = 200"), ("topfr50", "Take-or-pay, f/p = 0.5")]
MOD = [("N", "Nominal", "#0072B2", None), ("B1", "Belief-only (delta = 0.1)", "#CC79A7", "..."), ("BS", "Belief-only (calibrated delta)", "#009E73", "///"), ("OR", "Tail oracle", "#7F7F7F", "xxx")]
DS = ["0.05", "0.1", "0.2", "0.3"]
def load(cfg, nf):
    R = {}
    for f in sorted(glob.glob(O + "tkd/tkd_%s_g*_s*.csv" % cfg)):
        m = re.search(r"_g(\d+)_s(\d+)\.csv", f); key = (int(m.group(1)), int(m.group(2)))
        if not os.path.exists(f[:-4] + ".log") or "TKD_DONE" not in open(f[:-4] + ".log", encoding="utf-8", errors="replace").read(): continue
        T = list(csv.DictReader(open(O + "tk/tk_%s_g%d_s%d.csv" % ((cfg,) + key), encoding="utf-8")))
        xs = [r["x"] for r in T]; VN = np.array([float(r["VN"]) for r in T])
        adv = np.maximum.accumulate(np.array([[float(r["adv" + d]) for d in DS] for r in T]), axis=1); bq = np.array([float(r["bq%d" % nf]) for r in T])
        J = collections.defaultdict(list)
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        full = (int(np.argmin(VN)), int(np.argmin(VN + adv[:, 1])))
        keep = [i for i, x in enumerate(xs) if len(J[x]) > 0]   # a tied 8th candidate can differ between the two runs
        if len(keep) < len(xs):
            if not all(i in keep for i in full): print("    decision outside common candidates, skipped", cfg, key); continue
            xs = [xs[i] for i in keep]; VN = VN[keep]; adv = adv[keep]; bq = bq[keep]
        J = np.array([J[x] for x in xs])                      # candidates x draws
        n = int(np.argmin(VN)); k = next((i for i in range(4) if adv[n, i] >= bq[n] - 1e-9), 3)
        dec = {"N": n, "B1": int(np.argmin(VN + adv[:, 1])), "BS": int(np.argmin(VN + adv[:, k])), "OR": int(np.argmin(np.quantile(J, 0.9, axis=1)))}
        sc = abs(J.mean(axis=1).min()) / 100
        R[key] = {m_: (J[i] - J[n].mean()) / sc for m_, i in dec.items()}
    return R
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
q90 = lambda v: np.quantile(v, 0.9)
for nf in [5, 10]:
    ALL = {c: load(c, nf) for c, _ in CFG}; cfgs = [(c, l) for c, l in CFG if len(ALL[c]) >= 72]
    print("N_F", nf, {c: len(ALL[c]) for c, _ in CFG})
    sp = 3.0; off = [-0.75, -0.25, 0.25, 0.75]
    # figure 1: pooled cost distribution over retailer belief draws
    fig, ax = plt.subplots(figsize=(2.6 * len(cfgs) + 2.6, 4.8))
    for gi, (c, lab) in enumerate(cfgs):
        for mi, (m, ml, col, h) in enumerate(MOD):
            d = np.concatenate([r[m] for r in ALL[c].values()]); vbox(ax, gi * sp + off[mi], d, col, h)
            print("  pooled %-9s %-30s mean %+6.2f q75 %+6.2f q90 %+6.2f q95 %+6.2f" % (c, ml, d.mean(), np.quantile(d, .75), q90(d), np.quantile(d, .95)))
    ax.set_xticks([i * sp for i in range(len(cfgs))]); ax.set_xticklabels([l for _, l in cfgs], fontsize=10); style(ax)
    ax.set_ylabel("Leader cost over retailer beliefs (% of cost)"); legend(ax, MOD)
    fig.tight_layout(); fig.savefig(OUT + "tk_cost_boxplot_nf%d.png" % nf, dpi=200); plt.close(fig)
    # figure 2: per-run paired differences to Nominal
    fig, axs = plt.subplots(1, 2, figsize=(2.3 * len(cfgs) * 2 + 3.2, 4.8), sharey=True); off3 = [-0.5, 0.0, 0.5]
    for ax, (sl, st) in zip(axs, [("Mean over retailer beliefs", np.mean), ("q90 over retailer beliefs", q90)]):
        for gi, (c, lab) in enumerate(cfgs):
            for mi, (m, ml, col, h) in enumerate(MOD[1:]):
                d = np.array([st(r[m]) - st(r["N"]) for r in ALL[c].values()]); vbox(ax, gi * sp + off3[mi], d, col, h)
                print("  paired %-27s %-9s %-30s mean %+5.2f median %+5.2f better %2.0f%% worse %2.0f%% same %2.0f%%" % (sl, c, ml, d.mean(), np.median(d), 100 * (d < -0.01).mean(), 100 * (d > 0.01).mean(), 100 * (abs(d) <= 0.01).mean()))
        ax.set_xticks([i * sp for i in range(len(cfgs))]); ax.set_xticklabels([l.replace(", ", ",\n") for _, l in cfgs], fontsize=9); ax.set_title(sl, fontsize=11); style(ax)
    axs[0].set_ylabel("Difference to Nominal per run (% of cost)"); legend(axs[1], MOD[1:])
    fig.tight_layout(); fig.savefig(OUT + "tk_paired_boxplot_nf%d.png" % nf, dpi=200); plt.close(fig)
    # figure 3: per geography (5 seeds pooled), Nominal vs Belief-only (calibrated), manuscript vs take-or-pay
    if nf == 5 and all(len(ALL[c]) == 75 for c in ["base", "top"]):
        fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        for ax, c, lab in [(axs[0], "base", "Manuscript"), (axs[1], "top", "Take-or-pay")]:
            for g in range(1, 16):
                for mi, (m, ml, col, h) in enumerate([MOD[0], MOD[2]]):
                    d = np.concatenate([r[m] for (gg, s), r in ALL[c].items() if gg == g]); vbox(ax, g * 1.6 + [-0.28, 0.28][mi], d, col, h)
            ax.set_ylabel(lab + ": cost (% of cost)"); style(ax)
        axs[1].set_xticks([g * 1.6 for g in range(1, 16)]); axs[1].set_xticklabels(["geo %d" % g for g in range(1, 16)]); legend(axs[0], [MOD[0], MOD[2]])
        fig.tight_layout(); fig.savefig(OUT + "tk_cost_by_geo_nf5.png", dpi=170); plt.close(fig)
