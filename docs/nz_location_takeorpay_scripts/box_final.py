# final summary boxplots (TK world, take-or-pay): sites only at opening cost 750 vs sites + per-site capacity tiers (c/v 0.4, 0.5)
import csv, collections, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/"
DS = ["0.05", "0.1", "0.2", "0.3"]; q90 = lambda v: np.quantile(v, 0.9)
MOD = [("N", "Nominal", "#0072B2", None), ("B1", "Belief-only (delta = 0.1)", "#CC79A7", "..."), ("BS", "Belief-only (calibrated delta)", "#009E73", "///"), ("CL", "Retailer belief known (best decision per belief)", "#7F7F7F", "xxx")]
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else None
def pick(v, A, M):
    tied = np.where(v <= v.min() + 1e-6)[0]; n = int(tied[np.argmin([q90(M[i]) for i in tied])]); sc = abs(v[n]) / 100
    bq = q90(M[n]) - v[n]; k = next((i for i in range(4) if A[n, i] >= bq - 1e-9), 3)
    dec = {"N": n, "B1": int(np.argmin(v + A[:, 1])), "BS": int(np.argmin(v + A[:, k]))}
    out = {m: (M[i] - M[n].mean()) / sc for m, i in dec.items()}; out["CL"] = (M.min(axis=0) - M[n].mean()) / sc; return out
def loc_only(g, s, nf, Q=750):   # as box_q.py
    C = collections.OrderedDict()
    for cfg in ["top", "topr2", "topr1"]:
        T = rd(O + "tk/tk_%s_g%d_s%d.csv" % (cfg, g, s)); D = rd(O + "tkd/tkd_%s_g%d_s%d.csv" % (cfg, g, s)); J = collections.defaultdict(list)
        for r in (D or []):
            if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        if not J and cfg == "top":
            for r in rd(O + "tkd/tkd_topnf%d_g%d_s%d.csv" % (nf, g, s)):
                if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        for r in T:
            if r["x"] not in C: C[r["x"]] = (float(r["VN"]), np.maximum.accumulate([float(r["adv" + d]) for d in DS]), np.array(J[r["x"]]), r["x"].count(";") + 1)
    n_ = np.array([t[3] for t in C.values()]); add = (Q - 500) * n_
    return pick(np.array([t[0] for t in C.values()]) + add, np.array([t[1] for t in C.values()]), np.array([t[2] for t in C.values()]) + add[:, None])
def tier(g, s, nf, q0, c):       # as box_tier.py
    T = []; J = collections.defaultdict(list)
    for cf in ["toptier", "toptier2"]:
        T += rd(O + "tk/tk_%s_g%d_s%d.csv" % (cf, g, s))
        for r in rd(O + "tkd/tkd_%s_g%d_s%d.csv" % (cf, g, s)):
            if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
    def cst(x): pr = [p.split(":") for p in x.strip("{}").split(";")]; return q0 * len(pr) + c * sum(float(b) for _, b in pr)
    cost = np.array([cst(r["x"]) for r in T] + [0.0]); nd = len(J[T[0]["x"]])
    v = np.array([float(r["V0"]) for r in T] + [0.0]) + cost; A = np.array([np.maximum.accumulate([float(r["adv" + d]) for d in DS]) for r in T] + [np.zeros(4)])
    return pick(v, A, np.array([J[r["x"]] for r in T] + [np.zeros(nd)]) + cost[:, None])
GR = [("Sites only\n(opening cost 750)", lambda g, s, nf: loc_only(g, s, nf)), ("Sites + capacity tiers\n(c/v = 0.4)", lambda g, s, nf: tier(g, s, nf, 150.0, 2.0)), ("Sites + capacity tiers\n(c/v = 0.5)", lambda g, s, nf: tier(g, s, nf, 0.0, 2.5))]
def vbox(ax, x, d, col, hatch, w=0.2, cw=0.1):
    q05, q25, med, q75, q95 = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
    ax.plot([x, x], [q05, q25], color=col, lw=1.5, ls="--"); ax.plot([x, x], [q75, q95], color=col, lw=1.5, ls="--")
    ax.plot([x - cw, x + cw], [q05, q05], color=col, lw=2); ax.plot([x - cw, x + cw], [q95, q95], color=col, lw=2)
    fc = matplotlib.colors.to_rgba(col, 0.15) if hatch is None else "none"
    ax.add_patch(Rectangle((x - w, q25), 2 * w, q75 - q25, facecolor=fc, edgecolor=col, lw=2, hatch=hatch))
    ax.plot([x - w, x + w], [med, med], color=col, lw=3); ax.plot([x], [d.mean()], marker="D", color=col, ms=6)
def legend(ax, mods): ax.legend(handles=[Patch(facecolor=(matplotlib.colors.to_rgba(col, 0.15) if h is None else "none"), edgecolor=col, lw=2, hatch=h, label=ml) for _, ml, col, h in mods], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)
def style(ax): ax.grid(axis="y", color="0.85"); ax.set_axisbelow(True); ax.axhline(0, color="0.5", lw=0.8)
KEYS = [(g, s) for g in range(1, 16) for s in range(1, 6)]
ALL = {nf: [{k: f(k[0], k[1], nf) for k in KEYS} for _, f in GR] for nf in (5, 20)}
sp = 3.0; off = [-0.75, -0.25, 0.25, 0.75]
fig, axs = plt.subplots(1, 2, figsize=(18.5, 5.0), sharey=True)
for ax, nf in zip(axs, (5, 20)):
    for gi, R in enumerate(ALL[nf]):
        for mi, (m, ml, col, h) in enumerate(MOD):
            d = np.concatenate([r[m] for r in R.values()]); vbox(ax, gi * sp + off[mi], d, col, h)
            print("cost  N_F %2d %-38s %-30s mean %+6.2f q05 %+6.2f q25 %+6.2f med %+6.2f q75 %+6.2f q90 %+6.2f q95 %+6.2f" % (nf, GR[gi][0].replace("\n", " "), ml, d.mean(), *np.quantile(d, [.05, .25, .5, .75, .9, .95])))
    ax.set_xticks([i * sp for i in range(3)]); ax.set_xticklabels([l for l, _ in GR], fontsize=10); ax.set_title("Retailer belief from %d samples" % nf, fontsize=11); style(ax)
axs[0].set_ylabel("Leader cost over retailer beliefs (% of cost)"); legend(axs[1], MOD)
fig.tight_layout(); fig.savefig(OUT + "tk_final_cost_boxplot.png", dpi=200); plt.close(fig)
fig, axs = plt.subplots(2, 2, figsize=(17.0, 8.4), sharey="row"); off2 = [-0.55, 0.0, 0.55]
for ri, nf in enumerate((5, 20)):
    for ci, (sl, st) in enumerate([("Mean over retailer beliefs", np.mean), ("Upper tail (q90) over retailer beliefs", q90)]):
        ax = axs[ri, ci]
        for gi, R in enumerate(ALL[nf]):
            for mi, (m, ml, col, h) in enumerate(MOD[1:]):
                d = np.array([st(r[m]) - st(r["N"]) for r in R.values()]); vbox(ax, gi * sp + off2[mi], d, col, h)
                print("pair  N_F %2d %-38s %-30s %-38s mean %+5.2f median %+5.2f better %2.0f%% worse %2.0f%% same %2.0f%%" % (nf, GR[gi][0].replace("\n", " "), ml, sl, d.mean(), np.median(d), 100 * (d < -0.01).mean(), 100 * (d > 0.01).mean(), 100 * (abs(d) <= 0.01).mean()))
        ax.set_xticks([i * sp for i in range(3)]); ax.set_xticklabels([l for l, _ in GR], fontsize=9); ax.set_title("%s, %d samples" % (sl, nf), fontsize=11); style(ax)
    axs[ri, 0].set_ylabel("Belief-only minus Nominal per run (% of cost)"); legend(axs[ri, 1], MOD[1:])
fig.tight_layout(); fig.savefig(OUT + "tk_final_paired_boxplot.png", dpi=200); plt.close(fig)
