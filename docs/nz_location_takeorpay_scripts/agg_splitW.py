# -*- coding: utf-8 -*-
# Split of the loss N - N* (splitW.jl output geo/tw/{split,eval2}_g<g>.csv + truthW.jl output eval_g<g>.csv): python agg_splitW.py
#   N  = Nominal from the 10 training scenarios (demand and the retailer's commitment both from the sample)
#   Nr = same demand sample, but the leader knows the commitment of the truth-knowing retailer (only the demand error is left)
#   Nd = the leader knows the true demand, but predicts the commitment from the sample (only the order-prediction error is left)
import csv, os, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/tw/"
q90 = lambda v: np.quantile(v, 0.9)
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";")]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def ms(v):
    v = np.array(v, float); return "%+6.2f +- %.2f" % (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else "%+6.2f" % v.mean()
G = {}
for g in range(1, 16):
    lg = O + "sp_g%d.log" % g
    if not (os.path.exists(lg) and "SP_DONE" in open(lg, encoding="utf-8", errors="replace").read()): continue
    J = collections.defaultdict(lambda: collections.defaultdict(dict))
    for fn in ("eval_g%d.csv", "eval2_g%d.csv"):
        for r in csv.DictReader(open(O + fn % g, encoding="utf-8")): J[r["x"]][r["world"]][int(r["k"])] = float(r["J"])
    G[g] = (list(csv.DictReader(open(O + "split_g%d.csv" % g, encoding="utf-8"))), J)
print("geographies done: %d %s" % (len(G), sorted(G)))
bad = [(g, r["seed"], r["scen"]) for g, (sp, _) in G.items() for r in sp if r["who"] == "N" and r["okN"] != "true"]; dm = max(float(r["dmax"]) for sp, _ in G.values() for r in sp)
print("check: recomputed N differs from stored N in %d (geo, seed, scen) cases %s; max |V_ss - stored| = %.5f" % (len(bad), bad[:6], dm))
for scen in ("1", "2"):
    if not G: break
    r0 = [r for r in next(iter(G.values()))[0] if r["scen"] == scen][0]; q0, c = float(r0["q0"]), float(r0["c"]); cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1]); cap = lambda x: sum(par(x)[1])
    print("\n==================== c/v %.1f" % (c / 5))
    D = {}
    for g, (sp, J) in G.items():
        d = collections.defaultdict(dict)
        for r in sp:
            if r["scen"] == scen: d[int(r["seed"])][r["who"]] = r
        def val(x, w, J=J): return np.array([J[x][w][k] for k in sorted(J[x][w])]) + cost(x)
        D[g] = (d, val, abs(val(d[1]["N*"]["x"], "T")[0]) / 100)
    X = lambda g, s, w: D[g][0][s][w]["x"]
    n = sum(len(d) for d, _, _ in D.values())
    print("runs %d | N = N* %.0f%% | Nr = N* %.0f%% | Nd = N* %.0f%% | Nr = N %.0f%% | Nd = N %.0f%%" % ((n,) + tuple(100 * np.mean([X(g, s, a) == X(g, s, b) for g in D for s in D[g][0]]) for a, b in (("N", "N*"), ("Nr", "N*"), ("Nd", "N*"), ("Nr", "N"), ("Nd", "N")))))
    for w, wl in (("T", "retailer knows the truth"), ("5", "retailer own sample of 5"), ("20", "retailer own sample of 20")):
        print("-- %s" % wl)
        for lab, a in (("N  - N* (both errors)", "N"), ("Nr - N* (demand error only)", "Nr"), ("Nd - N* (order error only)", "Nd")):
            m = {g: np.array([(D[g][1](X(g, s, a), w).mean() - D[g][1](X(g, s, "N*"), w).mean()) / D[g][2] for s in sorted(D[g][0])]) for g in D}; al = np.concatenate(list(m.values()))
            print("   %-30s mean %s | q90 over samples %s | max %5.1f | runs >= 2%%: %3.0f%%" % (lab, ms([v.mean() for v in m.values()]), ms([q90(v) for v in m.values()]), al.max(), 100 * np.mean(al >= 2)))
    # ---- how the leader's estimate ranks N against N* (% of N* cost; negative = N looks better than N*). V_B gap = the loss on the 100-scenario truth
    gp = lambda g, s, col: (float(D[g][0][s]["N"][col]) + cost(X(g, s, "N")) - float(D[g][0][s]["N*"][col]) - cost(X(g, s, "N*"))) / D[g][2]
    rows = np.array([[gp(g, s, col) for col in ("Vss", "Vrs", "Vds", "VB")] for g in D for s in sorted(D[g][0])]); dif = rows[:, 3] > 1e-9
    print("-- estimated cost of N minus N* under each view (runs with N != N*: %d)" % dif.sum())
    for lab, j in (("own sample for demand and orders (what N sees)", 0), ("own demand sample, true orders", 1), ("true demand, orders predicted from the sample", 2), ("truth for both (actual loss, 100 scenarios)", 3)):
        print("   %-50s all runs %+6.2f | runs with N != N* %+6.2f" % (lab, rows[:, j].mean(), rows[dif, j].mean()))
    tot = rows[:, 3] - rows[:, 0]
    print("   misranking (truth - what N sees) %+.2f = orders first: orders %+.2f + demand %+.2f | demand first: demand %+.2f + orders %+.2f" % (tot.mean(), (rows[:, 1] - rows[:, 0]).mean(), (rows[:, 3] - rows[:, 1]).mean(), (rows[:, 2] - rows[:, 0]).mean(), (rows[:, 3] - rows[:, 2]).mean()))
    # ---- predicted and actual total commitment at the leader's sites, for the decision N
    hh = np.array([[float(D[g][0][s]["N"][col]) for col in ("hB_pred", "hB_true", "hB_l1")] + [cap(X(g, s, "N")), rows_i] for (g, s), rows_i in zip([(g, s) for g in D for s in sorted(D[g][0])], rows[:, 3])])
    big = hh[:, 4] >= 2
    print("-- commitment at the leader's sites for N: predicted %.0f, actual %.0f (capacity %.0f); predicted - actual: mean %+.1f, mean |.| %.1f, site-by-site |.| %.1f" % (hh[:, 0].mean(), hh[:, 1].mean(), hh[:, 3].mean(), (hh[:, 0] - hh[:, 1]).mean(), np.abs(hh[:, 0] - hh[:, 1]).mean(), hh[:, 2].mean()))
    for lab, m in (("loss >= 2%", big), ("loss < 2%", ~big)):
        if m.sum(): print("      %-10s %3d runs: predicted - actual %+6.1f | |.| %5.1f | site-by-site |.| %5.1f" % (lab, m.sum(), (hh[m, 0] - hh[m, 1]).mean(), np.abs(hh[m, 0] - hh[m, 1]).mean(), hh[m, 2].mean()))
    print("      correlation of the loss with: predicted - actual %+.2f, |predicted - actual| %+.2f, site-by-site |.| %+.2f" % (np.corrcoef(hh[:, 4], hh[:, 0] - hh[:, 1])[0, 1], np.corrcoef(hh[:, 4], np.abs(hh[:, 0] - hh[:, 1]))[0, 1], np.corrcoef(hh[:, 4], hh[:, 2])[0, 1]))
