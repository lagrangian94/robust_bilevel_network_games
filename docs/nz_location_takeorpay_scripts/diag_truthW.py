# -*- coding: utf-8 -*-
# Where the loss N - N* comes from (truthW.jl output): python diag_truthW.py
import csv, os, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/tw/"
q90 = lambda v: np.quantile(v, 0.9)
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";")]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def ms(v):
    v = np.array(v, float); return "%+6.2f +- %.2f" % (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else "%+6.2f" % v.mean()
G = {}
for g in range(1, 16):
    dec = list(csv.DictReader(open(O + "dec_g%d.csv" % g, encoding="utf-8"))); J = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(open(O + "eval_g%d.csv" % g, encoding="utf-8")): J[r["x"]][r["world"]][int(r["k"])] = float(r["J"])
    G[g] = (dec, J)
for scen in ("1", "2"):
    q0, c = [(float(r["q0"]), float(r["c"])) for r in G[1][0] if r["scen"] == scen][0]; cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1]); cap = lambda x: sum(par(x)[1])
    print("\n==================== c/v %.1f" % (c / 5))
    D = {}
    for g, (dec, J) in G.items():
        d = {"seeds": collections.defaultdict(dict)}
        for r in dec:
            if r["scen"] != scen: continue
            if r["who"] in ("N*", "B*"): d[r["who"]] = r["x"]
            elif r["who"] != "O": d["seeds"][int(r["seed"])][r["who"]] = r["x"]
        def val(x, w, J=J): return np.array([J[x][w][k] for k in sorted(J[x][w])]) + cost(x)
        d["val"] = val; d["sc"] = abs(val(d["N*"], "T")[0]) / 100; D[g] = d
    for w in ("T", "5", "20"):
        print("-- world %s" % w)
        rows = []
        for g, d in D.items():
            for s in sorted(d["seeds"]):
                n = d["seeds"][s]["N"]; ns = d["N*"]; sn, ss = set(par(n)[0]), set(par(ns)[0])
                rows.append((g, s, cap(n) - cap(ns), len(sn) - len(ss), len(ss - sn), (d["val"](n, w).mean() - d["val"](ns, w).mean()) / d["sc"]))
        R = np.array(rows, float); L = R[:, 5]; big = L >= 2
        print("   all runs %d: loss mean %+.2f; share of total loss from runs >= 2%%: %.0f%%" % (len(R), L.mean(), 100 * L[big].sum() / L.sum()))
        for lab, m in (("N = N*", (R[:, 4] == 0) & (R[:, 2] == 0) & (L == 0)), ("same sites, other capacity", (R[:, 4] == 0) & (R[:, 3] == 0) & (L != 0)), ("one site differs", (R[:, 4] == 1) & (R[:, 3] == 0)), ("two+ sites differ", (R[:, 4] >= 2) & (R[:, 3] == 0)), ("fewer sites than N*", R[:, 3] < 0), ("more sites than N*", R[:, 3] > 0)):
            if m.sum(): print("      %-28s %3d runs (%2.0f%%): loss %+5.2f | share of total loss %3.0f%% | cap diff %+5.0f" % (lab, m.sum(), 100 * m.mean(), L[m].mean(), 100 * L[m].sum() / L.sum(), R[m, 2].mean()))
        print("      among loss >= 2%% (%d runs): capacity differs %.0f%%, sites differ %.0f%%, built less %.0f%%, built more %.0f%%" % (big.sum(), 100 * np.mean(R[big, 2] != 0), 100 * np.mean(R[big, 4] > 0), 100 * np.mean(R[big, 2] < 0), 100 * np.mean(R[big, 2] > 0)))
        # per geography: loss mean over seeds
        print("      loss by geography (mean over 10 samples): " + " ".join("%.1f" % R[R[:, 0] == g, 5].mean() for g in sorted(D)))
        if w != "T":
            # B - N, D - N, C - N by seed group, belief tail (q90 over retailer draws) and mean
            for grp, lab in ((lambda s: s <= 3, "samples 1-3"), (lambda s: s > 3, "samples 4-10"), (lambda s: True, "all")):
                out = "      %-12s" % lab
                for mdl in ("B0.1", "D0.3", "C"):
                    a = []; t = []
                    for g, d in D.items():
                        for s in sorted(d["seeds"]):
                            if not grp(s) or mdl not in d["seeds"][s]: continue
                            vm, vn = d["val"](d["seeds"][s][mdl], w), d["val"](d["seeds"][s]["N"], w); a.append((vm.mean() - vn.mean()) / d["sc"]); t.append((q90(vm) - q90(vn)) / d["sc"])
                    if a: out += " | %s - N: mean %s, belief-tail %s (%d)" % (mdl, ms(a), ms(t), len(a))
                print(out)
