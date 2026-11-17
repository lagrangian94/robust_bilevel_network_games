# -*- coding: utf-8 -*-
# Tails over training samples: python agg_expoS.py <cfg> [nseeds]. Reads geo/expo/expoS_<cfg>_g<g>_s<s>.csv
import sys, csv, os
import numpy as np
OLD = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/expo/"
cfg = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 10; MINS = int(os.environ.get("AS_MIN", "5"))
D = {}
for g in range(1, 16):
    for s in range(1, NS + 1):
        f = OLD + "expoS_%s_g%d_s%d.csv" % (cfg, g, s)
        if not os.path.exists(f): continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            D.setdefault(r["cv"], {}).setdefault(g, {})[s] = r
def ms(v):
    v = np.array(v, float); return "%+.2f +- %.2f" % (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else "%+.2f" % v.mean()
for cv in sorted(D, reverse=True):
    G = {g: R for g, R in D[cv].items() if len(R) >= MINS}
    n = sum(len(R) for R in G.values())
    print("== c/v %s : %d geos, %d runs" % (cv, len(G), n))
    reg = []; big = {}
    for g, R in G.items():
        sc = np.mean([abs(float(r["JN"])) for r in R.values()]) / 100
        rg = [(float(r["JN"]) - float(r["Jbest"])) / sc for r in R.values()]; reg += rg; big[g] = sum(x >= 2 for x in rg)
    reg = np.array(reg)
    print("  N regret: mean %.2f median %.2f q90 %.2f max %.2f | runs>=2%%: %.0f%% | geos with any>=2%%: %d | per geo count>=2%%: %s" % (
        reg.mean(), np.median(reg), np.quantile(reg, 0.9), reg.max(), 100 * (reg >= 2).mean(), sum(v > 0 for v in big.values()), " ".join("g%d:%d" % (g, big[g]) for g in sorted(big))))
    for k, lab in (("1", "eps0.1"), ("2", "eps0.2"), ("3", "eps0.3")):
        dm = []; dq = []; dw = []; diff = 0; tot = 0
        for g, R in G.items():
            JN = np.array([float(r["JN"]) for r in R.values()]); JD = np.array([float(r["JD" + k]) for r in R.values()]); sc = abs(JN).mean() / 100
            dm.append((JD.mean() - JN.mean()) / sc); dq.append((np.quantile(JD, 0.9) - np.quantile(JN, 0.9)) / sc); dw.append((JD.max() - JN.max()) / sc)
            diff += sum(r["xD" + k] != r["xN"] for r in R.values()); tot += len(R)
        dq = np.array(dq)
        print("  D(%s)-N: mean %s | q90 %s [W %d / L %d / T %d] | worst %s | D!=N %.0f%%" % (lab, ms(dm), ms(dq), (dq < -0.05).sum(), (dq > 0.05).sum(), (abs(dq) <= 0.05).sum(), ms(dw), 100 * diff / tot))
        if k == "3": print("     per geo q90 diff: " + " ".join("g%d:%+.1f" % (g, x) for g, x in zip(sorted(G), [dq[list(G).index(g)] for g in sorted(G)])))
