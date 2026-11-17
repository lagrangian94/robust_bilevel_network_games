# -*- coding: utf-8 -*-
import sys, csv, os
import numpy as np
OLD = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/expo/"
cfg = sys.argv[1]
D = {}
for g in range(1, 16):
    for s in range(1, 11):
        f = OLD + "expoS_%s_g%d_s%d.csv" % (cfg, g, s)
        for r in csv.DictReader(open(f, encoding="utf-8")): D.setdefault(r["cv"], {}).setdefault(g, {})[s] = r
for cv in sorted(D, reverse=True):
    print("== c/v", cv)
    for k in ("1", "2", "3"):
        rN = []; rD = []; dc = []; opt = []
        for g, R in D[cv].items():
            sc = np.mean([abs(float(r["JN"])) for r in R.values()]) / 100
            for r in R.values():
                rN.append((float(r["JN"]) - float(r["Jbest"])) / sc); rD.append((float(r["JD" + k]) - float(r["Jbest"])) / sc)
                dc.append(float(r["capD" + k]) - float(r["capN"])); opt.append((float(r["JN"]) - float(r["VN"])) / sc)
        rN = np.array(rN); rD = np.array(rD); dc = np.array(dc); opt = np.array(opt); d = rD - rN
        print("  eps 0.%s: regret N %.2f D %.2f | D better %d same %d worse %d | cap D<N %d  D>N %d  mean dcap %+.0f" % (k, rN.mean(), rD.mean(), (d < -0.05).sum(), (abs(d) <= 0.05).sum(), (d > 0.05).sum(), (dc < 0).sum(), (dc > 0).sum(), dc.mean()))
        for lab, m in (("N regret>=2", rN >= 2), ("N regret<2", rN < 2), ("optimistic sample (JN-VN>+3%)", opt > 3), ("pessimistic sample (JN-VN<-3%)", opt < -3), ("middle", abs(opt) <= 3)):
            if m.sum(): print("     %-32s n=%3d  regret N %.2f  D %.2f  D-N %+.2f" % (lab, m.sum(), rN[m].mean(), rD[m].mean(), d[m].mean()))
    if cv == "0.2":
        for g in (2, 7):
            print("  g%d:" % g); R = D[cv][g]
            for s in sorted(R): r = R[s]; print("    s%2d VN %8.1f JN %8.1f JD3 %8.1f Jbest %8.1f capN %4s capD3 %4s capbest %s" % (s, float(r["VN"]), float(r["JN"]), float(r["JD3"]), float(r["Jbest"]), r["capN"], r["capD3"], r["xbest"]))
