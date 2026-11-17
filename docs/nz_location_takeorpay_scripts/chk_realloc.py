# How do the decisions differ from Nominal site by site?  config dwps3c (box r/sqrt(S)), scen 2
import csv, os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/dw/"
def par(x): return {int(p.split(":")[0]): float(p.split(":")[1]) for p in x.strip("{}").split(";") if p}
R = []
for g in range(1, 16):
    for s in range(1, 11):
        f = O + "sel_dwps3c_g%d_s%d.csv" % (g, s)
        if not os.path.exists(f): f = O + "sel_dwps3_g%d_s%d.csv" % (g, s)
        if not os.path.exists(f): continue
        d = {}
        for l in open(f, encoding="utf-8").read().splitlines()[1:]:
            p = l.split(",", 4)
            if p[0] == "2": d[p[3]] = par(p[4])
        R.append(d)
print("runs", len(R))
for w in ("B0.1", "D0.1", "D0.2", "C"):
    dif = [(r["N"], r[w]) for r in R if w in r and r[w] != r["N"]]; n = len(dif); cat = {"only down (some site smaller or closed, none larger or new)": 0, "only up": 0, "both: some site larger/new and some smaller/closed": 0}
    tot_down = tot_same = 0
    for a, b in dif:
        ks = set(a) | set(b); up = any(b.get(k, 0) > a.get(k, 0) for k in ks); dn = any(b.get(k, 0) < a.get(k, 0) for k in ks)
        cat["both: some site larger/new and some smaller/closed" if up and dn else "only down (some site smaller or closed, none larger or new)" if dn else "only up"] += 1
    print("%-5s differs %3d: " % (w, n) + " | ".join("%s %.0f%%" % (k, 100 * v / n) for k, v in cat.items()))
