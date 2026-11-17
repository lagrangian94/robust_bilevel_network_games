# decisions of Coupled so far (config dwps3c: N, B0.1, B0.2, D0.1, D0.2 with box r/sqrt(S); C = Coupled eps 0.1/0.1, delta 0.033), cost table scen 2
import csv, os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/dw/"
def par(x): return {int(p.split(":")[0]): float(p.split(":")[1]) for p in x.strip("{}").split(";") if p}
for scen in ("2", "1"):
    R = []
    for g in range(1, 16):
        for s in range(1, 11):
            f = O + "sel_dwps3c_g%d_s%d.csv" % (g, s)
            if not os.path.exists(f): continue
            d = {}
            for l in open(f, encoding="utf-8").read().splitlines()[1:]:
                p = l.split(",", 4)
                if p[0] == scen: d[p[3]] = par(p[4])
            if "C" in d: R.append((g, s, d))
    W = ("N", "B0.1", "B0.2", "D0.1", "D0.2", "C"); n = len(R); seeds = sorted(set(r[1] for r in R))
    print("scen %s: %d runs (seeds %s)" % (scen, n, seeds))
    for w in W:
        cap = np.mean([sum(r[2][w].values()) for r in R]); ns = np.mean([len(r[2][w]) for r in R]); dif = [r for r in R if r[2][w] != r[2]["N"]]
        if w == "N": print("  %-5s cap %4.0f sites %.2f" % (w, cap, ns)); continue
        k = max(len(dif), 1); cn = lambda r: sum(r[2]["N"].values()); cw = lambda r: sum(r[2][w].values())
        print("  %-5s cap %4.0f sites %.2f | differs from N %3.0f%% | of those: same sites %2.0f%%, cap less %2.0f%% same %2.0f%% more %2.0f%%" % (w, cap, ns, 100 * len(dif) / n,
              100 * sum(set(r[2][w]) == set(r[2]["N"]) for r in dif) / k, 100 * sum(cw(r) < cn(r) for r in dif) / k, 100 * sum(cw(r) == cn(r) for r in dif) / k, 100 * sum(cw(r) > cn(r) for r in dif) / k))
    for w in ("N", "B0.1", "B0.2", "D0.1", "D0.2"): print("  C same decision as %-5s %3.0f%%" % (w, 100 * np.mean([r[2]["C"] == r[2][w] for r in R])))
