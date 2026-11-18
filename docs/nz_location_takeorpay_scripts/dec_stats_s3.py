# -*- coding: utf-8 -*-
# decision statistics from zsW dec files: python dec_stats.py <dir> [scen]   (c/f 0.4 = scen 2 by default)
import csv, sys, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/" + sys.argv[1] + "/"
SC = sys.argv[2] if len(sys.argv) > 2 else "2"
def par(x): return {int(a): float(b) for a, b in (p.split(":") for p in x.strip("{}").split(";") if p)}
R = collections.defaultdict(dict)
for g in range(1, 16):
    for s in range(1, 4):
        try: rows = list(csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")))
        except FileNotFoundError: continue
        for r in rows:
            if r["scen"] == SC: R[(g, s)][r["who"]] = par(r["x"])
print("runs %d, scen %s" % (len(R), SC))
WHO = [w for w in ("N", "B0.1", "B0.2", "D0.1", "D0.2", "C") if all(w in d for d in R.values())]
for w in WHO:
    cap = np.array([sum(d[w].values()) for d in R.values()]); ns = np.array([len(d[w]) for d in R.values()]); u4 = np.mean([400.0 in d[w].values() for d in R.values()])
    dif = [k for k, d in R.items() if d[w] != d["N"]]
    line = "%-5s cap %5.0f  sites %.2f  uses400 %3.0f%%  differs %3.0f%% (%d)" % (w, cap.mean(), ns.mean(), 100 * u4, 100 * len(dif) / len(R), len(dif))
    if dif and w != "N":
        dc = np.array([sum(R[k][w].values()) - sum(R[k]["N"].values()) for k in dif])
        same_sites = np.mean([set(R[k][w]) == set(R[k]["N"]) for k in dif])
        # site-level: some site up and some down (over the union of sites)
        mix = 0
        for k in dif:
            a, b = R[k][w], R[k]["N"]; d = [a.get(i, 0) - b.get(i, 0) for i in set(a) | set(b)]
            mix += (max(d) > 0 and min(d) < 0)
        line += " | among differing: cap less %3.0f%% same %3.0f%% more %3.0f%%, mean dcap %+.0f, same sites %3.0f%%, some up & some down %3.0f%%" % (
            100 * np.mean(dc < 0), 100 * np.mean(dc == 0), 100 * np.mean(dc > 0), dc.mean(), 100 * same_sites, 100 * mix / len(dif))
    print(line)
for a, b in (("C", "D0.1"), ("C", "B0.1"), ("B0.1", "B0.2"), ("D0.1", "D0.2")):
    if a in WHO and b in WHO: print("%s == %s: %3.0f%%" % (a, b, 100 * np.mean([d[a] == d[b] for d in R.values()])))
