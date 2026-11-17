# -*- coding: utf-8 -*-
# Is "retailer keeps believing q̂" optimistic for the leader when the truth is p?  compare J(x; p, q̂) (Vds) with J(x; p, p) (Vpp) for the stored decisions,
# split by whether the retailer who knows p commits less than predicted under q̂ (hB_true < hB_pred).  data: geo/onp (price-based) or geo/on (old), d > 0.
import csv, os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
for d_, lab in (("onp", "price-based"), ("on", "old formula")):
    for who in ("N", "D0.3"):
        rows = []
        for g in range(1, 16):
            for s in range(1, 11):
                f = O + d_ + "/dec_g%d_s%d.csv" % (g, s)
                if not os.path.exists(f): continue
                R = list(csv.DictReader(open(f, encoding="utf-8")))
                sc = {r["scen"]: abs(float(r["Vss"]) + float(r["q0"]) * 0 ) for r in R if r["who"] == "N"}   # scale ~ |V_ss(N)| (store cost omitted: same on both sides)
                for r in R:
                    if r["who"] != who or float(r["d"]) == 0 or r["scen"] != "2": continue
                    rows.append((float(r["d"]), (float(r["Vds"]) - float(r["Vpp"])) / sc[r["scen"]] * 100, float(r["hB_true"]) - float(r["hB_pred"])))
        A = np.array(rows)
        print("== %s, decision %s, c/f 0.4: %d (truth, run) pairs.  diff = J(x; p, q̂) - J(x; p, p) in %% (negative = believing q̂ is better for the leader than knowing p)" % (lab, who, len(A)))
        for nm, m in (("all", np.ones(len(A), bool)), ("retailer who knows p commits LESS (by > 5)", A[:, 2] < -5), ("commits MORE (by > 5)", A[:, 2] > 5), ("about the same", np.abs(A[:, 2]) <= 5)):
            if m.sum() == 0: continue
            print("   %-44s n %4d | mean diff %+5.2f | share diff < 0: %3.0f%% | share diff > 0: %3.0f%%" % (nm, m.sum(), A[m, 1].mean(), 100 * np.mean(A[m, 1] < -1e-6), 100 * np.mean(A[m, 1] > 1e-6)))
        for d in (0.1, 0.2, 0.3):
            m = (A[:, 0] == d) & (A[:, 2] < -5); m2 = (A[:, 0] == d) & (A[:, 2] > 5)
            print("   d %.1f: commits less: mean diff %+5.2f (n %d) | commits more: mean diff %+5.2f (n %d)" % (d, A[m, 1].mean(), m.sum(), A[m2, 1].mean(), m2.sum()))
