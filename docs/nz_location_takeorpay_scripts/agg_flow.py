import csv, os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
oth = []; ncust = []; dB = []; dA = []; shareB = []; nregS = []; shB_own = []; shB_oth = []; nearest = []; hshare = []
for g in range(1, 16):
    for s in range(1, 4):
        f = O + "flow_g%d_s%d.csv" % (g, s)
        if not os.path.exists(f): continue
        K = ["i", "kind", "sreg", "cap", "h", "j", "creg", "dist", "mu", "yR", "yS"]   # the decision label itself contains ";": read the last 11 fields
        R = [dict(zip(K, l.split(";")[-11:])) for l in open(f, encoding="utf-8").read().splitlines()[1:]]
        for r in R: r["y"] = float(r["yR"]) + float(r["yS"]); r["dist"] = float(r["dist"])
        B = [r for r in R if r["kind"] == "B"]; A = [r for r in R if r["kind"] == "A"]; tb = sum(r["y"] for r in B); ta = sum(r["y"] for r in A)
        oth.append(sum(r["y"] for r in B if r["sreg"] != r["creg"]) / tb); shareB.append(tb / (tb + ta))
        dB.append(sum(r["y"] * r["dist"] for r in B) / tb); dA.append(sum(r["y"] * r["dist"] for r in A) / max(ta, 1e-9))
        for i in set(r["i"] for r in B):
            ri = [r for r in B if r["i"] == i]; ti = sum(r["y"] for r in ri)
            if ti <= 0: continue
            ncust.append(sum(r["y"] > 0.02 * ti for r in ri)); nregS.append(len(set(r["creg"] for r in ri if r["y"] > 0.02 * ti)))
        breg = set(r["sreg"] for r in B)
        for j in set(r["j"] for r in R):
            rj = [r for r in R if r["j"] == j]; tj = sum(r["y"] for r in rj); bj = sum(r["y"] for r in rj if r["kind"] == "B") / tj
            (shB_own if rj[0]["creg"] in breg else shB_oth).append(bj)
            # is the customer supplied mainly by its nearest open site?
            bysite = {}
            for r in rj: bysite[r["i"]] = bysite.get(r["i"], 0) + r["y"]
            near = min(rj, key=lambda r: r["dist"])["i"]; nearest.append(bysite[near] / tj)
n = len(oth); m = np.mean
print("runs %d" % n)
import statistics
print("leader's share of all units sold: %.0f%%" % (100 * m(shareB)))
print("leader's sales going to customers of another region than the store: %.0f%% (median %.0f%%)" % (100 * m(oth), 100 * np.median(oth)))
print("customers served by one leader store (> 2%% of its sales): %.1f, regions served by one leader store: %.2f" % (m(ncust), m(nregS)))
print("average distance of a shipped unit: leader %.0f, existing operator %.0f" % (m(dB), m(dA)))
print("leader's share of a customer's purchases: customers in a region with a leader store %.0f%%, customers in a region without one %.0f%%" % (100 * m(shB_own), 100 * m(shB_oth)))
print("share of a customer's purchases that comes from its nearest open site: %.0f%%" % (100 * m(nearest)))
