import os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
K = ["i", "kind", "sreg", "cap", "h", "j", "creg", "dist", "mu", "yR", "yS"]
shA = []; nearA = []; shA_near = []; shA_far = []; useA = []; useB = []; resA = []; spotA = []; resB = []; spotB = []; hA = []; hB = []
for g in range(1, 16):
    for s in range(1, 4):
        R = [dict(zip(K, l.split(";")[-11:])) for l in open(O + "flow_g%d_s%d.csv" % (g, s), encoding="utf-8").read().splitlines()[1:]]
        for r in R: r["yR"] = float(r["yR"]); r["yS"] = float(r["yS"]); r["y"] = r["yR"] + r["yS"]; r["dist"] = float(r["dist"])
        A = [r for r in R if r["kind"] == "A"]; B = [r for r in R if r["kind"] == "B"]; ta = sum(r["y"] for r in A); tb = sum(r["y"] for r in B); shA.append(ta / (ta + tb))
        capA = sum({r["i"]: float(r["cap"]) for r in A}.values()); capB = sum({r["i"]: float(r["cap"]) for r in B}.values()); useA.append(ta / capA); useB.append(tb / capB)
        resA.append(sum(r["yR"] for r in A)); spotA.append(sum(r["yS"] for r in A)); resB.append(sum(r["yR"] for r in B)); spotB.append(sum(r["yS"] for r in B))
        hA.append(sum({r["i"]: float(r["h"]) for r in A}.values())); hB.append(sum({r["i"]: float(r["h"]) for r in B}.values()))
        for j in set(r["j"] for r in R):
            rj = [r for r in R if r["j"] == j]; tj = sum(r["y"] for r in rj); near = min(rj, key=lambda r: r["dist"]); a = sum(r["y"] for r in rj if r["kind"] == "A") / tj
            nearA.append(near["kind"] == "A"); (shA_near if near["kind"] == "A" else shA_far).append(a)
m = np.mean
print("existing operator's share of units: mean %.0f%%, min %.0f%%, max %.0f%% over %d runs" % (100 * m(shA), 100 * min(shA), 100 * max(shA), len(shA)))
print("customers whose nearest open site is an existing-operator site: %.0f%%; they buy %.0f%% from it; the other customers buy %.0f%% from it" % (100 * m(nearA), 100 * m(shA_near), 100 * m(shA_far)))
print("capacity used on average: existing operator %.0f%%, leader %.0f%%" % (100 * m(useA), 100 * m(useB)))
print("existing operator: commitment %.0f, shipped from commitment %.0f, at-once %.0f | leader: commitment %.0f, shipped from commitment %.0f, at-once %.0f" % (m(hA), m(resA), m(spotA), m(hB), m(resB), m(spotB)))
