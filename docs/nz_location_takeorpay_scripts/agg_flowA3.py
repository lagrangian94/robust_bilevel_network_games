# existing operator's share by the number of leader stores and by leader capacity (45 runs with flows) + how many stores Nominal opens in all 150 runs
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
K = ["i", "kind", "sreg", "cap", "h", "j", "creg", "dist", "mu", "yR", "yS"]; rows = []
for g in range(1, 16):
    for s in range(1, 4):
        R = [dict(zip(K, l.split(";")[-11:])) for l in open(O + "flow_g%d_s%d.csv" % (g, s), encoding="utf-8").read().splitlines()[1:]]
        ta = sum(float(r["yR"]) + float(r["yS"]) for r in R if r["kind"] == "A"); tb = sum(float(r["yR"]) + float(r["yS"]) for r in R if r["kind"] == "B")
        caps = {r["i"]: float(r["cap"]) for r in R if r["kind"] == "B"}; rows.append((len(caps), sum(caps.values()), ta / (ta + tb), ta + tb))
A = np.array(rows)
for k in (1, 2, 3):
    m = A[:, 0] == k
    if m.any(): print("leader opens %d store(s): %2d runs | leader capacity %.0f | total demand %.0f | A share %.0f%% | leader share %.0f%%" % (k, m.sum(), A[m, 1].mean(), A[m, 3].mean(), 100 * A[m, 2].mean(), 100 * (1 - A[m, 2].mean())))
print("corr(leader share, leader capacity / total demand) = %.2f" % np.corrcoef(1 - A[:, 2], A[:, 1] / A[:, 3])[0, 1])
n = {1: 0, 2: 0, 3: 0}; c4 = 0; tot = 0
for g in range(1, 16):
    for s in range(1, 11):
        d = {r["who"]: r for r in csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        x = d["N"]["x"].strip("{}").split(";"); n[len(x)] += 1; tot += 1; c4 += any(p.endswith(":400") for p in x)
print("Nominal, 150 runs: opens 1 store %.0f%%, 2 stores %.0f%%, 3 stores %.0f%%; uses a size-400 store in %.0f%%" % (100 * n[1] / tot, 100 * n[2] / tot, 100 * n[3] / tot, 100 * c4 / tot))
