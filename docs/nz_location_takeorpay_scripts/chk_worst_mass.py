# Where does the worst-case distribution of Data-DR put its probability?  Decision = Nominal's (scen 2), belief q.  Scenarios ranked by total demand (xi_g.csv).
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
rng = np.random.default_rng(1); S = 10
def worst_a(Z, r, box):          # greedy = LP optimum for CVaR_0.4: add mass to the worst scenarios (box per scenario), remove it from the best ones
    a = np.full(S, 0.1); o = np.argsort(-Z); rest = r
    for s in o:
        m = min(box, rest); a[s] += m; rest -= m
        if rest <= 1e-12: break
    rest = r
    for s in o[::-1]:
        m = min(box, rest, a[s]); a[s] -= m; rest -= m
        if rest <= 1e-12: break
    return a
res = {k: [] for k in ("add3", "add5", "rem3", "rem5", "Etot", "ov3", "ov6", "floor3", "floor5", "nfl")}
for g in range(1, 16):
    T = {}
    for r in csv.reader(open(O + "xi_g%d.csv" % g)):
        if r[1] == "tot": T[int(r[0])] = np.array([float(x) for x in r[2:]])
    for s in range(1, 11):
        dec = {r["who"]: r for r in csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        for r in csv.DictReader(open(O + "mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
            if r["x"] == dec["N"]["x"] and r["bel"] == "q": Z = np.array([float(r["Z%d" % k]) for k in range(1, 11)]); hB = float(r["hB"]); v = float(r["v"])
        t = T[s]; lowrank = np.argsort(np.argsort(t))        # 0 = lowest total demand
        fl = (-Z) <= v * hB + 1e-4 * v * hB + 1e-6; res["nfl"].append(fl.sum())
        if fl.any(): res["floor3"].append(np.mean(lowrank[fl] < 3)); res["floor5"].append(np.mean(lowrank[fl] < 5))
        acc = {k: [] for k in ("add3", "add5", "rem3", "rem5", "Etot", "ov3", "ov6")}
        for _ in range(200):                                  # ties broken at random
            Zr = Z + 1e-7 * abs(Z).max() * rng.standard_normal(S); a = worst_a(Zr, 0.1, 0.2 / np.sqrt(S)); d = a - 0.1
            acc["add3"].append(d[lowrank < 3].clip(0).sum() / 0.1); acc["add5"].append(d[lowrank < 5].clip(0).sum() / 0.1)
            acc["rem3"].append((-d[lowrank >= 7]).clip(0).sum() / 0.1); acc["rem5"].append((-d[lowrank >= 5]).clip(0).sum() / 0.1)
            acc["Etot"].append(100 * (a @ t / t.mean() - 1)); w3 = np.argsort(-Zr)[:3]; w6 = np.argsort(-Zr)[:6]
            acc["ov3"].append(np.mean(lowrank[w3] < 3)); acc["ov6"].append(np.mean(lowrank[w6] < 6))
        for k in acc: res[k].append(np.mean(acc[k]))
m = lambda k: np.mean(res[k])
print("150 runs, Nominal's decision, Data-DR radius 0.1 (old box 0.063 per scenario)")
print("added mass 0.1: share going to the 3 lowest-total-demand scenarios %.0f%%, to the 5 lowest %.0f%%  (random: 30%%, 50%%)" % (100 * m("add3"), 100 * m("add5")))
print("removed mass 0.1: share taken from the 3 highest-total scenarios %.0f%%, from the 5 highest %.0f%%" % (100 * m("rem3"), 100 * m("rem5")))
print("expected total demand under the worst distribution vs under q: %+.1f%%" % m("Etot"))
print("leader's 3 lowest-revenue scenarios that are among the 3 lowest-total: %.0f%% (random 30%%);  6 lowest-revenue among the 6 lowest-total: %.0f%% (random 60%%)" % (100 * m("ov3"), 100 * m("ov6")))
print("floor scenarios (only the commitment sells): %.1f per run; among the 3 lowest-total %.0f%%, among the 5 lowest-total %.0f%%" % (m("nfl"), 100 * m("floor3"), 100 * m("floor5")))
