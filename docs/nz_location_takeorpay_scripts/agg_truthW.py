# -*- coding: utf-8 -*-
# Loss from not knowing the true distribution (truthW.jl output, geo/tw/{dec,eval}_g<g>.csv): python agg_truthW.py
#   N*, B* = decisions of a leader who knows the truth. N, B, D = decisions from 10 training scenarios (seeds 1..10). Costs under the true F.
#   Worlds: T = retailer knows the truth, 5 / 20 = retailer holds its own sample (40 common draws per geography).
import csv, os, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/tw/"
q90 = lambda v: np.quantile(v, 0.9)
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";")]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def ms(v):
    v = np.array(v, float); return "%+6.2f +- %.2f" % (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else "%+6.2f" % v.mean()
G = {}
for g in range(1, 16):
    if not (os.path.exists(O + "dec_g%d.csv" % g) and os.path.exists(O + "tw_g%d.log" % g) and "TW_DONE" in open(O + "tw_g%d.log" % g, encoding="utf-8", errors="replace").read()): continue
    dec = list(csv.DictReader(open(O + "dec_g%d.csv" % g, encoding="utf-8"))); J = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(open(O + "eval_g%d.csv" % g, encoding="utf-8")): J[r["x"]][r["world"]][int(r["k"])] = float(r["J"])
    G[g] = (dec, J)
print("geographies done: %d %s" % (len(G), sorted(G)))
WORLDS = [("T", "retailer knows the truth"), ("5", "retailer own sample of 5"), ("20", "retailer own sample of 20")]
for scen in ("1", "2"):
    if not G: break
    q0, c = [(float(r["q0"]), float(r["c"])) for r in next(iter(G.values()))[0] if r["scen"] == scen][0]; cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1]); cap = lambda x: sum(par(x)[1])
    print("\n==================== c/v %.1f" % (c / 5))
    D = {}
    for g, (dec, J) in G.items():
        d = {"seeds": collections.defaultdict(dict), "O": []}
        for r in dec:
            if r["scen"] != scen: continue
            if r["who"] in ("N*", "B*"): d[r["who"]] = r["x"]
            elif r["who"] == "O": d["O"].append(r["x"])
            else: d["seeds"][int(r["seed"])][r["who"]] = r["x"]
        def val(x, w, J=J): return np.array([J[x][w][k] for k in sorted(J[x][w])]) + cost(x)
        d["val"] = val; d["sc"] = abs(val(d["N*"], "T")[0]) / 100; D[g] = d
    ns = np.mean([len(d["seeds"]) for d in D.values()])
    print("N* = B* in %d of %d geographies; seeds per geography %.1f" % (sum(d["N*"] == d["B*"] for d in D.values()), len(D), ns))
    print("N = N* in %.0f%% of runs, same sites %.0f%%; B = B* in %.0f%%; B = N in %.0f%%; D0.3 = N in %.0f%%" % tuple(100 * np.mean([f(d, s) for d in D.values() for s in d["seeds"]]) for f in (
        lambda d, s: d["seeds"][s]["N"] == d["N*"], lambda d, s: par(d["seeds"][s]["N"])[0] == par(d["N*"])[0], lambda d, s: d["seeds"][s]["B0.1"] == d["B*"], lambda d, s: d["seeds"][s]["B0.1"] == d["seeds"][s]["N"], lambda d, s: d["seeds"][s]["D0.3"] == d["seeds"][s]["N"])))
    for w, wl in WORLDS:
        print("-- %s" % wl)
        def diffs(a, b, stat):   # per geography: list over seeds of stat(cost of a) - stat(cost of b), in % ; a, b = functions (d, seed) -> decision label
            return {g: np.array([(stat(d["val"](a(d, s), w)) - stat(d["val"](b(d, s), w))) / d["sc"] for s in sorted(d["seeds"])]) for g, d in D.items()}
        def line(lab, a, b):
            m = diffs(a, b, np.mean); out = "   %-22s mean over samples %s | q90 over samples %s | max %5.1f | runs >= 2%%: %3.0f%%" % (lab, ms([v.mean() for v in m.values()]), ms([q90(v) for v in m.values()]), max(v.max() for v in m.values()), 100 * np.mean(np.concatenate(list(m.values())) >= 2))
            if w != "T":
                t = diffs(a, b, q90); out += " || belief-tail (q90 over retailer draws): mean %s | q90 %s" % (ms([v.mean() for v in t.values()]), ms([q90(v) for v in t.values()]))
            print(out)
        N = lambda d, s: d["seeds"][s]["N"]; B = lambda d, s: d["seeds"][s]["B0.1"]; D3 = lambda d, s: d["seeds"][s]["D0.3"]; D1 = lambda d, s: d["seeds"][s]["D0.1"]; NS = lambda d, s: d["N*"]; BS = lambda d, s: d["B*"]
        line("N - N*", N, NS); line("B - B*", B, BS); line("B* - N* (truth known)", BS, NS); line("B - N", B, N); line("D0.3 - N", D3, N); line("D0.1 - N", D1, N); line("B - N*", B, NS)
        # where does N lose: capacity relative to N*
        rows = [(cap(N(d, s)) - cap(d["N*"]), par(N(d, s))[0] == par(d["N*"])[0], (d["val"](N(d, s), w).mean() - d["val"](d["N*"], w).mean()) / d["sc"], (d["val"](D3(d, s), w).mean() - d["val"](N(d, s), w).mean()) / d["sc"], (d["val"](B(d, s), w).mean() - d["val"](N(d, s), w).mean()) / d["sc"]) for d in D.values() for s in sorted(d["seeds"])]
        R = np.array([(a, b, c_, e, f) for a, b, c_, e, f in rows], float)
        for lab, m in (("N built less than N*", R[:, 0] < 0), ("same total capacity", R[:, 0] == 0), ("N built more than N*", R[:, 0] > 0), ("same sites as N*", R[:, 1] == 1), ("different sites", R[:, 1] == 0), ("N loss >= 2%", R[:, 2] >= 2)):
            if m.sum(): print("      %-22s %3d runs (%2.0f%%): N - N* %+5.2f | D0.3 - N %+5.2f | B - N %+5.2f" % (lab, m.sum(), 100 * m.mean(), R[m, 2].mean(), R[m, 3].mean(), R[m, 4].mean()))
        if w != "T":   # best achievable among the evaluated decisions (oracle), relative to N*
            om = []; ot = []
            for d in D.values():
                xs = set(d["O"]) | {d["N*"], d["B*"]}; om.append((min(d["val"](x, w).mean() for x in xs) - d["val"](d["N*"], w).mean()) / d["sc"]); ot.append((min(q90(d["val"](x, w)) for x in xs) - q90(d["val"](d["N*"], w))) / d["sc"])
            print("   best of the top truth-known candidates - N*: mean %s | belief-tail %s" % (ms(om), ms(ot)))
