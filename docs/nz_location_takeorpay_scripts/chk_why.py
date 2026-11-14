import csv, glob, re, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
# (a) value of the best decision by number of stores opened (truth), take-or-pay
best = collections.defaultdict(list); nopenN = collections.Counter(); gap8 = []
for f in sorted(glob.glob(O + "tk/vn_top_g*_s*.csv")):
    L = [(r["x"], float(r["VN"])) for r in csv.DictReader(open(f, encoding="utf-8"))]
    by = collections.defaultdict(list)
    for x, v in L: by[x.count(";") + 1].append(v)
    for k in (1, 2, 3): best[k].append(min(by[k]))
    xb = min(L, key=lambda t: t[1])[0]; nopenN[xb.count(";") + 1] += 1
    vs = sorted(v for _, v in L); gap8.append((vs[7] - vs[0]) / abs(vs[0]) * 100)
print("(a) runs %d | best V_N with 1 / 2 / 3 stores: %.0f / %.0f / %.0f | N opens %s | 8th-best decision is %.1f%% worse than the best" % (len(best[1]), np.mean(best[1]), np.mean(best[2]), np.mean(best[3]), dict(nopenN), np.mean(gap8)))
# overlap among the top-8 candidates with N
ov = []
for f in sorted(glob.glob(O + "tk/tk_top_g*_s*.csv")):
    xs = [set(r["x"].strip("{}").split(";")) for r in csv.DictReader(open(f, encoding="utf-8"))]
    VN = [float(r["VN"]) for r in csv.DictReader(open(f, encoding="utf-8"))]; n = int(np.argmin(VN))
    ov += [len(xs[n] & xs[i]) for i in range(len(xs)) if i != n]
print("    stores shared with N among the other 7 candidates: %s" % dict(sorted(collections.Counter(ov).items())))
# (b) who absorbs the change of the retailer's total reservation: A or B?  (N decision, take-or-pay)
sh, tot, hAq, hBq, bA, bB = [], [], [], [], [], []
for f in sorted(glob.glob(O + "tkh/tkh_top_g*_s*.csv")):
    D = collections.OrderedDict()
    for r in csv.DictReader(open(f, encoding="utf-8")):
        e = D.setdefault(r["x"], {"q": None, "J": [], "hA": [], "hB": []})
        if r["k"] == "0": e["q"] = (float(r["J"]), float(r["hA"]), float(r["hB"]))
        else: e["hA"].append(float(r["hA"])); e["hB"].append(float(r["hB"]))
    xs = list(D); n = int(np.argmin([D[x]["q"][0] for x in xs])); e = D[xs[n]]
    dA = np.array(e["hA"]) - e["q"][1]; dB = np.array(e["hB"]) - e["q"][2]; dT = dA + dB
    hAq.append(e["q"][1]); hBq.append(e["q"][2]); tot.append(np.quantile(-dT, 0.9))
    if np.var(dT) > 1e-9: bB.append(np.cov(dB, dT)[0, 1] / np.var(dT)); bA.append(np.cov(dA, dT)[0, 1] / np.var(dT))
print("(b) under truth: reservation at A %.0f, at B %.0f (B share %.0f%%) | per unit change of total reservation across beliefs: B moves %.2f, A moves %.2f | q90 drop of total %.0f" % (np.mean(hAq), np.mean(hBq), 100 * np.mean(hBq) / (np.mean(hAq) + np.mean(hBq)), np.mean(bB), np.mean(bA), np.mean(tot)))
