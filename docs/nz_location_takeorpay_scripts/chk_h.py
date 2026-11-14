import csv, glob, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
# TV distance between an N_F-sample empirical belief and the uniform truth on S = 10 scenarios
rng = np.random.default_rng(1)
for nf in [5, 10, 30]:
    tv = [0.5 * np.abs(np.bincount(rng.integers(0, 10, nf), minlength=10) / nf - 0.1).sum() for _ in range(20000)]
    print("N_F=%d: TV(belief, truth) mean %.2f, q10 %.2f, q90 %.2f, min %.2f" % (nf, np.mean(tv), np.quantile(tv, .1), np.quantile(tv, .9), np.min(tv)))
rel, relq, corr, dN, dO, hq_all = [], [], [], [], [], []
for f in sorted(glob.glob(O + "tkh/tkh_top_g*_s*.csv")):
    D = collections.OrderedDict()
    for r in csv.DictReader(open(f, encoding="utf-8")):
        e = D.setdefault(r["x"], {"q": None, "J": [], "hB": []})
        if r["k"] == "0": e["q"] = (float(r["J"]), float(r["hB"]))
        else: e["J"].append(float(r["J"])); e["hB"].append(float(r["hB"]))
    xs = list(D); n = int(np.argmin([D[x]["q"][0] for x in xs])); e = D[xs[n]]; hq = e["q"][1]; h = np.array(e["hB"]); hq_all.append(hq)
    rel.append((h - hq) / hq * 100)
    dmgN = np.array(e["J"]) - e["q"][0]
    for i, x in enumerate(xs):
        if i == n: continue
        d = np.array(D[x]["J"]) - D[x]["q"][0]
        if d.std() > 1e-9 and dmgN.std() > 1e-9: corr.append(np.corrcoef(dmgN, d)[0, 1])
R = np.concatenate(rel)
print("top, N decision, %d runs: total B reservation under truth mean %.0f (cap 900)" % (len(rel), np.mean(hq_all)))
print("  change of total B reservation under a sampled belief (%% of its level under truth): mean %+.1f | q05 %+.1f q10 %+.1f q25 %+.1f median %+.1f q75 %+.1f q90 %+.1f q95 %+.1f" % ((R.mean(),) + tuple(np.quantile(R, [.05, .1, .25, .5, .75, .9, .95]))))
print("  share of draws with |change| < 1%%: %.0f%%, drop > 5%%: %.0f%%, drop > 10%%: %.0f%%, drop > 20%%: %.0f%%" % (100 * (abs(R) < 1).mean(), 100 * (R < -5).mean(), 100 * (R < -10).mean(), 100 * (R < -20).mean()))
print("  correlation across belief draws between N's damage and each other candidate's damage: mean %.2f, q10 %.2f, q90 %.2f" % (np.mean(corr), np.quantile(corr, .1), np.quantile(corr, .9)))
