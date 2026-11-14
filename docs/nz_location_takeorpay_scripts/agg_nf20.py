# N_F = 20 (or any dumped nf) with up to 30 candidates: Belief-only decisions come from the top-8 adversarial values in geo/tk, oracles use all dumped candidates
import csv, glob, re, collections, os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
DS = ["0.05", "0.1", "0.2", "0.3"]; q90 = lambda v: np.quantile(v, 0.9)
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()])
    return np.mean(list(d.values())), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan")), int((m < -0.01).sum()), int((m > 0.01).sum())
def load(cfg, suf="nf20", nf=20):
    R = {}
    for f in sorted(glob.glob(O + "tkd/tkd_%s%s_g*_s*.csv" % (cfg, suf))):
        lg = f[:-4] + ".log"
        if not os.path.exists(lg) or "TKD_DONE" not in open(lg, encoding="utf-8", errors="replace").read(): continue
        m = re.search(r"_g(\d+)_s(\d+)\.csv", f); key = (int(m.group(1)), int(m.group(2)))
        J = collections.OrderedDict(); VN = {}
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if int(r["nf"]) == nf: J.setdefault(r["x"], []).append(float(r["J"])); VN[r["x"]] = float(r["VN"])
        xs = list(J); M = np.array([J[x] for x in xs]); vn = np.array([VN[x] for x in xs]); n = int(np.argmin(vn)); sc = abs(M.mean(axis=1).min()) / 100
        T = {r["x"]: r for r in csv.DictReader(open(O + "tk/tk_%s_g%d_s%d.csv" % ((cfg,) + key), encoding="utf-8"))}
        c8 = [i for i, x in enumerate(xs) if x in T]
        if xs[n] not in T: print("  N not in top-8 file", cfg, key); continue
        adv = np.maximum.accumulate(np.array([[float(T[xs[i]]["adv" + d]) for d in DS] for i in c8]), axis=1); v8 = vn[c8]; n8 = c8.index(n)
        bqN = q90(M[n] - vn[n]); k = next((i for i in range(4) if adv[n8, i] >= bqN - 1e-9), 3)
        dec = {"B1": c8[int(np.argmin(v8 + adv[:, 1]))], "B2": c8[int(np.argmin(v8 + adv[:, 2]))], "BS": c8[int(np.argmin(v8 + adv[:, k]))],
               "OR8": c8[int(np.argmin(np.quantile(M[c8], 0.9, axis=1)))], "OR": int(np.argmin(np.quantile(M, 0.9, axis=1)))}
        out = {"dmgq": (q90(M[n]) - vn[n]) / sc, "dmgm": (M[n].mean() - vn[n]) / sc, "k": k, "ncand": len(xs)}
        for a, i in dec.items(): out["m" + a] = (M[i].mean() - M[n].mean()) / sc; out["q" + a] = (q90(M[i]) - q90(M[n])) / sc; out["ch" + a] = float(i != n)
        for lab, idx in [("CL8", c8), ("CL", list(range(len(xs))))]:
            c = M[idx].min(axis=0); out["m" + lab] = (c.mean() - M[n].mean()) / sc; out["q" + lab] = (q90(c) - q90(M[n])) / sc
        out["rank_or"] = int(np.argsort(np.argsort(vn))[dec["OR"]]) + 1; out["corr"] = np.mean([np.corrcoef(M[n], M[i])[0, 1] for i in range(len(xs)) if i != n and M[i].std() > 1e-9])
        R[key] = out
    return R
for cfg in sys.argv[1:]:
    R = load(cfg)
    if not R: print(cfg, "no data"); continue
    f = lambda k: np.mean([r[k] for r in R.values()])
    print("===== %s  N_F=20: %d runs, candidates %.0f | N damage mean %+.2f q90 %.2f | calibrated delta counts %s" % (cfg, len(R), f("ncand"), f("dmgm"), f("dmgq"), dict(collections.Counter(DS[r["k"]] for r in R.values()))))
    for a, lab in [("B1", "Belief-only d=0.1"), ("B2", "Belief-only d=0.2"), ("BS", "Belief-only calibrated"), ("OR8", "tail oracle (8 cands)"), ("OR", "tail oracle (all cands)")]:
        m, sm, _, _ = cl({k: r["m" + a] for k, r in R.items()}); q, sq, w, l = cl({k: r["q" + a] for k, r in R.items()})
        print("  %-24s mean %+.2f +- %.2f | q90 %+.2f +- %.2f [geo %d/%d] | changed %.0f%%" % (lab, m, sm, q, sq, w, l, 100 * f("ch" + a)))
    print("  per-belief clairvoyant: 8 cands mean %+.2f q90 %+.2f | all cands mean %+.2f q90 %+.2f | oracle's V_N rank (all cands) mean %.1f | corr(N damage, other cand damage) %.2f" % (f("mCL8"), f("qCL8"), f("mCL"), f("qCL"), f("rank_or"), f("corr")))
