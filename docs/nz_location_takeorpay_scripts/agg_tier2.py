# per-site capacity tiers (tkT.jl): decision label "{i:b;...}", V0 without opening cost; store cost q0 + c*b added post hoc
import csv, re, collections, os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
DS = ["0.05", "0.1", "0.2", "0.3"]; q90 = lambda v: np.quantile(v, 0.9)
CFG = os.environ.get("AT_CFG", "toptier"); CFG2 = [c for c in os.environ.get("AT_CFG2", "").split(",") if c]; NFS = [int(v) for v in os.environ.get("AQ_NF", "5,20").split(",")]; SHI = int(os.environ.get("AQ_SHI", "5"))
SC = [(0.0, 2.5), (150.0, 2.0), (450.0, 1.0)]
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()])
    return np.mean(list(d.values())), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan")), int((m < -0.01).sum()), int((m > 0.01).sum())
def load(key, nf):
    g, s = key; lg = O + "tk/tk_%s_g%d_s%d.log" % (CFG, g, s)
    if not (os.path.exists(lg) and "TKD_DONE" in open(lg, encoding="utf-8", errors="replace").read()): return None
    T = []; J = collections.defaultdict(list)
    for cf in [CFG] + CFG2:
        if cf != CFG and not "TKD_DONE" in open(O + "tk/tk_%s_g%d_s%d.log" % (cf, g, s), encoding="utf-8", errors="replace").read(): return None
        T += list(csv.DictReader(open(O + "tk/tk_%s_g%d_s%d.csv" % (cf, g, s), encoding="utf-8")))
        for r in csv.DictReader(open(O + "tkd/tkd_%s_g%d_s%d.csv" % (cf, g, s), encoding="utf-8")):
            if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
    assert len(set(r["x"] for r in T)) == len(T)
    C = []
    for r in T:
        pr = [p.split(":") for p in r["x"].strip("{}").split(";")]; sites = tuple(int(a) for a, _ in pr); lv = tuple(float(b) for _, b in pr)
        C.append((sites, lv, float(r["V0"]), np.maximum.accumulate([float(r["adv" + d]) for d in DS]), np.array(J[r["x"]])))
    C.append(((), (), 0.0, np.zeros(4), np.zeros(len(C[0][4])))); return C
def run(nf):
    L = {(g, s): load((g, s), nf) for g in range(1, 16) for s in range(1, SHI + 1)}; L = {k: v for k, v in L.items() if v is not None}
    print("\n===== %s+%s N_F = %d : %d runs, candidates %.1f" % (CFG, ",".join(CFG2), nf, len(L), np.mean([len(v) - 1 for v in L.values()])))
    for q0, c in SC:
        R = {}; out = 0
        for key, C in L.items():
            cost = np.array([q0 * len(t[0]) + c * sum(t[1]) for t in C]); cap = np.array([sum(t[1]) for t in C])
            v = np.array([t[2] for t in C]) + cost; A = np.array([t[3] for t in C]); M = np.array([t[4] for t in C]) + cost[:, None]
            tied = np.where(v <= v.min() + 1e-6)[0]; n = int(tied[np.argmin([q90(M[i]) for i in tied])])
            if len(C[n][0]) == 0: out += 1; continue
            sc = abs(v[n]) / 100; bq = q90(M[n]) - v[n]; k = next((i for i in range(4) if A[n, i] >= bq - 1e-9), 3)
            loc = np.array([i for i, t in enumerate(C) if sorted(t[1]) == sorted(C[n][1])]); cp = np.array([i for i, t in enumerate(C) if t[0] == C[n][0]])
            dec = {"B1": int(np.argmin(v + A[:, 1])), "BS": int(np.argmin(v + A[:, k])), "OR": int(np.argmin(np.quantile(M, 0.9, axis=1))),
                   "LS": int(loc[np.argmin((v + A[:, k])[loc])]), "CS": int(cp[np.argmin((v + A[:, k])[cp])])}
            o = {"capN": cap[n], "nN": len(C[n][0]), "dq": bq / sc, "sc": sc, "nloc": len(loc), "ncp": len(cp)}
            for a, i in dec.items():
                o["m" + a] = (M[i].mean() - M[n].mean()) / sc; o["q" + a] = (q90(M[i]) - q90(M[n])) / sc; o["lo" + a] = float(cap[i] < cap[n]); o["cap" + a] = cap[i]
                o["ss" + a] = float(C[i][0] == C[n][0]); o["ch" + a] = float(i != n)
            R[key] = o
        if not R: print("-- q0 %.0f c %.1f: N opens nothing" % (q0, c)); continue
        f = lambda k: np.mean([r[k] for r in R.values()])
        print("-- q0 %3.0f c %.1f (c/v %.1f) | N opens in %d runs (nothing %d), stores %.2f, capacity %.0f, cost scale %.0f, q90 damage %.1f%% | same-tier candidates %.1f, same-site candidates %.1f" % (q0, c, c / 5, len(R), out, f("nN"), f("capN"), 100 * f("sc"), f("dq"), f("nloc"), f("ncp")))
        for a, lab in [("LS", "location only calib  "), ("CS", "capacity only calib  "), ("B1", "loc + capacity d=0.1 "), ("BS", "loc + capacity calib "), ("OR", "tail oracle          ")]:
            m, sm, _, _ = cl({k: r["m" + a] for k, r in R.items()}); q, sq, w, l = cl({k: r["q" + a] for k, r in R.items()})
            print("   %s mean %+.2f +- %.2f | q90 %+.2f +- %.2f [geo %d/%d] | changed %.0f%%, less capacity %.0f%% (capacity %.0f), same sites as N %.0f%%" % (lab, m, sm, q, sq, w, l, 100 * f("ch" + a), 100 * f("lo" + a), f("cap" + a), 100 * f("ss" + a)))
for nf in NFS: run(nf)
