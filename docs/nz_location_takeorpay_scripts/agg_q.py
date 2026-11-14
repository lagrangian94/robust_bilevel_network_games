# post-hoc fixed-cost analysis: candidates = union of top-8 per opening cap (top, topr2, topr1); V' = V + (q-500)|x|
import csv, glob, re, collections, os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
DS = ["0.05", "0.1", "0.2", "0.3"]; q90 = lambda v: np.quantile(v, 0.9)
CFGS = os.environ.get("AQ_CFGS", "top,topr2,topr1").split(","); NFS = [int(v) for v in os.environ.get("AQ_NF", "5,10,20").split(",")]
QS = [int(v) for v in os.environ.get("AQ_Q", "500,600,700,800,900,1000").split(",")]
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()])
    return np.mean(list(d.values())), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan")), int((m < -0.01).sum()), int((m > 0.01).sum())
def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else None
def load(key, nf):
    g, s = key; C = collections.OrderedDict(); chk = 0
    for cfg in CFGS:
        T = rd(O + "tk/tk_%s_g%d_s%d.csv" % (cfg, g, s))
        lg = O + "tk/tk_%s_g%d_s%d.log" % (cfg, g, s)
        if cfg != "top" and not (os.path.exists(lg) and "TKD_DONE" in open(lg, encoding="utf-8", errors="replace").read()): return None
        if T is None: return None
        D = rd(O + "tkd/tkd_%s_g%d_s%d.csv" % (cfg, g, s)); J = collections.defaultdict(list)
        if D is not None:
            for r in D:
                if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        if not J and cfg == "top":
            D = rd(O + "tkd/tkd_topnf%d_g%d_s%d.csv" % (nf, g, s))
            if D is None: return None
            for r in D:
                if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        for r in T:
            x = r["x"]
            if x not in J: return None
            a = np.maximum.accumulate([float(r["adv" + d]) for d in DS]); j = np.array(J[x])
            if x in C: chk = max(chk, float(np.abs(C[x][2] - j).max())); continue
            C[x] = (float(r["VN"]), a, j, x.count(";") + 1)
    return C, chk
def run(nf):
    keys = [(g, s) for g in range(1, 16) for s in range(1, 1 + int(os.environ.get("AQ_SHI", "5")))]; L = {k: load(k, nf) for k in keys}; L = {k: v for k, v in L.items() if v is not None}
    print("\n===== N_F = %d : %d runs, max mismatch of shared decisions %.3f, candidates %.1f" % (nf, len(L), max(v[1] for v in L.values()), np.mean([len(v[0]) for v in L.values()])))
    for Q in QS:
        R = {}
        for key, (C, _) in L.items():
            xs = list(C); n_ = np.array([C[x][3] for x in xs]); v = np.array([C[x][0] for x in xs]) + (Q - 500) * n_
            A = np.array([C[x][1] for x in xs]); M = np.array([C[x][2] for x in xs]) + (Q - 500) * n_[:, None]
            tied = np.where(v <= v.min() + 1e-6)[0]; n = int(tied[np.argmin([q90(M[i]) for i in tied])]); sc = abs(v[n]) / 100
            bq = q90(M[n]) - v[n]; k = next((i for i in range(4) if A[n, i] >= bq - 1e-9), 3)
            dec = {"B1": int(np.argmin(v + A[:, 1])), "B2": int(np.argmin(v + A[:, 2])), "BS": int(np.argmin(v + A[:, k])), "OR": int(np.argmin(np.quantile(M, 0.9, axis=1)))}
            o = {"nN": n_[n], "dq": bq / sc, "sc": sc, "tie": len(tied) > 1}
            for a, i in dec.items():
                o["m" + a] = (M[i].mean() - M[n].mean()); o["q" + a] = (q90(M[i]) - q90(M[n])); o["n" + a] = n_[i]; o["ch" + a] = float(i != n); o["lt" + a] = float(n_[i] < n_[n]); o["t" + a] = (v[i] - v[n])
            c = M.min(axis=0); o["mCL"] = c.mean() - M[n].mean(); o["qCL"] = q90(c) - q90(M[n])
            R[key] = o
        f = lambda k: np.mean([r[k] for r in R.values()])
        print("-- q = %d | N opens %.2f stores (3: %d, 2: %d, 1: %d) | N cost scale %.0f | N q90 damage %.1f%% | ties %d" % (Q, f("nN"), sum(r["nN"] == 3 for r in R.values()), sum(r["nN"] == 2 for r in R.values()), sum(r["nN"] == 1 for r in R.values()), 100 * f("sc"), f("dq"), sum(r["tie"] for r in R.values())))
        for a, lab in [("B1", "Belief-only d=0.1"), ("B2", "Belief-only d=0.2"), ("BS", "Belief-only calib "), ("OR", "tail oracle       ")]:
            m, sm, _, _ = cl({k: r["m" + a] / r["sc"] for k, r in R.items()}); q, sq, w, l = cl({k: r["q" + a] / r["sc"] for k, r in R.items()}); t, _, _, _ = cl({k: r["t" + a] / r["sc"] for k, r in R.items()})
            print("   %s mean %+.2f +- %.2f | q90 %+.2f +- %.2f [geo %d/%d] (abs %+.0f) | truth price %+.2f | changed %.0f%% fewer stores %.0f%% (opens %.2f)" % (lab, m, sm, q, sq, w, l, f("q" + a), t, 100 * f("ch" + a), 100 * f("lt" + a), f("n" + a)))
        print("   clairvoyant        mean %+.2f | q90 %+.2f" % (np.mean([r["mCL"] / r["sc"] for r in R.values()]), np.mean([r["qCL"] / r["sc"] for r in R.values()])))
for nf in NFS: run(nf)
