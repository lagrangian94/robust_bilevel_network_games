# location + common capacity level as the leader decision (post-hoc): candidates = top-8 locations at each capacity level b; store cost = qfix + c*b
import csv, glob, re, collections, os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
DS = ["0.05", "0.1", "0.2", "0.3"]; q90 = lambda v: np.quantile(v, 0.9)
LV = {200: "topbb200", 225: "topbb225", 250: "topbb250", 275: "topbb275", 300: "topbb300", 483: "top"}
NFS = [int(v) for v in os.environ.get("AQ_NF", "5,20").split(",")]; SHI = int(os.environ.get("AQ_SHI", "2"))
SC = [(750 - 300 * c, c) for c in (0.0, 1.0, 2.0, 2.5)] + [(500.0, c) for c in (1.5, 2.5, 3.5)]
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()])
    return np.mean(list(d.values())), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan")), int((m < -0.01).sum()), int((m > 0.01).sum())
def rd(path): return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else None
def load(key, nf):
    g, s = key; C = []
    for b, cfg in LV.items():
        T = rd(O + "tk/tk_%s_g%d_s%d.csv" % (cfg, g, s)); lg = O + "tk/tk_%s_g%d_s%d.log" % (cfg, g, s)
        if T is None or (cfg != "top" and not (os.path.exists(lg) and "TKD_DONE" in open(lg, encoding="utf-8", errors="replace").read())): return None
        D = rd(O + "tkd/tkd_%s_g%d_s%d.csv" % (cfg, g, s)); J = collections.defaultdict(list)
        for r in (D or []):
            if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        if not J and cfg == "top":
            for r in (rd(O + "tkd/tkd_topnf%d_g%d_s%d.csv" % (nf, g, s)) or []):
                if int(r["nf"]) == nf: J[r["x"]].append(float(r["J"]))
        for r in T:
            if r["x"] not in J: return None
            C.append((b, r["x"].count(";") + 1, float(r["VN"]), np.maximum.accumulate([float(r["adv" + d]) for d in DS]), np.array(J[r["x"]])))
    nd = len(C[0][4]); C.append((0, 0, 0.0, np.zeros(4), np.zeros(nd)))   # open nothing
    return C
def run(nf):
    L = {(g, s): load((g, s), nf) for g in range(1, 16) for s in range(1, SHI + 1)}; L = {k: v for k, v in L.items() if v is not None}
    print("\n===== N_F = %d : %d runs" % (nf, len(L)))
    for qf, c in SC:
        R = {}
        for key, C in L.items():
            b = np.array([t[0] for t in C]); n_ = np.array([t[1] for t in C]); add = (qf - 500 + c * b) * n_
            v = np.array([t[2] for t in C]) + add; A = np.array([t[3] for t in C]); M = np.array([t[4] for t in C]) + add[:, None]
            tied = np.where(v <= v.min() + 1e-6)[0]; n = int(tied[np.argmin([q90(M[i]) for i in tied])])
            if n_[n] == 0: R[key] = None; continue
            sc = abs(v[n]) / 100; bq = q90(M[n]) - v[n]; k = next((i for i in range(4) if A[n, i] >= bq - 1e-9), 3)
            same = np.where(b == b[n])[0]
            dec = {"B1": int(np.argmin(v + A[:, 1])), "BS": int(np.argmin(v + A[:, k])), "OR": int(np.argmin(np.quantile(M, 0.9, axis=1))),
                   "L1": int(same[np.argmin((v + A[:, 1])[same])]), "LS": int(same[np.argmin((v + A[:, k])[same])])}
            o = {"bN": b[n], "nN": n_[n], "dq": bq / sc, "sc": sc}
            for a, i in dec.items():
                o["m" + a] = (M[i].mean() - M[n].mean()) / sc; o["q" + a] = (q90(M[i]) - q90(M[n])) / sc; o["b" + a] = b[i]
                o["lo" + a] = float(b[i] * n_[i] < b[n] * n_[n]); o["cap" + a] = b[i] * n_[i]
            R[key] = o
        out = sum(r is None for r in R.values()); R = {k: r for k, r in R.items() if r is not None}
        if not R: print("-- qfix %.0f c %.1f : N opens nothing in all runs" % (qf, c)); continue
        f = lambda k: np.mean([r[k] for r in R.values()])
        print("-- qfix %4.0f c %.1f (c/v %.1f) | N: %d runs open (nothing %d), level %s, stores %.2f, cost scale %.0f, q90 damage %.1f%%" % (qf, c, c / 5, len(R), out, dict(sorted(collections.Counter(int(r["bN"]) for r in R.values()).items())), f("nN"), 100 * f("sc"), f("dq")))
        for a, lab in [("L1", "location only d=0.1  "), ("B1", "loc + capacity d=0.1 "), ("LS", "location only calib  "), ("BS", "loc + capacity calib "), ("OR", "tail oracle          ")]:
            m, sm, _, _ = cl({k: r["m" + a] for k, r in R.items()}); q, sq, w, l = cl({k: r["q" + a] for k, r in R.items()})
            print("   %s mean %+.2f +- %.2f | q90 %+.2f +- %.2f [geo %d/%d] | less total capacity %.0f%% (capacity %.0f vs N %.0f)" % (lab, m, sm, q, sq, w, l, 100 * f("lo" + a), f("cap" + a), np.mean([r["bN"] * r["nN"] for r in R.values()])))
for nf in NFS: run(nf)
