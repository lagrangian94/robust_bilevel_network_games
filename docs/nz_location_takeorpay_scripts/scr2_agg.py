# screening aggregate (take-or-pay knobs, TK world): python scr2_agg.py <ref> <cfg> ...   keys = geo 1-15 x seeds SEEDS (env SCR_S, default 1,2)
import sys, csv, os, collections, numpy as np
G = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/tk/"
DS = ["0.05", "0.1", "0.2", "0.3"]; SEEDS = [int(s) for s in os.environ.get("SCR_S", "1,2").split(",")]
def one(f, nf):
    L = list(csv.DictReader(open(f, encoding="utf-8")))
    VN = np.array([float(r["VN"]) for r in L]); adv = np.maximum.accumulate(np.array([[float(r["adv" + d]) for d in DS] for r in L]), axis=1)
    bm = np.array([float(r["bm%d" % nf]) for r in L]); bq = np.array([float(r["bq%d" % nf]) for r in L])
    n = int(np.argmin(VN)); k = next((i for i in range(4) if adv[n, i] >= bq[n] - 1e-9), 3)
    Tm, Tq = VN + bm, VN + bq; sc = abs(Tm.min()) / 100; out = {}
    for lab, kk in [("s", k), ("1", 1), ("2", 2)]:
        c = int(np.argmin(VN + adv[:, kk])); out["m" + lab] = (Tm[c] - Tm[n]) / sc; out["q" + lab] = (Tq[c] - Tq[n]) / sc; out["ch" + lab] = float(c != n)
    out["dmgq"] = bq[n] / sc; out["dmgm"] = bm[n] / sc; out["orc"] = (Tq[n] - Tq.min()) / sc; out["orcm"] = (Tm[n] - Tm.min()) / sc; out["cost"] = abs(Tm[n])
    return out
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()])
    return np.mean(list(d.values())), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan")), int((m < -0.01).sum()), int((m > 0.01).sum())
ref = sys.argv[1]
for nf in [5, 10]:
    print("==== N_F = %d   (C0 - N, %% of cost; negative = belief hedging better)" % nf)
    R = {}
    for cfg in sys.argv[1:]:
        R[cfg] = {}
        for g in range(1, 16):
            for s in SEEDS:
                f = G + "tk_%s_g%d_s%d.csv" % (cfg, g, s); lg = f[:-4] + ".log"
                if os.path.exists(f) and os.path.exists(lg) and "TK_DONE" in open(lg, encoding="utf-8", errors="replace").read(): R[cfg][(g, s)] = one(f, nf)
    for cfg in sys.argv[1:]:
        r = R[cfg]
        if not r: print("%-9s no data" % cfg); continue
        col = lambda k: {key: v[k] for key, v in r.items()}
        ms, ss, _, _ = cl(col("ms")); qs, sqs, w, l = cl(col("qs")); m1, s1, _, _ = cl(col("m1")); q1, sq1, _, _ = cl(col("q1")); m2, _, _, _ = cl(col("m2")); q2, sq2, _, _ = cl(col("q2"))
        line = "%-9s n=%2d | dmg N mean %+5.1f q90 %5.1f | oracle %4.2f | d*: mean %+5.2f+-%.2f q90 %+5.2f+-%.2f [%d/%d] | d=.1: %+5.2f, %+5.2f+-%.2f | d=.2: %+5.2f, %+5.2f+-%.2f" % (
            cfg, len(r), np.mean(list(col("dmgm").values())), np.mean(list(col("dmgq").values())), np.mean(list(col("orc").values())), ms, ss, qs, sqs, w, l, m1, q1, sq1, m2, q2, sq2)
        if cfg != ref:
            keys = [k for k in r if k in R[ref]]
            if keys:
                dq, sdq, w2, l2 = cl({k: r[k]["qs"] - R[ref][k]["qs"] for k in keys}); dm, sdm, _, _ = cl({k: r[k]["ms"] - R[ref][k]["ms"] for k in keys}); do, sdo, _, _ = cl({k: r[k]["orc"] - R[ref][k]["orc"] for k in keys})
                line += " || vs %s (n=%d): q90 %+5.2f+-%.2f [%d/%d] mean %+5.2f+-%.2f oracle %+5.2f+-%.2f" % (ref, len(keys), dq, sdq, w2, l2, dm, sdm, do, sdo)
        print(line)
