# paired comparison of knob configs vs base on same geo/seed (C0(delta*) - N, mean and q90), geo-clustered SE
import sys, csv, glob, re, collections, sys, numpy as np
CFGS = sys.argv[1:] or ["fr50", "vs25fr50", "vs125fr50", "vs0fr50"]
G = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/tk/"
DS = ["0.05", "0.1", "0.2", "0.3"]
def one(f, nf):
    L = list(csv.DictReader(open(f, encoding="utf-8")))
    VN = np.array([float(r["VN"]) for r in L]); adv = np.maximum.accumulate(np.array([[float(r["adv" + d]) for d in DS] for r in L]), axis=1)
    bm = np.array([float(r[f"bm{nf}"]) for r in L]); bq = np.array([float(r[f"bq{nf}"]) for r in L])
    n = int(np.argmin(VN)); k = next((i for i in range(4) if adv[n, i] >= bq[n] - 1e-9), 3)
    c = int(np.argmin(VN + adv[:, k])); Tm, Tq = VN + bm, VN + bq; sc = abs(Tm.min()) / 100
    return (Tm[c] - Tm[n]) / sc, (Tq[c] - Tq[n]) / sc, bq[n] / sc, (Tq[n] - Tq.min()) / sc
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()]); return np.mean(list(d.values())), m.std(ddof=1) / np.sqrt(len(m)), int((m < -0.01).sum()), int((m > 0.01).sum())
for nf in [3, 5, 10, 30]:
    R = {}
    for cfg in ["top"] + CFGS:
        R[cfg] = {}
        for g in range(1, 16):
            for s in range(1, 6):
                R[cfg][(g, s)] = one(G + f"tk_{cfg}_g{g}_s{s}.csv", nf)
    for cfg in CFGS:
        for j, lab in [(0, "mean"), (1, "q90")]:
            d = {k: R[cfg][k][j] - R["top"][k][j] for k in R[cfg]}
            m, se, w, l = cl(d); print(f"NF={nf} {cfg:8s} minus top, C0-N {lab}: {m:+.3f} +- {se:.3f} [geo better {w} / worse {l}]")
        # share of N tail damage removed, share of oracle tail regret recovered
        dq = np.mean([R[cfg][k][1] for k in R[cfg]]); bqN = np.mean([R[cfg][k][2] for k in R[cfg]]); rq = np.mean([R[cfg][k][3] for k in R[cfg]])
        print(f"      {cfg}: tail gain {-dq:.3f} = {100*-dq/bqN:.0f}% of N q90 damage {bqN:.2f}, {100*-dq/rq:.0f}% of N tail regret {rq:.2f}")
    dq = np.mean([R['top'][k][1] for k in R['top']]); bqN = np.mean([R['top'][k][2] for k in R['top']]); rq = np.mean([R['top'][k][3] for k in R['top']])
    print(f"      top : tail gain {-dq:.3f} = {100*-dq/bqN:.0f}% of N q90 damage {bqN:.2f}, {100*-dq/rq:.0f}% of N tail regret {rq:.2f}")
