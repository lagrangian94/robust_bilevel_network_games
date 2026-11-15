# how much does not knowing the true distribution cost the leader? (data world, existing dw outputs)
# per run: N decision's in-sample value vs true-F value (optimism), regret of N vs the ex-post best candidate under the true F
import csv, os, sys, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
def cst(x, q0, c): pr = [p.split(":") for p in x.strip("{}").split(";")]; return q0 * len(pr) + c * sum(float(b) for _, b in pr)
def se(v):
    v = np.array(v); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
for cfg, ns in [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]:
    for sc, (q0, c) in enumerate([(0.0, 2.5), (150.0, 2.0)], 1):
        R = collections.defaultdict(list)
        for g in range(1, 16):
            for s in range(1, ns + 1):
                tag = "%s_g%d_s%d" % (cfg, g, s)
                if not os.path.exists(O + "oos/true/ot_%s.csv" % tag): continue
                V0 = {r["x"]: float(r["V0"]) + cst(r["x"], q0, c) for r in rd(O + "dw/dw_%s.csv" % tag)}
                xN = [r["x"] for r in rd(O + "dw/sel_%s.csv" % tag) if int(r["scen"]) == sc and r["model"] == "N"][0]
                J = collections.defaultdict(lambda: collections.defaultdict(list))
                for r in rd(O + "oos/true/ot_%s.csv" % tag): J[int(r["nf"])][r["x"]].append(float(r["J0"]) + cst(r["x"], q0, c))
                scl = abs(V0[xN]) / 100
                Jk = {x: np.mean(v) for x, v in J[0].items()}
                R["optimism"].append((Jk[xN] - V0[xN]) / scl)                      # true cost minus in-sample cost of N (belief = q-hat)
                R["regret known"].append((Jk[xN] - min(Jk.values())) / scl)       # N vs ex-post best candidate, retailer belief = q-hat
                R["N is best"].append(100.0 * (Jk[xN] <= min(Jk.values()) + 1e-6))
                for nf in (5, 20):
                    Jm = {x: np.mean(v) for x, v in J[nf].items()}; R["regret nf%d" % nf].append((Jm[xN] - min(Jm.values())) / scl)
                    R["belief gap nf%d" % nf].append((Jm[xN] - Jk[xN]) / scl)       # N's true cost: retailer's own belief vs belief = q-hat
        print("== %s c/v %.1f: %d runs" % (cfg, c / 5, len(R["optimism"])))
        for k, v in R.items(): print("   %-16s mean %+6.2f +- %.2f | median %+6.2f | q90 %+6.2f" % ((k,) + se(v) + (np.median(v), np.quantile(v, 0.9))))
