# existing operator's share by geography vs how central its two sites are
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
K = ["i", "kind", "sreg", "cap", "h", "j", "creg", "dist", "mu", "yR", "yS"]; rows = []
for g in range(1, 16):
    X = list(csv.DictReader(open(O + "xid_g%d_s1.csv" % g, encoding="utf-8"), delimiter=";"))
    St = [(float(r["x"]), float(r["y"]), r["kind"], r["name"]) for r in X if r["kind"] in "AB"]; C = [(float(r["x"]), float(r["y"]), float(r["mu"])) for r in X if r["kind"] == "C"]
    w = np.array([c[2] for c in C]); D = np.array([[np.hypot(s[0] - c[0], s[1] - c[1]) for c in C] for s in St]); cen = D @ w / w.sum()      # demand-weighted mean distance of each site
    rank = np.argsort(np.argsort(cen)) + 1; ra = sorted(rank[i] for i, s in enumerate(St) if s[2] == "A")
    # share of demand for which an A site is the nearest of all 15 sites / the nearest when A competes with the best 3 B sites for it (greedy not computed): use "A nearest among A + all B" as lower bound
    nearA_all = sum(w[j] for j in range(len(C)) if St[int(np.argmin(D[:, j]))][2] == "A") / w.sum()
    dA = D[[i for i, s in enumerate(St) if s[2] == "A"]].min(0) @ w / w.sum()
    sh = []
    for s in range(1, 4):
        R = [dict(zip(K, l.split(";")[-11:])) for l in open(O + "flow_g%d_s%d.csv" % (g, s), encoding="utf-8").read().splitlines()[1:]]
        ta = sum(float(r["yR"]) + float(r["yS"]) for r in R if r["kind"] == "A"); tb = sum(float(r["yR"]) + float(r["yS"]) for r in R if r["kind"] == "B"); sh.append(ta / (ta + tb))
    rows.append((g, np.mean(sh), ra[0], ra[1], dA)); 
rows.sort(key=lambda r: r[1])
print("geo | A share | centrality rank of the two A sites among the 15 cities (1 = most central) | demand-weighted distance to the nearer A site")
for r in rows: print("g%-2d | %3.0f%% | %2d, %2d | %5.0f" % (r[0], 100 * r[1], r[2], r[3], r[4]))
A = np.array(rows); print("corr(A share, mean rank) = %.2f, corr(A share, distance to nearer A site) = %.2f" % (np.corrcoef(A[:, 1], (A[:, 2] + A[:, 3]) / 2)[0, 1], np.corrcoef(A[:, 1], A[:, 4])[0, 1]))
