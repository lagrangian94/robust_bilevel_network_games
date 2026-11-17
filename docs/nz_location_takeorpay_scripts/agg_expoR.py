# capacity-ratio screen over geographies: band, direct regret, D gain (expoR csv) + spot share, belief exposure (expo csv)
import csv, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/expo/"
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
def rank(v): o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
for cfg in sys.argv[1:]:
    for cv in ("0.5", "0.4", "0.2"):
        rows = []
        for g in range(1, 16):
            try:
                a = [r for r in rd(O + "expoR_%s_g%d_s1.csv" % (cfg, g)) if r["cv"] == cv][0]; b = [r for r in rd(O + "expo_%s_g%d_s1.csv" % (cfg, g)) if r["cv"] == cv][0]
            except (FileNotFoundError, IndexError): continue
            rows.append((g, 100 * float(a["band"]), float(a["regretN"]), float(a["gainD"]), 100 * float(b["spot_mean"]) / float(b["sales_mean"]), float(b["adv0.1"]), float(a["optimism"]), int(b["same"]), float(a["capN"])))
        A = np.array([r[1:] for r in rows]); n = len(rows)
        print("== %s c/v %s: %d geos" % (cfg, cv, n))
        print("   band %%      mean %5.1f median %5.1f | per geo %s" % (A[:, 0].mean(), np.median(A[:, 0]), " ".join("%.0f" % v for v in A[:, 0])))
        print("   regret N    mean %5.2f +- %.2f median %5.2f | >=1%%: %d/%d | per geo %s" % (A[:, 1].mean(), A[:, 1].std(ddof=1) / np.sqrt(n), np.median(A[:, 1]), (A[:, 1] >= 1).sum(), n, " ".join("%.1f" % v for v in A[:, 1])))
        print("   D - N       mean %+5.2f +- %.2f | D better %d, worse %d, same %d | per geo %s" % (A[:, 2].mean(), A[:, 2].std(ddof=1) / np.sqrt(n), (A[:, 2] < -0.01).sum(), (A[:, 2] > 0.01).sum(), (abs(A[:, 2]) <= 0.01).sum(), " ".join("%+.1f" % v for v in A[:, 2])))
        print("   spot share  mean %5.1f | belief exposure d=0.1 mean %5.1f | optimism mean %+.1f | D = N in-sample %d/%d | capacity %.0f" % (A[:, 3].mean(), A[:, 4].mean(), A[:, 5].mean(), A[:, 6].sum(), n, A[:, 7].mean()))
        print("   rank corr(band, belief exposure) %+.2f | corr(band, regret) %+.2f" % (np.corrcoef(rank(A[:, 0]), rank(A[:, 4]))[0, 1], np.corrcoef(rank(A[:, 0]), rank(A[:, 1]))[0, 1]))
