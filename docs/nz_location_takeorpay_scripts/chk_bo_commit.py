# -*- coding: utf-8 -*-
# Is the retailer's commitment at the leader's sites less sensitive to the belief under the Belief-only decision than under Nominal?
# geo/onp dec files (price-based): for each truth p (TV d from q̂) the commitment of a retailer who believes p (hB_true) vs believes q̂ (hB_pred), site-by-site |.| (hB_l1).
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/onp/"
for scen, lab in (("2", "0.4"), ("1", "0.5")):
    R = {d: [] for d in ("0.1", "0.2", "0.3")}
    for g in range(1, 16):
        for s in range(1, 11):
            T = {}
            for r in csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")):
                if r["scen"] == scen and r["who"] in ("N", "B0.1") and float(r["d"]) > 0: T.setdefault((r["d"], r["m"]), {})[r["who"]] = r
            for (d, m), t in T.items():
                if t["N"]["x"] == t["B0.1"]["x"]: continue
                f = lambda w: (float(t[w]["hB_pred"]), float(t[w]["hB_true"]), float(t[w]["hB_l1"]), float(t[w]["cap"]))
                R[d].append(f("N") + f("B0.1"))
    print("== c/f %s (runs where Belief-only differs from Nominal)" % lab)
    for d in ("0.1", "0.2", "0.3"):
        A = np.array(R[d]); n = len(A)
        sN = np.maximum(0, A[:, 0] - A[:, 1]); sB = np.maximum(0, A[:, 4] - A[:, 5])
        print("   TV %s (n %d): site-by-site |change of commitment|: N %.1f, B %.1f (B smaller in %2.0f%%) | shortfall max(0, predicted - actual): N %.1f, B %.1f | q90 of shortfall: N %.1f, B %.1f | shortfall / predicted: N %.2f%%, B %.2f%%" % (
            d, n, A[:, 2].mean(), A[:, 6].mean(), 100 * np.mean(A[:, 6] < A[:, 2] - 0.5), sN.mean(), sB.mean(), np.quantile(sN, .9), np.quantile(sB, .9), 100 * sN.mean() / A[:, 0].mean(), 100 * sB.mean() / A[:, 4].mean()))
