# -*- coding: utf-8 -*-
# Does "retailer believes q̂" push the leader to less capacity than "retailer believes the truth"?  For each truth p (geo/onp, price-based):
#   Nd(p) = argmin_x J(x; p, q̂)  (true demand weights, orders predicted from q̂),   N*(p) = argmin_x J(x; p, p)  (retailer follows the truth).
# compare their capacities, split by the direction of p (commitment of the informed retailer for N minus the predicted one).
import csv, os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";") if p]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
for d_, lab in (("onp", "price-based"), ("on", "old formula")):
    for scen, cf in (("2", "0.4"), ("1", "0.5")):
        rows = []
        for g in range(1, 16):
            for s in range(1, 11):
                f = O + d_ + "/dec_g%d_s%d.csv" % (g, s)
                if not os.path.exists(f): continue
                T = {}
                for r in csv.DictReader(open(f, encoding="utf-8")):
                    if r["scen"] == scen and float(r["d"]) > 0: T.setdefault((r["d"], r["m"]), {})[r["who"]] = r
                for (d, m), t in T.items():
                    if not all(w in t for w in ("N", "N*", "Nd", "Nr", "D0.3")): continue
                    c = lambda w: float(t[w]["cap"]); ns = lambda w: len(par(t[w]["x"])[0])
                    rows.append((float(d), float(t["N"]["hB_true"]) - float(t["N"]["hB_pred"]), c("N"), c("Nd"), c("N*"), c("Nr"), c("D0.3"), ns("Nd"), ns("N*"), float(t["Nd"]["hB_pred"]), float(t["N*"]["hB_true"])))
        A = np.array(rows)
        print("== %s, c/f %s: %d (truth, run) pairs" % (lab, cf, len(A)))
        print("   %-34s  n   | cap N  | cap Nd (belief q̂) | cap N* (belief = truth) | cap Nr | cap D0.3 | Nd < N*  Nd = N*  Nd > N* | commitment: Nd predicts, N* gets" % "direction of the truth p")
        for nm, m in (("low-demand side (commits less > 5)", A[:, 1] < -5), ("about the same", np.abs(A[:, 1]) <= 5), ("high-demand side (commits more > 5)", A[:, 1] > 5), ("low-demand side, d = 0.3 only", (A[:, 1] < -5) & (A[:, 0] == 0.3))):
            B = A[m]
            print("   %-34s %4d | %5.0f  | %5.0f             | %5.0f                   | %5.0f  | %5.0f    | %3.0f%%    %3.0f%%    %3.0f%%  | %5.0f, %5.0f" % (nm, len(B), B[:, 2].mean(), B[:, 3].mean(), B[:, 4].mean(), B[:, 5].mean(), B[:, 6].mean(),
                  100 * np.mean(B[:, 3] < B[:, 4]), 100 * np.mean(B[:, 3] == B[:, 4]), 100 * np.mean(B[:, 3] > B[:, 4]), B[:, 9].mean(), B[:, 10].mean()))
