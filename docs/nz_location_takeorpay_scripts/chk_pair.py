# how different are Nominal's rows in the two evaluations (zss3 vs zss3c)?  same draws, different sets of decisions evaluated in the same process
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
D = {}; 
for g in range(1, 16):
    for s in range(1, 11):
        ra = {tuple(r[:5]): r for r in list(csv.reader(open(O + "zss3/eval_g%d_s%d.csv" % (g, s), encoding="utf-8")))[1:]}
        for r in list(csv.reader(open(O + "zss3c/eval_g%d_s%d.csv" % (g, s), encoding="utf-8")))[1:]:
            k = tuple(r[:5])
            if k in ra:
                for col, nm in ((5, "JK"), (6, "JU1"), (7, "JU.5"), (8, "JU0")):
                    D.setdefault((r[1], nm), []).append((float(r[col]) - float(ra[k][col])) / abs(float(ra[k][col])) * 100)
print("gamma | column | rows | share differing > 0.01% | mean signed diff (%) | mean abs diff (%)")
for gam in ("0.1", "0.3", "0.5", "1.0", "3.0", "10.0"):
    for nm in ("JK", "JU1"):
        v = np.array(D[(gam, nm)]); print("%5s | %-4s | %6d | %5.2f%% | %+.4f | %.4f" % (gam, nm, len(v), 100 * np.mean(np.abs(v) > 0.01), v.mean(), np.abs(v).mean()))
