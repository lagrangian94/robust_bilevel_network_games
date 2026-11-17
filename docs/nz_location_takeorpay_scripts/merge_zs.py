# merge the evaluations zss3 (N, B0.1, B0.2, D0.1, D0.2) and zss3c (N, C): same draws (seed base 9595).  Checks that Nominal's rows agree, writes geo/zss3m/{dec,eval}_g_s.csv
import csv, os, sys
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
A, B, M = (sys.argv + ["zss3", "zss3c", "zss3m"])[1:4]; os.makedirs(O + M, exist_ok=True); n = 0; bad = 0; mx = 0.0
for g in range(1, 16):
    for s in range(1, 11):
        fa, fb = O + A + "/eval_g%d_s%d.csv" % (g, s), O + B + "/eval_g%d_s%d.csv" % (g, s)
        if not (os.path.exists(fa) and os.path.exists(fb)): continue
        ra = list(csv.reader(open(fa, encoding="utf-8"))); rb = list(csv.reader(open(fb, encoding="utf-8"))); assert ra[0] == rb[0]
        da = {tuple(r[:5]): r for r in ra[1:]}
        for r in rb[1:]:
            k = tuple(r[:5])
            if k in da:
                d = max(abs(float(x) - float(y)) for x, y in zip(r[5:], da[k][5:])); mx = max(mx, d); bad += d > 1e-3
            else: da[k] = r
        with open(O + M + "/eval_g%d_s%d.csv" % (g, s), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(ra[0]); w.writerows(da.values())
        qa = list(csv.reader(open(O + A + "/dec_g%d_s%d.csv" % (g, s), encoding="utf-8"))); qb = list(csv.reader(open(O + B + "/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")))
        with open(O + M + "/dec_g%d_s%d.csv" % (g, s), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(qa[0]); w.writerows(qa[1:]); w.writerows([r for r in qb[1:] if r[4] == "C"])
        n += 1
print("merged %d runs; overlapping rows that disagree by more than 1e-3: %d (largest difference %.2e)" % (n, bad, mx))
