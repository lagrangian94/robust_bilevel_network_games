# how much could ANY decision rule gain? fixed-decision tail oracle vs per-belief clairvoyant (picks the best of the 8 candidates for each belief draw)
import csv, glob, re, collections, os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
q90 = lambda v: np.quantile(v, 0.9)
for cfg in sys.argv[1:]:
    for nf in [5, 10]:
        rows = []
        for f in sorted(glob.glob(O + "tkd/tkd_%s_g*_s*.csv" % cfg)):
            J = collections.OrderedDict(); VN = {}
            for r in csv.DictReader(open(f, encoding="utf-8")):
                if int(r["nf"]) == nf: J.setdefault(r["x"], []).append(float(r["J"])); VN[r["x"]] = float(r["VN"])
            xs = list(J); M = np.array([J[x] for x in xs]); n = int(np.argmin([VN[x] for x in xs])); sc = abs(M.mean(axis=1).min()) / 100
            o = int(np.argmin(np.quantile(M, 0.9, axis=1))); om = int(np.argmin(M.mean(axis=1))); cl = M.min(axis=0)
            rows.append(((q90(M[o]) - q90(M[n])) / sc, (M[om].mean() - M[n].mean()) / sc, (cl.mean() - M[n].mean()) / sc, (q90(cl) - q90(M[n])) / sc,
                         (q90(M[n]) - M[n].mean()) / sc, np.mean(np.argmin(M, axis=0) == n) * 100, np.mean((M[n] - cl) / sc)))
        A = np.array(rows)
        print("%-8s N_F=%2d n=%d | N: q90 - mean = %.1f | fixed-decision oracle: q90 %+.2f, mean %+.2f | per-belief clairvoyant: mean %+.2f, q90 %+.2f | N is best in %.0f%% of draws" % (cfg, nf, len(A), A[:, 4].mean(), A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(), A[:, 3].mean(), A[:, 5].mean()))
