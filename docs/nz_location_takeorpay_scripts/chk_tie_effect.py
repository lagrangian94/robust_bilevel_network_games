# -*- coding: utf-8 -*-
# size of the tie effect at the decisions that matter: V_ss recomputed with a warm-started model (splitW.jl) vs the stored value from a fresh model (vn files)
import csv, numpy as np
B = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
out = {}
for g in range(1, 16):
    sp = list(csv.DictReader(open(B + "tw/split_g%d.csv" % g, encoding="utf-8")))
    for s in range(1, 11):
        cfg = "dw75" if s <= 3 else "dwl75"; vn = {}
        for l in open(B + "dw/vn_%s_g%d_s%d.csv" % (cfg, g, s), encoding="utf-8").read().splitlines()[1:]:
            f = l.split(","); vn[f[0]] = float(f[3])
        for r in sp:
            if int(r["seed"]) == s and r["scen"] == "1": out.setdefault(r["who"], []).append((float(r["Vss"]) - vn[r["x"]]) / (abs(float(r["VB"]) + 2.5 * float(r["cap"])) / 100))
for w, v in out.items():
    v = np.abs(np.array(v)); print("%-3s %3d rows: |V_ss warm - fresh| in %% of cost: mean %.3f, median %.3f, q90 %.3f, max %.3f, share > 0.1%%: %.0f%%, > 0.5%%: %.0f%%" % (w, len(v), v.mean(), np.median(v), np.quantile(v, .9), v.max(), 100 * np.mean(v > 0.1), 100 * np.mean(v > 0.5)))
