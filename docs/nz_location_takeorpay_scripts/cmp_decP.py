# -*- coding: utf-8 -*-
# old formula (geo/on) vs price-based (geo/onp) decisions: capacity, tier-400 use, share different from N
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";") if p]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def load(d):
    R = {}
    for g in range(1, 16):
        for s in range(1, 11):
            for r in csv.DictReader(open(O + d + "/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")):
                if r["who"] in ("N", "B0.1", "D0.1", "D0.3", "C") and (g, s, r["scen"], r["who"]) not in R: R[(g, s, r["scen"], r["who"])] = r
    return R
A, B = load("on"), load("onp")
for scen in ("1", "2"):
    print("== scen", scen)
    for nm, R in (("old", A), ("price", B)):
        keys = [(g, s) for g in range(1, 16) for s in range(1, 11)]
        capN = np.array([sum(par(R[(g, s, scen, "N")]["x"])[1]) for g, s in keys]); nst = np.array([len(par(R[(g, s, scen, "N")]["x"])[0]) for g, s in keys])
        t400 = np.mean([400.0 in par(R[(g, s, scen, "N")]["x"])[1] for g, s in keys]); hb = np.mean([float(R[(g, s, scen, "N")]["hB_pred"]) for g, s in keys])
        line = "%-5s N: stores %.2f capacity %.0f tier400 %.0f%% predicted commitment %.0f |" % (nm, nst.mean(), capN.mean(), 100 * t400, hb)
        for w in ("B0.1", "D0.1", "D0.3", "C"):
            ks = [k for k in keys if (k[0], k[1], scen, w) in R]
            if not ks: continue
            df = np.mean([R[(g, s, scen, w)]["x"] != R[(g, s, scen, "N")]["x"] for g, s in ks]); cp = np.mean([sum(par(R[(g, s, scen, w)]["x"])[1]) for g, s in ks])
            less = np.mean([sum(par(R[(g, s, scen, w)]["x"])[1]) < sum(par(R[(g, s, scen, "N")]["x"])[1]) for g, s in ks]); more = np.mean([sum(par(R[(g, s, scen, w)]["x"])[1]) > sum(par(R[(g, s, scen, "N")]["x"])[1]) for g, s in ks])
            t4 = np.mean([400.0 in par(R[(g, s, scen, w)]["x"])[1] for g, s in ks])
            line += " %s: differs %.0f%% cap %.0f (less %.0f%% more %.0f%%) t400 %.0f%% n %d |" % (w, 100 * df, cp, 100 * less, 100 * more, 100 * t4, len(ks))
        print(line)
    keys = [(g, s) for g in range(1, 16) for s in range(1, 11)]
    ca = np.array([sum(par(A[(g, s, scen, "N")]["x"])[1]) for g, s in keys]); cb = np.array([sum(par(B[(g, s, scen, "N")]["x"])[1]) for g, s in keys])
    print("   N price vs old: same decision %.0f%% | capacity larger %.0f%% smaller %.0f%% | same sites %.0f%%" % (100 * np.mean([A[(g, s, scen, "N")]["x"] == B[(g, s, scen, "N")]["x"] for g, s in keys]), 100 * np.mean(cb > ca), 100 * np.mean(cb < ca),
          100 * np.mean([par(A[(g, s, scen, "N")]["x"])[0] == par(B[(g, s, scen, "N")]["x"])[0] for g, s in keys])))
