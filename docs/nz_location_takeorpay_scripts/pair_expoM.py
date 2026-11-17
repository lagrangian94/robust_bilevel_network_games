# -*- coding: utf-8 -*-
# Paired comparison expo15 (multiplier 1) vs expo16 (1.3333): python pair_expoM.py
import csv
import numpy as np
OLD = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/expo/"
def load(pre, cfg):
    D = {}
    for g in range(1, 16):
        for s in range(1, 11):
            for r in csv.DictReader(open(OLD + "%s_%s_g%d_s%d.csv" % (pre, cfg, g, s), encoding="utf-8")): D[(r["cv"], g, s)] = r
    return D
A = {c: (load("expoS", c), load("expoR", c), load("expo", c)) for c in ("expo15", "expo16")}
def se(v): v = np.array(v, float); return v.std(ddof=1) / np.sqrt(len(v))
for cv in ("0.5", "0.4", "0.2"):
    print("== c/v", cv)
    out = {}
    for c in A:
        S, R, X = A[c]; band = []; cap = []; spot = []; adv = []; advp = []; reg = {}; dq = {}
        for g in range(1, 16):
            JN = np.array([float(S[(cv, g, s)]["JN"]) for s in range(1, 11)]); JD = np.array([float(S[(cv, g, s)]["JD3"]) for s in range(1, 11)]); JB = np.array([float(S[(cv, g, s)]["Jbest"]) for s in range(1, 11)])
            sc = abs(JN).mean() / 100; reg[g] = ((JN - JB) / sc).mean(); dq[g] = (np.quantile(JD, 0.9) - np.quantile(JN, 0.9)) / sc
            band.append(np.mean([float(R[(cv, g, s)]["band"]) for s in range(1, 11)])); cap.append(np.mean([float(R[(cv, g, s)]["capN"]) for s in range(1, 11)])); spot.append(np.mean([float(R[(cv, g, s)]["spotT"]) for s in range(1, 11)]))
            adv.append(np.mean([float(X[(cv, g, s)]["adv0.1"]) for s in range(1, 11)]))
            advp.append(np.mean([100 * float(X[(cv, g, s)]["adv0.1"]) / abs(float(X[(cv, g, s)]["VN"])) if abs(float(X[(cv, g, s)]["adv0.1"])) < 1e9 else np.nan for s in range(1, 11)]))
        out[c] = dict(band=np.array(band), cap=np.array(cap), spot=np.array(spot), adv=np.array(adv), advp=np.array(advp), reg=np.array([reg[g] for g in range(1, 16)]), dq=np.array([dq[g] for g in range(1, 16)]))
    a, b = out["expo15"], out["expo16"]
    for k, lab in (("band", "band"), ("cap", "capN"), ("spot", "spotT"), ("reg", "N regret %"), ("adv", "adv0.1 (raw col)"), ("advp", "adv0.1 as 100*col/|VN|"), ("dq", "D0.3-N q90 %")):
        d = b[k] - a[k]
        print("  %-24s m1 %8.3f  m1.33 %8.3f  diff %+8.3f +- %.3f  ratio %.2f  | geos up %d down %d" % (lab, a[k].mean(), b[k].mean(), d.mean(), se(d), b[k].mean() / a[k].mean() if abs(a[k].mean()) > 1e-9 else float("nan"), (d > 1e-6).sum(), (d < -1e-6).sum()))
    print("  D-N q90 more negative at m1.33 in %d of 15 geographies" % (b["dq"] < a["dq"] - 0.05).sum())
