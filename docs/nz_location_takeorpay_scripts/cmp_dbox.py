# -*- coding: utf-8 -*-
# Data-DR decisions under the old box (vn_dwpl75_*: width 2r/sqrt(S)) and the new box (vn_<CFG>_*: width 4 r q̂), from the enumeration files only
# (vn_* is written right after the enumeration stage, so this works while the runs are still in the candidate stage).  python cmp_dbox.py [cfg]
import csv, os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/dw/"
CFG = sys.argv[1] if len(sys.argv) > 1 else "dwps2b"; SC = [(0.0, 2.5, "0.5"), (150.0, 2.0, "0.4")]
def load(cfg, g, s):
    f = O + "vn_%s_g%d_s%d.csv" % (cfg, g, s)
    if not os.path.exists(f): return None
    R = list(csv.DictReader(open(f, encoding="utf-8")))
    return R if len(R) == 8463 else None
def pick(R, col, q0, c):
    v = [(float(r[col]) + q0 * int(r["n"]) + c * float(r["cap"]), r["x"]) for r in R]; i = min(range(len(v)), key=lambda k: v[k]); return R[i]
rows = {k: [] for k in ("0.5", "0.4")}; n = 0
for g in range(1, 16):
    for s in range(1, 11):
        A, B = load("dwpl75", g, s), load(CFG, g, s)
        if A is None or B is None: continue
        n += 1
        for q0, c, lab in SC:
            N = pick(A, "V0", q0, c); N2 = pick(B, "V0", q0, c); assert N["x"] == N2["x"], "Nominal differs"
            rec = {"N": float(N["cap"])}
            for e in ("0.1", "0.2", "0.3"):
                o, w = pick(A, "VD" + e, q0, c), pick(B, "VD" + e, q0, c)
                rec["o" + e] = (float(o["cap"]), o["x"] != N["x"]); rec["w" + e] = (float(w["cap"]), w["x"] != N["x"], w["x"] == o["x"])
            rows[lab].append(rec)
print("runs with both enumerations: %d (config %s)" % (n, CFG))
for lab in ("0.4", "0.5"):
    R = rows[lab]
    if not R: continue
    print("== c/f %s | Nominal capacity %.0f" % (lab, np.mean([r["N"] for r in R])))
    for e in ("0.1", "0.2", "0.3"):
        print("   radius %s: old box: capacity %.0f, differs from N %3.0f%% | new box: capacity %.0f, differs from N %3.0f%% | same decision old/new %3.0f%% | new has more capacity than old %3.0f%%, less %3.0f%%" % (
            e, np.mean([r["o" + e][0] for r in R]), 100 * np.mean([r["o" + e][1] for r in R]), np.mean([r["w" + e][0] for r in R]), 100 * np.mean([r["w" + e][1] for r in R]),
            100 * np.mean([r["w" + e][2] for r in R]), 100 * np.mean([r["w" + e][0] > r["o" + e][0] for r in R]), 100 * np.mean([r["w" + e][0] < r["o" + e][0] for r in R])))
