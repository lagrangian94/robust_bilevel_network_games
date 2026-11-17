# summarize a per-cell output of agg_zsW_cells.py: c/f 0.4 section, average of the 20 cells with gamma <= 1, the calm cell (gamma 10, zeta 0), range over the 30 cells
import re, sys, numpy as np
L = open(sys.argv[1], encoding="utf-8").read().splitlines(); i0 = next(i for i, l in enumerate(L) if "c/f 0.4" in l); cur = None; T = {}
for l in L[i0:]:
    if l.startswith("-- world"): cur = l[3:].split("(rows")[0].strip(); T[cur] = {}
    elif l.strip().startswith("gamma") and cur: p = l.split(); T[cur][float(p[1])] = [float(re.split("±", x)[0]) for x in p[2:]]
for k, t in T.items():
    if "level" in k or len(t) < 6: continue
    if not any(m in k for m in ("| mean", "| q95")): continue
    a = np.array([t[g] for g in (0.1, 0.3, 0.5, 1.0)]); al = np.array([t[g] for g in sorted(t)])
    print("%-52s gamma<=1 avg %+5.2f | calm cell %+5.2f | 30 cells min %+5.2f max %+5.2f" % (k, a.mean(), t[10.0][0], al.min(), al.max()))
