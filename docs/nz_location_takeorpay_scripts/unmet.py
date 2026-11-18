import os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/" + sys.argv[1] + "/"
sh = []; ncut = []
for g in range(1, 16):
    for s in range(1, 4):
        f = O + "flow_g%d_s%d.csv" % (g, s)
        if not os.path.exists(f): continue
        U = {}
        for l in open(f, encoding="utf-8").read().splitlines()[1:]:
            p = l.split(";")[-11:]
            if p[1] == "U": U[int(p[5])] = (float(p[8]), float(p[10]))
        if not U: continue
        mu = sum(a for a, _ in U.values()); u = sum(b for _, b in U.values()); sh.append(u / mu); ncut.append(np.mean([b > 0.5 * a for a, b in U.values()]))
print("runs %d | unmet demand / mean demand: mean %.1f%%, min %.1f%%, max %.1f%% | customers with more than half unmet: %.0f%%" % (len(sh), 100 * np.mean(sh), 100 * min(sh), 100 * max(sh), 100 * np.mean(ncut)))
