# -*- coding: utf-8 -*-
# sel_dwps3d_g_s.csv = sel_dwps3x_g_s.csv + rows of Coupled at radius 0.3 with coupling radius 0, 0.3, 0.6 (labels Cd0.0, Cd0.3, Cd0.6; selx_dwps3_<lab>_g_s.csv).  Delta sweep (ledger 4.34).
import os
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/dw/"
LABS = ("Cd0.0", "Cd0.3", "Cd0.6"); n = 0; miss = []
for g in range(1, 16):
    for s in range(1, 11):
        base = open(O + "sel_dwps3x_g%d_s%d.csv" % (g, s), encoding="utf-8").read().splitlines(); add = []
        for lab in LABS:
            f = O + "selx_dwps3_%s_g%d_s%d.csv" % (lab, g, s)
            if not os.path.exists(f): miss.append((lab, g, s)); continue
            r = open(f, encoding="utf-8").read().splitlines()[1:]; assert len(r) == 2 and all(x.split(",")[3] == lab for x in r), f; add += r
        if any(m[1:] == (g, s) for m in miss): continue
        with open(O + "sel_dwps3d_g%d_s%d.csv" % (g, s), "w", encoding="utf-8", newline="\n") as fo: fo.write("\n".join(base + add) + "\n")
        n += 1
print("SELD written %d runs, missing pieces %d %s" % (n, len(miss), miss[:5]))
