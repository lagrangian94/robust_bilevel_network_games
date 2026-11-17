# N_B = 2 decisions of Nominal and Data-DR from the stored full enumeration (vn_dwps3_g_s.csv): subset with at most 2 sites.  Writes geo/dw/sel_nb2_g_s.csv.
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/dw/"
def par(x): return {int(p.split(":")[0]): float(p.split(":")[1]) for p in x.strip("{}").split(";") if p}
SC = {"1": (0.0, 2.5), "2": (150.0, 2.0)}; ST = {k: [] for k in SC}
for g in range(1, 16):
    for s in range(1, 11):
        R = list(csv.DictReader(open(O + "vn_dwps3_g%d_s%d.csv" % (g, s), encoding="utf-8")))
        n = np.array([int(r["n"]) for r in R]); cap = np.array([float(r["cap"]) for r in R]); lab = [r["x"] for r in R]
        V = {"N": np.array([float(r["V0"]) for r in R]), "D0.1": np.array([float(r["VD0.1"]) for r in R]), "D0.2": np.array([float(r["VD0.2"]) for r in R])}
        out = ["scen,q0,c,model,x"]
        for k, (q0, c) in SC.items():
            cost = q0 * n + c * cap; pick = {}
            for nb in (3, 2):
                m = n <= nb
                for w, v in V.items():
                    t = np.where(m, v + cost, np.inf); i = min(np.flatnonzero(t <= t.min() + 1e-9), key=lambda j: lab[j]); pick[(nb, w)] = i
                    if nb == 2: out.append("%s,%s,%s,%s,%s" % (k, q0, c, w, lab[i]))
            ST[k].append((g, s, {kk: (lab[i], cap[i], n[i], V["N"][i] + cost[i]) for kk, i in pick.items()}))
        open(O + "sel_nb2_g%d_s%d.csv" % (g, s), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
for k in ("2", "1"):
    R = ST[k]; print("== cost table scen %s, %d runs" % (k, len(R)))
    for nb in (3, 2):
        for w in ("N", "D0.1", "D0.2"):
            cap = np.mean([r[2][(nb, w)][1] for r in R]); ns = np.mean([r[2][(nb, w)][2] for r in R]); u4 = np.mean([":400" in r[2][(nb, w)][0] for r in R])
            dif = [r for r in R if r[2][(nb, w)][0] != r[2][(nb, "N")][0]]; s_ = "  N_B<=%d %-5s cap %4.0f stores %.2f uses400 %2.0f%%" % (nb, w, cap, ns, 100 * u4)
            if w != "N" and dif:
                a = [(par(r[2][(nb, "N")][0]), par(r[2][(nb, w)][0])) for r in dif]
                less = np.mean([sum(b.values()) < sum(x.values()) for x, b in a]); same = np.mean([set(b) == set(x) for x, b in a])
                both = np.mean([any(b.get(i, 0) > x.get(i, 0) for i in set(x) | set(b)) and any(b.get(i, 0) < x.get(i, 0) for i in set(x) | set(b)) for x, b in a])
                s_ += " | differs %2.0f%%: cap less %2.0f%%, same sites %2.0f%%, some site up and some down %2.0f%%" % (100 * len(dif) / len(R), 100 * less, 100 * same, 100 * both)
            print(s_)
    loss = [100 * (r[2][(2, "N")][3] - r[2][(3, "N")][3]) / abs(r[2][(3, "N")][3]) for r in R]
    print("  Nominal objective: N_B<=2 is worse than N_B<=3 by %.2f%% on average (0 in %.0f%% of the runs)" % (np.mean(loss), 100 * np.mean(np.array(loss) < 1e-9)))
