import csv, os
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";") if p]; return {int(a): float(b) for a, b in pr}
for w in ("B0.1", "D0.1", "D0.3"):
    n = same = less = eq = more = sub = 0
    for g in range(1, 16):
        for s in range(1, 11):
            d = {r["who"]: r for r in csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
            a, b = par(d["N"]["x"]), par(d[w]["x"])
            if a == b: continue
            n += 1; same += set(a) == set(b); ca, cb = sum(a.values()), sum(b.values()); less += cb < ca; eq += cb == ca; more += cb > ca
            sub += (set(b) <= set(a)) and set(a) != set(b)
    print("%s: differs %d | same sites (only sizes) %.0f%% | subset of N sites %.0f%% | cap less %.0f%% same %.0f%% more %.0f%%" % (w, n, 100*same/n, 100*sub/n, 100*less/n, 100*eq/n, 100*more/n))
