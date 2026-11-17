import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
W = []; X = []; NA = []; N2 = []; ALL = []
for g in range(1, 16):
    rows = list(csv.DictReader(open(O + "xid_g%d_s1.csv" % g, encoding="utf-8"), delimiter=";"))
    St = [(float(r["x"]), float(r["y"]), int(r["region"]), r["kind"]) for r in rows if r["kind"] in "AB"]; C = [(float(r["x"]), float(r["y"]), int(r["region"])) for r in rows if r["kind"] == "C"]
    for (x, y, rc) in C:
        d = np.array([np.hypot(x - s[0], y - s[1]) for s in St]); reg = np.array([s[2] for s in St]); isA = np.array([s[3] == "A" for s in St])
        pos = d > 0
        if (pos & (reg == rc)).any(): W.append(d[pos & (reg == rc)].mean())
        if (reg != rc).any(): X.append(d[reg != rc].mean())
        NA.append(d[isA].min()); N2.append(np.sort(d)[1]); ALL.append(d.mean())
print("distance from a customer to a store site: same region mean %.0f | other region mean %.0f | all sites mean %.0f" % (np.mean(W), np.mean(X), np.mean(ALL)))
print("to the nearest existing-operator site: mean %.0f, median %.0f | to the nearest other city: mean %.0f" % (np.mean(NA), np.median(NA), np.mean(N2)))
