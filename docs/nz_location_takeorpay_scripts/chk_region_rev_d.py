# Is the leader's revenue driven by the demand of the regions where its stores are, or by total demand?  Nominal's decision (scen 2), belief q, 150 runs.
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/" + __import__("os").environ.get("AM_DIR", "mech") + "/"; OX = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
def par(x): return [int(p.split(":")[0]) for p in x.strip("{}").split(";") if p]
out = []; nreg = []; creg = []; areg = []; shr = []
for g in range(1, 16):
    rows = list(csv.DictReader(open(OX + "xid_g%d_s1.csv" % g, encoding="utf-8"), delimiter=";"))
    sreg = [int(r["region"]) for r in rows if r["kind"] in "AB"]; A = [int(r["region"]) for r in rows if r["kind"] == "A"]
    cust = [(int(r["region"]), float(r["mu"])) for r in rows if r["kind"] == "C"]
    T = {}
    for r in csv.reader(open(OX + "xi_g%d.csv" % g)):
        T.setdefault(int(r[0]), {})[r[1].split(":")[0]] = np.array([float(x) for x in r[2:]])
    for s in range(1, 4):
        dec = {r["who"]: r for r in csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        for r in csv.DictReader(open(O + "mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
            if r["x"] == dec["N"]["x"] and r["bel"] == "q": Z = np.array([float(r["Z%d" % k]) for k in range(1, 11)]); cap = float(r["cap"])
        own = sorted(set(sreg[len(A) + i - 1] for i in par(dec["N"]["x"]))); t = T[s]; tot = t["tot"]
        d_own = sum(t["reg%d" % r] for r in own); d_oth = tot - d_own
        nreg.append(len(own)); shr.append(d_own.mean() / tot.mean()); areg.append(len(set(A)))
        c = lambda a, b: np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
        out.append((c(-Z, tot), c(-Z, d_own), c(-Z, d_oth), cap / d_own.mean(), cap / tot.mean()))
out = np.array(out); nreg = np.array(nreg)
print("regions covered by the Nominal stores: 1 region %.0f%%, 2 regions %.0f%%, 3 regions %.0f%%;  own regions hold %.0f%% of total demand" % (100 * np.mean(nreg == 1), 100 * np.mean(nreg == 2), 100 * np.mean(nreg == 3), 100 * np.mean(shr)))
print("leader capacity / demand of own regions %.2f, / total demand %.2f" % (np.nanmean(out[:, 3]), np.nanmean(out[:, 4])))
print("corr over scenarios of the leader's revenue with: total demand %.2f | demand of the regions with a leader store %.2f | demand of the other regions %.2f" % tuple(np.nanmean(out[:, k]) for k in range(3)))
for k in (1, 2, 3):
    m = nreg == k
    if m.any(): print("  stores in %d region(s) (%3d runs): total %.2f | own %.2f | other %.2f | own share of demand %.0f%%" % (k, m.sum(), np.nanmean(out[m, 0]), np.nanmean(out[m, 1]), np.nanmean(out[m, 2]), 100 * np.mean(np.array(shr)[m])))
