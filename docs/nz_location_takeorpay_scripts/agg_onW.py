# -*- coding: utf-8 -*-
# Truth-unknown world ON THE SUPPORT (onW.jl output geo/on/{dec,eval}_g<g>_s<s>.csv): python agg_onW.py   (AO_ALL=1 also reads runs without ON_DONE)
#   truth p = probabilities on the same 10 scenarios at TV distance d from the uniform q̂; retailer belief = p (world T) or frequencies of 5 / 20 draws from p.
#   all values in % of the leader's nominal cost of N, |V_ss(N) + store cost|; negative = better than the reference.
import csv, os, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/" + os.environ.get("AO_DIR", "on") + "/"
ALL = os.environ.get("AO_ALL", "0") == "1"; q90 = lambda v: np.quantile(v, 0.9)
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";")]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def ms(v):
    v = np.array(v, float); return "%+6.2f +- %.2f" % (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else "%+6.2f" % v.mean()
R = {}   # (g, s) -> (dec rows, J[(d, m)][x][world] -> list)
for g in range(1, 16):
    for s in range(1, 11):
        lg = O + "on_g%d_s%d.log" % (g, s); fd = O + "dec_g%d_s%d.csv" % (g, s)
        done = os.path.exists(lg) and "ON_DONE" in open(lg, encoding="utf-8", errors="replace").read()
        if not (done or (ALL and os.path.exists(fd))): continue
        J = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        for r in csv.DictReader(open(O + "eval_g%d_s%d.csv" % (g, s), encoding="utf-8")): J[(r["d"], r["m"])][r["x"]][r["world"]].append(float(r["J"]))
        R[(g, s)] = (list(csv.DictReader(open(fd, encoding="utf-8"))), J)
print("runs done: %d (geographies %d)" % (len(R), len({g for g, _ in R})))
if not R: raise SystemExit
DS = sorted({r["d"] for dec, _ in R.values() for r in dec}, key=float); WORLDS = (("T", "retailer knows the truth"), ("5", "retailer sample of 5"), ("20", "retailer sample of 20"))
for scen in ("1", "2"):
    r0 = [r for r in next(iter(R.values()))[0] if r["scen"] == scen][0]; q0, c = float(r0["q0"]), float(r0["c"]); cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1])
    print("\n==================== c/v %.1f" % (c / 5))
    # T[(g, s)][(d, m)][who] = dec row
    T = {}
    for key, (dec, J) in R.items():
        t = collections.defaultdict(dict)
        for r in dec:
            if r["scen"] == scen: t[(r["d"], r["m"])][r["who"]] = r
        T[key] = t
    scale = {key: abs(float(next(iter(T[key].values()))["N"]["Vss"]) + cost(next(iter(T[key].values()))["N"]["x"])) / 100 for key in T}
    def costs(key, dm, who, w):   # all costs of the decision `who` under truth dm in world w (one per belief draw)
        x = T[key][dm][who]["x"]; return np.array(R[key][1][dm][x][w]) + cost(x)
    def table(a, b, w, dsel, keys):   # per run: mean and q90 (over truths and belief draws) of cost(a) - cost(b), difference of the pooled q90s
        mn, tl = [], []
        for key in keys:
            dms = [dm for dm in T[key] if dm[0] in dsel and a in T[key][dm] and b in T[key][dm]]
            if not dms: continue
            A = np.concatenate([costs(key, dm, a, w) for dm in dms]); B = np.concatenate([costs(key, dm, b, w) for dm in dms])
            mn.append((A.mean() - B.mean()) / scale[key]); tl.append((q90(A) - q90(B)) / scale[key])
        return mn, tl
    keys = sorted(T)
    print("runs %d | truths per run %d" % (len(keys), len(next(iter(T.values())))))
    for dsel, dl in [([d], "d = " + d) for d in DS] + [(DS, "all d")]:
        print("---- %s" % dl)
        same = lambda a, b: 100 * np.mean([T[k][dm][a]["x"] == T[k][dm][b]["x"] for k in keys for dm in T[k] if dm[0] in dsel])
        print("   N = N* %.0f%% | Nr = N* %.0f%% | Nd = N* %.0f%%" % (same("N", "N*"), same("Nr", "N*"), same("Nd", "N*")))
        for w, wl in WORLDS:
            print("   -- %s" % wl)
            for lab, a, b in (("N  - N*", "N", "N*"), ("Nr - N* (own demand weights, true orders)", "Nr", "N*"), ("Nd - N* (true demand weights, orders from q̂)", "Nd", "N*"),
                              ("B0.1 - N", "B0.1", "N"), ("D0.1 - N", "D0.1", "N"), ("D0.3 - N", "D0.3", "N"), ("C - N", "C", "N"), ("C - B0.1", "C", "B0.1")):
                mn, tl = table(a, b, w, dsel, keys)
                if not mn: continue
                line = "      %-46s mean %s | q90 %s | n %3d" % (lab, ms(mn), ms(tl), len(mn))
                if a in ("B0.1", "D0.3") and b == "N":
                    for sl, ks in (("seeds 1-3", [k for k in keys if k[1] <= 3]), ("seeds 4-10", [k for k in keys if k[1] > 3])):
                        m2, t2 = table(a, b, w, dsel, ks)
                        if m2: line += " | %s: mean %s q90 %s" % (sl, ms(m2), ms(t2))
                print(line)
        # how the leader's estimate ranks N against N* under the four views (negative = N looks better than N*)
        gp = lambda k, dm, col: (float(T[k][dm]["N"][col]) + cost(T[k][dm]["N"]["x"]) - float(T[k][dm]["N*"][col]) - cost(T[k][dm]["N*"]["x"])) / scale[k]
        rows = np.array([[gp(k, dm, col) for col in ("Vss", "Vrs", "Vds", "Vpp")] for k in keys for dm in T[k] if dm[0] in dsel])
        print("   -- estimated cost of N minus N*: what N sees %+.2f | own weights + true orders %+.2f | true weights + orders from q̂ %+.2f | truth %+.2f" % tuple(rows.mean(axis=0)))
        tot = rows[:, 3] - rows[:, 0]
        print("      misranking %+.2f = orders first: orders %+.2f + demand %+.2f | demand first: demand %+.2f + orders %+.2f" % (tot.mean(), (rows[:, 1] - rows[:, 0]).mean(), (rows[:, 3] - rows[:, 1]).mean(), (rows[:, 2] - rows[:, 0]).mean(), (rows[:, 3] - rows[:, 2]).mean()))
        hh = np.array([[float(T[k][dm]["N"][col]) for col in ("hB_pred", "hB_true", "hB_l1")] for k in keys for dm in T[k] if dm[0] in dsel])
        print("      commitment at the leader's sites for N: predicted %.0f, actual %.0f, mean |.| %.1f, site-by-site |.| %.1f" % (hh[:, 0].mean(), hh[:, 1].mean(), np.abs(hh[:, 0] - hh[:, 1]).mean(), hh[:, 2].mean()))
