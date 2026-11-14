# reservation-cap diagnostic: per config, N decision's reservation at the cap under q, direction of moves under sampled beliefs, link to damage
import csv, glob, re, collections, os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
W = 300.0
def rank(v): return np.argsort(np.argsort(v)).astype(float)
def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12: return np.nan
    return float(np.corrcoef(rank(a), rank(b))[0, 1])
def load(cfg):
    R = {}
    for f in sorted(glob.glob(O + "tkh/tkh_%s_g*_s*.csv" % cfg)):
        lg = f[:-4] + ".log"
        if not os.path.exists(lg) or "TKH_DONE" not in open(lg, encoding="utf-8", errors="replace").read(): continue
        m = re.search(r"_g(\d+)_s(\d+)\.csv", f); D = collections.OrderedDict()
        for r in csv.DictReader(open(f, encoding="utf-8")):
            e = D.setdefault(r["x"], {"q": None, "J": [], "hB": [], "ncap": [], "hl": []})
            hl = [float(z) for z in r["hBlist"].split(";")]
            if r["k"] == "0": e["q"] = (float(r["J"]), float(r["hB"]), int(r["ncap"]), int(r["nopen"]), hl, float(r["hA"]))
            else: e["J"].append(float(r["J"])); e["hB"].append(float(r["hB"])); e["ncap"].append(int(r["ncap"])); e["hl"].append(hl)
        R[(int(m.group(1)), int(m.group(2)))] = D
    return R
ALL = {}
for cfg in sys.argv[1:]:
    R = load(cfg); ALL[cfg] = R
    if not R: print(cfg, "no data"); continue
    A = collections.defaultdict(list)
    for key, D in R.items():
        xs = list(D); VN = np.array([D[x]["q"][0] for x in xs]); n = int(np.argmin(VN)); e = D[xs[n]]
        Jq, hq, nc, no, hlq, hA = e["q"]; J = np.array(e["J"]); hB = np.array(e["hB"]); HL = np.array(e["hl"]); sc = abs(np.mean([np.mean(D[x]["J"]) for x in xs][0:1] + [min(np.mean(D[x]["J"]) for x in xs)])) / 100
        sc = abs(min(np.mean(D[x]["J"]) for x in xs)) / 100
        A["capshare"].append(nc / no); A["fill"].append(hq / (W * no))
        A["up"].append(np.mean(hB > hq + 1)); A["down"].append(np.mean(hB < hq - 1)); A["same"].append(np.mean(abs(hB - hq) <= 1))
        drop = (hq - hB) / (W * no) * 100; dmg = (J - Jq) / sc
        A["drop_mean"].append(drop.mean()); A["drop_q90"].append(np.quantile(drop, 0.9)); A["rise_q90"].append(np.quantile(-drop, 0.9))
        A["dmg_mean"].append(dmg.mean()); A["dmg_q90"].append(np.quantile(dmg, 0.9)); A["dmg_q10"].append(np.quantile(dmg, 0.1))
        A["rho_drop_dmg"].append(spear(drop, dmg))
        # store level: stores at cap under q vs interior under q -> share of draws moving down / up
        atc = np.array(hlq) >= W - 0.5
        if atc.any(): A["cap_store_down"].append(np.mean(HL[:, atc] < np.array(hlq)[atc] - 1))
        if (~atc).any(): A["int_store_down"].append(np.mean(HL[:, ~atc] < np.array(hlq)[~atc] - 1)); A["int_store_up"].append(np.mean(HL[:, ~atc] > np.array(hlq)[~atc] + 1))
        # across the 8 candidates: fill under q vs q90 damage, and the tail-best candidate's fill vs N's
        fills = np.array([D[x]["q"][1] / (W * D[x]["q"][3]) for x in xs]); caps = np.array([D[x]["q"][2] / D[x]["q"][3] for x in xs])
        dq90 = np.array([np.quantile(np.array(D[x]["J"]) - D[x]["q"][0], 0.9) / sc for x in xs]); tq = np.array([np.quantile(D[x]["J"], 0.9) for x in xs])
        A["rho_fill_dmg_cands"].append(spear(fills, dq90)); o = int(np.argmin(tq))
        A["oracle_fill_minus_N"].append(fills[o] - fills[n]); A["oracle_cap_minus_N"].append(caps[o] - caps[n]); A["oracle_differs"].append(float(o != n))
        A["cand_dmg_sd"].append(dq90.std()); A["oracle_gain"].append((tq[n] - tq[o]) / sc); A["key"].append(key)
    f = lambda k: np.nanmean(A[k])
    print("===== %s: %d runs" % (cfg, len(R)))
    print("  N decision under q: cap share %.2f, fill %.2f | draws: down %.2f same %.2f up %.2f | drop mean %+.1f%% q90 %+.1f%% rise q90 %+.1f%% (of cap)" % (f("capshare"), f("fill"), f("down"), f("same"), f("up"), f("drop_mean"), f("drop_q90"), f("rise_q90")))
    print("  N damage: mean %+.2f q10 %+.2f q90 %+.2f | within-run rank corr(drop, damage) %+.2f" % (f("dmg_mean"), f("dmg_q10"), f("dmg_q90"), f("rho_drop_dmg")))
    print("  stores at cap under q: P(down) %.2f | interior stores: P(down) %.2f P(up) %.2f" % (f("cap_store_down"), f("int_store_down") if A["int_store_down"] else float("nan"), f("int_store_up") if A["int_store_up"] else float("nan")))
    print("  across 8 candidates: rank corr(fill under q, q90 damage) %+.2f | sd of q90 damage %.2f | oracle gain %.2f, oracle differs %.0f%%, oracle fill - N fill %+.3f, cap share diff %+.3f" % (f("rho_fill_dmg_cands"), f("cand_dmg_sd"), f("oracle_gain"), 100 * f("oracle_differs"), f("oracle_fill_minus_N"), f("oracle_cap_minus_N")))
    print("  across runs: rank corr(drop q90, N q90 damage) %+.2f | rank corr(cap share, oracle gain) %+.2f | runs with all stores at cap %.0f%%: oracle gain %.2f vs others %.2f" % (
        spear(A["drop_q90"], A["dmg_q90"]), spear(A["capshare"], A["oracle_gain"]), 100 * np.mean(np.array(A["capshare"]) > 0.99),
        np.mean([g for g, c in zip(A["oracle_gain"], A["capshare"]) if c > 0.99] or [np.nan]), np.mean([g for g, c in zip(A["oracle_gain"], A["capshare"]) if c <= 0.99] or [np.nan])))
if "base" in ALL and "top" in ALL:
    d = []
    for k in ALL["base"]:
        if k not in ALL["top"]: continue
        for x, e in ALL["base"][k].items():
            if x in ALL["top"][k]: d.append(abs(e["q"][1] - ALL["top"][k][x]["q"][1]))
    if d: print("H38d base vs top, same x: |hB(q) diff| max %.2f mean %.3f over %d decisions" % (max(d), np.mean(d), len(d)))
