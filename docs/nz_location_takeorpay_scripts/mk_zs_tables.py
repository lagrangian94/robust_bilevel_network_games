# -*- coding: utf-8 -*-
# parse agg_zsW_cells_out.txt -> markdown tables (cell = "mean / q95") for the report; python mk_zs_tables.py <c/f>
import re, sys, io
cf = sys.argv[1]; T = {}; scen = None; key = None
for l in io.open("agg_zsW_cells_out.txt", encoding="utf-8"):
    m = re.match(r"=+ c/f ([\d.]+)", l)
    if m: scen = m.group(1); continue
    m = re.match(r"-- world (.+?) \| (\S+) - N \| (\w+)", l)
    if m: key = (scen, m.group(1), m.group(2), m.group(3)); T[key] = {}; continue
    if l.startswith("-- world"): key = None; continue
    m = re.match(r"\s+gamma (\S+)\s+(.*)", l)
    if m and key: T[key][m.group(1)] = [(float(a), float(b)) for a, b in re.findall(r"([+-]?\d+\.\d+)±(\d+\.\d+)", m.group(2))]
ZET = ["0", "0.5", "1", "2", "5"]; GAM = ["0.1", "0.3", "0.5", "1.0"]
def tab(world, who):
    mn, q = T[(cf, world, who, "mean")], T[(cf, world, who, "q95")]
    out = ["| $\gamma$ \ $\zeta$ | " + " | ".join(ZET) + " |", "|---|" + "---|" * 5]
    for g in GAM: out.append("| %s | " % g.rstrip("0").rstrip(".") + " | ".join("$%+.2f$ / $%+.2f$" % (mn[g][i][0], q[g][i][0]) for i in range(5)) + " |")
    se = lambda d: (min(v[1] for g in GAM for v in d[g]), max(v[1] for g in GAM for v in d[g]))
    rng = lambda d: (min(v[0] for g in GAM for v in d[g]), max(v[0] for g in GAM for v in d[g]))
    return "\n".join(out), se(mn), se(q), rng(mn), rng(q)
for world in ["K (truth q̂, belief drawn)", "U rho = 1.0", "U rho = 0.5", "U rho = 0.0"]:
    for who in ["B0.1", "D0.1", "D0.3"]:
        if (cf, world, who, "mean") not in T: continue
        t, sm, sq, rm, rq = tab(world, who)
        print("### %s | %s | mean range %+.2f..%+.2f (SE %.2f-%.2f) | q95 range %+.2f..%+.2f (SE %.2f-%.2f)" % (world, who, rm[0], rm[1], sm[0], sm[1], rq[0], rq[1], sq[0], sq[1])); print(t); print()
