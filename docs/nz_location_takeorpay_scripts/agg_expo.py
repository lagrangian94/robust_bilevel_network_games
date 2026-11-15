import csv, glob, os, re, collections
D = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/expo/"
rows = collections.defaultdict(list)
for f in sorted(glob.glob(D + "expo_expo*_g*_s1.csv")):
    m = re.search(r"expo_(expo\d+)_g(\d+)_s1", f)
    if m is None: continue
    for r in csv.DictReader(open(f, encoding="utf-8")):
        r["g"] = int(m.group(2)); rows[(m.group(1), r["cv"])].append(r)
def F(x):
    try: return float(x)
    except: return float("nan")
print("cfg cv | g: capN hB hint spot_mean spot_cvar sales_mean sales_cvar expo_data same prem_D adv0.1 adv0.2")
for k in sorted(rows):
    rs = rows[k]
    print("==", k)
    for r in sorted(rs, key=lambda r: r["g"]):
        sm = F(r["sales_mean"]); sp = F(r["spot_mean"])
        print("  g%d cap %s hB %s hint %s spot %.1f/%.1f sales %.1f/%.1f share %.1f%% data %.2f same %s prem %s adv %.1f %.1f  xN %s xD %s" % (
            r["g"], r["capN"], r["hB"], r["hint"], sp, F(r["spot_cvar"]), sm, F(r["sales_cvar"]), 100*sp/sm if sm else 0, F(r["expo_data"]), r["same"], r["prem_D"], F(r["adv0.1"]), F(r["adv0.2"]), r["xN"], r["xD"]))
    n = len(rs)
    print("  MEAN share %.1f%% data %.2f adv0.1 %.1f adv0.2 %.1f same %d/%d" % (
        sum(100*F(r["spot_mean"])/F(r["sales_mean"]) for r in rs)/n, sum(F(r["expo_data"]) for r in rs)/n,
        sum(F(r["adv0.1"]) for r in rs)/n, sum(F(r["adv0.2"]) for r in rs)/n, sum(1 for r in rs if r["same"] in ("true","1","True")), n))
