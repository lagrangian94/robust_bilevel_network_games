# A 한도 스윕 (sgb15_qasweep.jl) 집계 (2026-11-08): 결정 × 한도 설정별 증폭 구간 [wit_C − UB_D, UB_C − wit_D] 와 예약 (D → C).
# % 는 p 스윕 기준과 같은 분모 |N({1;4;12})| (p = 600, 한도 300) = 1477.17.
import csv, os, sys
from collections import defaultdict
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sgb15_fac")
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SP, "qa15", "qa_s10_p600.csv")
if not os.path.exists(fn): fn += ".part"
BASE = 1477.166667
g = defaultdict(dict); sets = []
for r in csv.DictReader(open(fn, encoding="utf-8")):
    k = (round(float(r["qSL"])), round(float(r["qSJ"])))
    if k not in sets: sets.append(k)
    g[(k, r["x"])][r["model"]] = r
xs = sorted({x for (_, x) in g}, key=lambda t: [int(v) for v in t.strip("{}").split(";")])
def amp(k, x):
    m = g.get((k, x), {})
    if "C" not in m or "D" not in m: return None
    D, C = m["D"], m["C"]
    lo = float(C["witness"]) - float(D["UB"]); hi = float(C["UB"]) - float(D["witness"])
    return lo, hi, C["exact"] == "true" and D["exact"] == "true"
lab = {k: f"{k[0]}:{k[1]}" for k in sets}
print("증폭 (C − D), [witness, UB] 구간의 하한~상한 (% of |N({1;4;12})|)   * = 인증 안 됨")
print(f"{'x':6}" + "".join(f"| {lab[k]:>17} " for k in sets))
for x in xs:
    line = f"{x:6}"
    for k in sets:
        a = amp(k, x)
        if a is None: line += f"| {'':>17} "; continue
        lo, hi, ex = a
        line += f"| {max(lo,0):6.1f}~{max(hi,0):6.1f} ({100*max(lo,0)/BASE:4.1f}){' ' if ex else '*'}"
    print(line)
print("\n증폭 > 0.5 인 결정 수: " + ", ".join(f"{lab[k]} {sum(1 for x in xs if (a := amp(k, x)) and a[0] > 0.5)}/{sum(1 for x in xs if amp(k, x))}" for k in sets))
print("\nData-DR 값 (D, witness) — 한도가 B 판매 수준을 얼마나 바꾸는지")
print(f"{'x':6}" + "".join(f"| {lab[k]:>9} " for k in sets))
for x in xs:
    print(f"{x:6}" + "".join(f"| {float(g[(k, x)]['D']['witness']):9.1f} " if "D" in g.get((k, x), {}) else f"| {'':>9} " for k in sets))
print("\n예약 h (D: belief q̂ → C: 최악 belief), SL/SJ/B")
for x in xs:
    parts = []
    for k in sets:
        m = g.get((k, x), {})
        if "C" not in m or "D" not in m: continue
        f = lambda r: f"{float(r['hSL']):.0f}/{float(r['hSJ']):.0f}/{float(r['hB']):.0f}"
        parts.append(f"{lab[k]} {f(m['D'])}→{f(m['C'])}")
    print(f"{x:6} " + " | ".join(parts))
