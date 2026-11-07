# p 스윕 (sgb15_psweep.jl) 집계 (2026-11-07). λ 와 무관한 구간 [witness, UB] 로 V* 를 잡는다
#   (p ≥ 400 에서 B&B 하한 = Ω_λ 값이 witness 보다 커지는 경우가 있어 λᵁ 부족 가능 → 하한으로 witness 만 씀).
# 증폭 (C − D) 구간 = [wit_C − UB_D, UB_C − wit_D], 진짜 헤지 (D − N) 구간도 같은 방식. % 는 그 p 에서 |N({1;4;12})| (s10 Nominal 결정) 기준.
import csv, glob, os, sys
from collections import defaultdict
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sgb15_fac")
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SP, "ps15", "ps_s10.csv")
if not os.path.exists(fn): fn += ".part"
V = defaultdict(dict); CS = {}
for r in csv.DictReader(open(fn, encoding="utf-8")):
    p, x = int(float(r["p"])), r["x"]
    lo = float(r["witness"]); hi = float(r["UB"])
    V[(p, x)][r["model"]] = (lo, hi, r["exact"] == "true", float(r["LB"]))
    CS[(p, x)] = float(r["contested"])
ps = sorted({p for p, _ in V}); xs = list(dict.fromkeys(x for _, x in V))
def br(a, b):   # a − b 구간
    return (a[0] - b[1], a[1] - b[0])
print(f"{'결정':10s} " + " ".join(f"| p={p:<4d} 경합  증폭 구간 (%)        " for p in ps))
for x in xs:
    cells = []
    for p in ps:
        m = V.get((p, x), {})
        if not all(k in m for k in "NDC"): cells.append(f"| {'':38s}"); continue
        base = abs(V[(p, "{1;4;12}")]["N"][0]) / 100 if "N" in V.get((p, "{1;4;12}"), {}) else float("nan")
        a = br(m["C"], m["D"]); flag = "" if m["C"][2] else "*"
        cells.append(f"| {100*CS[(p,x)]:4.0f}% [{a[0]:7.1f},{a[1]:7.1f}] [{a[0]/base:5.1f},{a[1]/base:5.1f}]{flag:1s}")
    print(f"{x:10s} " + " ".join(cells))
print("* = 결합 평가가 시간 한계 (구간이 넓음)")
