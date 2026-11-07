# 지역 요인 SGB15 (S = 10) 상위 3 개 결정: α-B&B 정확값 vs 표본 분해 (fe15/fe_region_s*.csv, fa15/fa_region_s*.csv) (2026-11-06)
# 값은 LB (찾은 최악, 정확한 달성값), 괄호는 UB−LB 폭. % 는 |N(1 등)| 대비. 증폭 = C − D (belief 몫), 표본 증폭 = max Δ_D − max Δ_T.
# 상위 3 개 안의 선택: argmin, 차이가 TIE (%) 이하면 동률 표시.
import csv, glob, os, re
from collections import defaultdict
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sgb15_fac"); TIE = 0.3
def samp(seed):
    mx = defaultdict(lambda: {"T": -1e18, "D": -1e18, "B": -1e18}); L0 = {}
    for r in csv.DictReader(open(os.path.join(SP, "fa15", f"fa_region_s{seed}.csv"), encoding="utf-8")):
        L0[r["x"]] = float(r["L0"]); mx[r["x"]][r["kind"]] = max(mx[r["x"]][r["kind"]], float(r["delta"]))
    return mx, L0
rows = []
# 시드마다 fe_region_s<seed>.csv 또는 (멈춘 시드) _a.csv + 다시 돌린 _r.csv (뒤 파일이 같은 (x, 모형) 을 덮음)
files = defaultdict(list)
for f in glob.glob(os.path.join(SP, "fe15", "fe_region_s*.csv")):
    m = re.search(r"_s(\d+)(_a|_r|_x)?\.csv$", f)
    if m.group(2) == "_x": continue     # in-sample 교차 평가 (fi15_agg.py 용)
    files[int(m.group(1))].append((m.group(2) or "", f))
for seed in sorted(files):
    mx, L0 = samp(seed)
    V = defaultdict(dict); W = defaultdict(dict); rk = {}
    for _, f in sorted(files[seed]):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            V[r["x"]][r["model"]] = float(r["LB"]); W[r["x"]][r["model"]] = float(r["UB"]) - float(r["LB"]); rk[r["x"]] = int(r["rank"])
    if any(set(V[x]) != {"N", "D", "C"} for x in V):
        print(f"s{seed}: 아직 빠진 모형 있음 — 건너뜀"); continue
    xs = sorted(V, key=lambda x: rk[x]); b = abs(min(V[x]["N"] for x in xs)) / 100
    pick = {}
    for m in "NDC":
        best = min(V[x][m] for x in xs); pick[m] = "/".join(x for x in xs if (V[x][m] - best) / b <= TIE)
    print(f"s{seed}: 선택 N {pick['N']}  D {pick['D']}  C {pick['C']}")
    for x in xs:
        eT = (V[x]["D"] - V[x]["N"]) / b; eA = (V[x]["C"] - V[x]["D"]) / b; sT = mx[x]["T"] / b; sA = (mx[x]["D"] - mx[x]["T"]) / b
        print(f"   [{rk[x]}] {x:10s} 진짜 몫 정확 {eT:5.1f} (표본 {sT:5.1f}) | 증폭 정확 {eA:5.1f} ±{max(W[x]['C'], W[x]['D']) / b:3.1f} (표본 {sA:4.1f}, belief 만 표본 {mx[x]['B'] / b:4.1f})")
        rows.append((seed, x, eT, sT, eA, sA))
    D_pick = [x for x in xs if (V[x]["D"] - min(V[y]["D"] for y in xs)) / b <= TIE]
    C_best = min(V[x]["C"] for x in xs)
    print(f"   결합 기준 Data-DR 선택의 손해: " + ", ".join(f"{x} {(V[x]['C'] - C_best) / b:4.1f}%" for x in D_pick))
if rows:
    n = len(rows); miss = [r for r in rows if r[4] - r[5] >= 0.5]
    print(f"\n결정 {n} 개: 정확 증폭 평균 {sum(r[4] for r in rows) / n:4.2f}%p (표본 {sum(r[5] for r in rows) / n:4.2f}), ≥1%p {sum(r[4] >= 1 for r in rows)}/{n} (표본 {sum(r[5] >= 1 for r in rows)}/{n}), "
          f"표본이 0.5%p 이상 놓침 {len(miss)}/{n}; 진짜 몫 정확−표본 평균 {sum(r[2] - r[3] for r in rows) / n:4.2f}%p")
