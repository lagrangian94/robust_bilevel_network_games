# 지역 요인 수요 vs 독립 수요 (SGB15 S = 10) 분해 집계 (fa15/fa_<dem>_s*.csv, 2026-11-06). 표본 최대 = 공 안 최악의 하한.
# 상위 12 개 결정 안에서: Nominal = argmin L0, Data-DR ≈ argmin L0 + max Δ_T, 결합 δ=0 ≈ argmin L0 + max Δ_D. % 는 |L0(1 등)| 대비.
# 증폭 = max Δ_D − max Δ_T (follower 가 알 때 추가로 다치는 양).
import csv, glob, os, re
from collections import defaultdict
import numpy as np
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sgb15_fac")
TIE = 0.5
res = defaultdict(list)
for dem in ("iid", "region"):
    print(f"\n== {dem}")
    for f in sorted(glob.glob(os.path.join(SP, "fa15", f"fa_{dem}_s*.csv")), key=lambda s: int(re.search(r"_s(\d+)\.csv", s).group(1))):
        seed = int(re.search(r"_s(\d+)\.csv", f).group(1))
        mx = defaultdict(lambda: {"T": (-1e18, ""), "D": (-1e18, ""), "B": (-1e18, "")}); L0 = {}
        for r in csv.DictReader(open(f, encoding="utf-8")):
            x = r["x"]; L0[x] = float(r["L0"]); d = float(r["delta"])
            if d > mx[x][r["kind"]][0]: mx[x][r["kind"]] = (d, r["tag"])
        b = abs(min(L0.values())) / 100
        V = {"N": lambda x: L0[x], "D": lambda x: L0[x] + mx[x]["T"][0], "C": lambda x: L0[x] + mx[x]["D"][0]}
        xN, xD, xC = (min(L0, key=V[k]) for k in "NDC")
        amp = lambda x: (mx[x]["D"][0] - mx[x]["T"][0]) / b
        gapCD = (V["C"](xD) - V["C"](xC)) / b; gapDN = (V["D"](xN) - V["D"](xD)) / b
        res[dem].append(dict(seed=seed, ampD=amp(xD), ampN=amp(xN), ampmax=max(amp(x) for x in L0), CD=gapCD, DN=gapDN,
                             tagD=mx[xD]["D"][1], prem=(L0[xC] - L0[xN]) / b))
        print(f"  s{seed:<2d} N {xN:10s} D {xD:10s} C {xC:10s} | D 의 선택: 진짜만 {mx[xD]['T'][0] / b:4.1f} 앎 {mx[xD]['D'][0] / b:4.1f} ({mx[xD]['D'][1]}) belief만 {mx[xD]['B'][0] / b:4.1f} "
              f"→ 증폭 {amp(xD):4.1f} | 결합 기준 이득 vs D {gapCD:4.1f}% 보험료 {(L0[xC] - L0[xN]) / b:4.1f}%")
print("\n요약 (시드 10 개)")
for dem in ("iid", "region"):
    R = res[dem]; n = len(R)
    if not n: continue
    print(f"  {dem:7s} Data-DR 의 선택 증폭 평균 {np.mean([r['ampD'] for r in R]):4.1f}%p (≥2%p {sum(r['ampD'] >= 2 for r in R)}/{n}), "
          f"상위 12 개 중 최대 증폭 평균 {np.mean([r['ampmax'] for r in R]):4.1f}%p | 결합 때문에 갈림 (이득 > {TIE}%) {sum(r['CD'] > TIE for r in R)}/{n}, "
          f"이득 평균 {np.mean([r['CD'] for r in R]):4.1f}% | Data-DR ≠ Nominal {sum(r['DN'] > TIE for r in R)}/{n} | "
          f"D 의 '앎' 최악이 지역 방향인 시드 {sum(r['tagD'].startswith('R') for r in R)}/{n}")
