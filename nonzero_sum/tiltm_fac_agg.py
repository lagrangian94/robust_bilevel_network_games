# 지역 요인 SGB15 의 방향 tilt OOS 집계 (oosd_fac/tiltm/sgb15_s*.csv, 2026-11-07). tiltm_agg.py 사본, 결정은 fi15 in-sample (N·D3·C3).
# 방향 = 시장 15 × ± + 총수요 ± + 지역 3 × ± (draw ±17~19). "전체" = 38 방향, "지역" = 지역 방향 6 개만. 리더 목적 (낮을수록 좋음).
# 결합의 이득 % = (지표(x_m) − 지표(x_C)) / |지표(x_N)| × 100, 결정이 다른 시드만. 방향 평균 | 가장 나쁜 방향.
import csv, os, glob, re
from collections import defaultdict
import numpy as np
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sgb15_fac")
data = {}; X = {}
for f in glob.glob(os.path.join(SP, "oosd_fac", "tiltm", "sgb15_s*.csv")):
    s = int(re.search(r"_s(\d+)\.csv", f).group(1)); R = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(f, encoding="utf-8")):
        R[(r["param"], r["level"])][r["model"]][int(r["draw"])] = float(r["objective"]); X.setdefault(s, {})[r["model"]] = r["x"]
    data[s] = R
seeds = sorted(data); rs = sorted({k[0] for k in data[seeds[0]]}, key=float)
def met(s, r, mode, m, which):
    d = data[s][(r, mode)][m]
    v = np.array([d[k] for k in d if (which == "all" or abs(k) >= 17)]); return v.mean(), v.max()
for which, wn in (("all", "전체 38 방향"), ("reg", "지역 방향 6 개")):
    for mode, nm in (("learn", "follower 가 앎"), ("fixed", "follower 는 q̂")):
        print(f"\n== {wn}, {nm}: 결합의 이득 % (결정 다른 시드 평균, 승/패), 방향 평균 | 가장 나쁜 방향")
        for m, mn in (("N", "Nominal"), ("D3", "Data-DR")):
            diff = [s for s in seeds if X[s][m] != X[s]["C3"]]
            print(f"  vs {mn:8s} (n={len(diff)} {diff})")
            for r in rs:
                g = np.array([100 * (met(s, r, mode, m, which)[0] - met(s, r, mode, "C3", which)[0]) / abs(met(s, r, mode, "N", which)[0]) for s in diff])
                w = np.array([100 * (met(s, r, mode, m, which)[1] - met(s, r, mode, "C3", which)[1]) / abs(met(s, r, mode, "N", which)[1]) for s in diff])
                print(f"      r={float(r):.1f} {g.mean():+5.1f}({(g > 0.1).sum()}/{(g < -0.1).sum()}) | {w.mean():+5.1f}({(w > 0.1).sum()}/{(w < -0.1).sum()})")
print("\n시드별 follower 가 앎, 결합 vs Data-DR (전체 방향), r = 0.1/0.3/0.5: 평균 | 가장 나쁜")
for s in seeds:
    if X[s]["D3"] == X[s]["C3"]: continue
    cells = []
    for r in ("0.10", "0.30", "0.50"):
        d = met(s, r, "learn", "D3", "all"); c = met(s, r, "learn", "C3", "all"); n = met(s, r, "learn", "N", "all")
        cells.append(f"{100 * (d[0] - c[0]) / abs(n[0]):+5.1f} | {100 * (d[1] - c[1]) / abs(n[1]):+5.1f}")
    print(f"  s{s:<2d} D {X[s]['D3']:10s} C {X[s]['C3']:10s}: " + "   ".join(cells))
