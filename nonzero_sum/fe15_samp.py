# fa15 표본 분해에서 상위 3 개 결정의 표본 최대 Δ_T, Δ_D, Δ_B (L0 1 등 대비 %) (2026-11-06)
import csv, glob, os, re
from collections import defaultdict
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sgb15_fac")
for f in sorted(glob.glob(os.path.join(SP, "fa15", "fa_region_s*.csv")), key=lambda s: int(re.search(r"_s(\d+)\.csv", s).group(1))):
    seed = int(re.search(r"_s(\d+)\.csv", f).group(1))
    mx = defaultdict(lambda: {"T": -1e18, "D": -1e18, "B": -1e18}); L0 = {}; rk = {}
    for r in csv.DictReader(open(f, encoding="utf-8")):
        x = r["x"]; L0[x] = float(r["L0"]); rk[x] = int(r["rank"])
        mx[x][r["kind"]] = max(mx[x][r["kind"]], float(r["delta"]))
    b = abs(min(L0.values())) / 100
    for x in sorted(L0, key=lambda x: rk[x])[:3]:
        print(f"s{seed} [{rk[x]}] {x:10s} L0 {L0[x]:8.1f} T {mx[x]['T']/b:5.1f} D {mx[x]['D']/b:5.1f} B {mx[x]['B']/b:5.1f} amp {(mx[x]['D']-mx[x]['T'])/b:5.1f}")
