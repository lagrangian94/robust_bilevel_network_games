# 지역 요인 SGB15 (S = 10) in-sample 선택 (fi15/*.log RESULT) + 고정 평가 (fe15/fe_region_s*.csv, α-B&B) 교차표 (2026-11-06)
# 시드마다: Nominal·Data-DR·결합의 선택, Data-DR 선택에 점포 6 이 있나, 갈리면 서로의 모형으로 본 손해 (% = |Nominal 최적| 대비).
# 동률 = Data-DR 기준 차이가 Benders 허용오차 0.5% 이하.
import csv, glob, os, re
from collections import defaultdict
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sgb15_fac"); TOL = 0.5
norm = lambda x: "{" + ";".join(sorted(re.split(r"[,;]", x.strip("{}")), key=int)) + "}"
IS = defaultdict(dict)
for f in glob.glob(os.path.join(SP, "fi15", "*.log")):
    for l in open(f, encoding="utf-8", errors="replace"):
        if not l.startswith("RESULT"): continue
        kv = dict(t.split("=", 1) for t in l.split() if "=" in t)
        IS[int(kv["seed"])][kv["model"]] = (norm(kv["x"]), float(kv["LB"]), float(kv["UB"]), kv["cities"])
FX = defaultdict(dict)
for f in glob.glob(os.path.join(SP, "fe15", "fe_region_s*.csv")):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        FX[(int(r["seed"]), norm(r["x"]))][r["model"]] = float(r["LB"])
def val(seed, x, m):
    return FX.get((seed, x), {}).get(m)
print(f"{'시드':4s} {'Nominal':12s} {'Data-DR':12s} {'결합':12s} | 점포6 | 결합 기준 D 선택 손해 | D 기준 C 선택 손해 | 판정")
for seed in sorted(IS):
    R = IS[seed]
    if "datadr" not in R or "coupled" not in R: continue
    xN = R.get("nominal", ("?",))[0]; xD, vD = R["datadr"][0], R["datadr"][1]; xC, vC = R["coupled"][0], R["coupled"][1]
    base = abs(R["nominal"][1]) if "nominal" in R else abs(min((v["N"] for (s, _), v in FX.items() if s == seed and "N" in v), default=float("nan")))
    b = base / 100; s6 = "6" in xD.strip("{}").split(";")
    if xD == xC:
        print(f"s{seed:<3d} {xN:12s} {xD:12s} {xC:12s} | {'예' if s6 else '-':4s} | {'같음':>20s} | {'':>16s} | 같음"); continue
    cD = val(seed, xD, "C"); dC = val(seed, xC, "D")
    vC = val(seed, xC, "C") or vC; vD = val(seed, xD, "D") or vD     # 같은 방법 (고정 평가 LB) 끼리 비교, 없으면 in-sample LB
    lossC = (cD - vC) / b if cD is not None else None          # 결합 기준: Data-DR 선택의 값 − 결합 최적
    lossD = (dC - vD) / b if dC is not None else None          # Data-DR 기준: 결합 선택의 값 − Data-DR 최적
    tie = lossD is not None and lossD <= TOL
    judge = "?" if lossC is None or lossD is None else (f"D 동률 안에서 결합이 {lossC:.1f}% 나은 쪽" if tie and lossC > TOL else
             "둘 다 동률" if tie else f"진짜 갈림 (D 는 {lossD:.1f}% 이득)")
    fmt = lambda v: "  ?" if v is None else f"{v:5.1f}%"
    print(f"s{seed:<3d} {xN:12s} {xD:12s} {xC:12s} | {'예' if s6 else '-':4s} | {fmt(lossC):>20s} | {fmt(lossD):>16s} | {judge}")
