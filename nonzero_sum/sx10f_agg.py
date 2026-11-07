# 지역 요인 SGB15 S=10 κ OOS 집계 (sx10f/sx_kap_s*_k*.csv, 2026-11-07). sx10_agg.py 의 κ 부분 사본, 결정은 fi15 의 in-sample N·D·C.
# 진짜마다 belief 방향 평균 (무작위) / 최대 (적대적), 진짜에 대해 평균, 최악 = 최댓값. 결합의 이득 = (지표(x_m) − 지표(x_C)) / |지표(x_N)| × 100 (양수 = 결합이 나음).
import csv, glob, os, re
from collections import defaultdict
import numpy as np
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sgb15_fac"); LOG = os.path.join(SP, "fi15")
def logx(model, s):
    f = os.path.join(LOG, f"n_s{s}.log" if model == "nominal" else f"dc_s{s}.log")
    l = [z for z in open(f, encoding="utf-8", errors="replace") if z.startswith("RESULT") and f" model={model} " in z][0]
    return re.search(r" x=(\{[^}]*\})", l).group(1).replace(",", ";")
R = defaultdict(lambda: defaultdict(lambda: {"sh": [], "q": None}))
for f in glob.glob(os.path.join(SP, "sx10f", "sx_kap_s*_k*.csv")):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        d = R[(int(r["seed"]), r["x"], float(r["kappa"]))][int(r["i"])]
        if r["kind"] == "qhat": d["q"] = float(r["L"])
        else: d["sh"].append(float(r["L"]))
seeds = sorted({k[0] for k in R}); X = {s: {k: logx(t, s) for k, t in (("N", "nominal"), ("D", "datadr"), ("C", "coupled"))} for s in seeds}
ks = sorted({k[2] for k in R})
def met(s, x, k):
    P = list(R[(s, x, k)].values())
    return {"rand": np.mean([np.mean(d["sh"]) for d in P]), "adv": np.mean([np.max(d["sh"]) for d in P]), "worst": max(np.max(d["sh"]) for d in P),
            "qmean": np.mean([d["q"] for d in P]), "qworst": max(d["q"] for d in P)}
ok = [s for s in seeds if all((s, X[s][m], k) in R for m in "NDC" for k in ks)]
print(f"κ OOS: 시드 {ok}, κ {ks}")
for m in "ND":
    diff = [s for s in ok if X[s][m] != X[s]["C"]]
    print(f"\n결합 vs {dict(N='Nominal', D='Data-DR')[m]}: 결정이 다른 시드 {len(diff)} 개 {diff} 의 평균 이득 % (승/패)")
    for k in ks:
        cells = []
        for key in ("rand", "adv", "worst"):
            g = np.array([100 * (met(s, X[s][m], k)[key] - met(s, X[s]["C"], k)[key]) / abs(met(s, X[s]["N"], k)[key]) for s in diff])
            cells.append(f"{key} {g.mean():+5.1f} ({(g > 0.1).sum()}/{(g < -0.1).sum()})" if len(g) else f"{key} -")
        print(f"  κ={k:.0f}  " + "  ".join(cells))
    g = np.array([100 * (met(s, X[s][m], ks[0])["qmean"] - met(s, X[s]["C"], ks[0])["qmean"]) / abs(met(s, X[s]["N"], ks[0])["qmean"]) for s in diff])
    w = np.array([100 * (met(s, X[s][m], ks[0])["qworst"] - met(s, X[s]["C"], ks[0])["qworst"]) / abs(met(s, X[s]["N"], ks[0])["qworst"]) for s in diff])
    if len(g): print(f"  p̃=q̂  평균 {g.mean():+5.1f} ({(g > 0.1).sum()}/{(g < -0.1).sum()})  최악 {w.mean():+5.1f} ({(w > 0.1).sum()}/{(w < -0.1).sum()})")
    for s in diff:
        print(f"    s{s:<2d} {X[s][m]:10s} vs 결합 {X[s]['C']:10s}: " + "  ".join(
            f"κ={k:.0f} " + "/".join(f"{100 * (met(s, X[s][m], k)[key] - met(s, X[s]['C'], k)[key]) / abs(met(s, X[s]['N'], k)[key]):+5.1f}" for key in ("rand", "adv", "worst")) for k in ks)
            + f"  q̂ {100 * (met(s, X[s][m], ks[0])['qmean'] - met(s, X[s]['C'], ks[0])['qmean']) / abs(met(s, X[s]['N'], ks[0])['qmean']):+5.1f}")
