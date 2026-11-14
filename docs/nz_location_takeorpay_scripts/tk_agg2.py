# -*- coding: utf-8 -*-
# 진짜를 아는 리더 (TK) 집계: 설정 (base|w150|w600) × N_F. N vs C0 (δ* 규칙, 고정 δ). 오차는 지리 평균의 표준오차 (지리 군집).
import csv, glob, re, collections, sys, numpy as np
from scipy.stats import spearmanr
G = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/tk/"
import os; GLO = int(os.environ.get("TK_GLO", "1")); GHI = int(os.environ.get("TK_GHI", "1000")); SHI = int(os.environ.get("TK_SHI", "1000"))
DS = ["0.05", "0.1", "0.2", "0.3"]; NFS = [3, 5, 10, 30]
def gse(rows, key):   # 지리 군집 평균과 표준오차
    gg = collections.defaultdict(list)
    for r in rows: gg[r["g"]].append(r[key])
    m = np.array([np.mean(v) for v in gg.values()]); return np.mean([r[key] for r in rows]), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan")), m
for cfg in sys.argv[1:] or ["w150", "base", "w600"]:
    F = [f for f in glob.glob(G + f"tk_{cfg}_g*_s*.csv") if GLO <= int(re.search(r"_g(\d+)_s", f).group(1)) <= GHI and int(re.search(r"_s(\d+)\.csv", f).group(1)) <= SHI]; print(f"\n===== {cfg}: 파일 {len(F)} =====")
    if not F: continue
    for nf in NFS:
        rows = []; dstar = []
        for f in F:
            m = re.search(r"_g(\d+)_s(\d+)\.csv", f); L = list(csv.DictReader(open(f, encoding="utf-8")))
            if len(L) < 4: continue
            VN = np.array([float(r["VN"]) for r in L]); adv = np.maximum.accumulate(np.array([[float(r["adv" + d]) for d in DS] for r in L]), axis=1)   # 볼은 포함 관계 → 누적 최댓값
            bm = np.array([float(r[f"bm{nf}"]) for r in L]); bq = np.array([float(r[f"bq{nf}"]) for r in L]); bt = np.array([float(r[f"bt{nf}"]) for r in L])
            n = int(np.argmin(VN)); k = next((i for i in range(len(DS)) if adv[n, i] >= bq[n] - 1e-9), len(DS) - 1); dstar.append(DS[k])
            Tm, Tq, Tt = VN + bm, VN + bq, VN + bt; sc = abs(Tm.min()) / 100
            rec = dict(g=int(m.group(1)), rnm=(Tm[n] - Tm.min()) / sc, rnq=(Tq[n] - Tq.min()) / sc, bmN=bm[n] / sc, bqN=bq[n] / sc,
                       cq=spearmanr(adv[:, k], bq)[0] if adv[:, k].std() > 1e-9 and bq.std() > 1e-9 else np.nan, cm=spearmanr(adv[:, k], bm)[0] if adv[:, k].std() > 1e-9 and bm.std() > 1e-9 else np.nan)
            for lab, kk in [("s", k)] + [(d, i) for i, d in enumerate(DS)]:
                c = int(np.argmin(VN + adv[:, kk]))
                rec.update({f"sw{lab}": float(c != n), f"dm{lab}": (Tm[c] - Tm[n]) / sc, f"dq{lab}": (Tq[c] - Tq[n]) / sc, f"dt{lab}": (Tt[c] - Tt[n]) / sc,
                            f"pr{lab}": (VN[c] - VN[n]) / sc, f"pw{lab}": (adv[n, kk] - adv[c, kk]) / sc})
            rows.append(rec)
        A = lambda k: np.array([r[k] for r in rows])
        print(f"-- N_F={nf}: 시드 {len(rows)} | N 손해 평균기준 {A('rnm').mean():.3f}% 꼬리기준 {A('rnq').mean():.3f}% | N 결정 피해 평균 {A('bmN').mean():+.2f}% q90 {A('bqN').mean():+.2f}% | δ* 분포 {dict(collections.Counter(dstar))} | 순위상관 TV vs q90 {np.nanmean(A('cq')):+.2f}, vs 평균 {np.nanmean(A('cm')):+.2f}")
        for lab in ["s"] + DS:
            dm, sm, gm = gse(rows, f"dm{lab}"); dq, sq, gq = gse(rows, f"dq{lab}"); dt, st, _ = gse(rows, f"dt{lab}"); sw = A(f"sw{lab}") > 0
            pr = A(f"pr{lab}")[sw].mean() if sw.any() else 0; pw = A(f"pw{lab}")[sw].mean() if sw.any() else 0
            print(f"   C0(δ={'δ*' if lab == 's' else lab:>4}): 바뀜 {100*sw.mean():3.0f}% | C0−N 평균 {dm:+.3f} ± {sm:.3f} [지리 이김 {np.sum(gm < -0.01)}/짐 {np.sum(gm > 0.01)}] | q90 {dq:+.3f} ± {sq:.3f} [이김 {np.sum(gq < -0.01)}/짐 {np.sum(gq > 0.01)}] | 상위10% {dt:+.3f} ± {st:.3f} | 바뀐 시드 가격 {pr:.2f}% 최악보호 {pw:.2f}%")
