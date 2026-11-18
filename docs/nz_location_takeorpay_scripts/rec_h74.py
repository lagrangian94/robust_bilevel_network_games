# -*- coding: utf-8 -*-
# record: verdict of H74 c, d (two-store limit), ledger 4.28, hypothesis document 9.26, memory
import io, shutil
D = "C:/Users/user/robust_bilevel_network_games/docs/"; SP = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad/"
V = """
 H74 나머지 판정 (2026-11-18 05:40 무렵. 설정 `dwps3n2`, 결정 150 건 00:44~02:27, Coupled 150 건 02:27~03:14, 평가 150 건 03:14~05:14, 실패 0. 집계 `agg_zsW.py`·`agg_zsW_cells.py`·`sum_cells.py` (AZ_DIR=zsn2), 결정 통계 `dec_stats.py zsn2`, 출력 사본 `agg_zsn2_out_2026-11-18.txt`·`agg_zsn2_cells_out_2026-11-18.txt`. 용량비 0.4, 평가 씨앗 9595):
  c) 성립. Belief-only (0.1) 는 Nominal 과 48% 에서 다르고 (예측 35% 이상), 그중 총용량이 같은 건 60% (예측 50% 이상).
  d) 성립. 수요 분포를 아는 조건, γ ≤ 1 의 20 칸 평균에서 Belief-only (0.1) − Nominal 의 q95 는 −1.40 (예측 −0.4~−1.5), 평균 −0.19 (예측 ±0.4 안).
  읽기: 점포 2 곳 제한은 그림을 바꾸지 않는다. Data-DR 의 한 방향은 더 뚜렷해지고 (다른 결정 중 총용량 줄임 70 → 85%), belief 헤징과 Coupled 의 이득은 조금 커진다 (Coupled 의 ρ = 1 q95 −2.61 → −3.02, 평균 대가 +1.12 → +0.95).
"""
for p in (D + "nz_location_takeorpay_scripts/cf_predictions.txt", SP + "cf_predictions.txt"):
    s = io.open(p, encoding="utf-8").read(); k = s.index("\nH75 (2026-11-18 새벽"); io.open(p, "w", encoding="utf-8", newline="").write(s[:k].rstrip("\n") + "\n" + V + s[k:])
L = """- **2 의 결과 (150 건, 용량비 0.4, 30 칸 평균, 실패 0, 2026-11-18 05:14 끝).** Nominal 과 다른 결정: Belief-only 48% (0.1)·60% (0.2), Data-DR 31%·49%, Coupled 56% (3 곳일 때 47·58·33·50·55%). 총용량 Nominal 600, Belief-only 593·569, Data-DR 571·553, Coupled 568. 다른 결정 중 총용량을 줄인 건: Data-DR (0.1) 85% (3 곳 70%), Belief-only (0.1) 26% (23%), Coupled 54% (57%). 수요 분포를 아는 조건: Belief-only (0.1) 평균 −0.15 ± 0.15·q95 −1.38 ± 0.46, (0.2) +0.50·−2.63. 모르는 조건 (ρ = 1): Belief-only (0.1) −0.01·−0.40, (0.2) +0.82·−2.15, Data-DR (0.1) +1.12 ± 0.30·−1.22 ± 0.52, (0.2) +2.36·−1.74, Coupled +0.95 ± 0.27·−3.02 ± 0.48. ρ = 0.5 Coupled +1.37·−3.44, ρ = 0 +1.82·−2.89. H74 c·d 성립. 점포 2 곳 제한은 Data-DR 의 한 방향을 못 바꾼다.
- **상태 (2026-11-18).** 끝남.
"""
p = D + "nz_location_config_log.md"; s = io.open(p, encoding="utf-8").read(); k = s.index("\n### 4.29 "); io.open(p, "w", encoding="utf-8", newline="").write(s[:k].rstrip("\n") + "\n" + L + s[k:])
H = """
### 9.26 리더 점포 수 상한 2 곳: Belief-only·Coupled·평가 (2026-11-18 아침, 150 건, 실패 0)

9.23 의 나머지다. 설정 `dwps3n2` (점포 2 곳 이하, 나머지는 `dwps3` 과 같음), 용량비 0.4, 평가는 9.22·9.24 와 같은 격자·표본·씨앗. 괄호는 점포 3 곳 이하일 때의 값 (9.22·9.24).

| 결정 | 총용량 | Nominal 과 다른 건 | 다른 건 중 총용량 줄임 | 다른 건 중 어떤 거점은 늘리고 어떤 거점은 줄임 |
|---|---|---|---|---|
| Nominal | 600 (701) | | | |
| Belief-only (0.1) | 593 (694) | 48% (47%) | 26% (23%) | 86% (91%) |
| Belief-only (0.2) | 569 (685) | 60% (58%) | 50% (30%) | 77% (86%) |
| Data-DR (0.1) | 571 (673) | 31% (33%) | 85% (70%) | 63% (72%) |
| Data-DR (0.2) | 553 (653) | 49% (50%) | 82% (77%) | 67% (73%) |
| Coupled | 568 (667) | 56% (55%) | 54% (57%) | 82% (87%) |

평가 (30 칸 평균, 결정 − Nominal, %, ± 표준오차).

| 조건 | 결정 | 평균 | q95 |
|---|---|---|---|
| 수요 분포를 앎 | Belief-only (0.1) | −0.15 ± 0.15 (−0.04) | −1.38 ± 0.46 (−0.89) |
| 수요 분포를 앎 | Belief-only (0.2) | +0.50 ± 0.22 (0.00) | −2.63 ± 0.49 (−1.75) |
| 모름, ρ = 1 | Belief-only (0.1) | −0.01 ± 0.17 (+0.01) | −0.40 ± 0.47 (−0.51) |
| 모름, ρ = 1 | Belief-only (0.2) | +0.82 ± 0.29 (+0.21) | −2.15 ± 0.55 (−1.23) |
| 모름, ρ = 1 | Data-DR (0.1) | +1.12 ± 0.30 (+1.21) | −1.22 ± 0.52 (−0.66) |
| 모름, ρ = 1 | Data-DR (0.2) | +2.36 ± 0.46 (+2.46) | −1.74 ± 0.76 (−1.74) |
| 모름, ρ = 1 | Coupled | +0.95 ± 0.27 (+1.12) | −3.02 ± 0.48 (−2.61) |
| 모름, ρ = 0.5 | Coupled | +1.37 ± 0.28 (+1.39) | −3.44 ± 0.49 (−2.81) |
| 모름, ρ = 0 | Coupled | +1.82 ± 0.38 (+2.01) | −2.89 ± 0.49 (−1.90) |

- **점포 2 곳 제한은 그림을 바꾸지 않는다.** Data-DR 은 여전히 총용량을 줄이는 쪽으로 가고 그 몫은 더 커진다 (70 → 85%). 리더의 몫을 73 → 62% 로 내려도 리더 수입이 총수요를 따라가는 것은 그대로다 (9.23).
- **belief 헤징과 Coupled 의 값어치는 유지되거나 조금 커진다.** Coupled 는 Data-DR (0.1) 과 비슷한 평균 대가 (+0.95 대 +1.12) 로 꼬리를 2.5 배 줄인다 (−3.02 대 −1.22). 3 곳일 때 (네 배) 보다 배수는 작다. Data-DR (0.1) 의 꼬리 이득이 커졌기 때문이다 (−0.66 → −1.22).
- Belief-only (0.2) 는 2 곳에서 용량을 더 줄이고 (다른 결정 중 50%) 평균 대가가 생긴다 (+0.50, +0.82). 거점을 옮길 자리가 줄어서 용량으로 대응하는 것으로 읽힌다 (확인 안 함).
- 한계: Nominal 의 목적값이 2 곳 제한으로 평균 8.5% 나빠진다. 인스턴스를 바꾼 것이므로 3 곳 결과와 수준을 직접 견줄 수는 없다. Belief-only·Coupled 는 점 탐색 (하한).
"""
p = D + "nz_location_capacity_hypothesis_2026-11-14.md"; s = io.open(p, encoding="utf-8").read(); io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + H)
for a, b in (("agg_zsn2_out.txt", "agg_zsn2_out_2026-11-18.txt"), ("agg_zsn2_cells_out.txt", "agg_zsn2_cells_out_2026-11-18.txt"), ("dec_stats.py", "dec_stats.py"), ("rec_h74.py", "rec_h74.py")): shutil.copyfile(SP + a, D + "nz_location_takeorpay_scripts/" + b)
p = "C:/Users/user/.claude/projects/C--Users-user-robust-bilevel-network-games/memory/nz_location_takeorpay.md"; s = io.open(p, encoding="utf-8").read()
M = "\n- **리더 점포 2 곳 제한 전체 결과 (2026-11-18 아침, 가설 문서 9.26, 150 건).** 그림 안 바뀜. Data-DR (0.1) 다른 결정 중 총용량 줄임 70 → 85%. Coupled ρ = 1 평균 +0.95·q95 −3.02 (3 곳 +1.12·−2.61), Data-DR (0.1) +1.12·−1.22, Belief-only (0.1) 아는 조건 q95 −1.38. 리더 몫을 내려 Data-DR 의 한 방향을 바꾸는 길은 막힘. 남은 길은 주문 버리기 모형 (대장 4.29, 선별 실행 중).\n"
io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + M); print("recorded")
