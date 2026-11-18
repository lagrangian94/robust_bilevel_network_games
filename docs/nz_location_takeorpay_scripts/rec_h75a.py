# -*- coding: utf-8 -*-
# record: verdict of H75 for r = 1400 (unserved-demand model), ledger 4.29, memory
import io, shutil
D = "C:/Users/user/robust_bilevel_network_games/docs/"; SP = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad/"
V = """
 H75 판정, r = 1400 (2026-11-18 10:50 무렵. 설정 `dwps3r14`, 45 건, 결정 05:15~06:50, Coupled ~07:11, 평가 ~07:58, 분해·수송량 ~08:01, 실패 0. 집계 `agg_zsW_s.py` (AZ_SMAX=3, AZ_DIR=zs_dwps3r14), `dec_stats_s3.py`, `agg_flow.py`·`agg_flowA.py`·`unmet.py`·`chk_region_rev_d.py`·`chk_worst_mass2_s3.py` (AM_DIR=mech_dwps3r14). 비교는 같은 45 건의 `dwps3` (`agg_zss3m45_out.txt`). 용량비 0.4, 30 칸 평균):
  a) 대체로 성립. 안 채운 수요 13.4% (건별 1.4~24.5%). 다른 지역 고객에게 가는 몫 10% (지금 32%). 점포 하나가 파는 고객 4.5 곳 (예측 4 이하, 지금 5.3).
  b) 빗나감 (기각 기준에는 안 걸림). 총수요와의 상관 0.75 (예측 0.7 미만, 지금 0.83). 점포가 있는 지역의 수요와의 상관 0.74 로 총수요와 같은 크기. Nominal 이 세 지역에 다 여는 건이 11 → 44% 로 늘어 리더의 지역이 총수요의 91% 를 덮기 때문이다.
  c) 기각. Data-DR (0.1) 이 Nominal 과 다른 결정 (33%) 가운데 총용량을 줄인 건 80% (지금 78%). 한 방향 그대로다. 최악 분포가 더하는 확률 가운데 총수요가 가장 낮은 세 시나리오로 가는 몫은 91 → 72% 로 줄었다.
  d) 기각 (반대 방향). Belief-only (0.1) 의 수요 분포를 아는 조건 q95 이득 −3.25 ± 1.32 (같은 45 건의 지금 모형 −1.65 ± 0.51). 약해지지 않고 커졌다. 평균 −0.07.
  e) 빗나감 (기각 기준에는 안 걸림). Coupled 의 ρ = 1 q95 −3.99 ± 1.40 (지금 −3.29 ± 0.71), 평균 대가 +0.30 ± 0.33 (지금 +0.65). Data-DR (0.1) 은 +0.47·−2.00 (지금 +0.96·−1.45).
  읽기: 주문을 버릴 수 있게 하면 시장은 지역 단위로 갈리지만 (다른 지역 판매 32 → 10%) 리더가 세 지역에 나눠 열어서 수입은 여전히 총수요를 따라간다. Data-DR 의 한 방향은 안 바뀐다. belief 헤징과 Coupled 의 값어치는 남고 더 커 보이나 45 건이라 표준오차가 크다 (1.3~1.4).
"""
for p in (D + "nz_location_takeorpay_scripts/cf_predictions.txt", SP + "cf_predictions.txt"):
    s = io.open(p, encoding="utf-8").read(); io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + V)
L = """- **r = 1400 선별 결과 (45 건, 2026-11-18 08:01 끝, 실패 0. 괄호는 같은 45 건의 `dwps3`).** 안 채운 수요 13.4%, 다른 지역 판매 10% (32%), 리더의 몫 73% (73%), 리더 수입과 총수요의 상관 0.75 (0.83), Nominal 이 세 지역에 다 여는 건 44% (11%), 총용량 660 (729). Data-DR (0.1) 다른 결정 33% (40%), 그중 총용량 줄임 80% (78%), 최악 분포의 쏠림 72% (91%). 평가 (30 칸 평균, Nominal 대비 %): 아는 조건 Belief-only (0.1) 평균 −0.07·q95 −3.25 ± 1.32 (−0.22·−1.65). 모르는 조건 ρ = 1: Data-DR (0.1) +0.47·−2.00 (+0.96·−1.45), Coupled +0.30 ± 0.33·−3.99 ± 1.40 (+0.65·−3.29). H75 는 c·d 기각 (Data-DR 한 방향 그대로, belief 헤징은 약해지지 않고 커짐). r = 1800 은 평가 중.
"""
p = D + "nz_location_config_log.md"; s = io.open(p, encoding="utf-8").read(); io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + L)
for a, b in (("agg_zsr14_out.txt", "agg_zsr14_out_2026-11-18.txt"), ("agg_zss3m45_out.txt", "agg_zss3m45_out_2026-11-18.txt"), ("agg_zsW_s.py", "agg_zsW_s.py"), ("dec_stats_s3.py", "dec_stats_s3.py"), ("unmet.py", "unmet.py"),
             ("chk_region_rev_d.py", "chk_region_rev_d.py"), ("chk_worst_mass2_s3.py", "chk_worst_mass2_s3.py"), ("rec_h75a.py", "rec_h75a.py")): shutil.copyfile(SP + a, D + "nz_location_takeorpay_scripts/" + b)
p = "C:/Users/user/.claude/projects/C--Users-user-robust-bilevel-network-games/memory/nz_location_takeorpay.md"; s = io.open(p, encoding="utf-8").read()
M = "\n- **주문 버리기 모형 r = 1400 선별 (2026-11-18 오전, 45 건, 대장 4.29).** 안 채운 수요 13%, 다른 지역 판매 32 → 10% 로 시장은 지역 단위로 갈리지만 리더가 세 지역에 나눠 열어 (11 → 44%) 수입은 여전히 총수요를 따라감 (상관 0.75). Data-DR (0.1) 다른 결정 중 총용량 줄임 80% 로 한 방향 그대로. Belief-only (0.1) 아는 조건 q95 −3.25 ± 1.32, Coupled ρ = 1 +0.30·−3.99 ± 1.40 로 값어치는 남음 (표준오차 큼). r = 1800 결과와 채택 여부는 미정.\n"
io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + M); print("recorded")
