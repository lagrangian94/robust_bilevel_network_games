# -*- coding: utf-8 -*-
# record before the run: H79 (finer rho) in both prediction files, ledger 4.32
import io
D = "C:/Users/user/robust_bilevel_network_games/docs/"; SP = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad/"
V = """
H79 (2026-11-18 15:00, 실행 전 기록). belief 가 진짜 분포를 따라가는 정도 ρ 를 0.25, 0.75 에서 추가로 잼 (대장 4.32). 근거는 가설 문서 9.33 의 ρ = 0, 0.5, 1 (150 건, 용량비 0.4, 30 칸 평균, q95 의 Nominal 대비 %): Belief-only (0.1) −0.65, −0.62, −0.51. Belief-only (0.2) −1.23, −1.27, −1.23. Data-DR (0.1) −0.90, −1.06, −0.66. Coupled (0.1) −1.90, −2.81, −2.61. Nominal 의 q95 수준 −82.4, −75.4, −73.4.
 a) Belief-only 는 ρ 와 무관하다. ρ = 0.25 와 0.75 에서 Belief-only (0.1) 의 q95 는 −0.3 ~ −0.9, Belief-only (0.2) 는 −0.9 ~ −1.6.
 b) Coupled (0.1) 의 q95 는 ρ = 0.25 에서 ρ = 0 과 0.5 의 사이 (−1.9 ~ −2.8), ρ = 0.75 에서 −2.4 ~ −3.1. 우위는 ρ 가 0 에서 벗어나면 곧 커지고 0.5 위로는 평평하다. 기각 기준: ρ = 0.25 의 값이 ρ = 0 의 값보다 나쁨.
 c) Nominal 의 q95 수준은 ρ 에 오목하다. ρ = 0.25 에서 −79.5 ~ −77.0 (0 → 0.5 변화의 절반 이상이 0.25 까지에서 생김), ρ = 0.75 에서 −74.8 ~ −73.8.
 d) 두 ρ 모두에서 반지름 0.1 끼리는 Coupled 의 q95 가 가장 좋다 (Belief-only, Data-DR 과의 짝 차이가 음수이고 표준오차의 1 배 이상). 기각 기준: 어느 한쪽이라도 Coupled 보다 좋음.
 e) 점검: 표본이 같으므로 "아는 조건" 열은 `geo/zss3m`·`geo/zss3x` 의 값과 소수 넷째 자리까지 같아야 한다.
"""
for p in (D + "nz_location_takeorpay_scripts/cf_predictions.txt", SP + "cf_predictions.txt"):
    s = io.open(p, encoding="utf-8").read(); assert "H79 (" not in s; io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + V)
Lg = """
### 4.32 belief 가 진짜 분포를 따라가는 정도 ρ 를 더 촘촘히 (2026-11-18 오후, 사용자 요청)

- **왜.** 평가의 ρ 는 0, 0.5, 1 세 값뿐이다. Coupled 의 꼬리 우위가 ρ = 0 에서 작고 0.5 이상에서 큰데 (가설 문서 9.33), 그 사이가 어떻게 생겼는지 보려면 0.25, 0.75 가 필요하다.
- **무엇을.** 결정은 그대로 (`sel_dwps3x`, 새로 푸는 것 없음). 평가만 ρ = 0.25, 0.75 로 한 번 더 한다. 모형은 N, Belief-only 0.1·0.2·0.3, Data-DR 0.1·0.2·0.3, Coupled 0.1·0.2·0.3 열 가지.
- **어떻게.** `zsW.jl` 그대로, `ZS_RHO=0.25,0.75`, `ZS_CFG=dwps3x`, 격자 γ 0.1~10 × ζ 0~5, M 100, 씨앗 9595. 표본 (p, p₂) 은 ρ 목록과 무관하게 뽑히므로 4.24·4.31 과 같은 표본이다. 출력 `geo/zss3r`. 150 건, 11 개 동시. 반지름 스윕 (4.31) 이 끝난 뒤 `rho_chain.sh` 가 이어서 돌린다. 로그 `rho_run.log`.
- **예측.** H79.
- **상태.** 스윕 뒤 실행 예약.
"""
p = D + "nz_location_config_log.md"; s = io.open(p, encoding="utf-8").read(); assert "### 4.32" not in s; io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + Lg)
print("recorded")
