# -*- coding: utf-8 -*-
# records H80 (both copies of cf_predictions.txt) and ledger 4.33 (radius 0.5) BEFORE the run, and copies the new scripts to docs/nz_location_takeorpay_scripts.
import io, shutil, os
SP2 = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad/"
DOC = "C:/Users/user/robust_bilevel_network_games/docs/"
H80 = """
H80 (2026-11-18 18:50, 실행 전 기록). 반지름 0.5 를 끝점으로 추가 (대장 4.33, 사용자 결정 "0.5 돌려줘"). 설정 `dwps3` 의 집합 (TV ∩ 상자 r/√S), Belief-only·Data-DR·Coupled (δ = 0.5/3), 150 건, 용량비 0.4. 평가는 4.24·4.31 과 같은 격자·표본·씨앗.
 미리 본 것: 스크립트 시험 한 건 (지리 1, 시드 1, 후보 3 개만). 결정 {7:300;8:300;13:200} 의 값이 반지름 0.3 → 0.5 에서 Data-DR 최악값 −3746.31 → −3737.33 (+9, 0.1 → 0.3 구간은 0.1 마다 +55), belief 단독 피해 94.00 → 94.05, 결합 최악값 −3579.03 → −3380.51 (+199). 반지름 0.2·0.3 의 평가 결과 (H77) 는 아직 못 봄 (스윕 평가 진행 중).
 근거: 시나리오가 10 개이고 추정 확률이 0.1 씩이라 상자 폭 r/√10 이 0.1 이 되는 반지름 0.32 부터는 시나리오 확률의 아래 한도가 0 이다. 그 위로는 한쪽만 틀리는 집합이 거의 커지지 않고, 둘을 함께 움직이는 집합만 커진다.
 a) Data-DR (0.5) 의 결정은 Data-DR (0.3) 의 결정과 150 건 중 70% 이상에서 같다. 기각 기준: 50% 미만.
 b) Belief-only (0.5) 의 결정은 Belief-only (0.3) 의 결정과 80% 이상에서 같다. 기각 기준: 60% 미만.
 c) Coupled (0.5) 의 결정은 Coupled (0.3) 의 결정과 40% 이상에서 다르다. 기각 기준: 20% 미만.
 d) 평가 (수요 분포도 모르는 조건 ρ = 1, 30 칸 평균, Nominal 대비 %): Data-DR (0.5) 와 Belief-only (0.5) 는 평균·q95 모두 각자의 0.3 값에서 0.5%p 안. Coupled (0.5) 는 Coupled (0.3) 보다 평균 대가가 1.0%p 이상 크고, q95 는 1.0%p 넘게 좋아지지 않는다 (꼬리 이득은 포화, 대가만 늚). 기각 기준: Coupled (0.5) 의 q95 가 Coupled (0.3) 보다 1.5%p 넘게 좋음.
 e) 진짜 분포가 추정 근처인 칸 (γ = 10, ζ = 0) 에서 Coupled (0.5) 의 q95 는 양수다 (Nominal 보다 나쁨). 기각 기준: 음수.
 f) 점검: Data-DR (0.5) 의 비용 시나리오별 상위 결정이 저장된 후보 안에 있는 건이 80% 이상 (새 후보가 드물다). 전체 열거의 V0 는 저장된 열거값과 1e-3 안에서 같아야 한다 (스크립트가 어긋나면 멈춤).
"""
L433 = """
### 4.33 반지름 0.5 를 끝점으로 추가 (2026-11-18 저녁, 사용자 결정 "0.5 돌려줘")

- **왜.** 스윕 (4.31) 은 0.3 에서 멈춘다. 꼬리 이득이 반지름을 따라 계속 느는지, 포화되거나 꺾이는지를 말하려면 더 큰 반지름이 한 점 필요하다. 사용자가 올린 논문 세 편 (Delage–Saif, Chow 외, Ruan 외) 이 모두 조절 변수를 효과가 사라지거나 포화되는 곳까지 훑는다. 0.5 는 쓸 만한 반지름이 아니라 끝점이다 (상자 폭 $r/\\sqrt S$ 가 0.1 을 넘는 반지름 0.32 부터 시나리오 확률의 아래 한도가 0 이다). 계산 표의 반지름 0.3 은 그대로 둔다.
- **무엇을.** 집합과 인스턴스는 `dwps3` 그대로. 새로 푸는 칸은 Data-DR (0.5), Belief-only (0.5), Coupled (0.5, δ 0.167). 0.1~0.3 의 Data-DR 은 저장된 전체 열거에서 골랐지만 그 열거는 반지름 0.3 까지만 적어 두었으므로, 0.5 는 전체 열거를 다시 한다.
- **어떻게.**
  1. Data-DR (0.5): `dwD5_body.jl` → `dwD5W.jl` (= `dwLeanB.jl` 1~117 줄 + 본문). 결정 8463 개를 모두 다시 열거해 반지름 0.5 의 최악값을 재고 (`dwLeanB.jl` 1 절과 같은 규칙), V0 를 저장된 열거값 `vn_dwps3_*` 과 대조한다. 출력 `geo/dw/{vnx,selx}_dwps3_D0.5_g_s.csv`. 후보 집합도 넓힌다: 저장된 후보 + 반지름 0.5 최악값 기준 상위 결정 (비용 시나리오별 4·2 개, 기존 규칙) → `candx_dwps3_0.5_g_s.csv`.
  2. Belief-only (0.5): `dwB5_body.jl` → `dwB5W.jl`. 넓힌 후보에 반지름 0.5 의 belief 점 (씨앗 규칙은 `dwLeanB.jl` 과 같음) 으로 적대 피해를 재고 0.3 의 값과 큰 쪽을 쓴다. 저장된 후보의 0.3 값은 `dwx_dwps3_B0.3_*` 에서 읽고, 새 후보는 0.1·0.2·0.3 을 같은 점으로 여기서 잰다. 출력 `geo/dw/{dwx,selx}_dwps3_B0.5_g_s.csv`.
  3. Coupled (0.5): `coupW5.jl` (= `coupWE.jl` 에서 후보를 `candx_*` 에서 읽게 바꾼 사본, `mk_r5.py` 가 만듦), `CP_EH = CP_ET = 0.5`, δ 기본값 (반지름/3). 출력 `geo/dw/{cpx,selx}_dwps3_C0.5_g_s.csv`.
  4. 합치기 `mk_sely.py` → `sel_dwps3y_g_s.csv` (= `sel_dwps3x` + B0.5 + D0.5 + C0.5).
  5. 평가 `zsW.jl`, `ZS_CFG=dwps3y`, 모형 N·B0.5·D0.5·C0.5, 출력 `geo/zss3y`, 격자 γ 0.1~10 × ζ 0~5, M 100, 씨앗 9595 (4.24·4.31 과 같은 표본), ρ = 1, 0.5, 0.
  6. 분해 `mechW.jl` 로 새 결정의 약정·여유 용량 (`geo/mechy`).
  150 건, 11 개 동시, 순서대로. `r5_chain.sh` 가 ρ 평가 (4.32) 가 끝나기를 기다렸다가 이어서 돌린다. 로그 `r5_run.log`. 작업 스크립트 `dwD5_job.sh`, `dwB5_job.sh`, `coupW5_job.sh`, `mechy_job.sh`. 기존 스크립트와 기존 출력은 건드리지 않는다.
- **시험.** 지리 1, 시드 1 한 건을 줄여서 (열거 74 개, 후보 3 개) 세 스크립트 모두 통과. 시험 출력 (`*.test.*`) 은 지움.
- **한계.** Coupled 와 Belief-only 는 후보 안에서만 고른다 (0.1~0.3 과 같음). Coupled 의 값은 점 탐색이라 하한이다. Gurobi 를 쓴다 (라이선스 2026-11-20 까지).
- **예측.** H80.
- **상태.** ρ 평가 뒤 실행 예약.
"""
for p in (SP2 + "cf_predictions.txt", DOC + "nz_location_takeorpay_scripts/cf_predictions.txt"):
    s = io.open(p, encoding="utf-8").read(); assert "H80 (" not in s, p
    io.open(p, "w", encoding="utf-8", newline="\n").write(s.rstrip("\n") + "\n" + H80)
p = DOC + "nz_location_config_log.md"; s = io.open(p, encoding="utf-8").read(); assert "### 4.33" not in s
io.open(p, "w", encoding="utf-8", newline="\n").write(s.rstrip("\n") + "\n" + L433)
for f in ("dwD5_body.jl", "dwB5_body.jl", "mk_r5.py", "dwD5W.jl", "dwB5W.jl", "coupW5.jl", "dwD5_job.sh", "dwB5_job.sh", "coupW5_job.sh", "mechy_job.sh", "mk_sely.py", "r5_chain.sh", "rec_h80.py"):
    assert os.path.exists(SP2 + f), f; shutil.copy(SP2 + f, DOC + "nz_location_takeorpay_scripts/" + f)
print("H80 and ledger 4.33 recorded, scripts copied")
