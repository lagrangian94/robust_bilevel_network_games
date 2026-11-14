# 약정 물량 방식 실험 스크립트 사본 (2026-11-13 ~ 14)

세션 스크래치에서 돌린 스크립트를 기록용으로 옮긴 것이다. 경로가 스크래치 폴더로 박혀 있고 이전 세션의 사본 (`geo/nz_data_geo2.jl`, `geo/fac_builder_geo2.jl`, `box/nz_exact_eval_box.jl`, `geo/pick_cities.py`) 에 의존하므로 그대로는 돌지 않는다. 결과 파일 (csv) 은 용량이 커서 넣지 않았다.

- `cf_predictions.txt`: 실행 전 예측과 판정 (H26 ~ H46). 틀린 예측도 그대로 있다.
- `scr_cfg.txt`: 설정 이름과 환경 변수.
- 평가 (리더가 진짜 분포를 아는 조건): `tkC.jl` (기본, `SC_TOP=1` 이 약정 방식), `tkK.jl` (κ), `tkR.jl` (개설 수 상한 `SC_R`, belief 뽑기별 비용도 저장), `tkF.jl` (후보 선정용 개설비 이동 `SC_QADD`, 후보 수 `SC_NC`), `tkCap.jl` (B 거점 용량 `SC_BB`), `tkD.jl`·`tkD30.jl` (뽑기별 비용), `tkH.jl` (예약량 진단). `*_job.sh` 가 한 건 실행.
- 집계: `tk_agg2.py`, `tkS_pair3.py`, `tkS_pair_top.py`, `scr2_agg.py`, `agg_nf20.py`, `agg_q.py` (개설비 사후 계산), `agg_cap.py` (입지 + 용량 수준). `agg_q.out`, `agg_cap.out` 은 마지막 출력.
- 그림: `box_tk.py`, `box_dw.py`, `box_q.py`, `box_cap.py` → `docs/figs/nz_location_takeorpay/`.
- 진단: `chk_why.py`, `chk_clair.py`, `chk_h.py`.

요약은 `docs/nz_location_capacity_hypothesis_2026-11-14.md`, 경위는 `docs/nz_location_formulation_discussion_2026-11-14.md`.
