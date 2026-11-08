# ambiguity set 에 시나리오별 상자 (box + budget) 를 넣을 때 코드 영향 점검

2026-11-08. 코드는 고치지 않고 읽기만 함 (세 갈래 점검: 비제로섬 알고리즘, 비제로섬 벤치마크, zero-sum 전체). 핵심 지적은 직접 확인함.

## 배경

- 지금 집합: 진짜 분포 a 와 follower belief d 가 각각 q̂ 둘레 TV 공 (ε̂, ε̃), 결합 TV(a, d) ≤ δ. 리더 목적 CVaR_β (r ≤ a/(1−β)).
- TV 공은 질량을 한 시나리오에 몰 수 있음. S = 10, ε = 0.3, β = 0.4 에서 최악 시나리오 하나의 CVaR 가중이 (0.1 + 0.3)/0.6 = 2/3 → Data-DR 은 결정을 사실상 한 시나리오로 비교함 (비제로섬 SGB15 의 정확한 동률 원인, `robustness_value_todo.md` 6.6).
- 제안 (사용자, 미결정): Bertsimas–Sim 식 box + budget. 상자 l_s ≤ p_s ≤ u_s (예: p_s ≤ (1+κ) q̂_s) 를 a·d 에 더하고 결합은 TV 그대로.
- Sadana & Delage (arXiv 2003.07915, Assumption 1) 가 같은 형태 q = q̂ + diag(q̄) z, −1 ≤ z ≤ 1, Σ|z| ≤ Γ 를 씀. 다만 실험은 q̄ = 1 이라 상자가 걸리지 않음 (사실상 L1 공).

| 집합 | 최악 시나리오 가중 | Data-DR 이 보는 것 |
|---|---|---|
| TV 0.3 만 (지금) | 2/3 | 최악 1 개 중심 |
| + 상자 κ = 1 | 1/3 | 최악 3 개 평균 |
| + 상자 κ = 0.5 | 1/4 | 최악 4 개 평균 |
| Nominal | 1/6 | 최악 6 개 평균 |

## 결론

1. **두 게임 모두 ambiguity set 은 어디서도 쌍대화되지 않음.** a, b, r, d, e, cpl 은 Ω·KKT·정확값 LP 에서 모두 primal 변수. 그래서 상자는 x 와 무관한 범위·행만 더하고, 새 쌍대 승수나 상보성은 생기지 않음.
   - 예외 둘: zero-sum 구조 평가의 leader KKT 와 Piece-F recovery.
2. **θᵁ·λᵁ·쌍대 범위 조임·VI·outer cut 은 그대로 유효함.** 이 경계들은 recourse 쌍대나 시나리오별 양으로 만든 것이라 집합 모양과 무관함. 집합이 작아지면 상계는 유효하고 느슨해질 뿐.
3. **진짜 위험은 같은 집합을 따로 적은 사본들.** 하나라도 상자가 빠지면 오류 없이 다른 (더 큰) 모형을 풂. 결과는 다음 셋 중 하나:
   - 잘못된 cut 으로 LB 가 최적을 넘음;
   - 인증을 잘못 통과함;
   - 벤치마크 상계가 안 닫힘.
4. **α-B&B 는 오히려 조여짐.** RLT bound-factor 행이 a·d 의 변수 범위를 읽음 (`nz_alpha_bnb.jl:69-73`, `global_bilinear_solver.jl:44-48`). 그래서 상자를 범위로 넣으면 McCormick·RLT 가 자동으로 좁아짐.
5. **추천 구현:**
   - 상자를 데이터에 저장. 비제로섬은 `nd.meta[:box]` + `nz_box(nd)` + `nz_with_box` (δ 와 같은 방식), zero-sum 은 `TrueDROData` 필드.
   - `amin/amax/dmin/dmax` 를 상자와 교집합으로 둠.
   - 집합 행을 쓰는 helper 하나로 사본을 통합 (잊을 수 없게).
   - q̂ 가 상자 안에 있는지 assert.

## 오류 없이 틀리는 곳 (먼저 고칠 것)

### 비제로섬

| 파일:줄 | 무엇 | 상자가 빠지면 |
|---|---|---|
| `cert_lambda_free.jl:60-66, 74-79` | `mk()` 가 ε·β·δ 로만 인스턴스를 다시 만들고, RESULT 파싱에 상자 없음 | 큰 집합의 LB 를 씀 → `certified=yes` 오판 |
| `nz_exact_eval.jl:34-43` (`nz_exact_value`) | 집합 사본 | witness (λ 무관 하한) 무효 |
| `nz_omega.jl:330-357` (`nz_clean_belief`), `365-367` (`fix(force=true)`) | q̂ 쪽 축소를 TV 비율로만 함. `fix(force=true)` 가 변수 범위를 지움 | 상자 밖 belief 로 belief cut → LB 과대 (허용오차 수준). 상자 비율 t = min_s (u_s−q_s)/(a_s−q_s) 추가 (약 10 줄) |
| `nz_kkt_eval.jl:42-57` (`nz_kkt_value`, BD-KKT 전역 단계) | 집합 사본 | 벤치마크가 TV 모형 상계로 → gap 이 안 닫혀 인위적 타임아웃 (비교 왜곡) |
| `run_bd_kkt_pooled.jl:43-48` vs `pilot_pooled_store.jl:39-44` | 인스턴스 `mk()` 를 따로 가짐. 벤치마크는 `beta=0.4` 고정, 우리는 `PP_BETA`. 벤치마크 RESULT 에 ε·δ·β 없음 | 두 쪽이 다른 모형을 풀어도 로그로 안 보임 (상자와 무관하게 이미 있는 어긋남, β = 0.4 실행은 문제 없음) |
| 프리솔브 캐시 키 (`run_bd_kkt_pooled.jl:30`, `pilot_pooled_store.jl:48`), α-B&B resume (`nz_alpha_bnb_dr.jl:1074`) | 키에 집합 매개변수 없음 | 한 프로세스에서 κ 를 바꾸면 잘못 재사용 (지금은 프로세스당 한 설정이라 괜찮음) |
| `nz_benders.jl:101, 150`, `nz_ccg.jl:66`, `nz_bd_kkt.jl:65` | q̂ 또는 (q̂, q̂) 를 집합 안의 점으로 가정 | l ≤ q̂ ≤ u 면 유효 → assert 만 |

### zero-sum

| 파일:줄 | 무엇 | 상자가 빠지면 |
|---|---|---|
| ISP-L `true_dro_build_isp_leader.jl:43-50, 94-97`, ISP-F `true_dro_build_isp_follower.jl:39-44, 60-66`, ε̃ = 0 compact `true_dro_build_subproblem.jl:795-804` | 집합 사본 | mini-Benders·structured outer cut 이 집합 밖 (a, d) 로 만들어져 Z₀ 과대 → LB 가 최적을 넘을 수 있음 |
| `true_dro_structured_eval.jl` (leader 272-277, 307-315, 454-459; follower 540-545, 615-621, 1203-1208, 1253-1259, 1404-1410, 1760-1766) | 집합 사본 | 모든 경로가 TV 모형 V*. CCG 하한 과대 → 틀린 값으로 `:Optimal` |
| `bd_kkt_benders.jl:33-38, 45, 56-60` | 집합 사본 | 벤치마크 상계가 TV 모형 → 비교 왜곡. BD-G (`Og`, 136) 는 빌더에서 상속 |
| `true_dro_recover.jl:46-119` (Piece-F) | TV 쌍대 사용 | 상자 승수 필요 (`test_piece_f.jl` 만 씀) |
| `couple_clean` (`true_dro_data.jl:141-150`), `_clean_belief` (`belief_menu_benders.jl:60-78`), α-step clamp (`true_dro_benders.jl:885-887`) | TV 만 보고 정리, clamp 는 아래쪽만. `fix(force=true)` (`belief_menu_benders.jl:92`, `true_dro_benders.jl:911-921`) 가 범위를 지움 | 상자 비율 추가, 양쪽 clamp |
| 빌더 키워드 호출 약 15 곳 (`true_dro_benders.jl`, `belief_menu_benders.jl`, `global_bilinear_solver.jl`, `bd_kkt_benders.jl`, `run_nig_coupled_bdkkt.jl:92`, `diag_bound_compare_zs.jl:37`) | `delta_couple` 처럼 키워드로 넣으면 한 곳만 빠져도 조용히 큰 집합 | `TrueDROData` 필드로 넣을 것 |

## 그대로 유효함 (고칠 필요 없음)

- 비제로섬:
  - θᵁ 회로 정확값, ϖᵁ·πᵁ: recourse 만 봄.
  - λᵁ: 집합이 작아지면 임계 λ* 도 작아짐.
  - 쌍대 범위 조임 전부 (KKT-MIP, corner LP, DD, V-cut, KMC, FWD, comp): 시나리오별 recourse 쌍대라 (a, d) 와 무관.
  - 반응 프리솔브.
  - root tighten·infeas_pre·Farkas: 노드 LP 에서 상속.
  - TV RLT 행 (`nz_alpha_bnb.jl:74-90`): 상위 집합이라 유효, 약할 뿐.
  - 유효 차원 min(n_h, S−1), belief cell 의 d_s > 0 조건.
  - BD-G·CCG master·L-shaped·cut 식 (`nz_omega.jl:237-253`).
- zero-sum:
  - outer cut 의 π_x·절편, MW·Sherali.
  - objF·arcwise VI, φ̂ᵁ·φ̃ᵁ, local ρ ≤ 10, TV RLT.
  - min-cut VI (q̂ 가 두 상자 안이면).
  - `reduce_coupling`.
  - vertex-cover 정리 자체.

## 다시 유도할 곳

- **zero-sum 구조 평가:**
  - TV 꼭짓점 닫힌 식 `_tv_ball_vertices` (839-869), facet `_tv_tight_sets` (878-893), DD 초기화 (925-949, 1655-1683), 소속 검사 (997-1009). → 상자 facet 을 DD cut 으로 추가하고 소속 검사에 상자를 넣음.
  - leader KKT (297-380): 상자 쌍대 2S 개와 상보성 2S 쌍 추가.
- **선택 (지금도 유효, 느슨할 뿐):**
  - `_nz_tv_max` (`nz_alpha_bnb_dr.jl:434-441`, resp_fix, 기본 꺼짐): 상자를 지키는 greedy 로.
  - belief cell 다각형 (`nz_belief_cells.jl:79-84`): 상자 반평면으로 자르기.

## 증명·이론

- **zero-sum "β = 0, ε̃ ≥ ε̂ 이면 결합 효과 없음":** (a, a) ∈ D₀ 가 필요함. 즉 truth 상자 ⊆ belief 상자여야 하고, 같은 κ 면 성립. 다른 κ 면 "Coupled = Product" 결과를 다시 확인해야 함.
- **min-cut VI 증명** (`docs/omp_valid_inequality.md` Step 1): q̂ ∈ D̂∩D̃ 가 필요함. (1±κ)q̂ 상자면 성립.
- **Lemma geom** (`run_coupling_batch.jl:6`): B̃ ⊆ B̂ 가 필요함.

## 지금까지의 belief 발견은 무효인가

- **메커니즘과 조건 (왜·언제)** 은 follower 반응 h*(d) 와 recourse 구조의 성질이라 집합 모양과 무관함. 그대로 유지됨. 예: B 가 A 쪽 고객을 계약으로 가져오는 몫이 예측에 달림, 꼬리 조건, A 한도는 경로만 바꿈.
- **수치는 다시 계산해야 함.**
  - A 한도 스윕·시드 결과에서 증폭이 있던 60 사례 모두, 최악 a 와 d 가 한 시나리오에 0.2 를 넘게 몰려 있음 (최대 a 중앙값 0.40, 최대 d 중앙값 0.335). κ = 1 상자에서는 이 최악점이 모두 집합 밖.
  - 따라서 증폭 크기, 노출 결정 목록, Data-DR 동률은 상자에서 다시 계산해야 함.
  - 상자에서 증폭이 줄지, Data-DR 이 덜 평평해져 결정 차이가 보일지는 돌려 봐야 앎.
- **zero-sum Coupled = Product 와 β = 0 증명**은 같은 κ 면 유지.
- **계산 성능 결과** (belief cut, α-B&B) 는 TV 모형 기준. 상자를 넣으면 다시 재야 함.

## 공수 (점검 추정)

| 부분 | 코드 | 검증 |
|---|---|---|
| 비제로섬 알고리즘 + 벤치마크 | 약 1.5 일 (사본 약 8 곳 + helper + clean 비율 + 스크립트·RESULT) | 약 1 일. κ = ∞ 가 현재 값을 재현하는지, 작은 S 에서 Ω·`nz_kkt_value`·α-B&B 값이 같은지 |
| zero-sum (구조 평가 제외) | 약 2 일 | 포함 |
| zero-sum 구조 평가 (꼭짓점·leader KKT) | +2~3 일 | |
| 벤치마크 재실행 | 큐 시간이 주 비용 | |
