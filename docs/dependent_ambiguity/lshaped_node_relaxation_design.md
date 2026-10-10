# α-B&B 노드 완화의 시나리오 분해 (L-shaped) 설계안 (2026-10-10, 구현 전 검토용)

> **상태 (2026-10-10, 구현·측정 완료): 미채택.** 단일 노드 값은 정확히 일치하지만 600 s 누적 비교에서 S=50·200 모두 full 노드 LP 보다 나쁨
> (master 에 확률·곱 변수 ~80·S 가 남아 master LP 가 시간의 98%). 결과와 다음 후보 (라그랑지안 완화) 는 `alpha_bnb_nonzero_analysis.md` §7 A 결과.

목표: 큰 S (50, 200) 에서 α-B&B 노드 완화 LP (S=200: 변수 48k, 제약 45k, 루트 57 s) 를 시나리오 블록으로 분해해
(i) master 를 작게, (ii) 블록 cut 을 트리 전체에서 재사용해 노드마다 처음부터 풀지 않게 한다.
배경과 측정은 `alpha_bnb_nonzero_analysis.md` §5~7.

## 1. 분해 (어떤 변수·행을 어디에)

노드 완화 R(K) 의 변수 (nz_omega.jl 이름):

| 묶음 | 변수 | 위치 |
|---|---|---|
| 연결 | h^r (α), belief a·b·r·d·e·cpl, ζL·ζF·ζW, RLT 곱 Z_a·Z_b·Z_e·Z_c, **예약 행 인증서 ϖ_H** (H 행의 ϖ, 15/시나리오) | master |
| 리더 블록 s | ŷ_s, ϖ_s 중 H 행이 아닌 성분 (ϖ_N), ρ̂1·ρ̂2·ρ̂3 (x 행) | 시나리오 블록 LP |
| follower 블록 s | ȳ_s, π̃_s, ρ̃, (κ, ρ⁰ 는 시나리오 공통) | 1 단계: master 에 둠 / 2 단계: 분해 (§5) |

ϖ_H 를 master 에 두는 이유: 상자 K 에 의존하는 RLT 행 (ϖ ≥ 0, ϖ ≤ ϖᵁ r 에 (α_i − l_i), (u_i − α_i) 를 곱한 행: ζW − l_i ϖ_H ≥ 0 등) 이
ϖ_H 만 포함하게 되어 **리더 블록 LP 가 상자 K 와 무관** 해진다 → 블록 cut 이 트리 전체에서 유효.
(ϖ_H 를 블록에 두면 cut 은 그 노드의 subtree 에서만 유효해진다.)

리더 블록 LP (시나리오 s, master 값 m_s = (r_s, ζL_{·s}, ζW_{·s}, ϖ_{H,s}) 고정):

    Q_s(m_s) = max  (ℓ + θc)ᵀŷ − θ Σ_{k∈N} (u_ks + [k∈X] g_ks x̄_k) ϖ_ks − Σ_{k∈X} π̂ᵁ_k [x̄ ρ̂1 + (1−x̄) ρ̂3]
       s.t. Aŷ + [X 행](ρ̂2 − ρ̂3) ≤ u_s r_s + hcoef ζL_s                 (Lflow)   ← 우변이 master
            ρ̂1 + ρ̂2 − ρ̂3 ≥ −g_s r_s                                      (LMC)
            Σ_{k∈N} A_kj ϖ_kj ≥ c_j r_s − Σ_{k∈H} A_kj ϖ_{k,s}            (N1)      ← 우변이 master
            cᵀŷ − Σ_{k∈N} (u_ks + [X] g_ks x̄) ϖ_ks ≤ Σ_{k∈H} u_ks ϖ_{H,s} + Σ hcoef ζW_s   (WD) ← 우변이 master
            ŷ, ϖ_N, ρ̂ ≥ 0

목적의 −θ hcoef ζW, −θ u ϖ_H 항은 master 목적에 남긴다. Wcap (ϖ_H ≤ ϖᵁ r) 과 인증서 RLT 행은 master 행.
- 크기: 변수 ~65, 행 ~72. x̄ 는 노드 내내 고정 (α-B&B 는 고정 x̄ 에서 돌므로) → 블록 LP 의 계수는 노드·상자와 무관.
- 값 함수는 m_s 에 대해 **concave, 1 차 동차** (모든 우변이 m_s 에 선형, 상수항 없음) → cut 은 절편 없는 선형:
  η_s ≤ λ_sᵀ B_s m_s  (λ_s = 블록 LP 의 쌍대해). master 목적에 Σ_s η_s 를 더한다 (multi-cut).
- 실행가능성: ŷ 는 RCR 로 항상 존재, ϖ_N = 0 은 N1 을 만족 (c ≤ 0, A 의 H 행 계수 ≥ 0). WD 때문에 불가능해질 수 있는 master 값에는
  실행가능성 cut (Farkas 광선) 을 넣는다. 구현 초기엔 블록 LP 에 WD 위반 변수 (큰 벌점) 를 넣어 complete recourse 로 처리하는 방안도 비교.

## 2. 노드에서의 계산 (cut loop)

    입력: 상자 K, 전역 cut pool C (블록 s 마다 λ 목록), 안정화 중심 m̄
    1. master R_M(K, C) 를 dual simplex 로 (warm start) → (m*, η*), 상한 z_M
    2. 분리점 m̃ 선택 (§3 in-out), 블록 s = 1..S 를 병렬로: Q_s(m̃_s), 쌍대 λ_s (필요하면 Magnanti–Wong 으로 교체)
    3. η*_s > λ_sᵀ B_s m*_s + tol 인 블록에만 cut 추가 (C 에도 저장)
    4. 위반 cut 이 없거나 라운드 상한 (B_rounds) 에 닿으면 종료. 아니면 1 로.

- **어느 시점에 멈춰도 z_M 은 유효한 상한**: master 는 블록 값을 위에서 근사하므로 R(K) 의 완화 → B&B 에 그대로 쓸 수 있음.
  자식 노드는 라운드를 적게 (§B 실험의 child_rounds 와 같은 원리).
- RLT 분리 (기존 _nz_sep_solve!) 는 master 에 대해 그대로 수행. cut loop 와 RLT 분리를 번갈아 (또는 한 라운드에 둘 다).
- cut pool 관리: 활성 (최근 tight) cut 만 master 에, 나머지는 pool 에 두고 위반 시 복귀. 블록당 상한 (예 50 개).

## 3. 안정화 (Benders 의 진동·느린 수렴 대비)

1. **In-out (Ben-Ameur & Neto 2007)**: 분리점 m̃ = λ m* + (1 − λ) m̄ (λ ≈ 0.5). m̄ 는 직전 노드 (또는 부모) 의 master 해.
   m̃ 에서 cut 이 위반되지 않으면 λ → 1 로 (기본 L-shaped 로 복귀) 해서 유한 수렴 유지.
   LP Benders 에서 반복 수를 크게 줄이는 표준 기법이고 구현이 가볍다.
2. **Magnanti–Wong (Pareto-optimal cut)**: 블록 LP 가 쌍대 퇴화 (수송형이라 흔함) 라 같은 m* 에서 cut 이 여러 개.
   Q_s(m*_s) 를 먼저 풀고, 같은 최적값을 유지하는 쌍대해 중 core point m°_s 에서 가장 강한 것을 고른다 (블록 LP 한 번 더, 작음).
   core point 는 Papadakos (2008) 처럼 지금까지의 master 해들의 볼록조합으로 갱신. Benders·belief cut 에 이미 쓰는 MW 와 같은 발상.
3. 초기 cut: 루트에서 master 가 비어 있으면 첫 해가 엉뚱하므로, q̂ 기준 belief (r = q̂ 근처), 상자 중점 α 에서 블록을 풀어 시작 cut 을 만든다.

기본은 multi-cut + in-out, MW 는 옵션 (효과를 측정해 기본값 결정).

## 4. 기대 효과와 위험

- master 크기: 리더 블록만 분해하면 변수 ~241·S → ~176·S (follower 를 남기므로), 2 단계까지 하면 ~80·S + cut.
  S=200: 48k → 35k (1 단계) → 16k + cut (2 단계).
- 더 큰 이득은 **cut 재사용**: 블록 cut 이 상자와 무관하므로 노드가 바뀌어도 그대로 유효. 새 노드는 몇 라운드면 될 것으로 기대.
- 위험: (i) master 에 남는 곱 변수 (Z, ζ) 가 커서 1 단계만으로는 이득이 작을 수 있음, (ii) cut loop 가 RLT 분리와 상호작용해 라운드가 늘 수 있음,
  (iii) 블록 LP 가 많아 (S 개) 병렬화·오버헤드 관리 필요.
- 판정 기준: S=50, 200 에서 300 s 노드 수·UB 를 현재 최선 (child1) 과 비교. 루트 LP 시간 (S=200: 57 s) 단독 비교도.

## 5. 2 단계: follower 블록 분해

follower 블록은 시나리오 공통 행 3 종 (Fh: Σ_s Bᵀπ̃_s + c0 ≤ Wᵀκ, Flam: Σ_s (uᵀπ̃_s − cᵀȳ_s) + … ≤ c0ᵀα, Fpsi: Σ_s gᵀπ̃_s … ≤ 0) 로만 묶임.
집계량 t_s = Bᵀπ̃_s (n_h), w_s = u_sᵀπ̃_s − cᵀȳ_s (1), v_s = g_sᵀπ̃_s (n_x) 를 master 변수로 올리면
follower 블록 s 는 "(d_s, ζF_s) 가 주어졌을 때 (t_s, w_s, v_s) 를 만들 수 있는가" 의 실행가능성 집합 → 실행가능성·최적성 cut 으로 근사.
리더 블록과 같은 상자 무관성 조건 (ζF, d 만 master) 을 확인한 뒤 진행.

## 6. 구현 순서

1. 리더 블록 LP 빌더 + 쌍대 추출 + cut 공식, 단일 노드에서 R(K) 와 master+cut loop 의 값이 같은지 검증 (S=3, 20).
2. in-out, MW 추가, 단일 노드 수렴 라운드 측정 (S=50, 200 루트).
3. α-B&B 에 통합 (worker 별 master, 공유 cut pool), child_rounds 와 결합, 300 s 벤치마크.
4. 효과가 있으면 2 단계 (follower).
