# 꼭짓점 LP 쌍대 범위의 근거: pooled 예약 recourse 의 대체재 성질 (2026-10-28, 초안)

`nz_alpha_bnb_dr(...; range_lp=true)` (`_nz_dr_corner_ranges`) 는 상자 B = [l, u] 에서 예약 행 쌍대 σ_i 의 범위를
  σ_hi = max{σ_i : σ ∈ α = l 의 최적 쌍대면},   σ_lo = min{σ_i : σ ∈ α = u 의 최적 쌍대면}
로 잡는다. 이 범위가 유효하려면 **σ_i 가 모든 h 좌표에 대해 단조 감소** 해야 한다 (recourse 값함수가 submodular).
아래는 location (reservation = :pooled) recourse 에서 그것이 성립한다는 논증. 정리 번호는 원문 대조 전 (TODO).

## 1. recourse (시나리오 하나, x̄ 고정)

점포 i (용량 b_i, B 점포는 b_B x̄_i), 고객 j (수요 ξ_j), 예약분 y^R_ij (비용 d_ij), 현장분 y^S_ij (비용 d_ij + p, p > 0):

    C(h) = min Σ_ij d_ij (y^R_ij + y^S_ij) + p Σ_ij y^S_ij
           s.t. Σ_i (y^R_ij + y^S_ij) ≥ ξ_j,   Σ_j y^R_ij ≤ h_i (예약 풀),   Σ_j (y^R_ij + y^S_ij) ≤ b_i,   y ≥ 0

예약 행의 쌍대 σ_i ≥ 0 = 예약 풀 1 단위의 한계 절감액 (σ_i ∈ ∂(−C)(h) 의 i 성분). 코드의 recourse 는 max cᵀy, c = −비용 이므로 z = −C.

## 2. 단계 1 — 예약을 먼저 쓴다

같은 (i, j) 에서 예약분이 현장분보다 p 만큼 싸고 둘은 같은 용량을 쓴다. 점포 i 의 총 출하 Y_i = Σ_j (y^R_ij + y^S_ij) 가 정해지면
예약 사용량은 min(h_i, Y_i) 가 최적 (예약 풀에 여유가 있는데 현장분이 있으면 바꿔서 p 를 아낌; 어느 고객분을 바꾸든 비용 같음). 따라서

    C(h) = min_Y { Σ_ij d_ij Y_ij + Σ_i p (Y_i − h_i)⁺ : Σ_i Y_ij ≥ ξ_j,  Y_i ≤ b_i,  Y ≥ 0 }

(h_i ≤ b_i 가정 — 코드에서 hU = quota 300 ≤ 용량 450. 0 ≤ h.)

## 3. 단계 2 — M♮-볼록성 (Murota 2003, Discrete Convex Analysis; 실수값은 Murota & Shioura 2000 의 polyhedral 판)

공급량 s (점포별 출하) 의 수송비 T(s) = min{Σ d_ij Y_ij : Σ_j Y_ij = s_i, Σ_i Y_ij ≥ ξ_j, Y ≥ 0} 와 G(s) = T(s) + [s ≤ b] (지시함수) 로 두면

    C(h) = min_s { G(s) + Σ_i φ(s_i − h_i) },   φ(t) = p·t⁺   (1 변수 볼록)

  (a) T 는 M♮-볼록 (망을 통한 변환 / transformation by networks) — 정리 번호 TODO. 상자 지시함수를 더해도 M♮-볼록.
  (b) Φ̃(v) = Σ_i φ(−v_i) 는 분리 볼록 → M♮-볼록. C(h) = min_s G(s) + Φ̃(h − s) = (G □ Φ̃)(h) 는 M♮-볼록 함수의 infimal convolution → M♮-볼록 — 정리 번호 TODO.
  (c) M♮-볼록 함수는 supermodular — 정리 번호 TODO.

## 4. 단계 3 — 꼭짓점 범위

C 가 supermodular 이고 볼록이면 −C 의 방향 미분이 모든 좌표에 단조:
  α ≥ α′ (성분별) 이면  D⁻_i(−C)(α) ≤ D⁻_i(−C)(α′),  D⁺_i(−C)(α) ≤ D⁺_i(−C)(α′).
σ ∈ ∂(−C)(α) 이면 D⁺_i(−C)(α) ≤ σ_i ≤ D⁻_i(−C)(α). 그리고 최적 쌍대면 위의 max σ_i = D⁻_i(−C)(α), min σ_i = D⁺_i(−C)(α) (polyhedral).
따라서 α ∈ [l, u] 의 모든 최적 쌍대에 대해
  σ_i ≤ D⁻_i(−C)(α) ≤ D⁻_i(−C)(l) = max{σ_i : σ ∈ l 의 최적 쌍대면},
  σ_i ≥ D⁺_i(−C)(α) ≥ D⁺_i(−C)(u) = min{σ_i : σ ∈ u 의 최적 쌍대면}.
l_i = 0 (경계) 에서는 왼쪽 미분이 없을 수 있으나 코드의 쌍대 LP 는 σ_i ≤ ϖᵁ_i 로 막혀 있고 (Wcap 과 같은 논증: H 성분이 ϖᵁ 이하인 최적 쌍대가 있다) 그 상한 안에서 위 부등식이 성립.

## 5. 수치 확인

- `diag_root_l6.jl` (6): L6 seed 7 x̄={4,5}, 상자 3 개 — 63/63 성분이 범위 MIP 와 같음.
- `test_corner_ranges.jl`: 7 조합 × 상자 6 — 성분 1,446 개 중 꼭짓점이 더 좁음 0, 더 넓음 104 (MIP 시간 제한·W 행 차이로 추정), MIP 3,140 s vs LP 1.3 s.

## 6. 원고에서의 위치 (2026-10-28 사용자 방향)

일반론: α-B&B 의 노드 완화는 유효한 쌍대 범위가 주어지면 유효, 성능은 범위가 좌우. 범위는 KKT-MIP (일반, 비쌈) 또는 구조 활용.
"recourse 한계값이 단조이면 꼭짓점 LP 두 개" 는 일반 명제 (증명 한 줄). location 에서 성립한다는 것은 응용 예의 짧은 보조정리 (위 1~4).
이 성질을 깊게 부각하지 않는다 (문제 특화 방법론으로 보이지 않게).
