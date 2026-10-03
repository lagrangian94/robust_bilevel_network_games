# ccg.tex belief cover의 degeneracy 분석과 vertex cover

관련 코드: `true_dro/true_dro_structured_eval.jl` (`generate_belief_cover` = 논문 Alg.1,
`generate_belief_vertex_cover` = 본 문서 제안), 테스트 `true_dro/test_structured_eval.jl`.

## 1. 관측 (Alg.1 그대로 구현 시)

grid 3×3 (K=17), S=2, ε̃=0.2, x̄ ∈ {0, 3개 랜덤}:

| 항목 | 값 |
|---|---|
| 60초 동안 찾은 패턴 수 | 511 ~ 1027 (미완료) |
| 그 중 서로 다른 belief | **1개** (중심 q̃=(0.5,0.5), δ=0) |
| 패턴 간 값이 바뀌는 binary | zρ 0/1, zh 3/17, **zλ 23/34, zy 10/34** |

→ 열거가 중심 belief 하나에서 벗어나지 못함. 바뀌는 binary는 대부분 recourse 쪽(zλ, zy).

## 2. Degeneracy란

LP에서 (i) 최적해가 여러 개이거나(primal 비유일), (ii) 최적 쌍대해가 여러 개이거나(dual 비유일),
(iii) 어떤 상보쌍 (slack, dual)이 **둘 다 0**인 경우. (iii)이면 패턴 binary z가 0/1 아무 값이나 가능.

최대유량 follower에서의 출처:

| 출처 | 내용 | 영향받는 z |
|---|---|---|
| 같은 유량값의 다른 흐름 | 경로 분해/순환/0용량 arc | zy (y>0 여부), zλ (capacity 행 tight 여부) |
| 복수 min-cut | 쌍대 (λ,μ) 비유일 | zλ, zy (reduced cost=0 여부) |
| slack=0 & dual=0 | 포화됐지만 cut 밖 arc 등 | zλ, zy 자유 |
| h의 비유일 | 동률인 회복 배분 | zh |

Alg.1은 **패턴 z 단위**로 no-good cut을 넣으므로, 같은 면(face)·같은 belief에 대해
위 조합 수만큼(지수적) MILP를 풀어야 다음 belief로 넘어감. 그러나 decomposed/disjunctive 평가는
belief p̃ʲ에만 의존(ℱ(p̃ʲ)는 z를 쓰지 않음)하므로 이 라벨들은 전부 낭비.

## 3. 이론: belief 공간에서의 cover (degeneracy-free)

x̄ 고정. follower 가치함수 φ(p̃) := max_{h∈H} Σ_s p̃_s Q_s(h) — P_x̄의 support function을
(0,p̃) 방향으로 본 것이므로 p̃에 대해 convex piecewise-linear.
F(p̃) := (0,p̃) 방향 최적면, Ψ(x̄,p̃) = proj_h F(p̃).
Cell C_F := {p̃ ∈ Δ : F(p̃) = F} (normal cone의 relative interior ∩ Δ).
𝒞_x̄ := 이 cell들의 closure가 이루는 polyhedral complex (= φ의 linearity complex).

**Lemma A (상반연속).** p̃ ∈ C_F, v ∈ cl(C_F) ⇒ F ⊆ F(v), 따라서 Ψ(x̄,p̃) ⊆ Ψ(x̄,v).
*증명.* F가 (0,p̃ₙ)에 대해 최적이고 p̃ₙ→v이면, 최적성 조건 ⟨(0,p̃ₙ), u−u'⟩ ≥ 0 (u∈F, u'∈P)이
극한에서 유지되므로 F는 v에서도 최적, 즉 F ⊆ F(v). ∎

**Theorem B (vertex cover).**
Ψ(x̄, 𝒟̃) = ∪_{v ∈ vert(𝒞_x̄ ∩ 𝒟̃)} Ψ(x̄, v).
*증명.* (⊇) v ∈ 𝒟̃. (⊆) p̃ ∈ 𝒟̃, p̃ ∈ C_F. cl(C_F) ∩ 𝒟̃는 비어있지 않은 폴리토프이므로
꼭짓점 v를 가지며 v는 𝒞_x̄ ∩ 𝒟̃의 꼭짓점. Lemma A로 Ψ(x̄,p̃) ⊆ Ψ(x̄,v). ∎

**Corollary C.** V*(x̄) = max_{v} W(v), W(v) = max{V̄ᴸ(x̄,α): α∈Ψ(x̄,v)} (ccg.tex 식 evalMILP 그대로).

비고:
- ccg.tex Prop. menu는 cone N_I마다 q̃에 가장 가까운 대표 p̃_I를 쓰고 Alg.1이 이를 **패턴 z**로
  실현한다. Theorem B는 같은 역할을 **belief 공간의 꼭짓점**으로 실현하며, 꼭짓점은 z, y, (λ,μ)와
  무관하므로 degeneracy가 원천적으로 사라진다(φ는 값만으로 정의됨).
- 꼭짓점은 최소 면이므로 lower-dimensional cell(=더 큰 Ψ)을 자동으로 고른다. 최소성: 𝒞_x̄의
  0차원 cell(진짜 꼭짓점) v는 반드시 필요(그 Ψ(v)를 포함하는 다른 cell이 없음).
- 𝒞_x̄는 ε̃와 무관 → ε̃ sweep 시 complex를 재사용하고 볼과의 교집합만 다시 계산 (Remark thresholds 대응).

## 4. 계산: cutting-plane double description

Π_T := {(p̃,τ) : p̃∈Δ, (TV cuts), τ ≥ p̃ᵀθⁱ (i∈T), τ_min ≤ τ ≤ τ_max} 의 꼭짓점을 DD로 유지.
각 하단 꼭짓점 (p̃,τ)에 대해:
1. p̃ ∉ 𝒟̃ → TV facet Σ_{s∈A}(p̃_s − q̃_s) ≤ ε̃, A = {s: p̃_s > q̃_s} (TV = max_A Σ_A(p̃−q̃), 정확한 분리)
2. φ(p̃) > τ → envelope cut τ ≥ p̃ᵀθ, θ_s = Q_s(h*) (h*: p̃에서 follower LP 최적해, Q_s는 S개 max-flow)
3. 둘 다 만족 → verified.

종료 시 모든 하단 꼭짓점에서 φ ≤ φ_T, 그리고 θⁱ ∈ proj_θ P_x̄이므로 항상 φ ≥ φ_T.
φ_T는 각 영역에서 선형, φ는 convex → 영역 내부에서도 φ ≤ φ_T. 따라서 φ = φ_T on 𝒟̃,
하단 꼭짓점 = vert(𝒞_x̄ ∩ 𝒟̃). 유한 종료: θ는 follower LP의 BFS에서 오고 BFS는 유한.

오라클은 LP 1개 + max-flow S개. **MILP 없음, binary 없음.**
Benders에서 이전 x̄들의 h*를 warm start로 재사용: θ(h) = Q(x̄',h)는 x̄'에서도 유효한 envelope 조각.

### 4.1 구현 이슈: lazy TV cut → 중간 폴리토프 폭증
처음엔 Δ에서 시작해 TV facet을 lazy cut으로 넣었는데, S=10 Abilene에서 TV cut 66개만에
ray 10만 개 (중간 폴리토프가 최종 볼보다 훨씬 복잡). → 볼의 꼭짓점을 **닫힌 형태로 직접 열거**
(초과 좌표 1개 + 0까지 깎인 집합 J + 분수 부족 좌표 ≤1개; 양방향 섭동 불가 ⇒ 극점)하고,
tight facet A = P⁺ ∪ (Z의 부분집합)만 H-rep에 포함하여 DD 초기 상태로 사용. TV cut 0개.

### 4.2 한계: S가 크면 complex 꼭짓점 자체가 많음
S=10, q̃ 균등, ε̃=0.2 (Abilene, x̄=0): 볼 꼭짓점 360개, envelope 조각 11개 시점에 ray 13,421개,
DD 한 단계 135초 → 300초 제한 미완료. 9차원 arrangement 꼭짓점 수가 조각마다 수천 개씩 증가.
(S=2: 2개, S=3: 6개, S=5: 43개 — 작은 S에서는 매우 효과적)
꼭짓점이 많은 주된 이유: 볼 꼭짓점마다 p̃_s = 0인 시나리오 집합이 달라 follower가 무시하는
시나리오가 달라지고(→ 서로 다른 cell), 그 위에 envelope 조각들이 다시 분할.

## 5. Decomposed / Disjunctive 쪽 개선

| 항목 | 내용 |
|---|---|
| Decomposed prune | belief 순회 시 incumbent를 Gurobi `Cutoff`로 넘겨 W_j ≤ incumbent인 MILP는 조기 종료 (정확성 유지) |
| Disjunctive | J = 꼭짓점 수. belief 행 big-M은 활성 belief 대비 잔차로 유계 (|Δp̃| ≤ 1, SD 행 ≤ 2·max_s F_s) |
| 공통 VI | leader KKT를 indicator로 인코딩하면 u 무상한 → root relaxation unbounded. fᵀu = Σ r_s t_s ≤ Σ min(t_s, Tmax_s r_s) (McCormick 상한)를 추가하여 해결 |

## 6. 실험 결과 요약 (2026-10-02, `true_dro/test_structured_eval.jl`)

### 6.1 정확성 (global bilinear 대비, tol 1e-4)
| 범위 | 결과 |
|---|---|
| grid3×3 S=2,3 × {β 없음, β=0.4, ε̃=0} × x̄ 3~4개 × 3경로 × {indicator, bigM} | 실패 0 |
| grid4×4 S=5 × 동일 3설정 × x̄ 3개 × 3경로 × 2인코딩 | 실패 0 |
| Benders 전체 (grid3×3 S=3, β 없음/0.4): Z₀, x* | 3경로 모두 기존과 일치 |
| leader=:bilinear 변형 (grid3×3 S=3, 6 x̄) | 실패 0 |
모든 경우 α* 고정 LP(=cut) 값 = V, V̄F = 0 (cut이 x̄에서 tight).

### 6.2 속도 — S≤5 (vertex cover 완전 열거)
grid4×4 S=5 (K=31): global 42~402s / monolithic 0.2~14s / decomposed 1~13s / **disjunctive 0.1~1.8s**, cover 0.02~0.05s (J=23~47).
Benders grid3×3 S=3: disjunctive 0.35s vs 기존 0.40~0.66s (12 iter 동일).

### 6.3 속도 — S=10 (factor_5 batch 인스턴스, β=0.4, ε̂=ε̃=0.2, 제한 300s)
| 인스턴스 / x̄ | global bilinear | decomposed (부분 cover 361, leader bilinear) | disjunctive (부분) | monolithic (bilinear / KKT) |
|---|---|---|---|---|
| Abilene ∅ | 23.105698 (≈증명, 302s) | **23.105693, 241s, 357/361 pruned** | 23.0254 | 23.1057 / 23.1057 (미증명) |
| Abilene [8,11] | 18.6121, bd 19.0167 | **18.6457, 280s** | 18.5645 | **18.6457** / **18.6457** |
| Polska ∅ | 25.6329, bd 25.6475 | 25.6329 (시간초과) | 25.6329 | 25.6329 / 25.6329 |
| Polska [8,11] | 25.6329, bd 25.6426 | 25.6329 (시간초과) | 25.6329 | 25.6329 / 25.6329 |
| grid5×5 ∅ | 46.2890, bd 46.2901 | 46.2890 (시간초과) | 46.2890 | 46.2890 / 46.2890 |
| grid5×5 [17,23] | 42.2701, bd 42.2736 | 42.2701 (시간초과) | 42.2701 | 42.2701 / 42.2701 |
- 모든 구조 활용 방법이 300s 내 global incumbent 이상의 해를 찾음 (Abilene [8,11]은 global보다 좋음).
- 최적성 증명은 global이 가장 근접. 구조 활용 MILP는 relaxation이 약해 bound가 큼.
- vertex cover 전체 열거는 S=10에서 미완료 (J ≥ 15,975).

## 7. 발견한 문제와 해결 (시간순)
| # | 문제 | 원인 | 해결 |
|---|---|---|---|
| P1 | indicator 인코딩 root relaxation unbounded | leader 쌍대 u 무상한, 목적 fᵀu | fᵀu ≤ Σ min(t_s, Tmax_s r_s) McCormick VI |
| P2 | Alg.1 cover 미종료 (60s/패턴 1000+, belief 1개) | degeneracy (zλ 23/34, zy 10/34 변동) | Theorem B vertex cover (belief 공간, y·쌍대 무관) |
| P3 | vertex cover ray 10만 개 (S=10) | lazy TV cut 중간 폴리토프 폭증 | 볼 꼭짓점 닫힌 형태 + tight facet만 H-rep |
| P4 | S=10에서 complex 꼭짓점 수만 개 | 9차원 arrangement 본질적 크기 | **미해결** — 최소 cell 대표 이론 필요 |
| P5 | S=10 monolithic 600s 가능해 없음 | KKT binary ~690개 | LP primal-dual 기반 MIP start |
| P6 | belief 고정 W(p)도 느림 | leader KKT binary relaxation 약함 | `leader=:bilinear` (S개 bilinear 항, NonConvex=2) |
| P7 | decomposed가 시간 제한을 오류로 처리 | 구현 | 부분 결과(하한) + is_exact=false + 미방문 bound=∞ |
| P8 | indicator 해에서 V̄F=−5e−5 (Polska, KKT leader) | indicator 수치 허용오차 | 허용오차 내, 관찰만 |

## 8. Minimal cover — 국소 결정 정리 (P4 해결 시도, 2026-10-02)

### 8.1 설정
Θ := {θ ∈ ℝ^S : ∃h ∈ H, θ ≤ Q(h)} (polyhedral, down-closed, 위로 유계). σ(u) := sup_{θ∈Θ} uᵀθ 는
u ≥ 0 에서만 유한, φ(p) = σ(p) (p ∈ Δ). p ∈ Δ, S(p) := supp p, Z(p) := {s : p_s = 0}.
Θ_S := proj_S Θ, F_S(p) := argmax_{θ_S ∈ Θ_S} p_Sᵀθ_S.

**Lemma 1 (반응집합 = 노출면의 역상).** Ψ(x̄,p) = {h ∈ H : Q_{S(p)}(h) ∈ F_{S(p)}(p)}.
*증명.* h ∈ Ψ ⇔ pᵀQ(h) = φ(p) ⇔ p_Sᵀ Q_S(h) = σ_{Θ_S}(p_S) (Q_S(h) ∈ Θ_S) ⇔ Q_S(h) ∈ F_S(p). ∎

**Lemma 2 (국소 결정).** T ⊂ Θ 유한, φ_T(p) := max_{θ∈T} pᵀθ, A_T(p) := argmax_{θ∈T} pᵀθ.
𝒟̃⁺ := {p ∈ Δ : d_TV(p,q̃) ≤ ε̃ + δ} (δ > 0) 에서 φ_T = φ 이면, 모든 p ∈ 𝒟̃ 에 대해
  F_{S(p)}(p) = conv{θ_{S(p)} : θ ∈ A_T(p)}.
*증명.* S = S(p), g(u) := σ((u,0)), g_T(u) := max_{θ∈T} uᵀθ_S (u ∈ ℝ^S_+). p_S ∈ ℝ^S_{++}.
p_S의 충분히 작은 근방 U ⊂ ℝ^S_{++}의 u에 대해 (u,0)/1ᵀu 는 Δ의 면 {p_Z = 0} 위에 있고
d_TV ≤ d_TV(p,q̃) + o(1) ≤ ε̃ + δ 이므로 𝒟̃⁺에 속함 → 동차성으로 g = g_T on U.
g는 convex이고 p_S는 정의역 내부점이므로 ∂g(p_S)는 g의 국소 값만으로 결정: ∂g(p_S) = ∂g_T(p_S)
= conv{θ_S : θ ∈ A_T(p)} (Danskin). 한편 g = σ_{Θ_S} (Θ_S 닫힘) 이므로 ∂g(p_S) = F_S(p). ∎
(핵심: 미발견 vertex와의 동률은 𝒟̃의 경계에서만 문제였는데, 확대 볼에서 envelope를 보장하면
 경계점에서도 근방이 확대 볼 안에 있으므로 동률이 전부 T 안에서 드러남.)

**Lemma 3 (패턴 단조성).** π(p) := (Z(p), A_T(p)). π(p) ≤ π(p') (성분별 ⊆) 이면 Ψ(p) ⊆ Ψ(p').
*증명.* S' = S(p') ⊆ S = S(p). h ∈ Ψ(p) ⇒ Q_S(h) ∈ conv{θ_S : θ ∈ A_T(p)} ⇒ (S'로 사영)
Q_{S'}(h) ∈ conv{θ_{S'} : θ ∈ A_T(p)} ⊆ conv{θ_{S'} : θ ∈ A_T(p')} = F_{S'}(p') ⇒ h ∈ Ψ(p'). ∎

**Theorem 4 (minimal cover).** Π := {π(p) : p ∈ 𝒟̃} (유한). 각 극대 π ∈ Π 마다 p_π 하나를 고르면
Ψ(x̄,𝒟̃) = ∪_π Ψ(x̄,p_π). (유한 poset → 모든 π(p)는 어떤 극대 π 이하; Lemma 3.)
패턴 기준 최소: 모든 패턴을 지배하는 belief 집합은 각 극대 패턴의 실현점을 반드시 포함.

### 8.2 계산
**Prop 5 (envelope 분리, 정확).** v* := max { Σ_s p_s f_s − τ : p ∈ 𝒟̃⁺, (h,y,f) follower 가능해,
τ ≥ pᵀθ ∀θ∈T }. 고정 p에서 (h,y,f)에 대한 max = φ(p) 이므로 v* = max_{𝒟̃⁺}(φ − φ_T) ≥ 0.
v* = 0 ⇔ φ_T = φ on 𝒟̃⁺. bilinear 항은 p_s f_s 의 **S개뿐** (NonConvex=2).
v* > 0 이면 p*에서 follower LP (BFS) → θ = Q(h*) ∈ Θ, p*ᵀθ = φ(p*) > φ_T(p*) → 새 조각.
BFS는 유한 → 유한 종료.

**Prop 6 (극대 패턴 열거, 정확).** MILP: p ∈ 𝒟̃, τ ≥ pᵀθⁱ, τ − pᵀθⁱ ≤ M(1−δ_i), p_s ≤ 1 − ζ_s,
max Σδ + Σζ. 해를 패턴 고정 LP로 polish 후 실제 패턴 π_k 기록, cut "π ⊄ π_k" 추가.
(i) 각 π_k는 극대: π_k < π' ∈ Π 이면 π'는 이전 cut을 모두 통과(π' ⊆ π_j 이면 π_k ⊆ π_j 모순)하고
|π'| > |π_k| ≥ 최적값 → 모순. (ii) 극대 π는 자신과 같은 π_j가 나오기 전까지 항상 가능 →
종료(infeasible) 시 모두 열거. ∎  binary 수 = |T| + S.

### 8.3 Vertex cover와의 관계
Theorem B(꼭짓점)는 각 cell 폐포의 모든 꼭짓점을 취하나, 같은 패턴의 꼭짓점들은 같은 Ψ를 주므로
중복. Theorem 4는 패턴당 하나 → |minimal| ≤ |vertex|. 열거도 DD 대신 작은 MILP.

### 8.4 검증 결과 (minimal cover, envelope 인증 = DD on 𝒟̃⁺)
| 인스턴스 | J (minimal) | J (vertex) | cover 시간 | 결과 |
|---|---|---|---|---|
| grid3×3 S=2,3 (β 없음/0.4, x̄ 3개씩) | 1 | 2~6 | ≤0.7s | 전부 certified, global 일치 |
| grid4×4 S=5 (β 없음/0.4, x̄ 3개씩) | 5~8 | 23~47 | 0.1~0.3s | 전부 certified, global(42~300s)과 일치 |
| grid4×4 S=5 disjunctive leader=KKT | | | | **0.05~0.16s** certified |
| Abilene S=10 x̄=∅ | (envelope 미완) | — | DD 900s: 조각 31개, ray ~4.5만 | 미완료 |

관찰:
- leader 선택: S=5·소규모 cover에서는 KKT leader가 bilinear보다 수십~수천 배 빠름 (disjunctive 0.05s vs 150s).
  S=10 belief별 평가에선 bilinear가 나았음 → 규모별 선택.
- S=10 병목은 Lemma 2 전제(𝒟̃⁺에서 φ_T = φ) 인증. 조각 수는 30개 남짓이지만 TV 볼의 조합적 복잡도
  (ε⁺=0.21에서 꼭짓점 2520개, facet ~1000개)와 곱해져 하단 꼭짓점이 수만 개.
  꼭짓점 기반 인증은 convex 최대화와 동치(일반적으로 NP-hard) → 평가 자체의 Σ₂ᵖ-난이도가 이 단계로 이동.
- DD 개선: 제약 역인덱스 인접성 검사 (한 단계 135s → 수 초), 검증은 φ만(LP 1개), 위반 시에만 θ.
- 열린 과제: (a) ε⁺를 부분합 breakpoint로 골라 볼 꼭짓점 축소 (균등 q: ε⁺=0.3 → 840개),
  (b) Benders에서 x̄ 간 envelope 조각 warm start, (c) 인증 없이 패턴 cover를 heuristic으로 쓰고
  exact 단계에서만 인증.

### 8.5 envelope 인증: MILP (follower KKT) 버전 (`envelope_method=:milp`)
max wρ + Σcλ − τ s.t. p ∈ 𝒟̃⁺, 𝒦(p,z), τ ≥ pᵀθ (+ VI wρ+Σcλ ≤ Σ p_s Tmax_s). 위반 ≥ 1e-4면 조기 종료.
| 인스턴스 | DD | MILP indicator | MILP bigM |
|---|---|---|---|
| grid4×4 S=5 (3 x̄) | 0.03~0.05s, 인증 ✓ | 같은 조각 4개, 10~521s, 인증 ✗ | 같은 조각 4개, 2~89s, 인증 ✗ |
| Abilene S=10 x̄=∅ | 900s: 조각 31개, 미완 | **68s에 조각 36개** (위반점 탐색 빠름), bound 6.7에서 정체 → 인증 ✗ | (같은 양상, 중단) |
- S=5의 인증 실패는 bound가 1e-6까지 못 내려간 수치 문제(KKT 허용오차 × capacity 계수) 추정 — 허용오차 스케일링 필요.
- S=10: MILP는 조각 **발견**은 DD보다 훨씬 빠르지만 **인증**(bound ≤ 0)은 KKT relaxation이 약해 불가.
- 그 36조각으로 만든 패턴 cover(J=157)에서 disjunctive가 V=23.105694 (global 23.105698과 일치) 발견 → cover는 최적을 포함, 부족한 건 증명뿐.

## 9. Belief-bilinear 정식화 (bilinear 2S개) vs global bilinear (Ω, 2KS개)
`evaluate_belief_bilinear`: p 변수 + strong-duality 행 Σ p_s f'_s = wρ+Σcλ (S개) + leader Σ r_s t_s (S개).
p 고정 시 ℱ(p) → 정확. Gurobi spatial B&B가 belief(p)와 leader 가중치(r)에서만 분기.

| 인스턴스 | global(Ω) | belief-bilinear (leader bilinear) | belief-bilinear (leader KKT) |
|---|---|---|---|
| grid3×3 S=3 (6 x̄) | 0.02~1s | 0.04~0.4s | 0.04~1s (전부 증명) |
| grid4×4 S=5 (6 x̄) | 0.07s~**300s+ 미완 2건** | 2~7s, 1건 300s 미완 | **0.7~5.8s 전부 증명** |
| Abilene S=10 ∅ | 23.1057, bd 23.1061 | 23.1057, bd 23.33 | 23.1057, bd 28.0 |
| Abilene S=10 [8,11] | 18.6121, bd 19.06 | α* 기준 18.6121, bd 19.60 | α* 기준 18.6343, bd 25.98 |
| Polska S=10 ∅ / [8,11] | 25.6329, bd 25.67 | 25.14 / 25.03, bd 25.97~26.03 | 25.30 / 25.6329, bd 29.7 |
| grid5×5 S=10 ∅ | 46.2890, bd 46.370 | 46.2890, **bd 46.323** | 46.2890, bd 49.66 |
(S=10은 모두 300s 제한 미완. "실패" 3건은 incumbent가 최적이 아닐 때 LP(α*) > V인 정상 현상 — 판정 기준 문제)

결론: S≤5에서 belief-bilinear(KKT leader)·minimal cover가 global을 압도 (증명까지).
S=10에선 leader KKT relaxation이 약하고(bd 25~50), bilinear leader는 인스턴스별 혼재,
global(Ω)의 bound가 대체로 가장 조밀. 원인: p_s·f'_s의 McCormick에서 f' 범위(~25)가 넓음.

## 10. Belief-space B&B (Ω McCormick relaxation + belief 좌표만 분기 + 정확한 W 잎 평가) — 실패
`evaluate_belief_bnb`. grid3×3 S=3 β=0.4:
| x̄ | global(Ω) | belief B&B |
|---|---|---|
| ∅ | 1.04s | 10.1s, 노드 7,665 (정답) |
| [9,10] | **0.02s** | 96s, 노드 100,000 한도, gap 미종료 (rootUB 12.08 vs 9.72) |
원인: ζ = α·b 의 McCormick 오차 ∝ (b 상자 폭)·(α 범위 w). belief만 분기하면 α 범위가 그대로라
모든 belief 좌표를 아주 잘게 쪼개야 bound가 닫힘 (정밀도 기준 지수적 노드 수). 반면 실제 값은
belief cell 위에서 상수(조각상수)인데 McCormick은 이 구조를 모름. Gurobi는 α(최적 반응이 대개
예산을 몇 개 arc에 몰아주는 극단점)로도 분기하고 bound tightening을 하므로 훨씬 빠름.
→ "belief만 분기" 전제가 틀림. 중단.

## 11. Follower 반응 CCG (Zeng형, subproblem 내부) — `evaluate_response_ccg`
열 = follower 대안 반응 h̄ (조각 θ = Q(h̄)). master = relaxation(상한), pricing = follower LP, 상한=하한이면 종료.
master: `:bilinear` (p̃·f' S개 NonConvex) / `:milp` (T-complex ∩ TV볼 꼭짓점 disjunction, σ_v ⇒ vᵀf' ≥ φ_T(v), bigM=φ_T(v), leader KKT).
| 인스턴스 | global(Ω) | CCG bilinear | CCG MILP |
|---|---|---|---|
| grid3×3 S=3 (6 x̄) | 0.02~1.2s | 0.02~0.7s 증명 | 0.01~1.0s 증명 |
| grid4×4 S=5 (6 x̄) | 0.07~300s+ (미증명 2) | 0.4~2s, 2건 300s | **0.08~0.77s 전부 증명** (반복 1~2, 조각 1~2, |V|=20) |
| Abilene S=10 ∅ | 23.1057 | 22.954 bd 23.249 (300s) | 22.463 bd 28.04 (300s, 첫 master에서 소진) |
| Abilene S=10 [8,11] | 18.6121 bd 19.06 | 18.400 bd 19.16 | 18.044 bd 25.98 |
실패 0. S=10 병목 = leader 위험 평가 (max_α max_{q∈𝒬} qᵀQ(α)): q와 α가 같은 편(max–max) → CCG relaxation 불가.
Gurobi: 실험은 13.0.0 (Gurobi.jl 1.9.2 내장 artifact). 시스템엔 13.0.1 설치 (C:\gurobi1301) — 전환 안 함.
