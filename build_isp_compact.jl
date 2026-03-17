"""
build_isp_compact.jl

Dictionary-indexed compact LDR 버전의 ISP (Inner SubProblem) 빌더.

## 핵심 아이디어
비인접 arc pair의 LDR 변수를 **아예 생성하지 않는다**.
JuMP의 dictionary-indexed variable 기능을 활용하여, arc_adjacency/node_arc_incidence
행렬에서 true인 위치에 대해서만 변수를 만든다.

## fix() 접근법과의 차이
- fix(): 변수를 전부 생성한 후 0으로 고정 → JuMP 레벨 변수 수 동일, 성능 이득 없음
- dict-indexed: 변수 자체가 존재하지 않음 → 실질적 변수 수 감소, 메모리 절약

## Compact vs Dense 변수 구분
- **Compact (dict-indexed)**: U, P_Φ, P_Π, P_Y — arc adjacency/node incidence 기반 희소성
  - `@variable(model, Uhat1[(i,j) in afp] >= 0)` 형태
- **Dense (기존과 동일)**: M (PSD cone), Z, β, Γ — 블록 구조이므로 희소화 불가
  - `@variable(model, Mhat[s=1:S, 1:na1, 1:na1])` 형태
- **Dense (Yts)**: Ptilde_Yts — 1D per scenario, arc-pair 희소성 없음

## ISP의 S=1 특성
ISP는 initialize_isp에서 시나리오별로 1개씩 생성되므로 항상 S=1.
따라서 compact 변수에 s 인덱스가 불필요하다.

## 제공 함수
- build_isp_leader_compact(): ISP leader 모델 (hat 변수)
- build_isp_follower_compact(): ISP follower 모델 (tilde 변수)
- isp_leader_optimize_compact!(): compact objective 업데이트 + 풀이
- isp_follower_optimize_compact!(): compact objective 업데이트 + 풀이
- evaluate_master_opt_cut_compact(): compact → dense 변환 후 outer cut 구성
- initialize_isp_compact(): 모든 시나리오에 대한 compact ISP 인스턴스 생성

## 사용법 (compare_compact.jl에서)
    @eval Main initialize_isp = \$initialize_isp_compact
    @eval Main isp_leader_optimize! = \$isp_leader_optimize_compact!
    @eval Main isp_follower_optimize! = \$isp_follower_optimize_compact!
    @eval Main evaluate_master_opt_cut = \$evaluate_master_opt_cut_compact
"""

using JuMP
using LinearAlgebra
using SparseArrays
using Infiltrator
using Gurobi
using Mosek, MosekTools

include("network_generator.jl")
include("compact_ldr_utils.jl")
using .NetworkGenerator


# =============================================================================
# 1. build_isp_leader_compact
#    ISP Leader 문제 (hat 변수): 각 시나리오 s에 대한 inner subproblem의 leader 부분.
#    원본 nested_benders_trust_region.jl의 build_isp_leader()와 동일한 수학적 구조이나,
#    LDR 관련 dual 변수 (U, P)를 dictionary-indexed로 생성하여 변수 수를 줄인다.
# =============================================================================
function build_isp_leader_compact(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, true_S; πU=ϕU)
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs) - 1  # 마지막 arc는 dummy (t→s)
    N = network.N  # node-arc incidence 행렬
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    E = ones(num_arcs, num_arcs+1)
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]  # [I | 0] 행렬
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0  # d0 = [0,...,0,1] — intercept 추출용 단위 벡터
    na1 = num_arcs + 1  # intercept 열 인덱스 (num_arcs+1)

    # --- Compact 인덱스 셋 생성 ---
    # aap: slope 열의 인접 arc 쌍 (Φ_L, Ψ_L 제약에 사용)
    # afp: aap + intercept 열 (U, P_Φ 변수 인덱싱)
    # nap: slope 열의 incident node-arc 쌍 (Π_L 제약에 사용)
    # nfp: nap + intercept 열 (P_Π 변수 인덱싱)
    aap = arc_adj_pairs(network)
    afp = arc_full_pairs(network)
    nap = node_arc_inc_pairs(network)
    nfp = node_arc_full_pairs(network)

    println("Building ISP leader (COMPACT dict-indexed)...")
    println("  Arcs: $num_arcs, afp: $(length(afp)), nfp: $(length(nfp))")

    # =========================================================================
    # 결정 변수 (DECISION VARIABLES)
    # =========================================================================
    # OMP에서 전달받은 1단계 해 (ISP에서는 파라미터로 취급)
    λ, x, h, ψ0 = λ_sol, x_sol, h_sol, ψ0_sol
    α = α_sol  # IMP에서 전달받은 inner master 결정 변수

    # --- Dense 변수 (원본과 동일 구조, 축소 불가) ---
    # βhat1: Λ̂₁ dual의 β 부분, 3개 블록으로 분할
    #   블록1 (na1): Φ̂_0 제약 dual
    #   블록2 (num_nodes-1): Π̂_0 제약 dual
    #   블록3 (num_arcs): Ψ̂_0 제약 dual
    dim_Λhat1_rows = (num_arcs + 1) + (num_nodes - 1) + num_arcs
    dim_Λhat2_rows = num_arcs  # βhat2: μ̂ 제약 dual
    @variable(model, βhat1[s=1:S, 1:dim_Λhat1_rows] >= 0)
    @variable(model, βhat2[s=1:S, 1:dim_Λhat2_rows] >= 0)
    βhat1_1 = βhat1[:, 1:num_arcs+1]  # Φ̂_0 블록
    block2_start = num_arcs + 2
    block3_start = block2_start + num_nodes - 1
    βhat1_2 = βhat1[:, block2_start:block3_start-1]  # Π̂_0 블록
    βhat1_3 = βhat1[:, block3_start:end]  # Ψ̂_0 블록
    @assert sum([size(βhat1_1,2), size(βhat1_2,2), size(βhat1_3,2)]) == dim_Λhat1_rows

    # Mhat: S-lemma에 의한 SDP 행렬 (PSD cone), (na1 × na1) 차원 축소 불가
    @variable(model, Mhat[s=1:S, 1:na1, 1:na1])
    # Zhat: Λ̂ dual의 Z 부분 (불확실성 집합 R*ξ ≥ r 관련)
    dim_R_cols = size(R[1], 2)
    @variable(model, Zhat1[s=1:S, 1:dim_Λhat1_rows, 1:dim_R_cols])
    @variable(model, Zhat2[s=1:S, 1:dim_Λhat2_rows, 1:dim_R_cols])
    Zhat1_1 = Zhat1[:, 1:na1, :]  # Φ̂_L 블록
    block2_start = num_arcs + 2
    block3_start = block2_start + num_nodes - 1
    Zhat1_2 = Zhat1[:, block2_start:block3_start-1, :]  # Π̂_L 블록
    Zhat1_3 = Zhat1[:, block3_start:end, :]  # Ψ̂_L 블록
    @assert sum([size(Zhat1_1,2), size(Zhat1_2,2), size(Zhat1_3,2)]) == dim_Λhat1_rows

    # Γhat: SOC (second-order cone) 변수, ||Zhat_i|| ≤ Γhat_i 형태
    @variable(model, Γhat1[s=1:S, 1:dim_Λhat1_rows, 1:size(R[1],1)])
    @variable(model, Γhat2[s=1:S, 1:dim_Λhat2_rows, 1:size(R[1],1)])

    # --- Compact 변수 (dictionary-indexed) ---
    # 인접 arc pair에 대해서만 변수 생성. 비인접 (i,j)는 변수 자체가 없음.
    # Uhat1~3: Φ̂_L, Ψ̂_L dual (Big-M reformulation의 dual)
    #   Uhat1: Ψ̂ ≤ ϕU*x 의 dual
    #   Uhat2: Ψ̂ - Φ̂ ≤ 0 의 dual
    #   Uhat3: Φ̂ - Ψ̂ ≤ ϕU*(1-x) 의 dual
    @variable(model, Uhat1[(i,j) in afp] >= 0)
    @variable(model, Uhat2[(i,j) in afp] >= 0)
    @variable(model, Uhat3[(i,j) in afp] >= 0)
    # Phat: LDR 희소성 제약의 dual (P⁺, P⁻로 분해된 자유 변수)
    @variable(model, Phat1_Φ[(i,j) in afp] >= 0)   # Φ̂ 제약의 P⁺
    @variable(model, Phat2_Φ[(i,j) in afp] >= 0)   # Φ̂ 제약의 P⁻
    @variable(model, Phat1_Π[(i,j) in nfp] >= 0)   # Π̂ 제약의 P⁺ (node-arc incidence 기반)
    @variable(model, Phat2_Π[(i,j) in nfp] >= 0)   # Π̂ 제약의 P⁻

    # =========================================================================
    # 목적함수 (OBJECTIVE FUNCTION)
    # =========================================================================
    # 원본에서 sum(Uhat1 .* diag_x_E)를 compact 인덱스로 전개:
    #   diag_x_E[i,j] = x[i] (모든 j에 대해 동일) → Uhat1[(i,j)] * x[i]
    obj_term1 = -ϕU * sum(Uhat1[(i,j)] * x[i] for (i,j) in afp)
    obj_term2 = -ϕU * sum(Uhat3[(i,j)] * (1.0 - x[i]) for (i,j) in afp)
    # d0 = [0,...,0,1]: βhat1_1의 intercept 열만 추출
    obj_term3 = d0' * βhat1_1[1, :]
    # P 변수들의 합: LDR 제약의 dual bound 기여
    obj_ub = -ϕU * sum(Phat1_Φ[p] for p in afp) - πU * sum(Phat1_Π[p] for p in nfp)
    obj_lb = -ϕU * sum(Phat2_Φ[p] for p in afp) - πU * sum(Phat2_Π[p] for p in nfp)
    @objective(model, Max, obj_term1 + obj_term2 + obj_term3 + obj_ub + obj_lb)

    # intercept: x에 의존하지 않는 항 (outer cut 구성 시 사용)
    intercept = @expression(model, intercept, obj_term3 + obj_ub + obj_lb)

    # =========================================================================
    # 제약조건 (CONSTRAINTS)
    # =========================================================================
    # --- Conic 제약 (dense, 원본과 동일) ---
    @constraint(model, [s=1:S], Mhat[s,:,:] in PSDCone())  # S-lemma PSD
    @constraint(model, [s=1:S, i=1:dim_Λhat1_rows], Γhat1[s, i, :] in SecondOrderCone())  # SOC
    @constraint(model, [s=1:S, i=1:dim_Λhat2_rows], Γhat2[s, i, :] in SecondOrderCone())  # SOC

    # M̂의 (na1,na1) 원소: dual constant 제약 (확률 합 ≤ 1/S)
    @constraint(model, cons_dual_constant[s=1:S], Mhat[s, na1, na1] <= 1/true_S)
    # Trace 제약: tr(M̂₁₁) - M̂₂₂ * ε² ≤ 0 (불확실성 집합 크기 제약)
    @constraint(model, [s=1:S], tr(Mhat[s, 1:num_arcs, 1:num_arcs]) - Mhat[s, end, end] * (epsilon^2) <= 0)

    for s in 1:S
        D_s = diagm(xi_bar[s])  # D_s = diag(ξ̄_s): 명목 용량의 대각행렬
        Mhat_11 = Mhat[s, 1:num_arcs, 1:num_arcs]  # SDP 블록 (1,1)
        Mhat_12 = Mhat[s, 1:num_arcs, end]          # SDP 블록 (1,2) = Mhat의 마지막 열
        Mhat_22 = Mhat[s, end, end]                  # SDP 블록 (2,2) = 스칼라

        # --- Φ̂ 제약 (flow dual LDR) ---
        # Adj_L = -D_s * M̂₁₁ - M̂₁₂ * ξ̄ᵀ: Φ̂_L 열의 행렬 계수 (dense)
        Adj_L = -D_s * Mhat_11 + (-Mhat_12 * adjoint(xi_bar[s]))
        Adj_0 = -D_s * Mhat_12 + (-xi_bar[s] * Mhat_22)

        # Φ̂_L 제약: 인접 arc pair에 대해서만 (비인접은 변수 자체가 없으므로 제약 불필요)
        for (i,j) in aap
            @constraint(model, Adj_L[i,j] + Uhat2[(i,j)] - Uhat3[(i,j)] == 0)
        end
        # Φ̂_0 제약: intercept 열 (모든 arc i에 대해)
        dense_Φ0 = Adj_0 + I_0 * βhat1_1[s,:] + βhat1_3[s,:] - βhat2[s,:]
        for i in 1:num_arcs
            @constraint(model, dense_Φ0[i] + Uhat2[(i,na1)] - Uhat3[(i,na1)] + Phat1_Φ[(i,na1)] - Phat2_Φ[(i,na1)] == 0)
        end

        # --- Ψ̂ 제약 (interdiction LDR, McCormick dual) ---
        Adj_L_Ψ = v * D_s * Mhat_11 + v * (Mhat_12 * adjoint(xi_bar[s]))
        Adj_0_Ψ = v * D_s * Mhat_12 + v * xi_bar[s] * Mhat_22

        # Ψ̂_L 제약: 인접 arc pair에 대해서만
        for (i,j) in aap
            @constraint(model, Adj_L_Ψ[i,j] - Uhat1[(i,j)] - Uhat2[(i,j)] + Uhat3[(i,j)] <= 0)
        end
        # Ψ̂_0 제약: intercept 열
        for i in 1:num_arcs
            @constraint(model, Adj_0_Ψ[i] - Uhat1[(i,na1)] - Uhat2[(i,na1)] + Uhat3[(i,na1)] <= 0.0)
        end
    end

    # --- μ̂ 제약 (coupling: IMP의 α와 연결) ---
    # shadow_price로 cut 추출 시 사용되는 핵심 제약
    @constraint(model, coupling_cons[s=1:S, k=1:num_arcs], βhat2[s,k] <= α[k])

    # --- Π̂ 제약 (node price LDR) ---
    # Π̂_L 제약: incident (node, arc) pair에 대해서만
    for s in 1:S
        NZ = -N * Zhat1_1[s,:,:]  # N * Z 행렬곱 (dense)
        for (i,j) in nap
            @constraint(model, NZ[i,j] - Zhat1_2[s,i,j] + Phat1_Π[(i,j)] - Phat2_Π[(i,j)] == 0.0)
        end
    end
    # Π̂_0 제약: intercept 열
    for s in 1:S
        Nβ = N * βhat1_1[s,:] + βhat1_2[s,:]
        for i in 1:num_nodes-1
            @constraint(model, Nβ[i] + Phat1_Π[(i,na1)] - Phat2_Π[(i,na1)] == 0)
        end
    end

    # --- Λ̂ 제약 (불확실성 집합 dual: Z*R' + β*r' + Γ = 0) ---
    @constraint(model, [s=1:S], Zhat1[s,:,:] * R[s]' + βhat1[s,:] * r_dict[s]' + Γhat1[s,:,:] .== 0.0)
    @constraint(model, [s=1:S], Zhat2[s,:,:] * R[s]' + βhat2[s,:] * r_dict[s]' + Γhat2[s,:,:] .== 0.0)

    vars = Dict(
        :Mhat => Mhat,
        :Zhat1 => Zhat1,
        :Zhat2 => Zhat2,
        :Γhat1 => Γhat1,
        :Γhat2 => Γhat2,
        :Uhat1 => Uhat1,
        :Uhat3 => Uhat3,
        :Phat1_Φ => Phat1_Φ,
        :Phat1_Π => Phat1_Π,
        :Phat2_Φ => Phat2_Φ,
        :Phat2_Π => Phat2_Π,
        :βhat1_1 => βhat1_1,
        :intercept => intercept,
        :arc_full_pairs => afp,
        :node_arc_full_pairs => nfp,
    )

    return model, vars
end


# =============================================================================
# 2. build_isp_follower_compact
#    ISP Follower 문제 (tilde 변수): leader와 동일한 구조에 추가로
#    Y_tilde (flow LDR), Yts_tilde (dummy arc flow LDR) 변수가 있다.
#    Y_tilde는 arc adjacency 기반 compact, Yts_tilde는 1D이므로 dense 유지.
# =============================================================================
function build_isp_follower_compact(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, true_S; πU=ϕU, yU=ϕU, ytsU=ϕU)
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs) - 1
    N = network.N
    N_y = N[:, 1:num_arcs]
    N_ts = N[:, end]
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    E = ones(num_arcs, num_arcs+1)
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    na1 = num_arcs + 1

    # Index sets
    aap = arc_adj_pairs(network)
    afp = arc_full_pairs(network)
    nap = node_arc_inc_pairs(network)
    nfp = node_arc_full_pairs(network)

    println("Building ISP follower (COMPACT dict-indexed)...")
    println("  Arcs: $num_arcs, afp: $(length(afp)), nfp: $(length(nfp))")

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================
    λ, x, h, ψ0 = λ_sol, x_sol, h_sol, ψ0_sol
    α = α_sol

    # --- Dense 변수 (원본과 동일 구조) ---
    # βtilde1: Λ̃₁ dual의 β 부분, 6개 블록으로 분할 (follower는 leader보다 블록이 많음)
    #   블록1 (na1): Φ̃_0
    #   블록2 (num_nodes-1): Π̃_0
    #   블록3 (num_arcs): Ψ̃_0 / Y_0
    #   블록4 (num_nodes-1): Π̃ (follower 추가)
    #   블록5 (num_arcs): Φ̃ (follower 추가)
    #   블록6 (num_arcs): Ỹ (follower 추가)
    dim_Λtilde1_rows = num_arcs+1 + (num_nodes-1) + num_arcs + (num_nodes-1) + num_arcs + num_arcs
    dim_Λtilde2_rows = num_arcs  # βtilde2: μ̃ 제약 dual
    @variable(model, βtilde1[s=1:S, 1:dim_Λtilde1_rows] >= 0)
    @variable(model, βtilde2[s=1:S, 1:dim_Λtilde2_rows] >= 0)

    # 블록 인덱스 계산
    block2_start = num_arcs + 2
    block3_start = block2_start + num_nodes - 1
    block4_start = block3_start + num_arcs
    block5_start = block4_start + num_nodes - 1
    block6_start = block5_start + num_arcs
    βtilde1_1 = βtilde1[:, 1:na1]                           # Φ̃_0 블록
    βtilde1_2 = βtilde1[:, block2_start:block3_start-1]     # Π̃_0 블록
    βtilde1_3 = βtilde1[:, block3_start:block4_start-1]     # Ψ̃_0 / Ỹ_0 블록
    βtilde1_4 = βtilde1[:, block4_start:block5_start-1]     # Π̃ 블록 (follower 추가)
    βtilde1_5 = βtilde1[:, block5_start:block6_start-1]     # Φ̃ 블록 (follower 추가)
    βtilde1_6 = βtilde1[:, block6_start:end]                # Ỹ 블록 (follower 추가)
    @assert sum([size(βtilde1_1,2), size(βtilde1_2,2), size(βtilde1_3,2), size(βtilde1_4,2), size(βtilde1_5,2), size(βtilde1_6,2)]) == dim_Λtilde1_rows

    @variable(model, Mtilde[s=1:S, 1:na1, 1:na1])
    dim_R_cols = size(R[1], 2)
    @variable(model, Ztilde1[s=1:S, 1:dim_Λtilde1_rows, 1:dim_R_cols])
    @variable(model, Ztilde2[s=1:S, 1:dim_Λtilde2_rows, 1:dim_R_cols])

    block2_start = num_arcs + 2
    block3_start = block2_start + num_nodes - 1
    block4_start = block3_start + num_arcs
    block5_start = block4_start + num_nodes - 1
    block6_start = block5_start + num_arcs
    Ztilde1_1 = Ztilde1[:, 1:na1, :]
    Ztilde1_2 = Ztilde1[:, block2_start:block3_start-1, :]
    Ztilde1_3 = Ztilde1[:, block3_start:block4_start-1, :]
    Ztilde1_4 = Ztilde1[:, block4_start:block5_start-1, :]
    Ztilde1_5 = Ztilde1[:, block5_start:block6_start-1, :]
    Ztilde1_6 = Ztilde1[:, block6_start:end, :]
    @assert sum([size(Ztilde1_1,2), size(Ztilde1_2,2), size(Ztilde1_3,2), size(Ztilde1_4,2), size(Ztilde1_5,2), size(Ztilde1_6,2)]) == dim_Λtilde1_rows

    @variable(model, Γtilde1[s=1:S, 1:dim_Λtilde1_rows, 1:size(R[1],1)])
    @variable(model, Γtilde2[s=1:S, 1:dim_Λtilde2_rows, 1:size(R[1],1)])

    # Ptilde_Yts: dummy arc flow Ỹts의 dual — 1D per scenario, arc-pair 희소성 없으므로 dense
    @variable(model, Ptilde1_Yts[s=1:S, 1:na1] >= 0)
    @variable(model, Ptilde2_Yts[s=1:S, 1:na1] >= 0)

    # --- Compact 변수 (dictionary-indexed, 인접 쌍에 대해서만 생성) ---
    # U: Big-M dual (leader와 동일 패턴)
    @variable(model, Utilde1[(i,j) in afp] >= 0)
    @variable(model, Utilde2[(i,j) in afp] >= 0)
    @variable(model, Utilde3[(i,j) in afp] >= 0)
    # P_Φ, P_Π: flow/node price LDR 희소성 dual (leader와 동일)
    @variable(model, Ptilde1_Φ[(i,j) in afp] >= 0)
    @variable(model, Ptilde2_Φ[(i,j) in afp] >= 0)
    @variable(model, Ptilde1_Π[(i,j) in nfp] >= 0)
    @variable(model, Ptilde2_Π[(i,j) in nfp] >= 0)
    # P_Y: flow variable LDR 희소성 dual (follower에만 존재, arc adjacency 기반)
    @variable(model, Ptilde1_Y[(i,j) in afp] >= 0)
    @variable(model, Ptilde2_Y[(i,j) in afp] >= 0)

    # =========================================================================
    # 목적함수 (OBJECTIVE FUNCTION)
    # Follower는 leader보다 항이 많다: flow 변수 Y 관련 항 추가
    # =========================================================================
    diag_λ_ψ = Diagonal(λ * ones(num_arcs) - v .* ψ0)  # diag(λ - v*ψ₀)
    # Big-M dual 항 (leader와 동일 패턴, compact 합산)
    obj_term1 = -ϕU * sum(Utilde1[(i,j)] * x[i] for (i,j) in afp)
    obj_term2 = -ϕU * sum(Utilde3[(i,j)] * (1.0 - x[i]) for (i,j) in afp)
    # Follower 고유 항: Z̃₁₃ (flow LDR slope), β̃₁₁ (intercept), β̃₁₃ (capacity)
    obj_term4 = sum(Ztilde1_3[1, :, :] .* (diag_λ_ψ * diagm(xi_bar[1])))
    obj_term5 = (λ * d0') * βtilde1_1[1, :]
    obj_term6 = -(h + diag_λ_ψ * xi_bar[1])' * βtilde1_3[1, :]
    # P 변수들의 합: follower는 P_Y, P_Yts도 포함
    obj_ub = -ϕU * sum(Ptilde1_Φ[p] for p in afp) - πU * sum(Ptilde1_Π[p] for p in nfp) -
              yU * sum(Ptilde1_Y[p] for p in afp) - ytsU * sum(Ptilde1_Yts[1, :])
    obj_lb = -ϕU * sum(Ptilde2_Φ[p] for p in afp) - πU * sum(Ptilde2_Π[p] for p in nfp) -
              yU * sum(Ptilde2_Y[p] for p in afp) - ytsU * sum(Ptilde2_Yts[1, :])
    @objective(model, Max, obj_term1 + obj_term2 + obj_term4 + obj_term5 + obj_term6 + obj_ub + obj_lb)

    # intercept: x에 의존하지 않는 항 (P 합만 포함, follower는 obj_term4~6도 x-독립이지만
    #            원본 코드의 패턴을 따라 P 합만 intercept로 분리)
    intercept = @expression(model, intercept, obj_ub + obj_lb)

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================
    @constraint(model, [s=1:S], Mtilde[s,:,:] in PSDCone())
    @constraint(model, [s=1:S, i=1:dim_Λtilde1_rows], Γtilde1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λtilde2_rows], Γtilde2[s, i, :] in SecondOrderCone())

    @constraint(model, cons_dual_constant_pos[s=1:S], Mtilde[s, na1, na1] <= 1/true_S)
    @constraint(model, cons_dual_constant_neg[s=1:S], -Mtilde[s, na1, na1] <= -1/true_S)
    @constraint(model, [s=1:S], tr(Mtilde[s, 1:num_arcs, 1:num_arcs]) - Mtilde[s, end, end] * (epsilon^2) <= 0)

    for s in 1:S
        D_s = diagm(xi_bar[s])
        Mtilde_11 = Mtilde[s, 1:num_arcs, 1:num_arcs]
        Mtilde_12 = Mtilde[s, 1:num_arcs, end]
        Mtilde_22 = Mtilde[s, end, end]

        # --- From Φtilde ---
        Adj_L = -D_s * Mtilde_11 + (-Mtilde_12 * adjoint(xi_bar[s]))
        Adj_0 = -D_s * Mtilde_12 + (-xi_bar[s] * Mtilde_22)

        # Φtilde_L
        for (i,j) in aap
            @constraint(model, Adj_L[i,j] + Utilde2[(i,j)] - Utilde3[(i,j)] == 0)
        end
        # Φtilde_0
        dense_Φ0 = Adj_0 + I_0 * βtilde1_1[s,:] + βtilde1_5[s,:] - βtilde2[s,:]
        for i in 1:num_arcs
            @constraint(model, dense_Φ0[i] + Utilde2[(i,na1)] - Utilde3[(i,na1)] + Ptilde1_Φ[(i,na1)] - Ptilde2_Φ[(i,na1)] == 0)
        end

        # --- From Ψtilde ---
        Adj_L_Ψ = v * D_s * Mtilde_11 + v * (Mtilde_12 * adjoint(xi_bar[s]))
        Adj_0_Ψ = v * D_s * Mtilde_12 + v * xi_bar[s] * Mtilde_22

        # Ψtilde_L
        for (i,j) in aap
            @constraint(model, Adj_L_Ψ[i,j] - Utilde1[(i,j)] - Utilde2[(i,j)] + Utilde3[(i,j)] <= 0.0)
        end
        # Ψtilde_0
        for i in 1:num_arcs
            @constraint(model, Adj_0_Ψ[i] - Utilde1[(i,na1)] - Utilde2[(i,na1)] + Utilde3[(i,na1)] <= 0.0)
        end

        # --- Ỹts 제약 (dummy arc flow LDR) ---
        # Yts는 스칼라(1 × na1)이므로 arc-pair 희소성이 없어 dense로 유지
        Adj_L_ts = Mtilde_12
        Adj_0_ts = Mtilde_22
        # Ỹts_L: slope 열 (dense 벡터)
        @constraint(model, adjoint(Adj_L_ts) + N_ts' * Ztilde1_2[s,:,:] + Ptilde1_Yts[s, 1:num_arcs]' - Ptilde2_Yts[s, 1:num_arcs]' .== 0)
        # Ỹts_0: intercept 열 (스칼라)
        @constraint(model, Adj_0_ts - N_ts' * βtilde1_2[s,:] + Ptilde1_Yts[s, end]' - Ptilde2_Yts[s, end]' .== 0)
    end

    # --- From μtilde ---
    @constraint(model, coupling_cons[s=1:S, k=1:num_arcs], βtilde2[s,k] <= α[k])

    # --- From Πtilde ---
    # Πtilde_L: incident pairs only
    for s in 1:S
        NZ = -N * Ztilde1_1[s,:,:]
        for (i,j) in nap
            @constraint(model, NZ[i,j] - Ztilde1_4[s,i,j] + Ptilde1_Π[(i,j)] - Ptilde2_Π[(i,j)] == 0.0)
        end
    end
    # Πtilde_0: intercept column
    for s in 1:S
        Nβ = N * βtilde1_1[s,:] + βtilde1_4[s,:]
        for i in 1:num_nodes-1
            @constraint(model, Nβ[i] + Ptilde1_Π[(i,na1)] - Ptilde2_Π[(i,na1)] == 0)
        end
    end

    # --- Ỹ 제약 (flow variable LDR, follower 고유) ---
    # Ỹ_L: 인접 arc pair에 대해서만 (flow의 arc adjacency 희소성)
    for s in 1:S
        NY_Z = N_y' * Ztilde1_2[s,:,:]  # N_y' * Z (dense 행렬곱)
        for (i,j) in aap
            @constraint(model, NY_Z[i,j] + Ztilde1_3[s,i,j] - Ztilde1_6[s,i,j] + Ptilde1_Y[(i,j)] - Ptilde2_Y[(i,j)] == 0.0)
        end
    end
    # Ỹ_0: intercept 열 (모든 arc i에 대해)
    for s in 1:S
        yts_0 = -N_y' * βtilde1_2[s,:] - βtilde1_3[s,:] + βtilde1_6[s,:]
        for i in 1:num_arcs
            @constraint(model, yts_0[i] + Ptilde1_Y[(i,na1)] - Ptilde2_Y[(i,na1)] == 0)
        end
    end

    # --- From Λtilde1, Λtilde2 ---
    @constraint(model, [s=1:S], Ztilde1[s,:,:] * R[s]' + βtilde1[s,:] * r_dict[s]' + Γtilde1[s,:,:] .== 0.0)
    @constraint(model, [s=1:S], Ztilde2[s,:,:] * R[s]' + βtilde2[s,:] * r_dict[s]' + Γtilde2[s,:,:] .== 0.0)

    vars = Dict(
        :Mtilde => Mtilde,
        :Ztilde1 => Ztilde1,
        :Ztilde2 => Ztilde2,
        :Γtilde1 => Γtilde1,
        :Γtilde2 => Γtilde2,
        :Utilde1 => Utilde1,
        :Utilde3 => Utilde3,
        :Ptilde1_Φ => Ptilde1_Φ,
        :Ptilde1_Π => Ptilde1_Π,
        :Ptilde2_Φ => Ptilde2_Φ,
        :Ptilde2_Π => Ptilde2_Π,
        :Ptilde1_Y => Ptilde1_Y,
        :Ptilde1_Yts => Ptilde1_Yts,
        :Ptilde2_Y => Ptilde2_Y,
        :Ptilde2_Yts => Ptilde2_Yts,
        :βtilde1_1 => βtilde1_1,
        :βtilde1_3 => βtilde1_3,
        :Ztilde1_3 => Ztilde1_3,
        :intercept => intercept,
        :arc_full_pairs => afp,
        :node_arc_full_pairs => nfp,
    )

    return model, vars
end


# =============================================================================
# 3. isp_leader_optimize_compact!
#    IMP의 각 iteration에서 호출되어 x_sol, α_sol이 갱신될 때마다
#    목적함수를 업데이트하고 재풀이한다.
#    원본 isp_leader_optimize!와 동일한 로직이나, compact 변수에 맞는 합산 문법 사용.
#    반환값: (:OptimalityCut, cut_coeff) — μ̂ (subgradient), η̂ (intercept)
# =============================================================================
function isp_leader_optimize_compact!(isp_leader_model::Model, isp_leader_vars::Dict; isp_data=nothing, uncertainty_set::Dict=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    model, vars = isp_leader_model, isp_leader_vars
    E, ϕU, d0 = isp_data[:E], isp_data[:ϕU], isp_data[:d0]
    πU = get(isp_data, :πU, ϕU)
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    true_S = isp_data[:S]  # 전체 시나리오 수 (ISP는 S=1이지만 true_S로 정규화)

    # vars dict에서 compact 인덱스 셋 가져오기
    afp = vars[:arc_full_pairs]
    nfp = vars[:node_arc_full_pairs]
    Uhat1 = vars[:Uhat1]
    Uhat3 = vars[:Uhat3]
    Phat1_Φ = vars[:Phat1_Φ]
    Phat2_Φ = vars[:Phat2_Φ]
    Phat1_Π = vars[:Phat1_Π]
    Phat2_Π = vars[:Phat2_Π]
    βhat1_1 = vars[:βhat1_1]

    ## 목적함수 갱신: x_sol이 바뀌므로 obj_term1, obj_term2가 변함
    obj_term1 = -ϕU * sum(Uhat1[(i,j)] * x_sol[i] for (i,j) in afp)
    obj_term2 = -ϕU * sum(Uhat3[(i,j)] * (1.0 - x_sol[i]) for (i,j) in afp)
    obj_term3 = d0' * βhat1_1[1, :]
    obj_ub = -ϕU * sum(Phat1_Φ[p] for p in afp) - πU * sum(Phat1_Π[p] for p in nfp)
    obj_lb = -ϕU * sum(Phat2_Φ[p] for p in afp) - πU * sum(Phat2_Π[p] for p in nfp)
    @objective(model, Max, obj_term1 + obj_term2 + obj_term3 + obj_ub + obj_lb)

    ## coupling 제약 RHS 갱신: α_sol이 바뀌므로 βhat2[s,k] ≤ α[k] 업데이트
    coupling_cons = vec(model[:coupling_cons])
    set_normalized_rhs.(coupling_cons, α_sol)

    ## 풀이
    optimize!(model)
    st = MOI.get(model, MOI.TerminationStatus())
    if (st == MOI.OPTIMAL) || (st == MOI.SLOW_PROGRESS)
        ## Cut 추출: shadow price로 subgradient 계산
        # μ̂ = shadow_price(coupling_cons): α에 대한 subgradient
        # η̂ = shadow_price(cons_dual_constant): intercept 기여
        μhat = shadow_price.(coupling_cons)
        ηhat = shadow_price.(vec(model[:cons_dual_constant]))
        intercept, subgradient = (1/true_S) * sum(ηhat), μhat
        # 검증: dual_obj = intercept + α' * subgradient ≈ primal obj
        dual_obj = intercept + α_sol' * subgradient
        @assert abs(dual_obj - objective_value(model)) < 1e-4
        cut_coeff = Dict(:μhat => μhat, :intercept => intercept, :obj_val => dual_obj)
        return (:OptimalityCut, cut_coeff)
    else
        t_status = termination_status(model)
        @infiltrate
        error("Subproblem is not optimal")
    end
end


# =============================================================================
# 4. isp_follower_optimize_compact!
#    Leader와 동일 패턴. 추가로 η̃_pos, η̃_neg (follower의 dual constant는
#    양방향 제약이므로 pos-neg 분해)을 사용하여 intercept 계산.
# =============================================================================
function isp_follower_optimize_compact!(isp_follower_model::Model, isp_follower_vars::Dict; isp_data=nothing, uncertainty_set::Dict=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    model, vars = isp_follower_model, isp_follower_vars
    E, ϕU, d0 = isp_data[:E], isp_data[:ϕU], isp_data[:d0]
    πU, yU, ytsU = get(isp_data, :πU, ϕU), get(isp_data, :yU, ϕU), get(isp_data, :ytsU, ϕU)
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    num_arcs = length(x_sol)
    diag_λ_ψ = Diagonal(λ_sol * ones(num_arcs) - v .* ψ0_sol)
    true_S = isp_data[:S]

    afp = vars[:arc_full_pairs]
    nfp = vars[:node_arc_full_pairs]
    Utilde1 = vars[:Utilde1]
    Utilde3 = vars[:Utilde3]
    Ztilde1_3 = vars[:Ztilde1_3]
    Ptilde1_Φ = vars[:Ptilde1_Φ]
    Ptilde2_Φ = vars[:Ptilde2_Φ]
    Ptilde1_Π = vars[:Ptilde1_Π]
    Ptilde2_Π = vars[:Ptilde2_Π]
    Ptilde1_Y = vars[:Ptilde1_Y]
    Ptilde2_Y = vars[:Ptilde2_Y]
    Ptilde1_Yts = vars[:Ptilde1_Yts]
    Ptilde2_Yts = vars[:Ptilde2_Yts]
    βtilde1_1 = vars[:βtilde1_1]
    βtilde1_3 = vars[:βtilde1_3]

    S = 1
    ## update objective
    obj_term1 = -ϕU * sum(Utilde1[(i,j)] * x_sol[i] for (i,j) in afp)
    obj_term2 = -ϕU * sum(Utilde3[(i,j)] * (1.0 - x_sol[i]) for (i,j) in afp)
    obj_term4 = sum(Ztilde1_3[1, :, :] .* (diag_λ_ψ * diagm(xi_bar[1])))
    obj_term5 = (λ_sol * d0') * βtilde1_1[1, :]
    obj_term6 = -(h_sol + diag_λ_ψ * xi_bar[1])' * βtilde1_3[1, :]
    obj_ub = -ϕU * sum(Ptilde1_Φ[p] for p in afp) - πU * sum(Ptilde1_Π[p] for p in nfp) -
              yU * sum(Ptilde1_Y[p] for p in afp) - ytsU * sum(Ptilde1_Yts[1, :])
    obj_lb = -ϕU * sum(Ptilde2_Φ[p] for p in afp) - πU * sum(Ptilde2_Π[p] for p in nfp) -
              yU * sum(Ptilde2_Y[p] for p in afp) - ytsU * sum(Ptilde2_Yts[1, :])
    @objective(model, Max, obj_term1 + obj_term2 + obj_term4 + obj_term5 + obj_term6 + obj_ub + obj_lb)

    ## update constraints
    coupling_cons = vec(model[:coupling_cons])
    set_normalized_rhs.(coupling_cons, α_sol)

    ## optimize model
    optimize!(model)
    st = MOI.get(model, MOI.TerminationStatus())
    if (st == MOI.OPTIMAL) || (st == MOI.SLOW_PROGRESS)
        ## obtain cuts
        μtilde = shadow_price.(coupling_cons)
        ηtilde_pos = shadow_price.(vec(model[:cons_dual_constant_pos]))
        ηtilde_neg = shadow_price.(vec(model[:cons_dual_constant_neg]))
        intercept = sum((1/true_S) * (ηtilde_pos - ηtilde_neg))
        subgradient = μtilde
        dual_obj = intercept + α_sol' * subgradient
        if abs(dual_obj - objective_value(model)) > 1e-4
            @infiltrate
        end
        cut_coeff = Dict(:μtilde => μtilde, :intercept => intercept, :obj_val => dual_obj)
        return (:OptimalityCut, cut_coeff)
    else
        t_status = termination_status(model)
        @infiltrate
        error("Subproblem is not optimal")
    end
end


# =============================================================================
# 5. evaluate_master_opt_cut_compact
#    Outer loop에서 optimality cut을 구성할 때 호출.
#    α_sol을 고정하고 모든 시나리오의 ISP를 재풀이한 후,
#    dict-indexed 변수값을 dense 3D 배열 (S × num_arcs × num_arcs+1)로 변환.
#
#    이유: outer loop (tr_nested_benders_optimize!)의 cut 구성 코드가
#    dense 행렬 슬라이싱 (.* diag_x_E 등)을 사용하므로, compact → dense 변환이 필수.
#    비인접 항목은 0으로 채워지므로 수학적으로 동일한 cut을 생성한다.
# =============================================================================
function evaluate_master_opt_cut_compact(isp_leader_instances::Dict, isp_follower_instances::Dict, isp_data::Dict, cut_info::Dict, iter::Int; multi_cut_lf=false)
    S = isp_data[:S]
    α_sol = cut_info[:α_sol]
    num_arcs = length(isp_data[:network].arcs) - 1

    # Get index sets from first instance
    afp = isp_leader_instances[1][2][:arc_full_pairs]

    status = true
    for s in 1:S
        model_l = isp_leader_instances[s][1]
        model_f = isp_follower_instances[s][1]
        set_normalized_rhs.(vec(model_l[:coupling_cons]), α_sol)
        optimize!(model_l)
        st_l = MOI.get(model_l, MOI.TerminationStatus())

        set_normalized_rhs.(vec(model_f[:coupling_cons]), α_sol)
        optimize!(model_f)
        st_f = MOI.get(model_f, MOI.TerminationStatus())

        status = status && (st_l == MOI.OPTIMAL) && (st_f == MOI.OPTIMAL)
        if status == false
            if (st_l == MOI.SLOW_PROGRESS) || (st_f == MOI.SLOW_PROGRESS)
                status = true
            else
                @infiltrate
            end
        end
    end

    # --- Compact → Dense 변환 ---
    # dict-indexed U 변수값을 dense 3D 배열로 변환.
    # 비인접 (i,j)는 0으로 유지 → outer cut의 .* diag_x_E 연산과 호환.
    Uhat1 = zeros(S, num_arcs, num_arcs+1)
    Uhat3 = zeros(S, num_arcs, num_arcs+1)
    Utilde1 = zeros(S, num_arcs, num_arcs+1)
    Utilde3 = zeros(S, num_arcs, num_arcs+1)
    for s in 1:S
        # 각 시나리오의 인덱스 셋 (모두 동일하지만 인스턴스별로 접근)
        afp_l = isp_leader_instances[s][2][:arc_full_pairs]
        afp_f = isp_follower_instances[s][2][:arc_full_pairs]

        # Leader의 U 값 추출
        vals_Uhat1 = value.(isp_leader_instances[s][2][:Uhat1])
        vals_Uhat3 = value.(isp_leader_instances[s][2][:Uhat3])
        for (i,j) in afp_l
            Uhat1[s, i, j] = vals_Uhat1[(i,j)]
            Uhat3[s, i, j] = vals_Uhat3[(i,j)]
        end

        # Follower의 U 값 추출
        vals_Utilde1 = value.(isp_follower_instances[s][2][:Utilde1])
        vals_Utilde3 = value.(isp_follower_instances[s][2][:Utilde3])
        for (i,j) in afp_f
            Utilde1[s, i, j] = vals_Utilde1[(i,j)]
            Utilde3[s, i, j] = vals_Utilde3[(i,j)]
        end
    end

    # Dense 변수들 — 원본과 동일하게 cat으로 S차원 결합
    Ztilde1_3 = cat([value.(isp_follower_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
    βtilde1_1 = cat([value.(isp_follower_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
    βtilde1_3 = cat([value.(isp_follower_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)

    intercept_l = [value.(isp_leader_instances[s][2][:intercept]) for s in 1:S]
    intercept_f = [value.(isp_follower_instances[s][2][:intercept]) for s in 1:S]
    intercept = sum(intercept_l) + sum(intercept_f)

    leader_obj = sum(objective_value(isp_leader_instances[s][1]) for s in 1:S)
    follower_obj = sum(objective_value(isp_follower_instances[s][1]) for s in 1:S)
    avg_obj = (leader_obj + follower_obj) / S  # average over scenarios
    @assert abs(avg_obj - cut_info[:obj_val]) < 1e-3 "obj mismatch: avg=$avg_obj, cut_info=$(cut_info[:obj_val])"

    return Dict(
        :Uhat1 => Uhat1, :Utilde1 => Utilde1,
        :Uhat3 => Uhat3, :Utilde3 => Utilde3,
        :Ztilde1_3 => Ztilde1_3,
        :βtilde1_1 => βtilde1_1, :βtilde1_3 => βtilde1_3,
        :intercept => intercept,
        :intercept_l => intercept_l, :intercept_f => intercept_f,
    )
end


# =============================================================================
# 6. initialize_isp_compact
#    모든 시나리오에 대한 compact ISP 인스턴스를 생성한다.
#    각 시나리오 s에 대해 S=1인 ISP leader/follower를 build하여 Dict에 저장.
#    @eval Main으로 원본 initialize_isp를 대체하면 TR nested Benders에서 사용 가능.
# =============================================================================
"""
    initialize_isp_compact(...)

Dictionary-indexed compact ISP 인스턴스를 모든 시나리오에 대해 생성.
원본 initialize_isp()의 drop-in replacement.
compact optimize 함수들과 함께 사용해야 한다.
"""
function initialize_isp_compact(network, S, ϕU, λU, γ, w, v, uncertainty_set; conic_optimizer=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing, πU=ϕU, yU=ϕU, ytsU=ϕU)
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    leader_instances = Dict{Int, Tuple{Model, Dict}}()
    follower_instances = Dict{Int, Tuple{Model, Dict}}()
    for s in 1:S
        U_s = Dict(:R => Dict(1=>R[s]), :r_dict => Dict(1=>r_dict[s]), :xi_bar => Dict(1=>xi_bar[s]), :epsilon => epsilon)
        leader_instances[s] = build_isp_leader_compact(network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, S; πU=πU)
        follower_instances[s] = build_isp_follower_compact(network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, S; πU=πU, yU=yU, ytsU=ytsU)
    end
    return leader_instances, follower_instances
end
