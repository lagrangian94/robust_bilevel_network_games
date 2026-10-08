"""
true_dro_data.jl — True-DRO-Exact 데이터 준비.

TVData와 유사하지만 True-DRO formulation (Lagrangian decomposition,
true_dro_v5.md)에 맞춘 별도 struct.
"""

using LinearAlgebra

"""
    TrueDROData

True-DRO-Exact (true_dro_v5.md)에 필요한 데이터.

# Fields
- `Ny::Matrix{Float64}`: node-arc incidence (source-removed), (|V|-1) × |A|
- `Nts::Vector{Float64}`: dummy arc column, (|V|-1)
- `nv1::Int`: |V| - 1
- `num_arcs::Int`: |A| (dummy arc 제외)
- `S::Int`: 시나리오 수
- `xi_bar::Matrix{Float64}`: |A| × S, 시나리오별 capacity
- `q_hat::Vector{Float64}`: nominal probability, length S
- `eps_hat::Float64`: leader TV radius ε̂
- `eps_tilde::Float64`: follower TV radius ε̃
- `v::Matrix{Float64}`: interdiction effectiveness, |A| × S (scenario-dependent)
- `gamma::Int`: interdiction budget
- `w::Float64`: recovery budget weight
- `lambda_U::Float64`: λ upper bound (McCormick big-M for ψ⁰)
- `interdictable_arcs::Vector{Bool}`
- `phi_hat_U::Float64`: McCormick big-M for φ̂ (leader)
- `phi_tilde_U::Float64`: McCormick big-M for φ̃ (follower)
"""
struct TrueDROData
    Ny::Matrix{Float64}
    Nts::Vector{Float64}
    nv1::Int
    num_arcs::Int
    S::Int
    xi_bar::Matrix{Float64}    # |A| × S
    q_hat::Vector{Float64}     # S
    eps_hat::Float64
    eps_tilde::Float64
    v::Matrix{Float64}         # |A| × S
    gamma::Int
    w::Float64
    lambda_U::Float64
    interdictable_arcs::Vector{Bool}
    phi_hat_U::Float64
    phi_tilde_U::Float64
    beta::Union{Nothing, Float64}  # CVaR risk level β ∈ [0,1). nothing → 기존 expectation (r 없음). 0.0 → CVaR with r=a.
end


"""
    make_true_dro_data(network, scenarios, q_hat, eps_hat, eps_tilde;
                       w=1.0, lambda_U=10.0, gamma=2)

GridNetworkData + scenario 데이터로부터 TrueDROData 생성.
"""
function make_true_dro_data(network, scenarios, q_hat, eps_hat, eps_tilde;
                            w=1.0, lambda_U=10.0, gamma=2,
                            v_scenarios::Union{Matrix{Float64}, Nothing}=nothing,
                            beta::Union{Nothing, Float64}=nothing)
    num_arcs = length(network.arcs) - 1  # dummy arc 제외
    N_trunc = network.N
    Ny = N_trunc[:, 1:num_arcs]
    Nts = N_trunc[:, end]
    nv1 = size(Ny, 1)

    if scenarios isa Vector
        S = length(scenarios)
        xi_bar = hcat(scenarios...)
    else
        xi_bar = scenarios
        S = size(xi_bar, 2)
    end

    if size(xi_bar, 1) == num_arcs + 1
        xi_bar = xi_bar[1:num_arcs, :]
    end

    @assert size(xi_bar, 1) == num_arcs "xi_bar rows ($(size(xi_bar,1))) != num_arcs ($num_arcs)"
    @assert length(q_hat) == S "q_hat length ($(length(q_hat))) != S ($S)"
    @assert abs(sum(q_hat) - 1.0) < 1e-10 "q_hat must sum to 1"
    @assert 0 <= eps_hat <= 1 "eps_hat must be in [0,1]"
    @assert 0 <= eps_tilde <= 1 "eps_tilde must be in [0,1]"
    if beta !== nothing
        @assert 0 <= beta < 1 "beta must be in [0,1)"
    end

    if v_scenarios !== nothing
        @assert size(v_scenarios) == (num_arcs, S) "v_scenarios size ($(size(v_scenarios))) != ($num_arcs, $S)"
        v = v_scenarios
    else
        v = repeat(Float64.(network.interdictable_arcs[1:num_arcs]), 1, S)
    end

    # McCormick big-M
    # Leader: N_tsᵀ π̂ ≥ 1 → φ̂_k 가 1로 bound (loose)
    # Follower: N_tsᵀ π̃ ≥ λ → φ̃_k 가 λ_U로 bound
    phi_hat_U = 1.0
    phi_tilde_U = lambda_U

    return TrueDROData(Ny, Nts, nv1, num_arcs, S, xi_bar, q_hat,
                       eps_hat, eps_tilde, v, gamma, w, lambda_U,
                       network.interdictable_arcs, phi_hat_U, phi_tilde_U, beta)
end


# =====================================================================
# 종속 ambiguity set (belief 결합) 𝒟_δ = {(p, p̃) ∈ D̂ × D̃ : d_TV(p, p̃) ≤ δ} 공통 도우미
# (docs/dependent_ambiguity/). 결합 반경 δ 는 TrueDROData 에 넣지 않고 함수 키워드 delta_couple 로 넘긴다
# (Inf = 기존 rectangular).
# =====================================================================
"반경만 바꾼 사본"
td_with_eps(td::TrueDROData, eps_hat, eps_tilde) =
    TrueDROData(td.Ny, td.Nts, td.nv1, td.num_arcs, td.S, td.xi_bar, td.q_hat, Float64(eps_hat), Float64(eps_tilde),
                td.v, td.gamma, td.w, td.lambda_U, td.interdictable_arcs, td.phi_hat_U, td.phi_tilde_U, td.beta)

"""
    reduce_coupling(td, δ) → (td′, δ′)

결합이 곱집합으로 바뀌는 경우를 미리 정리해 결합 행 없는 (δ′ = Inf) 문제로 바꾼다 (Lemma geom, 노트 §1).
  δ ≥ ε̂ + ε̃ : 결합 비활성 (삼각부등식)        → (td, Inf)
  ε̂ = 0      : p = q̂ → p̃ 의 반경 min(ε̃, δ)    → (ε̃ ← min(ε̃, δ), Inf)
  ε̃ = 0      : p̃ = q̂ → p 의 반경 min(ε̂, δ)    → (ε̂ ← min(ε̂, δ), Inf)
그 외는 그대로 (td, δ). nominal / single-layer compact 빌더가 결합을 모르므로 Benders 진입 시 반드시 거친다.
"""
function reduce_coupling(td::TrueDROData, δ::Real)
    isfinite(δ) || return td, Inf
    δ >= td.eps_hat + td.eps_tilde && return td, Inf
    td.eps_hat == 0.0 && return td_with_eps(td, 0.0, min(td.eps_tilde, δ)), Inf
    td.eps_tilde == 0.0 && return td_with_eps(td, min(td.eps_hat, δ), 0.0), Inf
    return td, Float64(δ)
end

"""
(a, d) 를 결합 다면체 안으로: 음수 제거·합 1 후, ‖a−q̂‖₁ ≤ 2ε̂, ‖d−q̂‖₁ ≤ 2ε̃, ‖a−d‖₁ ≤ 2δ 를 모두 만족하도록
둘을 q̂ 쪽으로 같은 비율 t 만큼 줄인다 (세 거리가 모두 (1−t) 배가 됨). δ = Inf 면 각자 따로 줄인다 (기존 규칙).
"""
function couple_clean(a, d, q, ε̂, ε̃, δ)
    simp(y) = (y = max.(y, 0.0); y ./ sum(y))
    a = simp(a); d = simp(d)
    ratio(dev, ε) = dev > 2ε ? 2ε / dev : 1.0
    ta = ratio(sum(abs.(a .- q)), ε̂); td_ = ratio(sum(abs.(d .- q)), ε̃)
    if isfinite(δ)
        t = min(ta, td_, ratio(sum(abs.(a .- d)), δ)); ta = td_ = t
    end
    return q .+ (a .- q) .* ta, q .+ (d .- q) .* td_
end

"""CVaR 재가중 r 을 [0, a/(1−β)] 로 자르고 합 1 로 (부족분은 여유 비율로 배분, 초과는 비례 축소). belief_menu 의 규칙과 같음."""
function clean_r(r, a, β)
    cap = a ./ (1.0 - β)
    r = clamp.(r, 0.0, cap); sr = sum(r)
    if sr > 1.0
        r = r ./ sr
    elseif sr < 1.0
        slack = cap .- r; r = r .+ slack .* ((1.0 - sr) / sum(slack))
    end
    return r
end
