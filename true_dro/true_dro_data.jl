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
- `v::Vector{Float64}`: interdiction effectiveness, |A|
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
    v::Vector{Float64}         # |A|
    gamma::Int
    w::Float64
    lambda_U::Float64
    interdictable_arcs::Vector{Bool}
    phi_hat_U::Float64
    phi_tilde_U::Float64
end


"""
    make_true_dro_data(network, scenarios, q_hat, eps_hat, eps_tilde;
                       w=1.0, lambda_U=10.0, gamma=2)

GridNetworkData + scenario 데이터로부터 TrueDROData 생성.
"""
function make_true_dro_data(network, scenarios, q_hat, eps_hat, eps_tilde;
                            w=1.0, lambda_U=10.0, gamma=2)
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

    v = Float64.(network.interdictable_arcs)

    # McCormick big-M
    # Leader: N_tsᵀ π̂ ≥ 1 → φ̂_k 가 1로 bound (loose)
    # Follower: N_tsᵀ π̃ ≥ λ → φ̃_k 가 λ_U로 bound
    phi_hat_U = 1.0
    phi_tilde_U = lambda_U

    return TrueDROData(Ny, Nts, nv1, num_arcs, S, xi_bar, q_hat,
                       eps_hat, eps_tilde, v, gamma, w, lambda_U,
                       network.interdictable_arcs, phi_hat_U, phi_tilde_U)
end
