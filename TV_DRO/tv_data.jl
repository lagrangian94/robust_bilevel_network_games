"""
tv_data.jl — TV-DRO 데이터 준비 모듈.

TVData struct: network에서 TV-DRO에 필요한 데이터 추출.
GridNetworkData는 parent에서 재사용.
"""

using LinearAlgebra

"""
    TVData

TV-DRO에 필요한 모든 데이터를 담는 구조체.

# Fields
- `Ny::Matrix{Float64}`: node-arc incidence (source-removed), (|V|-1) × |A|
- `Nts::Vector{Float64}`: dummy arc column, (|V|-1)
- `nv1::Int`: |V| - 1
- `num_arcs::Int`: |A| (dummy arc 제외)
- `S::Int`: 시나리오 수
- `xi_bar::Matrix{Float64}`: |A| × S, 시나리오별 capacity
- `q_hat::Vector{Float64}`: nominal probability, length S
- `eps_hat::Float64`: leader TV radius
- `eps_tilde::Float64`: follower TV radius
- `v::Vector{Float64}`: interdiction effectiveness, |A|
- `gamma::Int`: interdiction budget
- `w::Float64`: recovery budget weight
- `lambda_U::Float64`: upper bound on λ
- `interdictable_arcs::Vector{Bool}`: which arcs can be interdicted
- `phi_U::Float64`: McCormick big-M for φ̂, φ̃ (= max_s Σ_k ξ̄_k^s + 1)
"""
struct TVData
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
    phi_U::Float64
end


"""
    make_tv_data(network, scenarios, q_hat, eps_hat, eps_tilde;
                 w=1.0, lambda_U=10.0, gamma=2)

GridNetworkData + scenario 데이터로부터 TVData 생성.

# Arguments
- `network`: GridNetworkData
- `scenarios`: Vector of Vector{Float64} (length S, each |A|) or Matrix |A|×S
- `q_hat`: nominal probabilities, length S
- `eps_hat`, `eps_tilde`: TV radii for leader/follower
"""
function make_tv_data(network, scenarios, q_hat, eps_hat, eps_tilde;
                      w=1.0, lambda_U=10.0, gamma=2)
    num_arcs = length(network.arcs) - 1  # dummy arc 제외
    N_trunc = network.N  # already source-removed: (|V|-1) × (|A|+1)
    Ny = N_trunc[:, 1:num_arcs]
    Nts = N_trunc[:, end]
    nv1 = size(Ny, 1)

    # scenarios를 Matrix로 변환
    if scenarios isa Vector
        S = length(scenarios)
        xi_bar = hcat(scenarios...)  # rows × S
    else
        xi_bar = scenarios
        S = size(xi_bar, 2)
    end

    # capacity_scenarios는 (num_arcs+1) × S (dummy arc 포함) 또는 num_arcs × S
    if size(xi_bar, 1) == num_arcs + 1
        # dummy arc (last row) 제거 → regular arcs만
        xi_bar = xi_bar[1:num_arcs, :]
    end

    @assert size(xi_bar, 1) == num_arcs "xi_bar rows ($( size(xi_bar,1))) != num_arcs ($num_arcs)"
    @assert length(q_hat) == S "q_hat length ($(length(q_hat))) != S ($S)"
    @assert abs(sum(q_hat) - 1.0) < 1e-10 "q_hat must sum to 1"
    @assert 0 <= eps_hat <= 1 "eps_hat must be in [0,1]"
    @assert 0 <= eps_tilde <= 1 "eps_tilde must be in [0,1]"

    v = Float64.(network.interdictable_arcs)  # v_k ∈ {0,1}

    # McCormick big-M for φ̂, φ̃ dual variables (max-flow dual UB = 1.0)
    phi_U = 2.0

    return TVData(Ny, Nts, nv1, num_arcs, S, xi_bar, q_hat,
                  eps_hat, eps_tilde, v, gamma, w, lambda_U,
                  network.interdictable_arcs, phi_U)
end
