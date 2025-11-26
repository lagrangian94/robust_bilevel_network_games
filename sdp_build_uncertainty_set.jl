"""
build_R_r.jl

연구 노트 1.5절 Robust Counterparts of Linear Uncertainty에서
Box-uncertainty set U_s = {ξ : Rξ ≥ r}의 R과 r을 생성하는 코드

where:
    R = [0 \\ D_s^{-1}]
    r = [-ε \\ D_s^{-1}ξ̂^s]

NOTE: |A|는 regular arcs만 포함 (dummy arc 제외)
"""

using LinearAlgebra
using Infiltrator
"""
    build_R_matrix(num_regular_arcs::Int)


Arguments:
- num_regular_arcs: regular arcs 개수 |A| (dummy arc 제외)
"""
function build_R_matrix(xi_hat::Vector{Float64})
    D = diagm(xi_hat)
    D_inv = inv(D)
    dim = length(xi_hat)
    R = vcat(zeros(1, dim), D_inv)
    return R
end

function build_R_matrix_rsoc(xi_hat::Vector{Float64})
    D = diagm(xi_hat)
    D_inv = inv(D)
    dim = length(xi_hat)
    R = vcat(zeros(2, dim), D_inv)
    return R
end
"""
    build_r_vector(xi_hat::Vector{Float64}, epsilon::Float64)
Arguments:
- xi_hat: capacity scenario ξ̂^s (크기 |A|, dummy arc 제외)
- epsilon: robustness parameter ε
"""
function build_r_vector(num_regular_arcs::Int, xi_hat::Vector{Float64}, epsilon::Float64)
    if size(xi_hat, 1) != num_regular_arcs
        error("xi_hat must have size $num_regular_arcs")
    end
    # D = diagm(xi_hat)
    # D_inv = inv(D)
    r_lower = ones(num_regular_arcs) # D_inv*xi_hat = ones(num_regular_arcs)
    r = vcat(-epsilon, r_lower)
    return r
end

function build_r_vector_rsoc(num_regular_arcs::Int, xi_hat::Vector{Float64}, epsilon::Float64)
    if size(xi_hat, 1) != num_regular_arcs
        error("xi_hat must have size $num_regular_arcs")
    end
    r_lower = ones(num_regular_arcs) # D_inv*xi_hat = ones(num_regular_arcs)
    r = vcat(-1, -(1/2)*epsilon^2, r_lower)
    return r
end
"""
    build_robust_counterpart_matrices(capacity_scenarios::Matrix{Float64}, epsilon::Float64)

모든 scenario에 대한 R과 r 생성

Arguments:
- capacity_scenarios: 각 열이 scenario (크기 |A| × S, dummy arc 제외)
- epsilon: robustness parameter

Returns:
- R: constraint matrix (모든 scenario 공통)
- r_dict: Dict{Int, Vector{Float64}} - 각 scenario의 r vector

Example:
    # network_generator.jl에서 생성한 capacity_scenarios에서 dummy arc(마지막 행) 제거
    capacity_scenarios_full, F, μ = generate_capacity_scenarios(num_arcs, num_scenarios)
    capacity_scenarios_regular = capacity_scenarios_full[1:end-1, :]  # dummy arc 제외
    
    R, r_dict = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
"""
function build_robust_counterpart_matrices(capacity_scenarios::Matrix{Float64}, epsilon::Float64, rsoc::Bool=false)
    num_regular_arcs, num_scenarios = size(capacity_scenarios)
    R = Dict{Int, Matrix{Float64}}()
    r_dict = Dict{Int, Vector{Float64}}()
    xi_bar = Dict{Int, Vector{Float64}}()
    for s in 1:num_scenarios
        if rsoc
            R[s] = build_R_matrix_rsoc(capacity_scenarios[:, s])
            r_dict[s] = build_r_vector_rsoc(num_regular_arcs, capacity_scenarios[:, s], epsilon)
        else
            R[s] = build_R_matrix(capacity_scenarios[:, s])
            r_dict[s] = build_r_vector(num_regular_arcs, capacity_scenarios[:, s], epsilon)
        end
        xi_bar[s] = capacity_scenarios[:,s]
    end
    
    return R, r_dict, xi_bar
end