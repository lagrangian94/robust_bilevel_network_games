"""
build_R_r.jl

연구 노트 1.5절 Robust Counterparts of Linear Uncertainty에서
Box-uncertainty set U_s = {ξ : Rξ ≥ r}의 R과 r을 생성하는 코드

where:
    R = [I; -I; e_{|A|+1}ᵀ; -e_{|A|+1}ᵀ]
    r = [ξ̂^s - εe; -ξ̂^s - εe; 1; -1]

NOTE: |A|는 regular arcs만 포함 (dummy arc 제외)
"""

using LinearAlgebra

"""
    build_R_matrix(num_regular_arcs::Int)

Constraint matrix R ∈ R^{2(|A|+1)+2 × (|A|+1)} 생성

Arguments:
- num_regular_arcs: regular arcs 개수 |A| (dummy arc 제외)
"""
function build_R_matrix(num_regular_arcs::Int)
    dim = num_regular_arcs + 1  # ξ = (ζ; τ) dimension, ζ ∈ R^|A|
    R = zeros(Float64, 2*dim + 2, dim)
    
    # Upper and lower bounds: I and -I
    R[1:dim, :] = Matrix{Float64}(I, dim, dim)
    R[dim+1:2*dim, :] = -Matrix{Float64}(I, dim, dim)
    
    # τ = 1 constraint
    e_last = zeros(Float64, dim)
    e_last[end] = 1.0
    R[2*dim+1, :] = e_last
    R[2*dim+2, :] = -e_last
    
    return R
end

"""
    build_r_vector(xi_hat::Vector{Float64}, epsilon::Float64)

Scenario s의 RHS vector r ∈ R^{2(|A|+1)+2} 생성

Arguments:
- xi_hat: capacity scenario ξ̂^s (크기 |A|, dummy arc 제외)
- epsilon: robustness parameter ε
"""
function build_r_vector(xi_hat::Vector{Float64}, epsilon::Float64)
    dim = length(xi_hat)
    e = ones(Float64, dim)
    
    r = zeros(Float64, 2*dim + 2)
    r[1:dim] = xi_hat - epsilon * e
    r[dim+1:2*dim] = -xi_hat - epsilon * e
    r[2*dim+1] = 1.0
    r[2*dim+2] = -1.0
    
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
function build_robust_counterpart_matrices(capacity_scenarios::Matrix{Float64}, epsilon::Float64)
    num_regular_arcs, num_scenarios = size(capacity_scenarios)
    
    R = build_R_matrix(num_regular_arcs)
    
    r_dict = Dict{Int, Vector{Float64}}()
    for s in 1:num_scenarios
        r_dict[s] = build_r_vector(capacity_scenarios[:, s], epsilon)
    end
    
    return R, r_dict
end