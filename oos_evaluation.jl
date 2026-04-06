"""
OOS (Out-of-Sample) Evaluation for Experiment 1: Value of Full Model

Follower의 2-stage 의사결정:
  Stage 1 (here-and-now): x* 관측 → P̃ 기반 nominal 2-stage SP → h* 결정
  Stage 2 (wait-and-see): ξ^(k) 실현 → h* 고정 → deterministic max-flow

Capacity formula: effective_cap[a] = ξ[a] * (1 - v*x[a]) + h[a]
"""

using JuMP
using HiGHS
using LinearAlgebra

"""
    solve_follower_response(network, x_star, v, w, follower_caps) -> h_star

Follower의 Stage 1: x* 관측 후 P̃ 기반 nominal SP로 h* 결정.

    max_{h} (1/S_f) Σ_s max_{y_s} y_ts(s)
    s.t. N*y(s) = 0,  ∀s
         y_a(s) ≤ ξ̃_a(s)*(1-v*x_a)+h_a,  ∀a, ∀s  (regular arcs)
         y(s) ≥ 0, h ≥ 0, Σh ≤ w

Args:
- network: GridNetworkData or RealWorldNetworkData
- x_star: Vector{Float64} — leader의 interdiction 결정 (num_arcs, dummy 제외)
- v: Float64 — interdiction effectiveness
- w: Float64 — recovery budget
- follower_caps: Matrix{Float64} — follower belief scenarios (num_arcs_with_dummy × S_f)

Returns:
- h_star: Vector{Float64} — optimal recovery allocation (num_arcs, dummy 제외)
"""
function solve_follower_response(network, x_star::Vector{Float64}, v::Float64, w::Float64,
                                  follower_caps::Matrix{Float64})
    num_arcs_total = length(network.arcs)       # dummy 포함
    num_arcs = num_arcs_total - 1               # dummy 제외
    num_nodes = length(network.nodes)
    N = network.N                                # (|V|-1) × |A|+1
    S_f = size(follower_caps, 2)

    # dummy arc index
    dummy_idx = findfirst(a -> a == ("t", "s"), network.arcs)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Variables
    @variable(model, h[1:num_arcs] >= 0)
    @variable(model, y[1:num_arcs_total, 1:S_f] >= 0)

    # Objective: max (1/S_f) Σ_s y_ts(s)  (dummy arc = ("t","s"))
    @objective(model, Max, (1/S_f) * sum(y[dummy_idx, s] for s in 1:S_f))

    # Recovery budget
    @constraint(model, sum(h) <= w)

    # Flow conservation: N * y(s) = 0, ∀s  (N includes dummy arc column)
    for s in 1:S_f
        @constraint(model, N * y[:, s] .== 0.0)
    end

    # Capacity constraints per scenario
    for s in 1:S_f
        for a in 1:num_arcs
            # effective capacity = ξ̃_a(s) * (1 - v*x_a) + h_a
            cap = follower_caps[a, s] * (1.0 - v * x_star[a]) + h[a]
            @constraint(model, y[a, s] <= cap)
        end
        # Dummy arc: unlimited (no upper bound constraint)
    end

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        @warn "Follower response LP status: $(termination_status(model))"
        return zeros(num_arcs)
    end

    return value.(h)
end


"""
    solve_deterministic_maxflow(network, x_star, h_star, v, xi) -> flow_value

Stage 2: ξ 실현 후 h* 고정 → deterministic max-flow.

    max y_ts  s.t. N*y=0, y_a ≤ ξ_a*(1-v*x_a)+h*_a, y≥0

Args:
- network: GridNetworkData or RealWorldNetworkData
- x_star: Vector{Float64} — interdiction decisions (num_arcs)
- h_star: Vector{Float64} — recovery allocation (num_arcs)
- v: Float64
- xi: Vector{Float64} — realized capacity (num_arcs, dummy 제외)

Returns:
- flow_value: Float64 — max-flow value (y_ts)
"""
function solve_deterministic_maxflow(network, x_star::Vector{Float64}, h_star::Vector{Float64},
                                      v::Float64, xi::Vector{Float64})
    num_arcs_total = length(network.arcs)
    num_arcs = num_arcs_total - 1
    N = network.N
    dummy_idx = findfirst(a -> a == ("t", "s"), network.arcs)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, y[1:num_arcs_total] >= 0)
    @objective(model, Max, y[dummy_idx])

    # Flow conservation
    @constraint(model, N * y .== 0.0)

    # Capacity constraints
    for a in 1:num_arcs
        cap = xi[a] * (1.0 - v * x_star[a]) + h_star[a]
        @constraint(model, y[a] <= max(cap, 0.0))  # cap이 음수면 0으로 clamp
    end

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        @warn "Deterministic max-flow status: $(termination_status(model))"
        return 0.0
    end

    return objective_value(model)
end


"""
    solve_deterministic_maxflow!(model, y_var, cap_con, xi, x_star, h_star, v, dummy_idx, num_arcs) -> flow_value

Batch-optimized: RHS만 업데이트하여 재풀기. 모델을 매번 빌드하지 않음.
"""
function solve_deterministic_maxflow!(model::Model, y_var, cap_con, xi::Vector{Float64},
                                       x_star::Vector{Float64}, h_star::Vector{Float64},
                                       v::Float64, dummy_idx::Int, num_arcs::Int)
    for a in 1:num_arcs
        cap = max(xi[a] * (1.0 - v * x_star[a]) + h_star[a], 0.0)
        set_normalized_rhs(cap_con[a], cap)
    end

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        return 0.0
    end

    return objective_value(model)
end


"""
    build_maxflow_template(network) -> (model, y_var, cap_con, dummy_idx, num_arcs)

Deterministic max-flow LP 템플릿. RHS만 바꿔서 K_test번 재사용.
"""
function build_maxflow_template(network)
    num_arcs_total = length(network.arcs)
    num_arcs = num_arcs_total - 1
    N = network.N
    dummy_idx = findfirst(a -> a == ("t", "s"), network.arcs)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, y[1:num_arcs_total] >= 0)
    @objective(model, Max, y[dummy_idx])
    @constraint(model, N * y .== 0.0)

    # Capacity constraints with placeholder RHS=1.0
    cap_con = Vector{ConstraintRef}(undef, num_arcs)
    for a in 1:num_arcs
        cap_con[a] = @constraint(model, y[a] <= 1.0)
    end

    return model, y, cap_con, dummy_idx, num_arcs
end


"""
    evaluate_oos(network, x_star, v, w, follower_caps, test_caps) -> (mean_flow, std_flow, h_star)

전체 OOS 평가 파이프라인:
1. solve_follower_response → h*
2. K_test개 test scenario에 대해 deterministic max-flow (batch)
3. (mean, std) 반환

Args:
- network: network struct
- x_star: Vector{Float64} — interdiction decisions
- v: Float64
- w: Float64
- follower_caps: Matrix{Float64} — follower belief (num_arcs_with_dummy × S_f)
- test_caps: Matrix{Float64} — test scenarios (num_arcs_with_dummy × K_test)

Returns:
- mean_flow: Float64
- std_flow: Float64
- h_star: Vector{Float64}
"""
function evaluate_oos(network, x_star::Vector{Float64}, v::Float64, w::Float64,
                       follower_caps::Matrix{Float64}, test_caps::Matrix{Float64})
    num_arcs = length(network.arcs) - 1
    K_test = size(test_caps, 2)

    # Stage 1: follower's h* decision
    h_star = solve_follower_response(network, x_star, v, w, follower_caps)
    println("    h* solved: sum(h*)=$(round(sum(h_star), digits=4)), " *
            "nnz=$(count(h_star .> 1e-6))/$(num_arcs)")

    # Stage 2: batch deterministic max-flow
    mf_model, y_var, cap_con, dummy_idx, na = build_maxflow_template(network)
    flows = zeros(K_test)

    for k in 1:K_test
        xi_k = test_caps[1:num_arcs, k]  # regular arcs only
        flows[k] = solve_deterministic_maxflow!(mf_model, y_var, cap_con, xi_k,
                                                 x_star, h_star, v, dummy_idx, na)
    end

    mean_flow = mean(flows)
    std_flow = std(flows)

    return mean_flow, std_flow, h_star
end
