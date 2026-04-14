"""
test_unbounded.jl — sub_model unbounded 재현 테스트.

Nobel-US, S=5, ε̂=ε̃=0.1 에서 특정 x̄로 fresh sub_model이 unbounded인지 확인.
"""

using JuMP
using Gurobi
using Printf
using LinearAlgebra
using Infiltrator
if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_subproblem.jl")

# ===== Instance setup (test_benders.jl의 setup_true_dro_instance 동일) =====
network = generate_nobel_us_network()
print_realworld_network_summary(network)

S = 5
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, 0.10 * num_interdictable)
capacities, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), S; seed=42)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(0.2 * γ * c_bar; digits=4)
λU = 2.0
q_hat = fill(1.0 / S, S)

td = make_true_dro_data(network, capacities, q_hat, 0.1, 0.1;
                         w=w, lambda_U=λU, gamma=γ)
println("  |A|=$num_arcs, S=$S, γ=$γ, w=$w, λU=$λU")

# ===== Test x̄ =====
# 수치 오차 포함된 원본 (OMP에서 나온 그대로)
x_raw = [-0.0, -0.0, -7.811371927613517e-7, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.000000781829077, 1.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0]
x_clean = round.(x_raw)

println("\n=== Test 0: fresh sub_model with x_raw (numerical noise) ===")
sub0, vars0 = build_true_dro_subproblem(td, x_raw; optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(sub0, "NonConvex", 2)
set_optimizer_attribute(sub0, "DualReductions", 0)
optimize!(sub0)
st0 = termination_status(sub0)
println("  Status: $st0")
if st0 == MOI.OPTIMAL
    @printf("  Z₀ = %.6f\n", objective_value(sub0))
end

x_test = x_clean

println("\n=== Test 1: fresh sub_model with x_clean (rounded) ===")
sub1, vars1 = build_true_dro_subproblem(td, x_test; optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(sub1, "NonConvex", 2)
set_optimizer_attribute(sub1, "DualReductions", 0)
optimize!(sub1)
st1 = termination_status(sub1)
println("  Status: $st1")
if st1 == MOI.OPTIMAL
    @printf("  Z₀ = %.6f\n", objective_value(sub1))
end

println("\n=== Test 2: fresh sub_model with x=0, then update to x_test ===")
sub2, vars2 = build_true_dro_subproblem(td, zeros(num_arcs); optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(sub2, "NonConvex", 2)
set_optimizer_attribute(sub2, "DualReductions", 0)
# 먼저 x=0으로 한 번 풀고
optimize!(sub2)
println("  x=0 status: $(termination_status(sub2))")
# objective를 x_test로 업데이트 후 다시 풀기
update_true_dro_subproblem_objective!(sub2, vars2, td, x_test)
optimize!(sub2)
st2 = termination_status(sub2)
println("  x_test status: $st2")
if st2 == MOI.OPTIMAL
    @printf("  Z₀ = %.6f\n", objective_value(sub2))
end

println("\n=== Test 3: fix/unfix cycle, then x_test ===")
sub3, vars3 = build_true_dro_subproblem(td, zeros(num_arcs); optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(sub3, "NonConvex", 2)
set_optimizer_attribute(sub3, "DualReductions", 0)
# 먼저 한 번 풀기
optimize!(sub3)
println("  Initial status: $(termination_status(sub3))")
# fix a, d (α-step 시뮬레이션)
a_vals = value.(vars3[:a])
d_vals = value.(vars3[:d])
fix.(vars3[:a], a_vals; force=true)
fix.(vars3[:d], d_vals; force=true)
optimize!(sub3)
println("  Fixed a,d status: $(termination_status(sub3))")
# unfix + restore bounds
unfix.(vars3[:a])
set_lower_bound.(vars3[:a], vars3[:a_min])
set_upper_bound.(vars3[:a], vars3[:a_max])
unfix.(vars3[:d])
set_lower_bound.(vars3[:d], vars3[:d_min])
set_upper_bound.(vars3[:d], vars3[:d_max])
# x_test로 풀기
update_true_dro_subproblem_objective!(sub3, vars3, td, x_test)
optimize!(sub3)
st3 = termination_status(sub3)
println("  After unfix + x_test status: $st3")
if st3 == MOI.OPTIMAL
    @printf("  Z₀ = %.6f\n", objective_value(sub3))
end

println("\n=== Summary ===")
println("  Test 0 (x_raw, noise):     $st0")
println("  Test 1 (x_clean, fresh):   $st1")
println("  Test 2 (reuse, no fix):    $st2")
println("  Test 3 (reuse, fix/unfix): $st3")
