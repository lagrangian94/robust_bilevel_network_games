"""
Debug: Polska S=1 CCG result 검증
하드코딩된 first-stage solution → OSP 풀어서 obj = -2.0 확인
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using LinearAlgebra
using Infiltrator
using Random

include("../network_generator.jl")
include("../build_uncertainty_set.jl")
include("../build_dualized_outer_subprob.jl")

using .NetworkGenerator: generate_polska_network, generate_capacity_scenarios_uniform_model, print_realworld_network_summary

# ===== Instance Setup (compare_benders.jl 동일) =====
network = generate_polska_network()
print_realworld_network_summary(network)

num_arcs = length(network.arcs) - 1
S = 1; epsilon = 0.5; ϕU = 1/epsilon; λU = ϕU; v_param = 1.0; seed = 42
γ = ceil(Int, 0.10 * sum(network.interdictable_arcs[1:num_arcs]))

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
capacity_scenarios_regular = capacities[1:end-1, :]
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(0.2 * γ * c_bar, digits=4)

R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
max_cap = maximum(capacity_scenarios_regular)
πU = ϕU; yU = min(max_cap, ϕU); ytsU = min(max_flow_ub, ϕU)

println("\nParams: S=$S, ϕU=$ϕU, λU=$λU, γ=$γ, w=$w, v=$v_param, πU=$πU, yU=$yU, ytsU=$ytsU")

# ===== Arc 확인 =====
println("\nArc list:")
for i in 1:num_arcs
    println("  Arc $i: $(network.arcs[i])  cap=$(Int(xi_bar[1][i]))")
end

# ===== 하드코딩된 CCG Solution =====
x_sol = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
λ_sol = 2.0
h_sol = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9999700558270899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.42222994417291, 0.0, 0.0]
ψ0_sol = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0]

interdicted = findall(x_sol .> 0.5)
println("\nInterdicted arcs: $interdicted")
for i in interdicted
    println("  Arc $i: $(network.arcs[i])  cap=$(Int(xi_bar[1][i]))  h=$(round(h_sol[i], digits=4))")
end
println("Σh = $(round(sum(h_sol), digits=6)),  λw = $(round(λ_sol*w, digits=6))")

#= ===== Tests A-G 주석처리 (Test H만 사용) =====
# ===== OSP 풀기 =====
println("\n" * "=" ^ 60)
println("OSP (Dualized Outer Subproblem) 풀기")
println("=" ^ 60)

osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
    πU=πU, yU=yU, ytsU=ytsU)

optimize!(osp_model)
osp_obj_free = objective_value(osp_model)
α_free = value.(osp_vars[:α])
println("\nStatus: $(termination_status(osp_model))")
println("OSP obj (α free): $osp_obj_free")
println("Expected (CCG): ≈ -2.0")

# α 분포 확인
println("\nOSP optimal α (>1e-4):")
for i in findall(α_free .> 1e-4)
    println("  α[$i] = $(round(α_free[i], digits=6))  → $(network.arcs[i])")
end
println("  Σα = $(round(sum(α_free), digits=6)),  w/S = $(round(w/S, digits=6))")

# ===== Test A: OSP에서 α를 vertex j=3에 fix =====
println("\n" * "=" ^ 60)
println("Test A: OSP with α fixed to vertex j=3 (α[3]=$(w/S), rest=0)")
println("=" ^ 60)

osp_model_v3, osp_vars_v3, _ = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
    πU=πU, yU=yU, ytsU=ytsU)

# α를 vertex j=3에 fix
α_v3 = osp_vars_v3[:α]
for k in 1:num_arcs
    if k == 3
        fix(α_v3[k], w/S; force=true)
    else
        fix(α_v3[k], 0.0; force=true)
    end
end
optimize!(osp_model_v3)
println("Status: $(termination_status(osp_model_v3))")
println("OSP obj (α fixed j=3): $(objective_value(osp_model_v3))")

# ===== Test B: 모든 CCG active vertex에 대해 OSP에서 α fix =====
println("\n" * "=" ^ 60)
println("Test B: OSP with α fixed to each CCG active vertex")
println("=" ^ 60)
println("CCG active vertices: J = {34, 13, 6, 33, 25, 18, 3, 23}")

ccg_vertices = [34, 13, 6, 33, 25, 18, 3, 23]
for j in ccg_vertices
    osp_vj, vars_vj, _ = build_dualized_outer_subproblem(
        network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
        Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
        πU=πU, yU=yU, ytsU=ytsU)
    α_vj = vars_vj[:α]
    for k in 1:num_arcs
        fix(α_vj[k], k == j ? w/S : 0.0; force=true)
    end
    optimize!(osp_vj)
    st = termination_status(osp_vj)
    obj = (st == MOI.OPTIMAL || st == MOI.ALMOST_OPTIMAL) ? round(objective_value(osp_vj), digits=6) : "$st"
    println("  vertex j=$j ($(network.arcs[j])): OSP obj = $obj")
end

# ===== Test C: ISP leader + follower 직접 빌드해서 evaluate =====
println("\n" * "=" ^ 60)
println("Test C: ISP leader + follower 직접 빌드 (vertex j=3)")
println("=" ^ 60)

include("../nested_benders_trust_region.jl")

α_j3 = zeros(num_arcs)
α_j3[3] = w / S

U_s1 = Dict(:R => Dict(:1=>R[1]), :r_dict => Dict(:1=>r_dict[1]),
            :xi_bar => Dict(:1=>xi_bar[1]), :epsilon => epsilon)

leader_model, leader_vars = build_isp_leader(
    network, 1, ϕU, λU, γ, w, v_param, U_s1,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_j3, S;
    πU=πU)

follower_model, follower_vars = build_isp_follower(
    network, 1, ϕU, λU, γ, w, v_param, U_s1,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_j3, S;
    πU=πU, yU=yU, ytsU=ytsU)

optimize!(leader_model)
optimize!(follower_model)

leader_obj = objective_value(leader_model)
follower_obj = objective_value(follower_model)
println("ISP leader obj: $leader_obj  (status: $(termination_status(leader_model)))")
println("ISP follower obj: $follower_obj  (status: $(termination_status(follower_model)))")
println("ISP total (leader+follower): $(leader_obj + follower_obj)")
println()
println("비교:")
println("  OSP α free:    $osp_obj_free")
println("  OSP α=vertex3: $(objective_value(osp_model_v3))")
println("  ISP leader+follower (j=3): $(leader_obj + follower_obj)")
println("  CCG reported:  -1.99997")

# ===== Test D: bound 키운 뒤 α free vs α=vertex 비교 =====
println("\n" * "=" ^ 60)
println("Test D: bound 키우면 vertex property 회복되는지?")
println("=" ^ 60)
println("현재: ϕU=$ϕU, λU=$λU, πU=$πU, yU=$yU, ytsU=$ytsU  (λ=λU에 binding)")
println()

header = rpad("Setting", 25) * rpad("OSP(α free)", 14) * rpad("OSP(α=v3)", 14) * rpad("|α>0|", 8) * "gap"
println(header)
println("-"^70)

for (ϕU_t, λU_t, πU_t, yU_t, ytsU_t, label) in [
    (2.0,  2.0,  2.0,  2.0,  2.0,  "ϕU=λU=2 (현재)"),
    (2.0,  2.0,  10.0, 10.0, 10.0, "ϕU=λU=2, P=10"),
    (2.0,  2.0, 100.0,100.0,100.0, "ϕU=λU=2, P=100"),
    (10.0, 10.0, 10.0, 10.0, 10.0, "ϕU=λU=P=10"),
    (100.0,100.0,100.0,100.0,100.0,"ϕU=λU=P=100"),
]
    # α free (suppress logs)
    old_stdout = stdout
    redirect_stdout(devnull)
    osp_f, vars_f, _ = build_dualized_outer_subproblem(
        network, S, ϕU_t, λU_t, γ, w, v_param, uncertainty_set,
        Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
        πU=πU_t, yU=yU_t, ytsU=ytsU_t)
    optimize!(osp_f)
    obj_free = objective_value(osp_f)
    α_opt = value.(vars_f[:α])
    n_nonzero = length(findall(α_opt .> 1e-4))

    # α = vertex j=3
    osp_v3, vars_v3, _ = build_dualized_outer_subproblem(
        network, S, ϕU_t, λU_t, γ, w, v_param, uncertainty_set,
        Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
        πU=πU_t, yU=yU_t, ytsU=ytsU_t)
    α_v3 = vars_v3[:α]
    for k in 1:num_arcs
        fix(α_v3[k], k == 3 ? w/S : 0.0; force=true)
    end
    optimize!(osp_v3)
    obj_v3 = objective_value(osp_v3)
    redirect_stdout(old_stdout)

    gap = obj_free - obj_v3
    println(rpad(label, 25) * rpad(round(obj_free, digits=4), 14) * rpad(round(obj_v3, digits=4), 14) * rpad(n_nonzero, 8) * "$(round(gap, digits=4))")

    # α 분포 출력
    for i in findall(α_opt .> 1e-4)
        println("    α[$i] = $(round(α_opt[i], digits=4))")
    end
end

# ===== Test E: bound 키운 뒤 vertex enforcement (전체 36 arcs sweep) =====
println("\n" * "=" ^ 60)
println("Test E: ϕU=λU=100 에서 전체 vertex sweep")
println("=" ^ 60)

ϕU_big = 100.0; λU_big = 100.0; πU_big = 100.0; yU_big = 100.0; ytsU_big = 100.0

# α free
old_stdout = stdout; redirect_stdout(devnull)
osp_big, vars_big, _ = build_dualized_outer_subproblem(
    network, S, ϕU_big, λU_big, γ, w, v_param, uncertainty_set,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
    πU=πU_big, yU=yU_big, ytsU=ytsU_big)
optimize!(osp_big)
obj_big_free = objective_value(osp_big)
α_big = value.(vars_big[:α])
redirect_stdout(old_stdout)

println("OSP(α free, ϕU=100): $(round(obj_big_free, digits=4))")
for i in findall(α_big .> 1e-4)
    println("  α[$i] = $(round(α_big[i], digits=4))  $(network.arcs[i])")
end

println("\nVertex sweep (α = w/S · e_j):")
global best_obj_e = -Inf
global best_j_e = -1
for j in 1:num_arcs
    local old_out = stdout; redirect_stdout(devnull)
    local osp_vj, vars_vj, _ = build_dualized_outer_subproblem(
        network, S, ϕU_big, λU_big, γ, w, v_param, uncertainty_set,
        Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
        πU=πU_big, yU=yU_big, ytsU=ytsU_big)
    for k in 1:num_arcs
        fix(vars_vj[:α][k], k == j ? w/S : 0.0; force=true)
    end
    optimize!(osp_vj)
    redirect_stdout(old_out)
    local vobj = objective_value(osp_vj)
    if vobj > best_obj_e
        global best_obj_e = vobj; global best_j_e = j
    end
    println("  j=$j $(rpad(string(network.arcs[j]), 30)) obj=$(round(vobj, digits=4))")
end
println("\nBest vertex: j=$best_j_e, obj=$(round(best_obj_e, digits=4))")
println("Gap (free - best vertex): $(round(obj_big_free - best_obj_e, digits=4))")

# ===== Test F: μhat, μtilde 비교 (α free vs α=vertex) =====
println("\n" * "=" ^ 60)
println("Test F: coupling shadow prices μhat, μtilde 비교")
println("=" ^ 60)

# --- α free (ϕU=2 original) ---
println("\n--- Case 1: α free (obj=3.567) ---")
old_stdout = stdout; redirect_stdout(devnull)
osp_f1, vars_f1, _ = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
    πU=πU, yU=yU, ytsU=ytsU)
optimize!(osp_f1)
redirect_stdout(old_stdout)

α_f1 = value.(vars_f1[:α])
# coupling_hat: βhat2[s,k] ≤ α[k]
μhat_f1 = [shadow_price(osp_f1[:coupling_hat][1, k]) for k in 1:num_arcs]
μtilde_f1 = [shadow_price(osp_f1[:coupling_tilde][1, k]) for k in 1:num_arcs]

println("obj = $(round(objective_value(osp_f1), digits=4))")
println(rpad("k", 4) * rpad("arc", 28) * rpad("α[k]", 10) * rpad("μhat", 12) * rpad("μtilde", 12) * "μhat+μtilde")
println("-"^78)
for k in 1:num_arcs
    println(rpad(k, 4) * rpad(string(network.arcs[k]), 28) *
            rpad(round(α_f1[k], digits=4), 10) *
            rpad(round(μhat_f1[k], digits=6), 12) *
            rpad(round(μtilde_f1[k], digits=6), 12) *
            "$(round(μhat_f1[k] + μtilde_f1[k], digits=6))")
end

# --- α = vertex j=3 ---
println("\n--- Case 2: α = vertex j=3 (obj=-2.0) ---")
old_stdout = stdout; redirect_stdout(devnull)
osp_v3b, vars_v3b, _ = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
    πU=πU, yU=yU, ytsU=ytsU)
for k in 1:num_arcs
    fix(vars_v3b[:α][k], k == 3 ? w/S : 0.0; force=true)
end
optimize!(osp_v3b)
redirect_stdout(old_stdout)

μhat_v3 = [shadow_price(osp_v3b[:coupling_hat][1, k]) for k in 1:num_arcs]
μtilde_v3 = [shadow_price(osp_v3b[:coupling_tilde][1, k]) for k in 1:num_arcs]

println("obj = $(round(objective_value(osp_v3b), digits=4))")
println(rpad("k", 4) * rpad("arc", 28) * rpad("α[k]", 10) * rpad("μhat", 12) * rpad("μtilde", 12) * "μhat+μtilde")
println("-"^78)
for k in 1:num_arcs
    α_k = (k == 3) ? w/S : 0.0
    println(rpad(k, 4) * rpad(string(network.arcs[k]), 28) *
            rpad(round(α_k, digits=4), 10) *
            rpad(round(μhat_v3[k], digits=6), 12) *
            rpad(round(μtilde_v3[k], digits=6), 12) *
            "$(round(μhat_v3[k] + μtilde_v3[k], digits=6))")
end

# ===== Test G: μhat, μtilde ≤ 1.5 bound via slack+penalty =====
println("\n" * "=" ^ 60)
println("Test G: OSP with μhat, μtilde ≤ 1.5 (slack+penalty)")
println("=" ^ 60)

μ_bound = 1.5

function build_osp_with_mu_bound(network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
        λ_sol, x_sol, h_sol, ψ0_sol, πU, yU, ytsU, μ_bound, num_arcs; fix_vertex=nothing)
    osp, vars, _ = build_dualized_outer_subproblem(
        network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
        Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
        πU=πU, yU=yU, ytsU=ytsU)
    if fix_vertex !== nothing
        for k in 1:num_arcs
            fix(vars[:α][k], k == fix_vertex ? w/S : 0.0; force=true)
        end
    end
    @variable(osp, slack_hat[1:S, 1:num_arcs] >= 0)
    @variable(osp, slack_tilde[1:S, 1:num_arcs] >= 0)
    for s in 1:S, k in 1:num_arcs
        delete(osp, osp[:coupling_hat][s, k])
        delete(osp, osp[:coupling_tilde][s, k])
    end
    unregister(osp, :coupling_hat)
    unregister(osp, :coupling_tilde)
    βhat2 = osp[:βhat2]; βtilde2 = osp[:βtilde2]; α_g = vars[:α]
    @constraint(osp, coupling_hat_new[s=1:S, k=1:num_arcs], βhat2[s,k] == α_g[k] + slack_hat[s,k])
    @constraint(osp, coupling_tilde_new[s=1:S, k=1:num_arcs], βtilde2[s,k] == α_g[k] + slack_tilde[s,k])
    orig_obj = objective_function(osp)
    @objective(osp, Max, orig_obj
        - μ_bound * sum(slack_hat[s,k] for s in 1:S, k in 1:num_arcs)
        - μ_bound * sum(slack_tilde[s,k] for s in 1:S, k in 1:num_arcs))
    return osp, vars
end

# --- α free ---
println("\n--- α free (μ≤$μ_bound) ---")
local old_stdout = stdout; redirect_stdout(devnull)
osp_gf, vars_gf = build_osp_with_mu_bound(network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
    λ_sol, x_sol, h_sol, ψ0_sol, πU, yU, ytsU, μ_bound, num_arcs)
optimize!(osp_gf)
redirect_stdout(old_stdout)
obj_gf = objective_value(osp_gf)
α_gf = value.(vars_gf[:α])
println("obj = $(round(obj_gf, digits=4))  (status: $(termination_status(osp_gf)))")
nonzero_α = findall(α_gf .> 1e-4)
println("α nonzero: $(["$k=$(round(α_gf[k],digits=4))" for k in nonzero_α])")

# --- vertex sweep ---
println("\nVertex sweep (μ≤$μ_bound):")
global best_obj_g = -Inf
global best_j_g = -1
for j in 1:num_arcs
    local old_out = stdout; redirect_stdout(devnull)
    local osp_vj, _ = build_osp_with_mu_bound(network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
        λ_sol, x_sol, h_sol, ψ0_sol, πU, yU, ytsU, μ_bound, num_arcs; fix_vertex=j)
    optimize!(osp_vj)
    redirect_stdout(old_out)
    local vobj = objective_value(osp_vj)
    if vobj > best_obj_g
        global best_obj_g = vobj; global best_j_g = j
    end
    println("  j=$j $(rpad(string(network.arcs[j]), 30)) obj=$(round(vobj, digits=4))")
end
println("\nBest vertex: j=$best_j_g ($(network.arcs[best_j_g])), obj=$(round(best_obj_g, digits=4))")
println("α free obj: $(round(obj_gf, digits=4))")
println("Gap (free - best vertex): $(round(obj_gf - best_obj_g, digits=4))")

# --- best vertex의 μ 출력 ---
println("\n--- Best vertex j=$best_j_g μhat/μtilde 상세 ---")
local old_stdout2 = stdout; redirect_stdout(devnull)
osp_best, vars_best = build_osp_with_mu_bound(network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
    λ_sol, x_sol, h_sol, ψ0_sol, πU, yU, ytsU, μ_bound, num_arcs; fix_vertex=best_j_g)
optimize!(osp_best)
redirect_stdout(old_stdout2)

μhat_best = [shadow_price(osp_best[:coupling_hat_new][1, k]) for k in 1:num_arcs]
μtilde_best = [shadow_price(osp_best[:coupling_tilde_new][1, k]) for k in 1:num_arcs]
println("obj = $(round(objective_value(osp_best), digits=4))")
println("μhat  range: [$(round(minimum(μhat_best), digits=4)), $(round(maximum(μhat_best), digits=4))]")
println("μtilde range: [$(round(minimum(μtilde_best), digits=4)), $(round(maximum(μtilde_best), digits=4))]")
println(rpad("k", 4) * rpad("arc", 28) * rpad("α[k]", 10) * rpad("μhat", 12) * rpad("μtilde", 12) *
        rpad("sl_hat", 10) * "sl_tilde")
println("-"^88)
for k in 1:num_arcs
    local α_k = (k == best_j_g) ? w/S : 0.0
    println(rpad(k, 4) * rpad(string(network.arcs[k]), 28) *
            rpad(round(α_k, digits=4), 10) *
            rpad(round(μhat_best[k], digits=6), 12) *
            rpad(round(μtilde_best[k], digits=6), 12) *
            rpad(round(value(osp_best[:slack_hat][1, k]), digits=6), 10) *
            "$(round(value(osp_best[:slack_tilde][1, k]), digits=6))")
end

# ===== Test H: OSP 풀고 α 외 모든 변수 fix → α LP on simplex =====
println("\n" * "=" ^ 60)
println("Test H: OSP optimal → α 외 fix → vertex property 검증")
println("=" ^ 60)

# Step 1: OSP 전체 풀기
println("\n--- Step 1: OSP 전체 풀기 ---")
local old_stdout_h = stdout; redirect_stdout(devnull)
osp_h, vars_h, _ = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
    πU=πU, yU=yU, ytsU=ytsU)
optimize!(osp_h)
redirect_stdout(old_stdout_h)

α_opt = value.(vars_h[:α])
obj_full = objective_value(osp_h)
println("OSP full obj = $(round(obj_full, digits=6))")
println("α opt: $(["$k=$(round(α_opt[k],digits=4))" for k in findall(α_opt .> 1e-4)])")

# Step 2: α 이외 모든 변수를 optimal 값으로 fix
println("\n--- Step 2: α 외 모든 변수 fix, α만 free → re-optimize ---")
local old_stdout_h2 = stdout; redirect_stdout(devnull)
osp_h2, vars_h2, _ = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
    πU=πU, yU=yU, ytsU=ytsU)
redirect_stdout(old_stdout_h2)

# 모든 변수 목록 가져오기
all_vars_h2 = all_variables(osp_h2)
α_h2_set = Set(vars_h2[:α])

# α 외 변수들의 optimal 값을 osp_h에서 가져와서 fix
for var in all_vars_h2
    if var ∉ α_h2_set
        # 같은 이름의 변수를 osp_h에서 찾아서 값 가져오기
        var_name = name(var)
        var_in_h = variable_by_name(osp_h, var_name)
        if var_in_h !== nothing
            fix(var, value(var_in_h); force=true)
        end
    end
end

optimize!(osp_h2)
α_h2_opt = value.(vars_h2[:α])
obj_h2 = objective_value(osp_h2)
println("α free re-opt obj = $(round(obj_h2, digits=6))  (status: $(termination_status(osp_h2)))")
println("α re-opt: $(["$k=$(round(α_h2_opt[k],digits=4))" for k in findall(α_h2_opt .> 1e-4)])")

# Step 3: 같은 모델에서 α를 각 vertex로 fix → obj 비교
println("\n--- Step 3: α를 각 vertex로 fix ---")
global best_obj_h = -Inf
global best_j_h = -1
for j in 1:num_arcs
    # α fix
    for k in 1:num_arcs
        fix(vars_h2[:α][k], k == j ? w/S : 0.0; force=true)
    end
    optimize!(osp_h2)
    local st = termination_status(osp_h2)
    local vobj = (st == MOI.OPTIMAL || st == MOI.ALMOST_OPTIMAL) ? objective_value(osp_h2) : NaN
    if !isnan(vobj) && vobj > best_obj_h
        global best_obj_h = vobj; global best_j_h = j
    end
    local status_str = (st == MOI.OPTIMAL || st == MOI.ALMOST_OPTIMAL) ? "$(round(vobj, digits=4))" : "$st"
    println("  j=$j $(rpad(string(network.arcs[j]), 30)) obj=$status_str")
end
println("\nBest vertex: j=$best_j_h, obj=$(round(best_obj_h, digits=4))")
println("Full OSP obj:         $(round(obj_full, digits=4))")
println("α free re-opt obj:    $(round(obj_h2, digits=4))")
println("Best vertex obj:      $(round(best_obj_h, digits=4))")
println("Gap (full - best vtx): $(round(obj_full - best_obj_h, digits=4))")
===== Tests A-G 주석처리 끝 =#

# ===== Test H: Primal ISP → α LP on simplex → vertex property 검증 =====
println("\n" * "=" ^ 60)
println("Test H: Primal ISP, 변수 fix 후 α LP on simplex")
println("=" ^ 60)

include("../build_primal_isp.jl")

# Step 1: OSP 전체 풀어서 α* 획득
println("\n--- Step 1: OSP → α* ---")
old_stdout_h1 = stdout; redirect_stdout(devnull)
osp_h, vars_h, _ = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v_param, uncertainty_set,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol;
    πU=πU, yU=yU, ytsU=ytsU)
optimize!(osp_h)
redirect_stdout(old_stdout_h1)
α_star = value.(vars_h[:α])
obj_osp = objective_value(osp_h)
println("OSP obj = $(round(obj_osp, digits=6))")
println("α*: $(["$k=$(round(α_star[k],digits=4))" for k in findall(α_star .> 1e-4)])")

# Step 2: α* fix → primal ISP leader + follower 풀기
println("\n--- Step 2: Primal ISP at α* ---")
U_s1 = Dict(:R => Dict(1=>R[1]), :r_dict => Dict(1=>r_dict[1]),
            :xi_bar => Dict(1=>xi_bar[1]), :epsilon => epsilon)

old_stdout_h2 = stdout; redirect_stdout(devnull)
pl_model, pl_vars = build_primal_isp_leader(
    network, 1, ϕU, λU, γ, w, v_param, U_s1, Mosek.Optimizer,
    x_sol, λ_sol, h_sol, ψ0_sol, S)
pf_model, pf_vars = build_primal_isp_follower(
    network, 1, ϕU, λU, γ, w, v_param, U_s1, Mosek.Optimizer,
    x_sol, λ_sol, h_sol, ψ0_sol, S)
redirect_stdout(old_stdout_h2)

# α* 를 objective coefficient로 설정하고 풀기
for k in 1:num_arcs
    set_objective_coefficient(pl_model, pl_vars[:μhat][1, k], α_star[k])
    set_objective_coefficient(pf_model, pf_vars[:μtilde][1, k], α_star[k])
end
optimize!(pl_model)
optimize!(pf_model)

pl_obj = objective_value(pl_model)
pf_obj = objective_value(pf_model)
println("Primal ISP leader obj  = $(round(pl_obj, digits=6))  ($(termination_status(pl_model)))")
println("Primal ISP follower obj = $(round(pf_obj, digits=6))  ($(termination_status(pf_model)))")
println("Sum = $(round(pl_obj + pf_obj, digits=6))  (OSP는 Max, primal ISP는 Min이므로 부호 주의)")

# μhat, μtilde 값 확인
μhat_star = value.(pl_vars[:μhat][1, :])
μtilde_star = value.(pf_vars[:μtilde][1, :])
println("\nμhat + μtilde (at α*):")
for k in findall((μhat_star .+ μtilde_star) .> 1e-4)
    println("  k=$k $(network.arcs[k]): μhat=$(round(μhat_star[k],digits=4)), μtilde=$(round(μtilde_star[k],digits=4)), sum=$(round(μhat_star[k]+μtilde_star[k],digits=4))")
end

# Step 3: 모든 변수 fix, α만 free → LP on simplex
println("\n--- Step 3: 변수 fix → α LP on simplex ---")

# 새 LP 모델 생성
lp_model = Model(optimizer_with_attributes(Gurobi.Optimizer, MOI.Silent() => true))
@variable(lp_model, α_lp[k=1:num_arcs] >= 0)
@constraint(lp_model, sum(α_lp) == w / S)

# Primal ISP의 objective는 Min (1/S)*Σ ηhat + Σ_k α_k * μhat_k
# 변수 fix → ηhat는 상수, μhat_k는 상수 → α에 대한 linear function
# leader + follower 합: Min const_l + Σ α_k * μhat_k* + const_f + Σ α_k * μtilde_k*
# = Min Σ α_k * (μhat_k* + μtilde_k*) + const
# OSP는 Max이므로 negate: Max -Σ α_k * (μhat_k* + μtilde_k*) - const
# 여기선 Min으로 풀자 (primal ISP와 같은 방향)

@objective(lp_model, Min, sum(α_lp[k] * (μhat_star[k] + μtilde_star[k]) for k in 1:num_arcs))

optimize!(lp_model)
α_lp_opt = value.(α_lp)
obj_lp = objective_value(lp_model)

println("LP obj (α part only) = $(round(obj_lp, digits=6))  ($(termination_status(lp_model)))")
println("α LP opt: $(["$k=$(round(α_lp_opt[k],digits=4))" for k in findall(α_lp_opt .> 1e-4)])")

# 이 α_lp_opt를 다시 primal ISP에 넣어서 실제 obj 확인
println("\n--- Step 4: LP optimal α를 primal ISP에 대입 → 실제 obj 확인 ---")
for k in 1:num_arcs
    set_objective_coefficient(pl_model, pl_vars[:μhat][1, k], α_lp_opt[k])
    set_objective_coefficient(pf_model, pf_vars[:μtilde][1, k], α_lp_opt[k])
end
optimize!(pl_model)
optimize!(pf_model)

pl_obj2 = objective_value(pl_model)
pf_obj2 = objective_value(pf_model)
println("Primal ISP at α_LP: leader=$(round(pl_obj2,digits=4)), follower=$(round(pf_obj2,digits=4)), sum=$(round(pl_obj2+pf_obj2,digits=4))")

# 비교
println("\n--- 비교 ---")
println("  OSP(α free):             $(round(obj_osp, digits=4))")
println("  Primal ISP(α*):          $(round(pl_obj+pf_obj, digits=4))")
println("  LP on simplex → α_LP:    coeff obj = $(round(obj_lp, digits=4))")
println("  Primal ISP(α_LP):        $(round(pl_obj2+pf_obj2, digits=4))")
println("  α_LP vertex?             $(count(α_lp_opt .> 1e-4)) nonzero components")

# Step 5: vertex sweep - 모든 vertex에서 primal ISP 직접 풀기
println("\n--- Step 5: 모든 vertex에서 Primal ISP 직접 풀기 ---")
global best_primal_obj = Inf  # Min이니까
global best_primal_j = -1
for j in 1:num_arcs
    local α_vj = zeros(num_arcs)
    α_vj[j] = w / S
    for k in 1:num_arcs
        set_objective_coefficient(pl_model, pl_vars[:μhat][1, k], α_vj[k])
        set_objective_coefficient(pf_model, pf_vars[:μtilde][1, k], α_vj[k])
    end
    optimize!(pl_model)
    optimize!(pf_model)
    local vobj = objective_value(pl_model) + objective_value(pf_model)
    if vobj < best_primal_obj
        global best_primal_obj = vobj; global best_primal_j = j
    end
    println("  j=$j $(rpad(string(network.arcs[j]), 30)) obj=$(round(vobj, digits=4))")
end
println("\nBest vertex: j=$best_primal_j, obj=$(round(best_primal_obj, digits=4))")
println("Primal ISP(α*):   $(round(pl_obj+pf_obj, digits=4))")
println("Gap: $(round(best_primal_obj - (pl_obj+pf_obj), digits=4))")

# ===== Test I: Section 6.6 — M=1000 bounds on ALL unbounded vars → vertex sweep =====
println("\n" * "=" ^ 60)
println("Test I: M_big=1000 bounds on ALL unbounded vars → vertex property 검증")
println("=" ^ 60)

M_big = 1000.0

function add_big_bounds!(model, vars_dict, M_big; is_leader=true)
    """Add M_big box bounds to all unbounded variables in primal ISP."""
    S = size(vars_dict[is_leader ? :μhat : :μtilde], 1)
    num_arcs = size(vars_dict[is_leader ? :μhat : :μtilde], 2)

    if is_leader
        # ηhat >= 0, add upper
        for s in 1:S
            @constraint(model, vars_dict[:ηhat][s] <= M_big)
        end
        # μhat >= 0, add upper
        for s in 1:S, k in 1:num_arcs
            @constraint(model, vars_dict[:μhat][s,k] <= M_big)
        end
        # ϑhat >= 0, add upper
        for s in 1:S
            @constraint(model, vars_dict[:ϑhat][s] <= M_big)
        end
        # Mhat (PSD): box bounds on all elements
        dim_M = size(vars_dict[:Mhat], 2)
        for s in 1:S, i in 1:dim_M, j in 1:dim_M
            @constraint(model, vars_dict[:Mhat][s,i,j] <= M_big)
            @constraint(model, vars_dict[:Mhat][s,i,j] >= -M_big)
        end
        # Λhat1 (SOC): box bounds
        for idx in eachindex(vars_dict[:Λhat1])
            @constraint(model, vars_dict[:Λhat1][idx] <= M_big)
            @constraint(model, vars_dict[:Λhat1][idx] >= -M_big)
        end
        # Λhat2 (SOC): box bounds
        for idx in eachindex(vars_dict[:Λhat2])
            @constraint(model, vars_dict[:Λhat2][idx] <= M_big)
            @constraint(model, vars_dict[:Λhat2][idx] >= -M_big)
        end
    else
        # ηtilde: FREE, add both bounds
        for s in 1:S
            @constraint(model, vars_dict[:ηtilde][s] <= M_big)
            @constraint(model, vars_dict[:ηtilde][s] >= -M_big)
        end
        # μtilde >= 0, add upper
        for s in 1:S, k in 1:num_arcs
            @constraint(model, vars_dict[:μtilde][s,k] <= M_big)
        end
        # ϑtilde >= 0, add upper
        for s in 1:S
            @constraint(model, vars_dict[:ϑtilde][s] <= M_big)
        end
        # Mtilde (PSD): box bounds
        dim_M = size(vars_dict[:Mtilde], 2)
        for s in 1:S, i in 1:dim_M, j in 1:dim_M
            @constraint(model, vars_dict[:Mtilde][s,i,j] <= M_big)
            @constraint(model, vars_dict[:Mtilde][s,i,j] >= -M_big)
        end
        # Λtilde1 (SOC): box bounds
        for idx in eachindex(vars_dict[:Λtilde1])
            @constraint(model, vars_dict[:Λtilde1][idx] <= M_big)
            @constraint(model, vars_dict[:Λtilde1][idx] >= -M_big)
        end
        # Λtilde2 (SOC): box bounds
        for idx in eachindex(vars_dict[:Λtilde2])
            @constraint(model, vars_dict[:Λtilde2][idx] <= M_big)
            @constraint(model, vars_dict[:Λtilde2][idx] >= -M_big)
        end
    end
end

# Build primal ISP with M_big bounds
println("\n--- Building primal ISP with M=$M_big bounds ---")
old_stdout_i1 = stdout; redirect_stdout(devnull)
pl_model_b, pl_vars_b = build_primal_isp_leader(
    network, 1, ϕU, λU, γ, w, v_param, U_s1, Mosek.Optimizer,
    x_sol, λ_sol, h_sol, ψ0_sol, S)
pf_model_b, pf_vars_b = build_primal_isp_follower(
    network, 1, ϕU, λU, γ, w, v_param, U_s1, Mosek.Optimizer,
    x_sol, λ_sol, h_sol, ψ0_sol, S)
redirect_stdout(old_stdout_i1)

add_big_bounds!(pl_model_b, pl_vars_b, M_big; is_leader=true)
add_big_bounds!(pf_model_b, pf_vars_b, M_big; is_leader=false)
println("  ✓ M=$M_big bounds added to all unbounded variables")

# Step I-1: Solve at α* (should match original)
println("\n--- Step I-1: Primal ISP (bounded) at α* ---")
for k in 1:num_arcs
    set_objective_coefficient(pl_model_b, pl_vars_b[:μhat][1, k], α_star[k])
    set_objective_coefficient(pf_model_b, pf_vars_b[:μtilde][1, k], α_star[k])
end
optimize!(pl_model_b)
optimize!(pf_model_b)
pl_obj_b = objective_value(pl_model_b)
pf_obj_b = objective_value(pf_model_b)
println("  Leader:   $(round(pl_obj_b, digits=6))  ($(termination_status(pl_model_b)))")
println("  Follower: $(round(pf_obj_b, digits=6))  ($(termination_status(pf_model_b)))")
println("  Sum:      $(round(pl_obj_b + pf_obj_b, digits=6))")
println("  Original: $(round(pl_obj + pf_obj, digits=6))")
println("  Bound binding? $(abs(pl_obj_b + pf_obj_b - (pl_obj + pf_obj)) > 1e-3 ? "YES ✗" : "NO ✓")")

# Step I-2: Full vertex sweep with M_big bounds
println("\n--- Step I-2: Full vertex sweep (M=$M_big bounds) ---")
global best_primal_obj_b = Inf
global best_primal_j_b = -1
for j in 1:num_arcs
    α_vj = zeros(num_arcs)
    α_vj[j] = w / S
    for k in 1:num_arcs
        set_objective_coefficient(pl_model_b, pl_vars_b[:μhat][1, k], α_vj[k])
        set_objective_coefficient(pf_model_b, pf_vars_b[:μtilde][1, k], α_vj[k])
    end
    optimize!(pl_model_b)
    optimize!(pf_model_b)
    st_l = termination_status(pl_model_b)
    st_f = termination_status(pf_model_b)
    ok_l = (st_l == MOI.OPTIMAL || st_l == MOI.ALMOST_OPTIMAL || st_l == MOI.SLOW_PROGRESS)
    ok_f = (st_f == MOI.OPTIMAL || st_f == MOI.ALMOST_OPTIMAL || st_f == MOI.SLOW_PROGRESS)
    if ok_l && ok_f
        vobj = objective_value(pl_model_b) + objective_value(pf_model_b)
    else
        vobj = NaN
    end
    if !isnan(vobj) && vobj < best_primal_obj_b
        global best_primal_obj_b = vobj
        global best_primal_j_b = j
    end
    status_str = isnan(vobj) ? "$(st_l)/$(st_f)" : "$(round(vobj, digits=4))"
    println("  j=$j $(rpad(string(network.arcs[j]), 30)) obj=$status_str")
end

println("\n" * "=" ^ 60)
println("Test I 결과 요약")
println("=" ^ 60)
println("  Primal ISP(α*, no bound):  $(round(pl_obj + pf_obj, digits=4))")
println("  Primal ISP(α*, M=$M_big):  $(round(pl_obj_b + pf_obj_b, digits=4))")
println("  Best vertex (M=$M_big):    j=$best_primal_j_b, obj=$(round(best_primal_obj_b, digits=4))")
println("  Best vertex (no bound):    j=$best_primal_j, obj=$(round(best_primal_obj, digits=4))")
gap_bounded = best_primal_obj_b - (pl_obj_b + pf_obj_b)
println("  Gap (bounded):             $(round(gap_bounded, digits=4))")
if abs(gap_bounded) < 1e-2
    println("  → Vertex property RECOVERED! Sion's theorem path is viable.")
else
    println("  → Vertex property still fails. Issue is deeper than compactness.")
end

@infiltrate
