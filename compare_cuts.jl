"""
compare_cuts.jl — 1회 정밀 비교: Strict (Full OSP) vs Nested (IMP↔ISP) cut quality

같은 (x, λ, h, ψ0) 입력에서 두 방법으로 cut을 생성하고:
1. α_sol 비교
2. intercept_l, intercept_f 비교
3. coefficient norms 비교
4. random feasible x에서 cut value 비교 (어느 cut이 더 restrictive한지)
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using LinearAlgebra
using Infiltrator
using Revise
using Pajarito
using Random

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("build_full_model.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Common Parameters (from compare_benders.jl) =====
S = 1
γ_ratio = 0.10
ρ = 0.2
v = 1.0
seed = 42
epsilon = 0.5
ϕU = 1/epsilon
λU = ϕU

# ===== Generate Network & Uncertainty Set =====
println("="^80)
println("GENERATING NETWORK AND UNCERTAINTY SET")
println("="^80)

network = generate_grid_network(3, 3, seed=seed)  # 3×3 for speed
print_network_summary(network)

num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)
println("  Interdiction budget: γ = $γ")

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)
println("  Recovery budget: w = $w")

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

# LDR bounds
source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
max_cap = maximum(capacity_scenarios_regular)
πU = ϕU
yU = min(max_cap, ϕU)
ytsU = min(max_flow_ub, ϕU)
println("  LDR bounds: ϕU=$ϕU, πU=$πU, yU=$yU, ytsU=$ytsU")

# ===== Step 1: Get a feasible (x, λ, h, ψ0) from initial OMP =====
println("\n" * "="^80)
println("STEP 1: Getting feasible point from OMP")
println("="^80)

# Run a few iterations of strict benders to get a reasonable point
model_s, vars_s = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut_lf=true)
result_s = strict_benders_optimize!(model_s, vars_s, network, ϕU, λU, γ, w, uncertainty_set;
    optimizer=Gurobi.Optimizer, multi_cut_lf=true, max_iter=5, πU=πU, yU=yU, ytsU=ytsU)

# Use the solution from the last iteration
x_test = result_s[:debug_α][1] !== nothing ? round.(value.(vars_s[:x])) : zeros(num_arcs)
# Actually, let's get from opt_sol if available, or from last iteration
if haskey(result_s, :opt_sol)
    x_test = result_s[:opt_sol][:x]
    h_test = result_s[:opt_sol][:h]
    λ_test = result_s[:opt_sol][:λ]
    ψ0_test = result_s[:opt_sol][:ψ0]
else
    # Use values from the model (last iteration)
    x_test = round.(value.(vars_s[:x]))
    h_test = value.(vars_s[:h])
    λ_test = value(vars_s[:λ])
    ψ0_test = value.(vars_s[:ψ0])
end

println("Test point:")
println("  x  = $x_test")
println("  λ  = $λ_test")
println("  h  = $(round.(h_test, digits=4))")
println("  ψ0 = $(round.(ψ0_test, digits=4))")

# ===== Step 2: Full OSP (Strict method) =====
println("\n" * "="^80)
println("STEP 2: Full OSP (Strict Benders method)")
println("="^80)

osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v, uncertainty_set, MosekTools.Optimizer,
    λ_test, x_test, h_test, ψ0_test; πU=πU, yU=yU, ytsU=ytsU)

(status_strict, cut_strict) = osp_optimize!(osp_model, osp_vars, osp_data,
    λ_test, x_test, h_test, ψ0_test; multi_cut_lf=true)

println("  Status: $status_strict")
println("  OSP objective (strict): $(cut_strict[:obj_val])")
println("  α_sol (strict): $(round.(cut_strict[:α_sol], digits=6))")
println("  Σ intercept_l: $(sum(cut_strict[:intercept_l]))")
println("  Σ intercept_f: $(sum(cut_strict[:intercept_f]))")

# ===== Step 3: Nested inner loop (IMP ↔ ISP) =====
println("\n" * "="^80)
println("STEP 3: Nested inner loop (IMP ↔ ISP)")
println("="^80)

E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1); d0[end] = 1.0
isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :πU => πU, :yU => yU, :ytsU => ytsU,
    :λU => λU, :γ => γ, :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S=>S)

imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
st_imp, α_sol_init = initialize_imp(imp_model, imp_vars)

leader_instances, follower_instances = initialize_isp(
    network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_test, x_sol=x_test, h_sol=h_test, ψ0_sol=ψ0_test,
    α_sol=α_sol_init, πU=πU, yU=yU, ytsU=ytsU)

imp_cuts = Dict{Symbol, Any}(:old_tr_constraints => nothing)
status_nested, cut_nested = tr_imp_optimize!(
    imp_model, imp_vars, leader_instances, follower_instances;
    isp_data=isp_data, λ_sol=λ_test, x_sol=x_test, h_sol=h_test, ψ0_sol=ψ0_test,
    outer_iter=1, imp_cuts=imp_cuts, inner_tr=true, tol=1e-4)

α_sol_nested = cut_nested[:α_sol]
println("  Status: $status_nested")
println("  Inner loop objective (nested): $(cut_nested[:obj_val])")
println("  Inner loop iterations: $(cut_nested[:iter])")
println("  α_sol (nested): $(round.(α_sol_nested, digits=6))")

# Now extract outer cut coefficients by re-evaluating ISP at converged α
outer_cut_nested = evaluate_master_opt_cut(
    leader_instances, follower_instances, isp_data, cut_nested, 1; multi_cut_lf=true)

println("  Σ intercept_l (nested): $(sum(outer_cut_nested[:intercept_l]))")
println("  Σ intercept_f (nested): $(sum(outer_cut_nested[:intercept_f]))")

# ===== Step 3b: Hybrid — OSP's α, ISP's coefficients =====
println("\n" * "="^80)
println("STEP 3b: Hybrid (OSP α → ISP coefficient extraction)")
println("="^80)

# Use α from Full OSP, but extract coefficients via ISP decomposition
α_strict_for_isp = cut_strict[:α_sol]
hybrid_cut_info = Dict(:α_sol => α_strict_for_isp, :obj_val => cut_strict[:obj_val])
outer_cut_hybrid = evaluate_master_opt_cut(
    leader_instances, follower_instances, isp_data, hybrid_cut_info, 1; multi_cut_lf=true)

println("  α used: OSP's α_sol (same as strict)")
println("  Σ intercept_l (hybrid): $(sum(outer_cut_hybrid[:intercept_l]))")
println("  Σ intercept_f (hybrid): $(sum(outer_cut_hybrid[:intercept_f]))")
println("  ||Uhat1|| (hybrid): $(round(norm(outer_cut_hybrid[:Uhat1]), digits=4))")
println("  cf. ||Uhat1|| strict=$(round(norm(cut_strict[:Uhat1]), digits=4)), nested=$(round(norm(outer_cut_nested[:Uhat1]), digits=4))")

# ===== Step 3c: Magnanti-Wong cuts =====
println("\n" * "="^80)
println("STEP 3c: Magnanti-Wong cuts (interior + arc-directed)")
println("="^80)

interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
core_points = generate_core_points(network, γ, λU, w, v;
    interdictable_idx=interdictable_idx, strategy=:interior_and_arcs)
println("  Generated $(length(core_points)) core points (1 interior + $(length(core_points)-1) arc-directed)")

# MW cut from interior core point
outer_cut_mw_interior = evaluate_mw_opt_cut(
    leader_instances, follower_instances, isp_data, cut_nested, 1;
    x_sol=x_test, λ_sol=λ_test, h_sol=h_test, ψ0_sol=ψ0_test,
    x_core=core_points[1].x, λ_core=core_points[1].λ,
    h_core=core_points[1].h, ψ0_core=core_points[1].ψ0,
    multi_cut_lf=true)

println("  MW-interior: ||Uhat1||=$(round(norm(outer_cut_mw_interior[:Uhat1]), digits=4))")
println("  Σ intercept_l (MW-int): $(sum(outer_cut_mw_interior[:intercept_l]))")
println("  Σ intercept_f (MW-int): $(sum(outer_cut_mw_interior[:intercept_f]))")

# MW cuts from arc-directed core points — pick best (highest cut value at random x)
mw_arc_cuts = []
for cp_idx in 2:length(core_points)
    cp = core_points[cp_idx]
    mw_arc = evaluate_mw_opt_cut(
        leader_instances, follower_instances, isp_data, cut_nested, 1;
        x_sol=x_test, λ_sol=λ_test, h_sol=h_test, ψ0_sol=ψ0_test,
        x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
        multi_cut_lf=true)
    push!(mw_arc_cuts, mw_arc)
    arc_idx = interdictable_idx[cp_idx - 1]
    println("  MW-arc$(arc_idx): ||Uhat1||=$(round(norm(mw_arc[:Uhat1]), digits=4))")
end

# Use best arc cut for comparison (highest cut value at current point)
best_arc_idx = 1
if !isempty(mw_arc_cuts)
    arc_vals = [eval_cut_value(mc, x_test, λ_test, h_test, ψ0_test)[1] for mc in mw_arc_cuts]
    best_arc_idx = argmax(arc_vals)
end
outer_cut_mw_arc = isempty(mw_arc_cuts) ? outer_cut_mw_interior : mw_arc_cuts[best_arc_idx]

# ===== Step 3d: Sherali ζ-perturbation cut =====
println("\n" * "="^80)
println("STEP 3d: Sherali ζ-perturbation cut (interior core point)")
println("="^80)

outer_cut_sherali = evaluate_sherali_opt_cut(
    leader_instances, follower_instances, isp_data, cut_nested, 1;
    x_sol=x_test, λ_sol=λ_test, h_sol=h_test, ψ0_sol=ψ0_test,
    x_core=core_points[1].x, λ_core=core_points[1].λ,
    h_core=core_points[1].h, ψ0_core=core_points[1].ψ0,
    ζ=1e-8, multi_cut_lf=true)

println("  Sherali: ||Uhat1||=$(round(norm(outer_cut_sherali[:Uhat1]), digits=4))")
println("  Σ intercept_l (Sherali): $(sum(outer_cut_sherali[:intercept_l]))")
println("  Σ intercept_f (Sherali): $(sum(outer_cut_sherali[:intercept_f]))")

# ===== Step 4: Comparison =====
println("\n" * "="^80)
println("COMPARISON: Strict vs Nested vs Hybrid vs MW-interior vs MW-arc vs Sherali")
println("="^80)

# α comparison
println("\n--- α_sol comparison ---")
α_strict = cut_strict[:α_sol]
α_nested = α_sol_nested
println("  ||α_strict - α_nested|| = $(norm(α_strict - α_nested))")
println("  ||α_strict|| = $(norm(α_strict)),  ||α_nested|| = $(norm(α_nested))")
for k in 1:min(num_arcs, 20)
    diff = abs(α_strict[k] - α_nested[k])
    marker = diff > 1e-4 ? " ← DIFF" : ""
    println("    arc $k: strict=$(round(α_strict[k], digits=6))  nested=$(round(α_nested[k], digits=6))$marker")
end

# Intercept comparison
println("\n--- Intercept comparison ---")
il_s = sum(cut_strict[:intercept_l])
if_s = sum(cut_strict[:intercept_f])
il_n = sum(outer_cut_nested[:intercept_l])
if_n = sum(outer_cut_nested[:intercept_f])
println("  intercept_l:  strict=$(round(il_s, digits=6))  nested=$(round(il_n, digits=6))  diff=$(round(il_s - il_n, digits=6))")
println("  intercept_f:  strict=$(round(if_s, digits=6))  nested=$(round(if_n, digits=6))  diff=$(round(if_s - if_n, digits=6))")
println("  total:        strict=$(round(il_s+if_s, digits=6))  nested=$(round(il_n+if_n, digits=6))  diff=$(round((il_s+if_s)-(il_n+if_n), digits=6))")

# Coefficient norms comparison (6-way)
println("\n--- Coefficient norms comparison (6-way) ---")
coeff_names = [:Uhat1, :Utilde1, :Uhat3, :Utilde3, :βtilde1_1, :βtilde1_3, :Ztilde1_3]
println("  " * rpad("Coeff", 13) * rpad("Strict", 12) * rpad("Nested", 12) * rpad("Hybrid", 12) * rpad("MW-int", 12) * rpad("MW-arc", 12) * "Sherali")
println("  " * "-"^85)
for cn in coeff_names
    ns = norm(cut_strict[cn])
    nn = norm(outer_cut_nested[cn])
    nh = norm(outer_cut_hybrid[cn])
    nmi = norm(outer_cut_mw_interior[cn])
    nma = norm(outer_cut_mw_arc[cn])
    nsh = norm(outer_cut_sherali[cn])
    println("  " * rpad(string(cn), 13) * rpad(round(ns, digits=4), 12) * rpad(round(nn, digits=4), 12) * rpad(round(nh, digits=4), 12) * rpad(round(nmi, digits=4), 12) * rpad(round(nma, digits=4), 12) * "$(round(nsh, digits=4))")
end

# ===== Step 4b: Uhat1/Utilde1 element-wise decomposition =====
println("\n--- Uhat1 element-wise analysis (s=1) ---")
Uhat1_s = cut_strict[:Uhat1][1,:,:]
Uhat1_n = outer_cut_nested[:Uhat1][1,:,:]
Uhat1_diff = Uhat1_s - Uhat1_n

# x=1인 arc (arc 9)과 x=0인 arc 구분
x_on = findall(x_test .== 1.0)
x_off = findall(x_test .== 0.0)

println("  Uhat1 shape: $(size(Uhat1_s))")
println("  ||Uhat1_strict||=$(round(norm(Uhat1_s), digits=4)),  ||Uhat1_nested||=$(round(norm(Uhat1_n), digits=4))")
println()

# x=1 rows vs x=0 rows 분해
norm_s_on = norm(Uhat1_s[x_on, :])
norm_n_on = norm(Uhat1_n[x_on, :])
norm_s_off = norm(Uhat1_s[x_off, :])
norm_n_off = norm(Uhat1_n[x_off, :])
println("  Rows where x=1 (arcs $x_on):")
println("    ||Uhat1_strict[x=1,:]|| = $(round(norm_s_on, digits=4))")
println("    ||Uhat1_nested[x=1,:]|| = $(round(norm_n_on, digits=4))")
println("    diff = $(round(norm_s_on - norm_n_on, digits=4))")
println("  Rows where x=0 ($(length(x_off)) arcs):")
println("    ||Uhat1_strict[x=0,:]|| = $(round(norm_s_off, digits=4))")
println("    ||Uhat1_nested[x=0,:]|| = $(round(norm_n_off, digits=4))")
println("    diff = $(round(norm_s_off - norm_n_off, digits=4))")

# cut에서 Uhat1이 x와 어떻게 곱해지는지: -ϕU * Uhat1 .* diag(x)E
# → x=0인 row의 Uhat1은 현재 점에서 cut value에 기여 안 함
# → 하지만 OMP에서 x가 변할 때는 기여함 (cut의 slope)
println()
println("  Key insight: x=0 rows contribute to cut SLOPE but not to cut VALUE at current point")
println("  → Strict has $(round((norm_s_off/norm_n_off - 1)*100, digits=1))% larger x=0 row norms")
println("  → This makes Strict cut more sensitive to x changes")

# Utilde1 동일 분석
println("\n--- Utilde1 element-wise analysis (s=1) ---")
Utilde1_s = cut_strict[:Utilde1][1,:,:]
Utilde1_n = outer_cut_nested[:Utilde1][1,:,:]
norm_s_on_t = norm(Utilde1_s[x_on, :])
norm_n_on_t = norm(Utilde1_n[x_on, :])
norm_s_off_t = norm(Utilde1_s[x_off, :])
norm_n_off_t = norm(Utilde1_n[x_off, :])
println("  Rows where x=1: strict=$(round(norm_s_on_t, digits=4))  nested=$(round(norm_n_on_t, digits=4))  diff=$(round(norm_s_on_t-norm_n_on_t, digits=4))")
println("  Rows where x=0: strict=$(round(norm_s_off_t, digits=4))  nested=$(round(norm_n_off_t, digits=4))  diff=$(round(norm_s_off_t-norm_n_off_t, digits=4))")

# PSD matrix (Mhat, Mtilde) 값 비교 — Full OSP에서만 직접 접근 가능
println("\n--- Mhat/Mtilde from Full OSP (PSD block values) ---")
Mhat_val = value.(osp_model[:Mhat][1,:,:])
Mtilde_val = value.(osp_model[:Mtilde][1,:,:])
println("  ||Mhat||=$(round(norm(Mhat_val), digits=6)),  tr(Mhat_11)=$(round(tr(Mhat_val[1:num_arcs,1:num_arcs]), digits=6)),  Mhat_22=$(round(Mhat_val[end,end], digits=6))")
println("  ||Mtilde||=$(round(norm(Mtilde_val), digits=6)),  tr(Mtilde_11)=$(round(tr(Mtilde_val[1:num_arcs,1:num_arcs]), digits=6)),  Mtilde_22=$(round(Mtilde_val[end,end], digits=6))")
eigvals_hat = eigen(Symmetric(Mhat_val)).values
eigvals_tilde = eigen(Symmetric(Mtilde_val)).values
println("  Mhat eigenvalues: min=$(round(minimum(eigvals_hat), digits=8)), max=$(round(maximum(eigvals_hat), digits=6))")
println("  Mtilde eigenvalues: min=$(round(minimum(eigvals_tilde), digits=8)), max=$(round(maximum(eigvals_tilde), digits=6))")

# Ψhat constraint (line 234): v*D_s*Mhat_11 + ... - Uhat1 <= 0
# → Uhat1[i,j] >= v*D_s*Mhat_11[i,j] + ... (when active)
# If Mhat values differ between joint/separate, Uhat1 lower bounds differ
println("\n--- Ψhat constraint analysis: Uhat1 >= f(Mhat) ---")
xi_bar_1 = uncertainty_set[:xi_bar][1]
D_s = diagm(xi_bar_1)
# Ψhat_L constraint (line 234): v*D_s*Mhat_11 + v*Mhat_12*ξ̄' - Uhat1_L - Uhat2_L + Uhat3_L <= 0
# → Uhat1_L >= v*D_s*Mhat_11 + v*Mhat_12*ξ̄' - Uhat2_L + Uhat3_L
Adj_Psi_hat = v * D_s * Mhat_val[1:num_arcs, 1:num_arcs] + v * Mhat_val[1:num_arcs, end] * xi_bar_1'
println("  ||v*D_s*Mhat_11 + v*Mhat_12*ξ̄'|| = $(round(norm(Adj_Psi_hat), digits=4))  (lower bound driver for Uhat1)")
println("  For reference, ||Uhat1_strict_L|| = $(round(norm(Uhat1_s[:,1:num_arcs]), digits=4)), ||Uhat1_nested_L|| = $(round(norm(Uhat1_n[:,1:num_arcs]), digits=4))")

# ===== Step 5: Cut value at random feasible x =====
println("\n--- Cut value at random feasible x points ---")
println("  (positive diff = nested cut more restrictive = nested is stronger)")

xi_bar_local = uncertainty_set[:xi_bar]

function eval_cut_value(cut_coeffs, x_eval, λ_eval, h_eval, ψ0_eval; intercept_l_key=:intercept_l, intercept_f_key=:intercept_f)
    """Evaluate the cut at a given (x, λ, h, ψ0) using the cut coefficients."""
    diag_x_E_eval = Diagonal(x_eval) * E
    diag_λ_ψ_eval = Diagonal(λ_eval*ones(num_arcs) - v .* ψ0_eval)

    val_l = 0.0
    val_f = 0.0
    for s in 1:S
        # Leader terms
        val_l += -ϕU * sum(cut_coeffs[:Uhat1][s,:,:] .* diag_x_E_eval)
        val_l += -ϕU * sum(cut_coeffs[:Uhat3][s,:,:] .* (E - diag_x_E_eval))
        # Follower terms
        val_f += -ϕU * sum(cut_coeffs[:Utilde1][s,:,:] .* diag_x_E_eval)
        val_f += -ϕU * sum(cut_coeffs[:Utilde3][s,:,:] .* (E - diag_x_E_eval))
        val_f += sum(cut_coeffs[:Ztilde1_3][s,:,:] .* (diag_λ_ψ_eval * diagm(xi_bar_local[s])))
        val_f += (d0' * cut_coeffs[:βtilde1_1][s,:]) * λ_eval
        val_f += -1 * (h_eval + diag_λ_ψ_eval * xi_bar_local[s])' * cut_coeffs[:βtilde1_3][s,:]
    end
    val_l += sum(cut_coeffs[intercept_l_key])
    val_f += sum(cut_coeffs[intercept_f_key])
    return val_l + val_f, val_l, val_f
end

# Verify tightness at current point
val_strict_at_pt, vl_s, vf_s = eval_cut_value(cut_strict, x_test, λ_test, h_test, ψ0_test)
val_nested_at_pt, vl_n, vf_n = eval_cut_value(outer_cut_nested, x_test, λ_test, h_test, ψ0_test)
val_hybrid_at_pt, vl_h, vf_h = eval_cut_value(outer_cut_hybrid, x_test, λ_test, h_test, ψ0_test)
val_mwi_at_pt, vl_mi, vf_mi = eval_cut_value(outer_cut_mw_interior, x_test, λ_test, h_test, ψ0_test)
val_mwa_at_pt, vl_ma, vf_ma = eval_cut_value(outer_cut_mw_arc, x_test, λ_test, h_test, ψ0_test)
val_sherali_at_pt, vl_sh, vf_sh = eval_cut_value(outer_cut_sherali, x_test, λ_test, h_test, ψ0_test)

println("\n  At current test point (should match OSP obj):")
println("    OSP obj (strict): $(round(cut_strict[:obj_val], digits=6))")
println("    cut_val (strict):  $(round(val_strict_at_pt, digits=6))  (l=$(round(vl_s, digits=4)), f=$(round(vf_s, digits=4)))")
println("    OSP obj (nested): $(round(cut_nested[:obj_val], digits=6))")
println("    cut_val (nested):  $(round(val_nested_at_pt, digits=6))  (l=$(round(vl_n, digits=4)), f=$(round(vf_n, digits=4)))")
println("    cut_val (hybrid):  $(round(val_hybrid_at_pt, digits=6))  (l=$(round(vl_h, digits=4)), f=$(round(vf_h, digits=4)))")
println("    cut_val (MW-int):  $(round(val_mwi_at_pt, digits=6))  (l=$(round(vl_mi, digits=4)), f=$(round(vf_mi, digits=4)))")
println("    cut_val (MW-arc):  $(round(val_mwa_at_pt, digits=6))  (l=$(round(vl_ma, digits=4)), f=$(round(vf_ma, digits=4)))")
println("    cut_val (Sherali): $(round(val_sherali_at_pt, digits=6))  (l=$(round(vl_sh, digits=4)), f=$(round(vf_sh, digits=4)))")
println("    MW validity: MW-int >= z*-ε? $(val_mwi_at_pt >= cut_nested[:obj_val] - 1e-3), MW-arc >= z*-ε? $(val_mwa_at_pt >= cut_nested[:obj_val] - 1e-3)")
println("    Sherali validity: Sherali >= z*-ε? $(val_sherali_at_pt >= cut_nested[:obj_val] - 1e-3)")

# Generate random feasible x points (binary, sum ≤ γ, only interdictable arcs)
rng = MersenneTwister(123)
n_random = 10
println("\n  Cut values at $n_random random feasible x points (6-way):")
println("  " * rpad("Point", 7) * rpad("Strict", 12) * rpad("Nested", 12) * rpad("Hybrid", 12) * rpad("MW-int", 12) * rpad("MW-arc", 12) * rpad("Sherali", 12) * "Best")
println("  " * "-"^91)

wins = Dict("Strict"=>0, "Nested"=>0, "Hybrid"=>0, "MW-int"=>0, "MW-arc"=>0, "Sherali"=>0)

for trial in 1:n_random
    global wins
    # Random binary x with sum ≤ γ, only on interdictable arcs
    x_rand = zeros(num_arcs)
    available = copy(interdictable_idx)
    n_select = rand(rng, 0:γ)
    selected = sort(shuffle(rng, available))[1:min(n_select, length(available))]
    x_rand[selected] .= 1.0

    # λ random in [0.001, λU]
    λ_rand = 0.001 + rand(rng) * (λU - 0.001)

    # h: random, sum ≤ λ*w
    h_rand = rand(rng, num_arcs) .* (λ_rand * w / num_arcs)
    h_rand = h_rand .* (λ_rand * w / max(sum(h_rand), 1e-10))  # normalize

    # ψ0: McCormick consistent
    ψ0_rand = [min(λU * x_rand[k], λ_rand, max(λ_rand - λU * (1 - x_rand[k]), 0.0)) for k in 1:num_arcs]

    val_s, _, _ = eval_cut_value(cut_strict, x_rand, λ_rand, h_rand, ψ0_rand)
    val_n, _, _ = eval_cut_value(outer_cut_nested, x_rand, λ_rand, h_rand, ψ0_rand)
    val_h, _, _ = eval_cut_value(outer_cut_hybrid, x_rand, λ_rand, h_rand, ψ0_rand)
    val_mi, _, _ = eval_cut_value(outer_cut_mw_interior, x_rand, λ_rand, h_rand, ψ0_rand)
    val_ma, _, _ = eval_cut_value(outer_cut_mw_arc, x_rand, λ_rand, h_rand, ψ0_rand)
    val_sh, _, _ = eval_cut_value(outer_cut_sherali, x_rand, λ_rand, h_rand, ψ0_rand)
    vals = [val_s, val_n, val_h, val_mi, val_ma, val_sh]
    names = ["Strict", "Nested", "Hybrid", "MW-int", "MW-arc", "Sherali"]
    best = names[argmax(vals)]
    wins[best] += 1
    println("  " * rpad(trial, 7) * rpad(round(val_s, digits=4), 12) * rpad(round(val_n, digits=4), 12) * rpad(round(val_h, digits=4), 12) * rpad(round(val_mi, digits=4), 12) * rpad(round(val_ma, digits=4), 12) * rpad(round(val_sh, digits=4), 12) * best)
end

println("\n  Wins: Strict=$(wins["Strict"]), Nested=$(wins["Nested"]), Hybrid=$(wins["Hybrid"]), MW-int=$(wins["MW-int"]), MW-arc=$(wins["MW-arc"]), Sherali=$(wins["Sherali"]) / $n_random")
println("  (Higher cut value = more restrictive = better for LB)")
println("="^80)

# @infiltrate
