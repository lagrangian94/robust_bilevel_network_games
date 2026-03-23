"""
debug_outer_tr.jl

outer_tr=true에서 Stage 1이 x*를 놓치는 원인 진단.
1) outer_tr=false로 x* 확인
2) x*에서 수동으로 subproblem 평가 (inner Benders)
3) outer_tr=true 실행
4) 결과 비교

출력은 콘솔 + debug_outer_tr_output.txt 동시 저장.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using LinearAlgebra
using Infiltrator
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Tee 출력: 콘솔 + 파일 동시 =====
const LOG_FILE = open("debug_outer_tr_output.txt", "w")

function tprintln(args...)
    println(stdout, args...)
    println(LOG_FILE, args...)
    flush(LOG_FILE)
end

function tprint(args...)
    print(stdout, args...)
    print(LOG_FILE, args...)
    flush(LOG_FILE)
end

# ===== Same setup as compare_benders.jl (5x5 grid, S=1) =====
seed = 42
S = 2
epsilon = 0.5
ϕU = 1/epsilon
λU = ϕU
γ_ratio = 0.10
ρ = 0.2
v = 1.0

network = generate_grid_network(4, 4, seed=seed)
print_network_summary(network)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)
tprintln("  γ = $γ")

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uset = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
max_cap = maximum(capacity_scenarios_regular)
πU = ϕU
yU = min(max_cap, ϕU)
ytsU = min(max_flow_ub, ϕU)
tprintln("  ϕU=$ϕU, πU=$πU, yU=$yU, ytsU=$ytsU")

strengthen_cuts = :mw

# ===== 1) outer_tr=false 결과 하드코딩 (STEP 1 skip) =====
tprintln("\n" * "="^80)
tprintln("STEP 1: outer_tr=false — 정답 x* (하드코딩)")
tprintln("="^80)

x_star = zeros(num_arcs)
x_star[20] = 1.0
x_star[22] = 1.0
h_star = zeros(num_arcs)
h_star[20] = 0.001079509888992789
h_star[21] = 0.0008620682187832367
h_star[22] = 1.5521892223974416e-5
λ_star = 0.001
ψ0_star = zeros(num_arcs)  # approximate (not critical for subproblem eval)
obj_ref = 3.1027347888957584

tprintln("  x* = $(findall(x_star .> 0.5))")
tprintln("  obj* = $obj_ref")
tprintln("  h* = $h_star")
tprintln("  λ* = $λ_star")

# ===== 2) x*에서 수동 subproblem 평가 =====
tprintln("\n" * "="^80)
tprintln("STEP 2: x*에서 수동으로 inner Benders (subproblem) 평가")
tprintln("="^80)

# Fresh IMP + ISP 구축
E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1)
d0[end] = 1.0

imp_model_manual, imp_vars_manual = build_imp(network, S, ϕU, λU, γ, w, v, uset; mip_optimizer=Gurobi.Optimizer)
st_manual, α_sol_manual = initialize_imp(imp_model_manual, imp_vars_manual)

leader_inst_manual, follower_inst_manual = initialize_isp(network, S, ϕU, λU, γ, w, v, uset;
    conic_optimizer=Mosek.Optimizer,
    λ_sol=λ_star, x_sol=x_star, h_sol=h_star, ψ0_sol=ψ0_star, α_sol=α_sol_manual,
    πU=πU, yU=yU, ytsU=ytsU)

isp_data_manual = Dict(:E => E, :network => network, :ϕU => ϕU, :πU => πU, :yU => yU, :ytsU => ytsU,
    :λU => λU, :γ => γ, :w => w, :v => v, :uncertainty_set => uset, :d0 => d0, :S => S)

imp_cuts_manual = Dict{Symbol, Any}(:old_tr_constraints => nothing)

tprintln("  Evaluating subproblem at x* with inner_tr=true...")
status_manual, cut_info_manual = tr_imp_optimize!(imp_model_manual, imp_vars_manual,
    leader_inst_manual, follower_inst_manual;
    isp_data=isp_data_manual,
    λ_sol=λ_star, x_sol=x_star, h_sol=h_star, ψ0_sol=ψ0_star,
    outer_iter=1, imp_cuts=imp_cuts_manual, inner_tr=true)

tprintln("  Manual subprob at x*: obj = $(cut_info_manual[:obj_val])")
tprintln("  Expected (from outer_tr=false): ≈ $obj_ref")
tprintln("  Difference: $(abs(cut_info_manual[:obj_val] - obj_ref))")

# ===== 3) outer_tr=true: 실행 =====
tprintln("\n" * "="^80)
tprintln("STEP 3: outer_tr=true — 전체 실행")
tprintln("="^80)

model_tr, vars_tr = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
result_tr = tr_nested_benders_optimize!(model_tr, vars_tr, network, ϕU, λU, γ, w, uset;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    outer_tr=true, inner_tr=true,
    πU=πU, yU=yU, ytsU=ytsU, strengthen_cuts=strengthen_cuts)

# ===== 4) 결과 비교 =====
tprintln("\n" * "="^80)
tprintln("STEP 4: 결과 비교")
tprintln("="^80)

obj_tr = haskey(result_tr, :past_local_lower_bound) ? minimum(result_tr[:past_local_lower_bound]) : minimum(result_tr[:past_upper_bound])
tprintln("  outer_tr=false obj: $obj_ref")
tprintln("  outer_tr=true  obj: $obj_tr")
tprintln("  Gap: $(abs(obj_ref - obj_tr))")

if haskey(result_tr, :past_local_center)
    centers_list = result_tr[:past_local_center]
    tprintln("\n  Stage별 center vs x* L1 거리:")
    for (i, c) in enumerate(centers_list)
        c_round = round.(c)
        dist = Int(sum(abs.(x_star .- c_round)))
        tprintln("    Stage $i: center arcs=$(findall(c_round .> 0.5)), ||x*-center||₁=$dist")
    end
end

if haskey(result_tr, :past_local_lower_bound)
    tprintln("\n  past_local_lower_bound: $(round.(result_tr[:past_local_lower_bound], digits=6))")
end

if haskey(result_tr, :past_local_optimizer)
    tprintln("\n  Stage별 local optimizer (x):")
    for (i, opt) in enumerate(result_tr[:past_local_optimizer])
        tprintln("    Stage $i: x arcs=$(findall(round.(opt[:x]) .> 0.5))")
    end
end

tprintln("\n  x* arcs: $(findall(x_star .> 0.5))")
tprintln("  Manual subprob at x*: $(round(cut_info_manual[:obj_val], digits=6))")

# ===== 5) x*를 Stage 1 center 기준으로 분석 =====
if haskey(result_tr, :past_local_center) && length(result_tr[:past_local_center]) >= 1
    c1 = round.(result_tr[:past_local_center][1])
    tprintln("\n" * "="^80)
    tprintln("STEP 5: Stage 1 분석")
    tprintln("="^80)
    tprintln("  Stage 1 center: arcs=$(findall(c1 .> 0.5))")
    tprintln("  x*:             arcs=$(findall(x_star .> 0.5))")
    dist_to_c1 = Int(sum(abs.(x_star .- c1)))
    tprintln("  ||x* - center₁||₁ = $dist_to_c1")

    B_bin_seq = result_tr[:tr_info][:bin_B_steps]
    tprintln("  B_bin_steps (stage 전환 iter): $B_bin_seq")

    if dist_to_c1 <= 1
        tprintln("\n  ⚠ x*는 Stage 1 영역 내 (distance ≤ B_bin=1)")
        tprintln("    Stage 1 LB = $(round(result_tr[:past_local_lower_bound][1], digits=6))")
        tprintln("    하지만 x*의 true obj = $obj_ref")
        tprintln("    → Stage 1이 x*를 놓치거나 subproblem을 잘못 평가함!")
    end
end

# ===== 6) 모든 OMP cut을 x*에서 평가 =====
tprintln("\n" * "="^80)
tprintln("STEP 6: OMP의 모든 cut을 x*에서 평가")
tprintln("="^80)

# x*에서의 OMP 변수값 세팅
y_star = Dict(
    [vars_tr[:x][k] => x_star[k] for k in 1:num_arcs]...,
    [vars_tr[:h][k] => h_star[k] for k in 1:num_arcs]...,
    vars_tr[:λ] => λ_star,
    [vars_tr[:ψ0][k] => ψ0_star[k] for k in 1:num_arcs]...
)

function evaluate_expr_safe(expr::AffExpr, var_values::Dict)
    eval_result = expr.constant
    for (var, coef) in expr.terms
        if haskey(var_values, var)
            eval_result += coef * var_values[var]
        else
            # t_0 등 epigraph 변수는 skip
            continue
        end
    end
    return eval_result
end

true_val_raw = obj_ref * S  # subprob_obj * S (raw, not averaged)
tprintln("  true subprob value at x* (raw, ×S): $(round(true_val_raw, digits=6))")
tprintln()

invalid_cuts = []
all_cuts = result_tr[:cuts]
for (name, con) in sort(collect(all_cuts), by=x->x[1])
    # constraint: t_0 >= rhs_expr  →  t_0 - rhs_expr >= 0
    # add_optimality_cuts! returns the rhs expression (sum of leader_s + follower_s)
    # but result[:cuts] stores the JuMP constraint reference, not the expression
    # We need to extract the cut value from the constraint

    # The constraint is: t_0 >= cut_expr, i.e., t_0 - cut_expr >= 0
    # In JuMP normalized form: t_0 - cut_expr >= 0
    # We can evaluate the cut_expr at x* by computing: t_0_coeff * t_0_val - (constraint evaluated)
    # Actually, let's just read the constraint's function and evaluate
    con_obj = constraint_object(con)
    func = con_obj.func  # AffExpr: t_0 - cut_expr
    # cut_expr = t_0 - func (since func = t_0 - cut_expr)
    # At x*, cut_val = t_0_val - func_val, but we don't know t_0_val
    # Instead: func has t_0 with coeff +1, and cut terms with coeff -1
    # cut_value_at_x* = -evaluate_expr_safe(func, y_star) (excluding t_0 term)

    # func = 1.0*t_0 + (-coeff)*x + ... + constant >= 0
    # So cut_expr = -( (-coeff)*x + ... + constant ) = coeff*x - constant + ...
    # Easier: just evaluate func without t_0, negate
    cut_val = -evaluate_expr_safe(func, y_star)

    is_invalid = cut_val > true_val_raw + 1e-4
    marker = is_invalid ? " ⚠ INVALID" : ""
    if is_invalid
        push!(invalid_cuts, (name, cut_val))
    end
    tprintln("  $(rpad(name, 25)) cut_val=$(round(cut_val, digits=6))$(marker)")
end

tprintln()
tprintln("  총 cuts: $(length(all_cuts)),  invalid at x*: $(length(invalid_cuts))")
if !isempty(invalid_cuts)
    tprintln("  ── Invalid cuts 상세 ──")
    for (name, val) in invalid_cuts
        tprintln("    $name: $(round(val, digits=6))  (excess=$(round(val - true_val_raw, digits=6)))")
    end
end

tprintln("\n" * "="^80)
tprintln("로그 저장 완료: debug_outer_tr_output.txt")
tprintln("="^80)

close(LOG_FILE)

@infiltrate
