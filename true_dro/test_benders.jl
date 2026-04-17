"""
test_benders.jl — True-DRO Benders 검증.

Interactive input으로 network config, S 선택.
compare_benders.jl의 setup_instance 패턴 참고.
"""

"""
5x5, S=10, 0.1, 0.1, Sub time =30, mini-bd=true
User-callback calls 243, time in user-callback 0.00 sec
  Sub: Z₀=11.500237 [LOCAL], α=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.297,0.720,2.316,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
  Iter 637: LB=8.133596  UB=8.133597  gap=1.46e-07
[ Info: True-DRO Benders converged at iter 637 (gap=1.455851768957056e-7)

----------------------------------------
True-DRO: status=Optimal, Z₀=8.133597, iters=637
  LB=8.133596, UB=8.133597, gap=1.46e-07
  x* = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  α* = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.112, 1.175, 0.000, 0.000, 1.047, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
============================================================
Primal Recovery: x* → α* → (h*, λ*, ψ⁰*)
============================================================

lambdaU=2.0 했을 시,
  Sub: Z₀=8.133597, α=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.074,1.222,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.038,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
  Iter 198: LB=8.133598  UB=8.133597  gap=8.90e-08
[ Info: True-DRO Benders converged at iter 198 (gap=8.904142162382194e-8)

----------------------------------------
True-DRO: status=Optimal, Z₀=8.133597, iters=198
  LB=8.133598, UB=8.133597, gap=8.90e-08
  x* = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  α* = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.074, 1.222, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.038, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
============================================================

lambdaU=2.0, omp VI (phase A) 했을 시,
  Sub: Z₀=8.133599, α=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.494,0.000,0.000,0.000,0.704,0.000,0.000,0.000,1.135,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
  Iter 102: LB=8.133598  UB=8.133599  gap=1.67e-07
[ Info: True-DRO Benders converged at iter 102 (gap=1.669338904071469e-7)

----------------------------------------
True-DRO: status=Optimal, Z₀=8.133599, iters=102
  LB=8.133598, UB=8.133599, gap=1.67e-07
  x* = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  α* = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.494, 0.000, 0.000, 0.000, 0.704, 0.000, 0.000, 0.000, 1.135, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
  Wall time: 362.06 sec


LambdaU=2.0 ,omp VI (phase A + 2B) 했을 시,
  Sub: Z₀=8.133600, α=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.891,0.000,0.000,0.000,0.865,0.000,0.000,0.000,0.578,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
  Iter 105: LB=8.133598  UB=8.133600  gap=2.32e-07
[ Info: True-DRO Benders converged at iter 105 (gap=2.3234024756858074e-7)

----------------------------------------
True-DRO: status=Optimal, Z₀=8.133600, iters=105
  LB=8.133598, UB=8.133600, gap=2.32e-07
  x* = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  α* = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.891, 0.000, 0.000, 0.000, 0.865, 0.000, 0.000, 0.000, 0.578, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
  Wall time: 564.02 sec

lambdaU=2.0, omp VI (phase 2B만) 했을 시,
User-callback calls 674, time in user-callback 0.00 sec
  Sub: Z₀=11.400161, α=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,2.092,0.000,0.000,0.819,0.423,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
  Iter 201: LB=8.132960  UB=8.133597  gap=7.83e-05
[ Info: True-DRO Benders converged at iter 201 (gap=7.834036344484369e-5)
----------------------------------------
True-DRO: status=Optimal, Z₀=8.133597, iters=201, time=2245.86s
  LB=8.132960, UB=8.133597, gap=7.83e-05
    x* = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  α* = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.074, 1.222, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.038, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
  Wall time: 2245.86 sec
  """



using Revise
using JuMP
using Gurobi
using HiGHS
using Printf
using LinearAlgebra
using Infiltrator
using Serialization
using Dates
using Logging

# ---- Load modules ----
if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

# True-DRO
includet("true_dro_data.jl")
includet("true_dro_build_omp.jl")
includet("true_dro_build_subproblem.jl")
includet("true_dro_build_isp_leader.jl")
includet("true_dro_build_isp_follower.jl")
includet("true_dro_benders.jl")
includet("true_dro_mincut_vi.jl")
includet("true_dro_recover.jl")

# ε calibration table
includet("oos_dirichlet.jl")

# TV-DRO (for sandwich comparison)
includet("../TV_DRO/tv_data.jl")
includet("../TV_DRO/tv_build_isp_leader.jl")
includet("../TV_DRO/tv_build_isp_follower.jl")
includet("../TV_DRO/tv_build_imp.jl")
includet("../TV_DRO/tv_build_omp.jl")
includet("../TV_DRO/tv_build_full_model.jl")
includet("../TV_DRO/tv_nested_benders.jl")


# ===== Network Instance Configs =====
network_configs = Dict(
    :grid_3x3   => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_5x5   => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us   => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene    => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska     => Dict(:type => :real_world, :generator => generate_polska_network),
)


"""
    compute_interdict_budget(config_key, num_interdictable, γ_ratio) → γ

네트워크별 interdiction budget. Sioux Falls/Abilene: source 근처 sparse → γ=2로 고정.
"""
function compute_interdict_budget(config_key::Symbol, num_interdictable::Int, γ_ratio::Float64)
    if config_key in (:sioux_falls, :abilene)
        return 2
    end
    return ceil(Int, γ_ratio * num_interdictable)
end


"""
    setup_true_dro_instance(config_key; S, γ_ratio, ρ, v, seed, epsilon_hat, epsilon_tilde)

네트워크 + 파라미터 → TrueDROData 생성. compare_benders.jl의 setup_instance 패턴.
"""
function setup_true_dro_instance(config_key::Symbol;
    S=10, γ_ratio=0.10, ρ=0.2, v=1.0, seed=42,
    epsilon_hat=0.5, epsilon_tilde=epsilon_hat)

    config = network_configs[config_key]

    # --- Network 생성 ---
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n]; seed=seed)
        print_network_summary(network)
    elseif config[:type] == :real_world
        network = config[:generator]()
        print_realworld_network_summary(network)
    end

    num_arcs = length(network.arcs) - 1

    # --- Parameters (compare_benders.jl 기준) ---
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = compute_interdict_budget(config_key, num_interdictable, γ_ratio)

    # capacities, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), S; seed=seed)
    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)

    # λU: leader's λ upper bound
    λU = 2.0

    q_hat = fill(1.0 / S, S)

    println("  |A|=$num_arcs, S=$S, γ=$γ, w=$(round(w, digits=4)), λU=$λU")
    println("  ε̂=$epsilon_hat, ε̃=$epsilon_tilde")

    td = make_true_dro_data(network, capacities, q_hat, epsilon_hat, epsilon_tilde;
                            w=w, lambda_U=λU, gamma=γ)

    return network, td
end


# ===== Interactive 설정 =====
println("=" ^ 70)
println("True-DRO Benders Test")
println("=" ^ 70)

# --- 입력 검증 헬퍼 ---
function _parse_yn(prompt::String, default::Bool)::Bool
    print(prompt); s = strip(readline()) |> lowercase
    isempty(s) && return default
    s in ("y", "n") || error("잘못된 입력: '$s' (y/n만 허용)")
    return s == "y"
end

function _parse_choice(prompt::String, allowed::Vector{String}, default::String)::String
    print(prompt); s = strip(readline()) |> lowercase
    isempty(s) && return default
    s in allowed || error("잘못된 입력: '$s' (허용: $(join(allowed, ", ")))")
    return s
end

# --- 네트워크 선택 ---
const DEFAULT_NETWORK_ORDER = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls]

# 축약 이름 → instance key 매핑
const NET_ALIAS = Dict(
    "3x3" => :grid_3x3, "grid_3x3" => :grid_3x3,
    "4x4" => :grid_4x4, "grid_4x4" => :grid_4x4,
    "5x5" => :grid_5x5, "grid_5x5" => :grid_5x5,
    "sioux_falls" => :sioux_falls, "sioux" => :sioux_falls, "sf" => :sioux_falls,
    "nobel_us" => :nobel_us, "nobel" => :nobel_us,
    "abilene" => :abilene, "ab" => :abilene,
    "polska" => :polska, "pl" => :polska,
)

println("네트워크 (Enter=기본순서: 5x5→polska→abilene→nobel_us→sioux_falls)")
println("  또는 직접 입력 (comma 구분, e.g. '5x5, polska' or 'abilene')")
print("선택: "); net_input = strip(readline())

if isempty(net_input)
    instance_keys = copy(DEFAULT_NETWORK_ORDER)
else
    tokens = [strip(t) |> lowercase for t in split(net_input, ",")]
    instance_keys = Symbol[]
    for tok in tokens
        haskey(NET_ALIAS, tok) || error("알 수 없는 네트워크: '$tok'  (허용: $(join(sort(collect(keys(NET_ALIAS))), ", ")))")
        key = NET_ALIAS[tok]
        # 4x4는 동적으로 추가
        if key == :grid_4x4 && !haskey(network_configs, :grid_4x4)
            network_configs[:grid_4x4] = Dict(:type => :grid, :m => 4, :n => 4)
        end
        push!(instance_keys, key)
    end
end
println("  → 풀 네트워크: $(instance_keys)")

# --- 파라미터 입력 ---
print("시나리오 수 S: "); S = parse(Int, readline())
S > 0 || error("S는 양수여야 합니다: $S")

println("β (Dirichlet concentration):")
println("  1. 0.1   2. 0.3   3. 0.5   4. 0.8")
print("선택 (1-4) [3]: "); β_choice_str = strip(readline())
β_choice = isempty(β_choice_str) ? 3 : parse(Int, β_choice_str)
β_map = Dict(1 => 0.1, 2 => 0.3, 3 => 0.5, 4 => 0.8)
haskey(β_map, β_choice) || error("잘못된 β 선택: $β_choice")
β_val = β_map[β_choice]

# ε lookup from calibration table (coverage=95%, round to 2 decimals)
ε_raw = lookup_epsilon(S, β_val; coverage=0.95)
ε_hat = round(ε_raw; digits=2)
ε_tilde = ε_hat
@printf("β=%.1f, S=%d → ε_raw=%.8f → ε=%.2f (leader=follower)\n", β_val, S, ε_raw, ε_hat)

print("Sub time limit (sec, 0=none) [30]: "); tl_str = strip(readline())
sub_tl = isempty(tl_str) ? 30.0 : parse(Float64, tl_str)
sub_tl >= 0 || error("time limit은 0 이상이어야 합니다: $sub_tl")
sub_tl = sub_tl <= 0 ? nothing : sub_tl

use_mini_benders = _parse_yn("Mini-Benders? (y/n) [n]: ", false)
max_mb_iter = 5

strengthen_cuts = Symbol(_parse_choice("Strengthen cuts? (none/mw) [none]: ",
    ["none", "mw"], "none"))

use_inexact = _parse_yn("Inexact mode? (y/n) [y]: ", true)

vi_sym = Symbol(_parse_choice("Valid inequality? (none/mincut) [none]: ",
    ["none", "mincut"], "none"))

# --- Log 폴더 ---
β_str = replace(@sprintf("%.1f", β_val), "." => "p")
log_dir = joinpath(@__DIR__, "S$(S)_beta$(β_str)_cov95")
mkpath(log_dir)


# ===== 네트워크 순차 루프 =====
for (net_idx, instance_key) in enumerate(instance_keys)

println("\n" * "#" ^ 70)
@printf("# [%d/%d] Network: %s\n", net_idx, length(instance_keys), instance_key)
println("#" ^ 70)

# ===== Log file setup (pipe-based tee: stdout + file 동시) =====
log_timestamp = Dates.format(now(), "yyyymmdd_HHMMss")
log_filename = joinpath(log_dir, "log_$(instance_key)_$(log_timestamp).txt")
log_io = open(log_filename, "w")
original_stdout = stdout
rd, wr = redirect_stdout()
log_task = @async begin
    try
        while isopen(rd)
            data = readavailable(rd)
            isempty(data) && break
            write(original_stdout, data)
            flush(original_stdout)
            write(log_io, data)
            flush(log_io)
        end
    catch e
        e isa EOFError || rethrow()
    end
end

println("Log file: $log_filename")
println("Started: $(now())")
println()

println("\n" * "=" ^ 70)
println("INSTANCE: $instance_key (S=$S, β=$β_val, ε̂=$ε_hat, ε̃=$ε_tilde)")
println("=" ^ 70)

network, td = setup_true_dro_instance(instance_key; S=S,
    epsilon_hat=ε_hat, epsilon_tilde=ε_tilde)


# ===== 1. True-DRO Benders =====
# ===== VI-only OMP diagnostic =====
if vi_sym == :mincut
    println("\n" * "-" ^ 40)
    println("VI-only OMP diagnostic (no Benders cuts)")
    vi_omp, vi_vars = build_true_dro_omp(td; optimizer=Gurobi.Optimizer, silent=false)
    add_phase1_mincut_vi!(vi_omp, vi_vars, td)
    optimize!(vi_omp)
    vi_st = termination_status(vi_omp)
    if vi_st == MOI.OPTIMAL
        vi_lb = objective_value(vi_omp)
        vi_x = round.(Int, [value(vi_vars[:x][k]) for k in 1:td.num_arcs])
        @printf("  VI-only LB = %.6f, x = %s\n", vi_lb, string(vi_x))
    else
        println("  VI-only OMP: $vi_st")
    end
    println("-" ^ 40)
end

println("\n" * "=" ^ 70)
println("1. True-DRO Benders (bilinear subproblem via Gurobi NonConvex=2)")
println("=" ^ 70)

result = true_dro_benders_optimize!(td;
    mip_optimizer=Gurobi.Optimizer,
    nlp_optimizer=Gurobi.Optimizer,
    inexact=use_inexact,
    max_iter=1000,
    tol=1e-4,
    verbose=true,
    sub_verbose=true,
    sub_time_limit=sub_tl,
    mini_benders=use_mini_benders,
    lp_optimizer=(use_mini_benders ? Gurobi.Optimizer : nothing),
    max_mini_benders_iter=max_mb_iter,
    strengthen_cuts=strengthen_cuts,
    valid_inequality=vi_sym)

gap = abs(result[:upper_bound] - result[:lower_bound]) /
      max(abs(result[:upper_bound]), 1e-10)

println("\n" * "-" ^ 40)
wt = get(result, :wall_time, NaN)
@printf("True-DRO: status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n", result[:status], result[:Z0], result[:iters], wt)
@printf("  LB=%.6f, UB=%.6f, gap=%.2e\n", result[:lower_bound], result[:upper_bound], gap)
x_int = round.(Int, result[:x])
println("  x* = $x_int")
α_str = join([@sprintf("%.3f", a) for a in result[:α]], ", ")
println("  α* = [$α_str]")

# ===== Save profile =====
profile_key = (net=instance_key, S=S, εh=ε_hat, εt=ε_tilde,
               tl=sub_tl, mb=use_mini_benders, sc=strengthen_cuts, vi=vi_sym)
profile_path = joinpath(@__DIR__, "profiles.jls")
profiles = isfile(profile_path) ? deserialize(profile_path) : Dict{NamedTuple, Dict}()
profiles[profile_key] = result
serialize(profile_path, profiles)
println("  Profile saved → $profile_path ($(length(profiles)) entries)")

# ===== Primal Recovery =====
rec = recover_and_print(td, result; optimizer=HiGHS.Optimizer)

# ===== Close log =====
println("\nFinished: $(now())")
redirect_stdout(original_stdout)
close(wr)
wait(log_task)
close(log_io)
println("Log saved → $log_filename")

end  # for instance_key

println("\n" * "=" ^ 70)
println("All $(length(instance_keys)) networks done.")
println("=" ^ 70)
@infiltrate

# # ===== 2. Sandwich: TV-DRO (V^Dir) =====
# println("\n" * "=" ^ 70)
# println("2. Sandwich comparison: V*(true) ≤ V^Dir(TV-DRO)")
# println("=" ^ 70)

# tv = make_tv_data(network, td.xi_bar, td.q_hat, td.eps_hat, td.eps_tilde;
#                   w=td.w, lambda_U=td.lambda_U, gamma=td.gamma)

# # TV-DRO full model (direct route)
# full_model, full_vars = build_full_tv_model(tv; optimizer=Gurobi.Optimizer)
# optimize!(full_model)

# V_true = result[:Z0]
# if termination_status(full_model) == MOI.OPTIMAL
#     V_dir = objective_value(full_model)
#     @printf("  V*(true)  = %.6f\n", V_true)
#     @printf("  V^Dir(TV) = %.6f\n", V_dir)
#     @printf("  Gap       = %.6f (%.2f%%)\n", V_dir - V_true,
#             100 * (V_dir - V_true) / max(abs(V_true), 1e-10))
#     if V_true <= V_dir + 1e-4
#         println("  ✓ V* ≤ V^Dir")
#     else
#         println("  ✗ V* > V^Dir — violation!")
#     end
# else
#     println("  TV-DRO full model: $(termination_status(full_model))")
# end


# # ===== 3. ε→0 nominal convergence =====
# println("\n" * "=" ^ 70)
# println("3. ε→0 nominal convergence")
# println("=" ^ 70)

# epsilons = [0.3, 0.1, 0.05, 0.01, 0.001]
# eps_results = []

# for ε in epsilons
#     network_eps, td_eps = setup_true_dro_instance(instance_key; S=S,
#         epsilon_hat=ε, epsilon_tilde=ε)
#     r = true_dro_benders_optimize!(td_eps;
#         mip_optimizer=Gurobi.Optimizer,
#         nlp_optimizer=Gurobi.Optimizer,
#         max_iter=1000, tol=1e-4, verbose=false,
#         sub_time_limit=sub_tl,
#         mini_benders=use_mini_benders,
#         lp_optimizer=(use_mini_benders ? Gurobi.Optimizer : nothing),
#         max_mini_benders_iter=max_mb_iter)
#     push!(eps_results, (ε=ε, Z0=r[:Z0], status=r[:status], iters=r[:iters]))
#     @printf("  ε=%.4f → Z₀=%.6f  (%s, %d iters)\n", ε, r[:Z0], r[:status], r[:iters])
# end

# # ε 작아질수록 Z₀ monotone non-increasing (tighter ambiguity → smaller worst-case)
# objs = [r.Z0 for r in eps_results]
# is_mono = all(objs[i] >= objs[i+1] - 1e-4 for i in 1:length(objs)-1)
# if is_mono
#     println("  ✓ Z₀ monotone non-increasing as ε→0")
# else
#     println("  ✗ Z₀ not monotone — check!")
# end

