"""
run_experiment2_calibration.jl — Experiment 2: PO/PP/NR/WR Analytical ε Calibration.

95% coverage ε이 과대하여 DRO structural over-conservatism 발생 → Rahimian et al. (2019)
analytical calibration으로 [ε^S, ε^D] range를 찾는 bisection.

핵심:
  evaluate_f(x, ε) = build & solve subproblem at ε, evaluate at x → Z0_val = f_ε(x)
  PO - PP = f_ε(x_neut) - f_ε(x_rob)     (싸다: evaluate_f만)
  NR - WR = [f_0(x*)-f_0(x_neut)] - [f_1(x*)-f_1(x_rob)]  (비싸다: DRO 풀어야)

Bisection:
  PO-PP < 0  →  ε < ε^S  →  lo = ε
  PO-PP ≥ 0 & NR-WR < 0  →  ε ∈ [ε^S, ε^D]  →  STOP
  PO-PP ≥ 0 & NR-WR ≥ 0  →  ε > ε^D  →  hi = ε

Usage:
    include("run_experiment2_calibration.jl")
"""

using Revise
using JuMP
using Gurobi
using Printf
using LinearAlgebra
using Serialization
using Dates

# ---- Load modules ----
if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

includet("true_dro_data.jl")
includet("true_dro_build_omp.jl")
includet("true_dro_build_subproblem.jl")
includet("true_dro_build_isp_leader.jl")
includet("true_dro_build_isp_follower.jl")
includet("true_dro_benders.jl")
includet("true_dro_mincut_vi.jl")
includet("oos_dirichlet.jl")

# Log 파싱 (run_oos_phase_a_wmax.jl에서 가져옴)
include("oos_evaluate.jl")  # compute_win_rate 등 (필요 시)

# ============================================================
# Log 파싱 유틸 (run_oos_phase_a_wmax.jl 에서 가져옴)
# ============================================================

"""
    parse_log_wmax(filepath) → Dict

로그에서 x*, Z0, status 파싱. Nominal / True-DRO / Single 지원.
"""
function parse_log_wmax(filepath::String)
    lines = readlines(filepath)

    x_star_summary = nothing
    x_star_omp = nothing
    Z0 = NaN
    status = "Unknown"
    iters = 0
    wall_time = NaN
    variant = :unknown

    for line in lines
        # True-DRO / Single-layer status
        m_td = match(r"(True-DRO|Single-layer):\s*status=(\w+),\s*Z₀=([\d.]+),\s*iters=(\d+),\s*time=([\d.]+)s", line)
        if m_td !== nothing
            variant = m_td.captures[1] == "Single-layer" ? :single : :two_layer
            status = m_td.captures[2]
            Z0 = parse(Float64, m_td.captures[3])
            iters = parse(Int, m_td.captures[4])
            wall_time = parse(Float64, m_td.captures[5])
        end

        # Nominal SP status
        m_nom = match(r"Nominal SP:\s*Z₀=([\d.]+),\s*time=([\d.]+)s", line)
        if m_nom !== nothing
            variant = :nominal
            status = "Optimal"
            Z0 = parse(Float64, m_nom.captures[1])
            wall_time = parse(Float64, m_nom.captures[2])
        end

        # OMP line: x=[...] (마지막 iteration 값 유지)
        m_omp = match(r"OMP:.*x=\[([0-9,\s]+)\]", line)
        if m_omp !== nothing
            x_star_omp = parse.(Int, split(m_omp.captures[1], r"[,\s]+"; keepempty=false))
        end

        # Summary x* = [...]
        if !isnan(Z0) && x_star_summary === nothing
            m_x = match(r"^\s*x\*\s*=\s*\[([0-9,\s]+)\]", line)
            if m_x !== nothing
                x_star_summary = parse.(Int, split(m_x.captures[1], r"[,\s]+"; keepempty=false))
            end
        end
    end

    # x* 결정: summary가 all-zeros이면 last OMP 사용
    x_star = x_star_summary
    x_source = :summary
    if x_star !== nothing && all(x_star .== 0) && x_star_omp !== nothing
        x_star = x_star_omp
        x_source = :last_omp
    end
    if x_star === nothing && x_star_omp !== nothing
        x_star = x_star_omp
        x_source = :last_omp
    end

    return Dict(:x_star => x_star, :x_source => x_source, :Z0 => Z0,
                :status => status, :variant => variant,
                :iters => iters, :wall_time => wall_time)
end

function find_log_wmax(base_dir::String, net_key::Symbol, suffix::String)
    !isdir(base_dir) && return nothing
    pattern_prefix = "log_$(net_key)_"
    files = filter(readdir(base_dir)) do f
        startswith(f, pattern_prefix) && endswith(f, "$(suffix).txt")
    end
    isempty(files) && return nothing
    sort!(files)
    return joinpath(base_dir, files[end])
end

"""
    load_nominal_x(net_key; S=20) → (x_neut::Vector{Float64}, Z0::Float64)

S20_nominal_wmax/ 로그에서 nominal x* 파싱.
"""
function load_nominal_x(net_key::Symbol; S::Int=EXP2_S)
    base_dir = joinpath(@__DIR__, "S$(S)_nominal_wmax")
    log_path = find_log_wmax(base_dir, net_key, "nominal_wmax")
    if log_path === nothing
        error("Nominal log not found for $net_key in $base_dir")
    end
    parsed = parse_log_wmax(log_path)
    if parsed[:x_star] === nothing
        error("Could not parse x* from $log_path")
    end
    x_neut = Float64.(parsed[:x_star])
    Z0 = parsed[:Z0]
    @printf("  Loaded x_neut from log: Z₀=%.6f, Σx=%d, source=%s\n",
            Z0, sum(x_neut), parsed[:x_source])
    println("  x_neut = $(round.(Int, x_neut))")
    return x_neut, Z0
end

"""
    load_or_solve_robust_x(td, net_key; S, kwargs...) → (x_rob, Z0, source)

S20_robust_wmax/ 로그가 있으면 로드, 없으면 Benders solve 후 로그 저장.
"""
function load_or_solve_robust_x(td::TrueDROData, net_key::Symbol;
        S::Int=EXP2_S,
        sub_time_limit::Float64=30.0,
        max_iter::Int=500,
        verbose::Bool=true)

    base_dir = joinpath(@__DIR__, "S$(S)_robust_wmax")
    log_path = find_log_wmax(base_dir, net_key, "robust_wmax")

    if log_path !== nothing
        parsed = parse_log_wmax(log_path)
        if parsed[:x_star] !== nothing
            x_rob = Float64.(parsed[:x_star])
            Z0 = parsed[:Z0]
            @printf("  Loaded x_rob from log: Z₀=%.6f, Σx=%d, source=%s\n",
                    Z0, sum(x_rob), parsed[:x_source])
            println("  x_rob = $(round.(Int, x_rob))")
            return x_rob, Z0, :log
        end
    end

    # 로그 없음 → Benders solve
    @printf("  No robust log found → solving DRO (ε=1.0)...\n")
    mkpath(base_dir)

    log_timestamp = Dates.format(now(), "yyyymmdd_HHMMSSs")
    log_filename = joinpath(base_dir, "log_$(net_key)_S$(S)_$(log_timestamp)_robust_wmax.txt")
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

    try
        println("Log file: $log_filename")
        println("Started: $(now())")
        println(@sprintf("Robust DRO: %s (S=%d, ε̂=%.2f, ε̃=%.2f)",
                net_key, S, td.eps_hat, td.eps_tilde))
        println()

        t0 = time()
        res = solve_dro(td; sub_time_limit=sub_time_limit,
                         max_iter=max_iter, verbose=verbose)
        wt = time() - t0

        x_rob = res[:x]
        Z0 = res[:Z0]
        x_int = round.(Int, x_rob)

        @printf("True-DRO: status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n",
                res[:status], Z0, res[:iters], wt)
        println("  x* = $x_int")
        println("\nFinished: $(now())")

        return x_rob, Z0, :solved
    catch err
        println("\nERROR: $err")
        bt = catch_backtrace()
        Base.showerror(stdout, err, bt)
        rethrow()
    finally
        redirect_stdout(original_stdout)
        close(wr)
        try; wait(log_task); catch; end
        close(log_io)
        println("  Robust log saved → $log_filename")
    end
end


# ===== Config =====
const EXP2_NETWORKS = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls]

const EXP2_NET_CONFIGS = Dict(
    :grid_5x5    => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us    => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene     => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska      => Dict(:type => :real_world, :generator => generate_polska_network),
)

const EXP2_S = 20
const EXP2_GAMMA_RATIO = 0.10
const EXP2_SEED = 42
const EXP2_LAMBDA_U = 2.0

function exp2_compute_interdict_budget(config_key::Symbol, num_interdictable::Int, γ_ratio::Float64)
    if config_key in (:sioux_falls, :abilene)
        return 2
    end
    return ceil(Int, γ_ratio * num_interdictable)
end

function exp2_regenerate_network(config_key::Symbol; S::Int=EXP2_S)
    config = EXP2_NET_CONFIGS[config_key]
    network = config[:type] == :grid ?
        generate_grid_network(config[:m], config[:n]; seed=EXP2_SEED) :
        config[:generator]()

    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = exp2_compute_interdict_budget(config_key, num_interdictable, EXP2_GAMMA_RATIO)

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=EXP2_SEED)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    w = round(maximum(capacities[interdictable_idx, :]); digits=4)

    return network, capacities, w, γ
end


# ============================================================
# 1. build_evaluate_f_model — subproblem을 한 번 build
# ============================================================

"""
    build_evaluate_f_model(td; optimizer, nonconvex_attr) → (model, vars, builder_type)

ε에 맞는 subproblem model을 한 번 build하여 반환.
- ε̂=ε̃=0: nominal (pure LP)
- ε̃=0:   single-layer compact
- else:   full bilinear (NonConvex=2)
"""
function build_evaluate_f_model(td::TrueDROData;
        optimizer=Gurobi.Optimizer,
        nonconvex_attr=("NonConvex" => 2))

    K = td.num_arcs
    x_dummy = zeros(K)  # dummy x for initial build

    if td.eps_hat == 0.0 && td.eps_tilde == 0.0
        model, vars = build_true_dro_subproblem_nominal(td, x_dummy;
            optimizer=optimizer_with_attributes(optimizer, MOI.Silent() => true),
            rho_upper_bound=10.0)
        return model, vars, :nominal
    elseif td.eps_tilde == 0.0
        model, vars = build_true_dro_subproblem_single(td, x_dummy;
            optimizer=optimizer_with_attributes(optimizer, MOI.Silent() => true,
                                                nonconvex_attr),
            rho_upper_bound=10.0)
        return model, vars, :single
    else
        model, vars = build_true_dro_subproblem(td, x_dummy;
            optimizer=optimizer_with_attributes(optimizer, MOI.Silent() => true,
                                                nonconvex_attr),
            rho_upper_bound=10.0)
        return model, vars, :full
    end
end


# ============================================================
# 2. evaluate_f — f_ε(x)를 계산
# ============================================================

"""
    evaluate_f(model, vars, td, x; sub_time_limit=3600.0, mip_gap=0.005) → Float64

기존 solve_true_dro_subproblem!을 호출하여 f_ε(x) = Z0_val 반환.
MIPGap 0.5%, TimeLimit 3600s 기본. TIME_LIMIT 시 incumbent 사용 + 경고.
"""
function evaluate_f(model, vars, td::TrueDROData, x::Vector{Float64};
        sub_time_limit::Float64=3600.0,
        mip_gap::Float64=0.005)

    set_optimizer_attribute(model, "TimeLimit", sub_time_limit)
    set_optimizer_attribute(model, "MIPGap", mip_gap)

    sub_info = solve_true_dro_subproblem!(model, vars, td, x)

    if !sub_info[:is_optimal]
        @printf("  ⚠ evaluate_f: TIME_LIMIT, Z0_val=%.6f, Z0_bound=%.6f, gap=%.2e\n",
                sub_info[:Z0_val], sub_info[:Z0_bound],
                abs(sub_info[:Z0_val] - sub_info[:Z0_bound]) / max(abs(sub_info[:Z0_val]), 1e-10))
    end

    return sub_info[:Z0_val]
end


# ============================================================
# 3. solve_dro — Benders로 x*(ε) solve
# ============================================================

"""
    solve_dro(td; kwargs...) → Dict(:x, :α, :Z0, :status, :iters, :wall_time)

true_dro_benders_optimize!를 호출하여 x*(ε) 산출.
"""
function solve_dro(td::TrueDROData;
        sub_time_limit::Float64=30.0,
        max_iter::Int=500,
        verbose::Bool=true)

    mip_opt = Gurobi.Optimizer
    nlp_opt = Gurobi.Optimizer
    lp_opt  = Gurobi.Optimizer

    result = true_dro_benders_optimize!(td;
        mip_optimizer = mip_opt,
        nlp_optimizer = nlp_opt,
        lp_optimizer  = lp_opt,
        max_iter = max_iter,
        tol = 1e-4,
        verbose = verbose,
        sub_time_limit = sub_time_limit,
        mini_benders = true,
        max_mini_benders_iter = 5,
        strengthen_cuts = :mw,
        valid_inequality = :mincut,
        inexact = true,
        nonconvex_attr = ("NonConvex" => 2))

    return Dict(
        :x      => result[:x],
        :α      => result[:α],
        :Z0     => result[:Z0],
        :status => result[:status],
        :iters  => result[:iters],
        :wall_time => result[:wall_time],
    )
end


# ============================================================
# 4. find_epsilon_range — Combined bisection
# ============================================================

"""
    find_epsilon_range(network, capacities, q_hat, w, γ; kwargs...) → Dict

Combined bisection으로 [ε^S, ε^D] 구간을 찾는다.

Returns Dict with:
  :eps_recommended  — bisection으로 찾은 ε (ε^S ≤ ε ≤ ε^D 구간 내)
  :eps_lo, :eps_hi  — final bisection bounds
  :region           — :in_range, :below_eps_S, :above_eps_D
  :history          — Vector of per-iteration Dicts
  :x_neut, :x_rob   — nominal/robust solutions
  :f0_neut, :f1_rob  — baseline values
  :dro_solves        — Dict(ε => solve result)
"""
function find_epsilon_range(network, capacities, q_hat, w, γ, net_key::Symbol;
        eps_max::Float64 = 1.0,
        lambda_U::Float64 = EXP2_LAMBDA_U,
        tol::Float64 = 0.01,
        max_iter::Int = 15,
        benders_sub_time_limit::Float64 = 30.0,
        benders_max_iter::Int = 500,
        verbose::Bool = true)

    S = size(capacities, 2)
    num_arcs = length(network.arcs) - 1

    # ---- Step 1: x_neut (ε=0) from log, x_rob (ε=ε_max) from DRO ----
    if verbose
        println("=" ^ 70)
        println("Step 1: Load x_neut (ε=0) from log, solve x_rob (ε=$(eps_max))")
        println("=" ^ 70)
    end

    # x_neut: load from S20_nominal_wmax/ log (이미 풀어놓은 결과)
    td_0 = make_true_dro_data(network, capacities, q_hat, 0.0, 0.0;
                               w=w, lambda_U=lambda_U, gamma=γ)

    if verbose; println("\n--- Loading x_neut (ε=0) from log ---"); end
    x_neut, Z0_neut_log = load_nominal_x(net_key; S=S)

    # x_rob: load from S20_robust_wmax/ or solve & save
    td_1 = make_true_dro_data(network, capacities, q_hat, eps_max, eps_max;
                               w=w, lambda_U=lambda_U, gamma=γ)

    if verbose; println("\n--- Loading/solving x_rob (ε=$(eps_max)) ---"); end
    x_rob, Z0_rob, rob_source = load_or_solve_robust_x(td_1, net_key;
        S=S, sub_time_limit=benders_sub_time_limit,
        max_iter=benders_max_iter, verbose=verbose)

    # Baseline evaluations: f_0(x_neut), f_1(x_rob) — 상수
    # f_0(x_neut): subproblem at ε=0, evaluate x_neut
    model_0, vars_0, _ = build_evaluate_f_model(td_0)
    f0_neut = evaluate_f(model_0, vars_0, td_0, x_neut)

    # f_1(x_rob): subproblem at ε=1.0, evaluate x_rob
    model_1, vars_1, _ = build_evaluate_f_model(td_1)
    f1_rob = evaluate_f(model_1, vars_1, td_1, x_rob)

    if verbose
        @printf("\nBaselines: f_0(x_neut)=%.6f, f_1(x_rob)=%.6f\n", f0_neut, f1_rob)
    end

    # ---- Step 2: Bisection loop ----
    if verbose
        println("\n" * "=" ^ 70)
        println("Step 2: Bisection (tol=$(tol), max_iter=$(max_iter))")
        println("=" ^ 70)
    end

    lo = 0.0
    hi = eps_max
    ε = (lo + hi) / 2.0

    history = Vector{Dict}()
    dro_solves = Dict{Float64, Dict}()
    found_region = :max_iter_reached
    eps_recommended = ε

    # Checkpoint 파일: 중단 후 재개용
    checkpoint_dir = joinpath(@__DIR__, "exp2_checkpoints")
    mkpath(checkpoint_dir)
    checkpoint_path = joinpath(checkpoint_dir, "checkpoint_$(net_key).jls")

    # 기존 checkpoint 있으면 로드하여 이어서 진행
    if isfile(checkpoint_path)
        ckpt = deserialize(checkpoint_path)
        lo = ckpt[:lo]
        hi = ckpt[:hi]
        ε = ckpt[:next_eps]
        history = ckpt[:history]
        dro_solves = ckpt[:dro_solves]
        start_iter = length(history) + 1
        if verbose
            @printf("  Checkpoint loaded: %d iters done, lo=%.4f, hi=%.4f, next ε=%.4f\n",
                    length(history), lo, hi, ε)
        end
    else
        start_iter = 1
    end

    function _save_checkpoint()
        ckpt = Dict(
            :net_key    => net_key,
            :lo         => lo,
            :hi         => hi,
            :next_eps   => ε,
            :history    => history,
            :dro_solves => dro_solves,
            :x_neut     => x_neut,
            :x_rob      => x_rob,
            :f0_neut    => f0_neut,
            :f1_rob     => f1_rob,
        )
        serialize(checkpoint_path, ckpt)
    end

    for iter in start_iter:max_iter
        if verbose
            @printf("\n--- Bisection iter %d: ε=%.4f [lo=%.4f, hi=%.4f] ---\n",
                    iter, ε, lo, hi)
        end

        # Build subproblem at current ε
        td_ε = make_true_dro_data(network, capacities, q_hat, ε, ε;
                                   w=w, lambda_U=lambda_U, gamma=γ)
        model_ε, vars_ε, _ = build_evaluate_f_model(td_ε)

        # PO - PP (싸다: evaluate_f만)
        f_ε_neut = evaluate_f(model_ε, vars_ε, td_ε, x_neut)
        f_ε_rob  = evaluate_f(model_ε, vars_ε, td_ε, x_rob)
        po_pp = f_ε_neut - f_ε_rob

        iter_info = Dict(
            :iter     => iter,
            :eps      => ε,
            :lo       => lo,
            :hi       => hi,
            :f_eps_neut => f_ε_neut,
            :f_eps_rob  => f_ε_rob,
            :po_pp    => po_pp,
            :nr       => NaN,
            :wr       => NaN,
            :nr_wr    => NaN,
            :region   => :unknown,
        )

        if verbose
            @printf("  f_ε(x_neut)=%.6f, f_ε(x_rob)=%.6f, PO-PP=%.6f\n",
                    f_ε_neut, f_ε_rob, po_pp)
        end

        if po_pp < 0
            # ε < ε^S → lo = ε, increase ε
            iter_info[:region] = :below_eps_S
            lo = ε
            ε = (ε + hi) / 2.0

            if verbose
                @printf("  PO-PP < 0 → ε < ε^S → lo=%.4f, next ε=%.4f\n", lo, ε)
            end

            push!(history, iter_info)
            _save_checkpoint()

            # 수렴 체크
            if (hi - lo) < tol
                found_region = :below_eps_S
                eps_recommended = ε
                break
            end
            continue
        end

        # PO-PP ≥ 0 → NR-WR 계산 필요 (비싸다: DRO solve)
        if verbose
            println("  PO-PP ≥ 0 → computing NR-WR (requires DRO solve)...")
        end

        # x*(ε) = solve DRO at current ε
        t0 = time()
        res_star = solve_dro(td_ε; sub_time_limit=benders_sub_time_limit,
                              max_iter=benders_max_iter, verbose=verbose)
        t_dro = time() - t0
        x_star = res_star[:x]
        dro_solves[ε] = res_star

        if verbose
            @printf("  x*(ε=%.4f) solved: Z₀=%.6f, iters=%d, time=%.1fs\n",
                    ε, res_star[:Z0], res_star[:iters], t_dro)
            println("  x* = $(round.(Int, x_star))")
        end

        # f_0(x*) and f_1(x*)
        f0_star = evaluate_f(model_0, vars_0, td_0, x_star)
        f1_star = evaluate_f(model_1, vars_1, td_1, x_star)

        nr = f0_star - f0_neut
        wr = f1_star - f1_rob
        nr_wr = nr - wr

        iter_info[:nr] = nr
        iter_info[:wr] = wr
        iter_info[:nr_wr] = nr_wr
        iter_info[:f0_star] = f0_star
        iter_info[:f1_star] = f1_star
        iter_info[:x_star] = round.(Int, x_star)

        if verbose
            @printf("  f_0(x*)=%.6f, f_1(x*)=%.6f\n", f0_star, f1_star)
            @printf("  NR=%.6f, WR=%.6f, NR-WR=%.6f\n", nr, wr, nr_wr)
        end

        if nr_wr < 0
            # ε ∈ [ε^S, ε^D] → STOP
            iter_info[:region] = :in_range
            found_region = :in_range
            eps_recommended = ε

            if verbose
                @printf("  NR-WR < 0 → ε=%.4f ∈ [ε^S, ε^D] → FOUND!\n", ε)
            end

            push!(history, iter_info)
            _save_checkpoint()
            break
        else
            # ε > ε^D → hi = ε, decrease ε
            iter_info[:region] = :above_eps_D
            hi = ε
            ε = (lo + ε) / 2.0

            if verbose
                @printf("  NR-WR ≥ 0 → ε > ε^D → hi=%.4f, next ε=%.4f\n", hi, ε)
            end
        end

        push!(history, iter_info)
        _save_checkpoint()

        # 수렴 체크
        if (hi - lo) < tol
            found_region = :converged
            eps_recommended = ε
            break
        end
    end

    # ---- ε_recommended에서 x* solve (아직 안 풀었으면) ----
    x_star_rec = nothing
    Z0_star_rec = NaN
    if !haskey(dro_solves, eps_recommended)
        if verbose
            @printf("\n--- Solving x*(ε_recommended=%.4f) ---\n", eps_recommended)
        end
        td_rec = make_true_dro_data(network, capacities, q_hat, eps_recommended, eps_recommended;
                                     w=w, lambda_U=lambda_U, gamma=γ)
        res_rec = solve_dro(td_rec; sub_time_limit=benders_sub_time_limit,
                             max_iter=benders_max_iter, verbose=verbose)
        dro_solves[eps_recommended] = res_rec
        x_star_rec = res_rec[:x]
        Z0_star_rec = res_rec[:Z0]
    else
        x_star_rec = dro_solves[eps_recommended][:x]
        Z0_star_rec = dro_solves[eps_recommended][:Z0]
    end
    if verbose
        @printf("  x*(ε_rec): Z₀=%.6f, x=%s\n", Z0_star_rec, string(round.(Int, x_star_rec)))
    end

    # Checkpoint 정리 (완료 시 삭제)
    if isfile(checkpoint_path)
        rm(checkpoint_path)
        if verbose; println("  Checkpoint removed (completed)"); end
    end

    # ---- Results ----
    result = Dict(
        :eps_recommended => eps_recommended,
        :eps_lo          => lo,
        :eps_hi          => hi,
        :region          => found_region,
        :history         => history,
        :x_neut          => x_neut,
        :x_rob           => x_rob,
        :x_star          => x_star_rec,
        :Z0_star         => Z0_star_rec,
        :f0_neut         => f0_neut,
        :f1_rob          => f1_rob,
        :Z0_neut_log     => Z0_neut_log,
        :Z0_rob          => Z0_rob,
        :rob_source      => rob_source,
        :dro_solves      => dro_solves,
    )

    return result
end


# ============================================================
# 5. print_bisection_summary — 결과 출력
# ============================================================

function print_bisection_summary(net_key::Symbol, result::Dict; betas=[0.1, 0.3, 0.5])
    println("\n" * "=" ^ 90)
    @printf("Experiment 2 Results: %s\n", net_key)
    println("=" ^ 90)

    # Bisection history table
    @printf("%-5s  %8s  %8s  %8s  %12s  %12s  %12s  %-12s\n",
            "Iter", "ε", "lo", "hi", "PO-PP", "NR-WR", "NR", "Region")
    println("-" ^ 90)
    for h in result[:history]
        nr_wr_str = isnan(h[:nr_wr]) ? "    ---" : @sprintf("%12.6f", h[:nr_wr])
        nr_str    = isnan(h[:nr])    ? "    ---" : @sprintf("%12.6f", h[:nr])
        @printf("%-5d  %8.4f  %8.4f  %8.4f  %12.6f  %s  %s  %-12s\n",
                h[:iter], h[:eps], h[:lo], h[:hi], h[:po_pp],
                nr_wr_str, nr_str, h[:region])
    end
    println("-" ^ 90)

    @printf("\nε_recommended = %.4f  (region: %s)\n",
            result[:eps_recommended], result[:region])
    @printf("ε bounds: [%.4f, %.4f]\n", result[:eps_lo], result[:eps_hi])
    @printf("f_0(x_neut)=%.6f, f_1(x_rob)=%.6f\n", result[:f0_neut], result[:f1_rob])
    println("x_neut = $(round.(Int, result[:x_neut]))")
    println("x_rob  = $(round.(Int, result[:x_rob]))")

    # Coverage-based ε 비교
    S = EXP2_S
    println("\n--- ε comparison: analytical vs coverage ---")
    @printf("%-6s  %10s  %10s  %10s\n", "β", "ε_coverage", "ε_analyt", "ratio")
    println("-" ^ 44)
    for β in betas
        ε_cov = round(lookup_epsilon(S, β; coverage=0.95); digits=4)
        ε_ana = result[:eps_recommended]
        ratio = ε_cov > 0 ? ε_ana / ε_cov : NaN
        @printf("%-6.2f  %10.4f  %10.4f  %10.4f\n", β, ε_cov, ε_ana, ratio)
    end
end


# ============================================================
# 6. Main script
# ============================================================

println("=" ^ 70)
println("Experiment 2: PO/PP/NR/WR Analytical ε Calibration")
println("  Networks: $EXP2_NETWORKS")
println("  S=$(EXP2_S), γ_ratio=$(EXP2_GAMMA_RATIO), λU=$(EXP2_LAMBDA_U)")
println("  Bisection: tol=0.01, max_iter=15")
println("=" ^ 70)

batch_start = time()
all_exp2_results = Dict{Symbol, Dict}()
summary_rows = []

for net_key in EXP2_NETWORKS
    println("\n\n" * "#" ^ 70)
    @printf("# NETWORK: %s\n", net_key)
    println("#" ^ 70)
    flush(stdout)

    network, capacities, w, γ = exp2_regenerate_network(net_key)
    num_arcs = length(network.arcs) - 1
    q_hat = fill(1.0 / EXP2_S, EXP2_S)

    @printf("  arcs=%d, γ=%d, w=%.4f\n", num_arcs, γ, w)

    t0 = time()
    result = find_epsilon_range(network, capacities, q_hat, w, γ, net_key;
        eps_max = 1.0,
        lambda_U = EXP2_LAMBDA_U,
        tol = 0.01,
        max_iter = 15,
        benders_sub_time_limit = 30.0,
        benders_max_iter = 500,
        verbose = true)
    t_net = time() - t0

    all_exp2_results[net_key] = result

    print_bisection_summary(net_key, result)
    @printf("\nTotal time for %s: %.1fs (%.1f min)\n", net_key, t_net, t_net / 60)

    # Summary row
    push!(summary_rows, (
        network = string(net_key),
        eps_recommended = result[:eps_recommended],
        eps_lo = result[:eps_lo],
        eps_hi = result[:eps_hi],
        region = string(result[:region]),
        n_iters = length(result[:history]),
        f0_neut = result[:f0_neut],
        f1_rob = result[:f1_rob],
        wall_time = t_net,
    ))

    # 네트워크별 즉시 저장
    serialize(joinpath(@__DIR__, "exp2_calibration_results.jls"), all_exp2_results)
    @printf("  Results saved (%d networks done)\n", length(all_exp2_results))

    flush(stdout)
end

batch_wall = time() - batch_start

# ---- Save results ----
jls_path = joinpath(@__DIR__, "exp2_calibration_results.jls")
serialize(jls_path, all_exp2_results)
println("\nRaw results saved → $jls_path")

csv_path = joinpath(@__DIR__, "exp2_calibration_summary.csv")
open(csv_path, "w") do io
    println(io, "network,eps_recommended,eps_lo,eps_hi,region,n_iters,f0_neut,f1_rob,wall_time")
    for r in summary_rows
        @printf(io, "%s,%.4f,%.4f,%.4f,%s,%d,%.6f,%.6f,%.1f\n",
                r.network, r.eps_recommended, r.eps_lo, r.eps_hi,
                r.region, r.n_iters, r.f0_neut, r.f1_rob, r.wall_time)
    end
end
println("Summary CSV saved → $csv_path")


# ---- Final summary table ----
println("\n\n" * "=" ^ 100)
println("Experiment 2 — Final Summary")
println("=" ^ 100)
@printf("%-14s  %10s  %8s  %8s  %-14s  %6s  %10s  %10s  %10s\n",
        "Network", "ε_recommend", "ε_lo", "ε_hi", "Region", "Iters",
        "f0(x_neut)", "f1(x_rob)", "Time(s)")
println("-" ^ 100)
for r in summary_rows
    @printf("%-14s  %10.4f  %8.4f  %8.4f  %-14s  %6d  %10.4f  %10.4f  %10.1f\n",
            r.network, r.eps_recommended, r.eps_lo, r.eps_hi,
            r.region, r.n_iters, r.f0_neut, r.f1_rob, r.wall_time)
end
println("-" ^ 100)
@printf("Total wall time: %.1f s (%.1f min)\n", batch_wall, batch_wall / 60)

# Per-network ε comparison
println("\n--- ε_recommended vs ε_coverage (95%) ---")
@printf("%-14s  %10s  %10s  %10s  %10s\n",
        "Network", "ε_analyt", "ε_cov(β=.1)", "ε_cov(β=.3)", "ε_cov(β=.5)")
println("-" ^ 60)
for r in summary_rows
    ε_01 = round(lookup_epsilon(EXP2_S, 0.1; coverage=0.95); digits=4)
    ε_03 = round(lookup_epsilon(EXP2_S, 0.3; coverage=0.95); digits=4)
    ε_05 = round(lookup_epsilon(EXP2_S, 0.5; coverage=0.95); digits=4)
    @printf("%-14s  %10.4f  %10.4f  %10.4f  %10.4f\n",
            r.network, r.eps_recommended, ε_01, ε_03, ε_05)
end
println("-" ^ 60)

println("\nDone! $(now())")
