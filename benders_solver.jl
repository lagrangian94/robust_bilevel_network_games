"""
Nested Benders Decomposition Solvers Module

객체지향적으로 구조화된 Nested Benders 알고리즘 구현.
- NestedBendersSolver: 기본 Nested Benders Decomposition
- TRNestedBendersSolver: Trust Region 안정화가 추가된 버전

Author: Seokwoo
"""

module NestedBendersSolvers

using JuMP
using LinearAlgebra
using SparseArrays
using Infiltrator

export AbstractNestedBendersSolver, NestedBendersSolver, TRNestedBendersSolver
export initialize!, solve!, get_result, SolverConfig, TrustRegionConfig, SolverHistory, SolverResult, ModelInstances

# ============================================================================
# Abstract Base Type
# ============================================================================
"""
    AbstractNestedBendersSolver

모든 Nested Benders Solver의 추상 베이스 타입.
"""
abstract type AbstractNestedBendersSolver end

# ============================================================================
# Configuration Structs
# ============================================================================
"""
    SolverConfig

Solver의 기본 설정을 담는 구조체.
"""
Base.@kwdef struct SolverConfig
    max_iter::Int = 1000
    tol::Float64 = 1e-4
    verbose::Bool = true
    multi_cut::Bool = false
end

"""
    TrustRegionConfig

Trust Region 관련 설정을 담는 구조체.
"""
Base.@kwdef struct TrustRegionConfig
    B_bin_sequence::Vector{Float64} = [0.05, 0.5, 1.0]
    B_con_max::Union{Float64, Nothing} = nothing
    β_relative::Float64 = 1e-4  # Serious step threshold
end

# ============================================================================
# History/Result Structs
# ============================================================================
"""
    SolverHistory

최적화 과정의 히스토리를 저장하는 구조체.
"""
mutable struct SolverHistory
    past_obj::Vector{Float64}
    past_subprob_obj::Vector{Float64}
    past_upper_bound::Vector{Float64}
    past_lower_bound::Vector{Float64}
    inner_iter::Vector{Int}
    cuts::Dict{String, Any}
    
    function SolverHistory()
        new(Float64[], Float64[], Float64[], Float64[], Int[], Dict{String, Any}())
    end
end

"""
    TRSolverHistory

Trust Region Solver의 추가 히스토리를 저장하는 구조체.
"""
mutable struct TRSolverHistory
    base::SolverHistory
    past_major_subprob_obj::Vector{Float64}
    past_minor_subprob_obj::Vector{Float64}
    past_model_estimate::Vector{Float64}
    past_local_lower_bound::Vector{Float64}
    past_local_optimizer::Vector{Dict{Symbol, Any}}
    major_iter::Vector{Int}
    bin_B_steps::Vector{Int}
    tr_info::Dict{Symbol, Any}
    
    function TRSolverHistory()
        new(
            SolverHistory(),
            Float64[], Float64[], Float64[], Float64[],
            Dict{Symbol, Any}[], Int[], Int[],
            Dict{Symbol, Any}()
        )
    end
end

"""
    SolverResult

최적화 결과를 담는 구조체.
"""
mutable struct SolverResult
    solution_time::Float64
    optimal_value::Float64
    x_sol::Union{Vector{Float64}, Nothing}
    h_sol::Union{Vector{Float64}, Nothing}
    λ_sol::Union{Float64, Nothing}
    ψ0_sol::Union{Vector{Float64}, Nothing}
    α_sol::Union{Vector{Float64}, Nothing}
    status::Symbol
    history::Union{SolverHistory, TRSolverHistory}
    
    function SolverResult(history::Union{SolverHistory, TRSolverHistory})
        new(0.0, Inf, nothing, nothing, nothing, nothing, nothing, :NotSolved, history)
    end
end

# ============================================================================
# Model Instances Container
# ============================================================================
"""
    ModelInstances

OMP, IMP, ISP 모델 인스턴스들을 담는 구조체.
"""
mutable struct ModelInstances
    # Outer Master Problem
    omp_model::Union{Model, Nothing}
    omp_vars::Union{Dict{Symbol, Any}, Nothing}
    
    # Inner Master Problem  
    imp_model::Union{Model, Nothing}
    imp_vars::Union{Dict{Symbol, Any}, Nothing}
    
    # Inner SubProblem instances (per scenario)
    isp_leader_instances::Union{Dict{Int, Tuple{Model, Dict}}, Nothing}
    isp_follower_instances::Union{Dict{Int, Tuple{Model, Dict}}, Nothing}
    
    # Shared data
    isp_data::Union{Dict{Symbol, Any}, Nothing}
    
    function ModelInstances()
        new(nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

# ============================================================================
# NestedBendersSolver
# ============================================================================
"""
    NestedBendersSolver

기본 Nested Benders Decomposition Solver.

# Fields
- `config`: Solver 설정
- `problem_data`: 문제 데이터 (network, parameters 등)
- `models`: 모델 인스턴스들
- `history`: 최적화 히스토리
- `result`: 최적화 결과
"""
mutable struct NestedBendersSolver <: AbstractNestedBendersSolver
    config::SolverConfig
    problem_data::Dict{Symbol, Any}
    models::ModelInstances
    history::SolverHistory
    result::SolverResult
    
    # Internal state
    upper_bound::Float64
    iter::Int
    imp_cuts::Dict{Symbol, Any}
    
    function NestedBendersSolver(;
        network,
        S::Int,
        ϕU::Float64,
        λU::Float64,
        γ::Float64,
        w::Float64,
        v::Float64,
        uncertainty_set::Dict,
        config::SolverConfig = SolverConfig()
    )
        problem_data = Dict{Symbol, Any}(
            :network => network,
            :S => S,
            :ϕU => ϕU,
            :λU => λU,
            :γ => γ,
            :w => w,
            :v => v,
            :uncertainty_set => uncertainty_set
        )
        
        history = SolverHistory()
        result = SolverResult(history)
        models = ModelInstances()
        imp_cuts = Dict{Symbol, Any}()
        
        new(config, problem_data, models, history, result, Inf, 0, imp_cuts)
    end
end

# ============================================================================
# TRNestedBendersSolver (Trust Region)
# ============================================================================
"""
    TRNestedBendersSolver

Trust Region 안정화가 추가된 Nested Benders Decomposition Solver.

# Fields
- `config`: Solver 설정
- `tr_config`: Trust Region 설정
- `problem_data`: 문제 데이터
- `models`: 모델 인스턴스들
- `history`: 최적화 히스토리
- `result`: 최적화 결과
- Trust region specific fields
"""
mutable struct TRNestedBendersSolver <: AbstractNestedBendersSolver
    config::SolverConfig
    tr_config::TrustRegionConfig
    problem_data::Dict{Symbol, Any}
    models::ModelInstances
    history::TRSolverHistory
    result::SolverResult
    
    # Internal state
    upper_bound::Float64
    lower_bound::Float64
    iter::Int
    imp_cuts::Dict{Symbol, Any}
    
    # Trust region state
    B_bin::Float64
    B_bin_stage::Int
    B_con::Union{Float64, Nothing}
    centers::Dict{Symbol, Any}
    tr_constraints::Dict{Symbol, Any}
    
    function TRNestedBendersSolver(;
        network,
        S::Int,
        ϕU::Float64,
        λU::Float64,
        γ::Float64,
        w::Float64,
        v::Float64,
        uncertainty_set::Dict,
        config::SolverConfig = SolverConfig(),
        tr_config::TrustRegionConfig = TrustRegionConfig()
    )
        problem_data = Dict{Symbol, Any}(
            :network => network,
            :S => S,
            :ϕU => ϕU,
            :λU => λU,
            :γ => γ,
            :w => w,
            :v => v,
            :uncertainty_set => uncertainty_set
        )
        
        history = TRSolverHistory()
        result = SolverResult(history)
        models = ModelInstances()
        imp_cuts = Dict{Symbol, Any}()
        
        # Trust region 초기화
        num_interdictable = sum(network.interdictable_arcs)
        B_bin = tr_config.B_bin_sequence[1] * num_interdictable
        B_bin_stage = 1
        B_con = tr_config.B_con_max
        
        centers = Dict{Symbol, Any}(
            :x => nothing,
            :h => nothing,
            :λ => nothing,
            :ψ0 => nothing
        )
        
        tr_constraints = Dict{Symbol, Any}(
            :binary => nothing,
            :continuous => nothing
        )
        
        new(
            config, tr_config, problem_data, models, history, result,
            Inf, -Inf, 0, imp_cuts,
            B_bin, B_bin_stage, B_con, centers, tr_constraints
        )
    end
end

# ============================================================================
# Interface Methods - Initialize
# ============================================================================
"""
    initialize!(solver::AbstractNestedBendersSolver; mip_optimizer, conic_optimizer)

Solver를 초기화하고 모든 모델 인스턴스를 생성합니다.
"""
function initialize!(solver::NestedBendersSolver; 
                     mip_optimizer=nothing, 
                     conic_optimizer=nothing,
                     omp_model::Model=nothing,
                     omp_vars::Dict=nothing)
    pd = solver.problem_data
    
    # OMP 모델이 외부에서 제공된 경우 사용
    if omp_model !== nothing && omp_vars !== nothing
        solver.models.omp_model = omp_model
        solver.models.omp_vars = omp_vars
    else
        error("OMP model must be provided externally via build_omp()")
    end
    
    # OMP 초기 해 구하기
    st, λ_sol, x_sol, h_sol, ψ0_sol = _initialize_omp(solver.models.omp_model, solver.models.omp_vars)
    
    # IMP 빌드
    solver.models.imp_model, solver.models.imp_vars = _build_imp(
        pd[:network], pd[:S], pd[:ϕU], pd[:λU], pd[:γ], pd[:w], pd[:v], 
        pd[:uncertainty_set]; mip_optimizer=mip_optimizer
    )
    
    # IMP 초기화
    st, α_sol = _initialize_imp(solver.models.imp_model, solver.models.imp_vars)
    
    # ISP 인스턴스들 초기화
    solver.models.isp_leader_instances, solver.models.isp_follower_instances = _initialize_isp(
        pd[:network], pd[:S], pd[:ϕU], pd[:λU], pd[:γ], pd[:w], pd[:v], 
        pd[:uncertainty_set];
        conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol
    )
    
    # ISP 공유 데이터 설정
    num_arcs = length(pd[:network].arcs) - 1
    E = ones(num_arcs, num_arcs + 1)
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    
    solver.models.isp_data = Dict{Symbol, Any}(
        :E => E,
        :network => pd[:network],
        :ϕU => pd[:ϕU],
        :λU => pd[:λU],
        :γ => pd[:γ],
        :w => pd[:w],
        :v => pd[:v],
        :uncertainty_set => pd[:uncertainty_set],
        :d0 => d0,
        :S => pd[:S]
    )
    
    solver.config.verbose && @info "NestedBendersSolver initialized"
    return solver
end

function initialize!(solver::TRNestedBendersSolver;
                     mip_optimizer=nothing,
                     conic_optimizer=nothing,
                     omp_model::Model=nothing,
                     omp_vars::Dict=nothing)
    pd = solver.problem_data
    
    # OMP 모델이 외부에서 제공된 경우 사용
    if omp_model !== nothing && omp_vars !== nothing
        solver.models.omp_model = omp_model
        solver.models.omp_vars = omp_vars
    else
        error("OMP model must be provided externally via build_omp()")
    end
    
    # OMP 초기 해 구하기
    st, λ_sol, x_sol, h_sol, ψ0_sol = _initialize_omp(solver.models.omp_model, solver.models.omp_vars)
    
    # Stability centers 초기화
    solver.centers[:x] = x_sol
    solver.centers[:h] = h_sol
    solver.centers[:λ] = λ_sol
    solver.centers[:ψ0] = ψ0_sol
    
    # IMP 빌드
    solver.models.imp_model, solver.models.imp_vars = _build_imp(
        pd[:network], pd[:S], pd[:ϕU], pd[:λU], pd[:γ], pd[:w], pd[:v],
        pd[:uncertainty_set]; mip_optimizer=mip_optimizer
    )
    
    # IMP 초기화
    st, α_sol = _initialize_imp(solver.models.imp_model, solver.models.imp_vars)
    
    # ISP 인스턴스들 초기화
    solver.models.isp_leader_instances, solver.models.isp_follower_instances = _initialize_isp(
        pd[:network], pd[:S], pd[:ϕU], pd[:λU], pd[:γ], pd[:w], pd[:v],
        pd[:uncertainty_set];
        conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol
    )
    
    # ISP 공유 데이터 설정
    num_arcs = length(pd[:network].arcs) - 1
    E = ones(num_arcs, num_arcs + 1)
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    
    solver.models.isp_data = Dict{Symbol, Any}(
        :E => E,
        :network => pd[:network],
        :ϕU => pd[:ϕU],
        :λU => pd[:λU],
        :γ => pd[:γ],
        :w => pd[:w],
        :v => pd[:v],
        :uncertainty_set => pd[:uncertainty_set],
        :d0 => d0,
        :S => pd[:S]
    )
    
    solver.config.verbose && @info "TRNestedBendersSolver initialized"
    return solver
end

# ============================================================================
# Interface Methods - Solve
# ============================================================================
"""
    solve!(solver::AbstractNestedBendersSolver)

Nested Benders 알고리즘을 실행합니다.
"""
function solve!(solver::NestedBendersSolver)
    time_start = time()
    pd = solver.problem_data
    models = solver.models
    config = solver.config
    history = solver.history
    
    omp_model = models.omp_model
    omp_vars = models.omp_vars
    
    # t_0 변수 참조
    t_0 = config.multi_cut ? (omp_vars[:t_0_l] + omp_vars[:t_0_f]) : omp_vars[:t_0]
    
    st = MOI.OPTIMAL  # 초기 상태
    
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        solver.iter += 1
        config.verbose && @info "Iteration $(solver.iter)"
        
        # Outer Master Problem 풀기
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())
        
        x_sol = value.(omp_vars[:x])
        h_sol = value.(omp_vars[:h])
        λ_sol = value(omp_vars[:λ])
        ψ0_sol = value.(omp_vars[:ψ0])
        t_0_sol = value(t_0)
        
        # Inner Benders (IMP + ISP) 풀기
        status, cut_info = _imp_optimize!(
            solver;
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
            outer_iter=solver.iter
        )
        
        push!(history.inner_iter, cut_info[:iter])
        solver.imp_cuts[:old_cuts] = cut_info[:cuts]
        solver.upper_bound = min(solver.upper_bound, cut_info[:obj_val])
        
        if status == :OptimalityCut
            # 종료 조건 체크
            if t_0_sol >= solver.upper_bound - config.tol
                time_end = time()
                solver.result.solution_time = time_end - time_start
                config.verbose && @info "Termination condition met"
                
                push!(history.past_obj, t_0_sol)
                push!(history.past_subprob_obj, cut_info[:obj_val])
                push!(history.past_upper_bound, solver.upper_bound)
                
                # 결과 저장
                solver.result.optimal_value = t_0_sol
                solver.result.x_sol = x_sol
                solver.result.h_sol = h_sol
                solver.result.λ_sol = λ_sol
                solver.result.ψ0_sol = ψ0_sol
                solver.result.status = :Optimal
                
                return solver.result
            end
            
            # Cut 추가
            _add_outer_cut!(solver, cut_info, solver.iter)
            
            # 히스토리 업데이트
            push!(history.past_obj, t_0_sol)
            push!(history.past_subprob_obj, cut_info[:obj_val])
            push!(history.past_upper_bound, solver.upper_bound)
        end
        
        # Maximum iteration 체크
        if solver.iter >= config.max_iter
            config.verbose && @warn "Maximum iterations reached"
            solver.result.status = :MaxIterations
            break
        end
    end
    
    solver.result.solution_time = time() - time_start
    return solver.result
end

function solve!(solver::TRNestedBendersSolver)
    time_start = time()
    pd = solver.problem_data
    models = solver.models
    config = solver.config
    tr_config = solver.tr_config
    history = solver.history
    network = pd[:network]
    
    omp_model = models.omp_model
    omp_vars = models.omp_vars
    
    # t_0 변수 참조
    t_0 = config.multi_cut ? (omp_vars[:t_0_l] + omp_vars[:t_0_f]) : omp_vars[:t_0]
    
    st = MOI.OPTIMAL
    gap = Inf
    
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL) && gap > config.tol
        solver.iter += 1
        config.verbose && @info "Outer Iteration $(solver.iter) (B_bin=$(solver.B_bin), Stage=$(solver.B_bin_stage)/$(length(tr_config.B_bin_sequence)))"
        
        # Outer Master Problem 풀기
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())
        
        if st == MOI.INFEASIBLE
            config.verbose && @info "Outer Master Problem infeasible: No search space left"
            gap = 0.0
            continue
        end
        
        x_sol = value.(omp_vars[:x])
        h_sol = value.(omp_vars[:h])
        λ_sol = value(omp_vars[:λ])
        ψ0_sol = value.(omp_vars[:ψ0])
        model_estimate = value(t_0)
        
        solver.lower_bound = max(solver.lower_bound, model_estimate)
        
        # Inner Benders 풀기
        status, cut_info = _imp_optimize!(
            solver;
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
            outer_iter=solver.iter
        )
        
        if status != :OptimalityCut
            @warn "Outer Subproblem not optimal"
            @infiltrate
        end
        
        push!(history.base.inner_iter, cut_info[:iter])
        solver.imp_cuts[:old_cuts] = cut_info[:cuts]
        if haskey(cut_info, :tr_constraints) && cut_info[:tr_constraints] !== nothing
            solver.imp_cuts[:old_tr_constraints] = cut_info[:tr_constraints]
        end
        
        subprob_obj = cut_info[:obj_val]
        solver.upper_bound = min(solver.upper_bound, subprob_obj)
        
        # Serious step test
        if solver.iter == 1
            push!(history.past_major_subprob_obj, subprob_obj)
        end
        
        gap = solver.upper_bound - solver.lower_bound
        tr_needs_update = false
        
        predicted_decrease = history.past_major_subprob_obj[end] - model_estimate
        β_dynamic = max(1e-8, tr_config.β_relative * predicted_decrease)
        improvement = history.past_major_subprob_obj[end] - subprob_obj
        is_serious_step = (improvement >= β_dynamic)
        
        if is_serious_step
            # Serious Step: Stability center 이동
            solver.centers[:x] = x_sol
            solver.centers[:h] = h_sol
            solver.centers[:λ] = λ_sol
            solver.centers[:ψ0] = ψ0_sol
            push!(history.major_iter, solver.iter)
            push!(history.past_major_subprob_obj, subprob_obj)
            tr_needs_update = true
        end
        
        # 히스토리 저장
        push!(history.base.past_lower_bound, solver.lower_bound)
        push!(history.past_model_estimate, model_estimate)
        push!(history.past_minor_subprob_obj, subprob_obj)
        push!(history.base.past_upper_bound, solver.upper_bound)
        
        # Gap 체크 및 Trust Region 확장
        if gap <= config.tol
            if solver.B_bin_stage < length(tr_config.B_bin_sequence)
                # Trust region 확장
                solver.B_bin_stage += 1
                B_bin_old = solver.B_bin
                solver.B_bin = tr_config.B_bin_sequence[solver.B_bin_stage] * sum(network.interdictable_arcs)
                push!(history.bin_B_steps, solver.iter)
                push!(history.past_local_lower_bound, solver.lower_bound)
                push!(history.past_local_optimizer, Dict(
                    :x => x_sol, :h => h_sol, :λ => λ_sol, :ψ0 => ψ0_sol
                ))
                
                config.verbose && @info "Local optimal reached! Expanding B_bin to $(solver.B_bin)"
                
                # Trust Region 제약 업데이트
                solver.tr_constraints = _update_outer_trust_region_constraints!(
                    omp_model, omp_vars, solver.centers, solver.B_bin, 
                    solver.B_con, solver.tr_constraints, network
                )
                
                # Reverse region constraint 추가
                _add_reverse_region_constraint!(omp_model, omp_vars[:x], solver.centers[:x], B_bin_old, network)
                
                solver.lower_bound = -Inf  # 영역 확장 후 리셋
                gap = Inf  # 계속 진행
            else
                # Global Optimality 달성
                time_end = time()
                solver.result.solution_time = time_end - time_start
                config.verbose && @info "GLOBAL OPTIMAL!"
                
                # 최적해 선택 (local optimizer들 중 최소)
                best_idx = argmin([opt[:obj] for opt in history.past_local_optimizer] ∪ [subprob_obj])
                
                solver.result.optimal_value = solver.upper_bound
                solver.result.x_sol = x_sol
                solver.result.h_sol = h_sol
                solver.result.λ_sol = λ_sol
                solver.result.ψ0_sol = ψ0_sol
                solver.result.status = :Optimal
                
                return solver.result
            end
        elseif tr_needs_update
            solver.tr_constraints = _update_outer_trust_region_constraints!(
                omp_model, omp_vars, solver.centers, solver.B_bin,
                solver.B_con, solver.tr_constraints, network
            )
        end
        
        # Cut 추가
        _add_outer_cut!(solver, cut_info, solver.iter)
        
        # Maximum iteration 체크
        if solver.iter >= config.max_iter
            config.verbose && @warn "Maximum iterations reached"
            solver.result.status = :MaxIterations
            break
        end
    end
    
    solver.result.solution_time = time() - time_start
    return solver.result
end

# ============================================================================
# Internal Helper Functions
# ============================================================================
function _initialize_omp(omp_model::Model, omp_vars::Dict)
    optimize!(omp_model)
    st = MOI.get(omp_model, MOI.TerminationStatus())
    @info "Initial OMP status: $st"
    return st, value(omp_vars[:λ]), value.(omp_vars[:x]), value.(omp_vars[:h]), value.(omp_vars[:ψ0])
end

function _build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=nothing)
    num_arcs = length(network.arcs) - 1
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    S = length(xi_bar)
    flow_upper = sum(sum(xi_bar[s]) for s in 1:S)
    
    model = Model(optimizer_with_attributes(mip_optimizer, MOI.Silent() => false))
    
    @variable(model, t_1_l[s=1:S], upper_bound=flow_upper)
    @variable(model, t_1_f[s=1:S], upper_bound=flow_upper)
    @variable(model, α[k=1:num_arcs] >= 0)
    @constraint(model, sum(α) == w * (1/S))
    @objective(model, Max, sum(t_1_l) + sum(t_1_f))
    
    vars = Dict(
        :t_1_l => t_1_l,
        :t_1_f => t_1_f,
        :α => α
    )
    return model, vars
end

function _initialize_imp(imp_model::Model, imp_vars::Dict)
    optimize!(imp_model)
    st = MOI.get(imp_model, MOI.TerminationStatus())
    α_sol = value.(imp_vars[:α])
    return st, α_sol
end

function _initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set; 
                         conic_optimizer=nothing, λ_sol=nothing, x_sol=nothing, 
                         h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], 
                                  uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    
    leader_instances = Dict{Int, Tuple{Model, Dict}}()
    follower_instances = Dict{Int, Tuple{Model, Dict}}()
    
    for s in 1:S
        U_s = Dict(
            :R => Dict(:1 => R[s]),
            :r_dict => Dict(:1 => r_dict[s]),
            :xi_bar => Dict(:1 => xi_bar[s]),
            :epsilon => epsilon
        )
        # Note: build_isp_leader와 build_isp_follower는 외부 함수로 가정
        # 실제 사용시 해당 함수들을 include하거나 이 모듈에 추가해야 함
        leader_instances[s] = build_isp_leader(network, 1, ϕU, λU, γ, w, v, U_s, 
                                                conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, S)
        follower_instances[s] = build_isp_follower(network, 1, ϕU, λU, γ, w, v, U_s,
                                                    conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, S)
    end
    
    return leader_instances, follower_instances
end

"""
Inner Master Problem + Inner SubProblem 최적화 (공통 로직)
"""
function _imp_optimize!(solver::AbstractNestedBendersSolver;
                        λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing,
                        outer_iter=nothing)
    models = solver.models
    pd = solver.problem_data
    config = solver.config
    
    imp_model = models.imp_model
    imp_vars = models.imp_vars
    isp_leader_instances = models.isp_leader_instances
    isp_follower_instances = models.isp_follower_instances
    isp_data = models.isp_data
    uncertainty_set = pd[:uncertainty_set]
    S = pd[:S]
    
    R = uncertainty_set[:R]
    r_dict = uncertainty_set[:r_dict]
    xi_bar = uncertainty_set[:xi_bar]
    epsilon = uncertainty_set[:epsilon]
    
    # 이전 iteration의 cuts 제거
    if outer_iter > 1 && haskey(solver.imp_cuts, :old_cuts)
        for (cut_name, cut) in solver.imp_cuts[:old_cuts]
            delete(imp_model, cut)
        end
        if haskey(solver.imp_cuts, :old_tr_constraints) && solver.imp_cuts[:old_tr_constraints] !== nothing
            for tr_cons in solver.imp_cuts[:old_tr_constraints]
                delete.(imp_model, tr_cons)
            end
        end
    end
    
    st = MOI.get(imp_model, MOI.TerminationStatus())
    iter = 0
    past_obj = Float64[]
    past_subprob_obj = Float64[]
    past_major_subprob_obj = Float64[]
    past_lower_bound = Float64[]
    major_iter = Int[]
    lower_bound = -Inf
    
    result = Dict{Symbol, Any}()
    result[:cuts] = Dict{String, Any}()
    
    # Trust region for inner loop (TRNestedBendersSolver인 경우)
    B_conti_max = isp_data[:w] / S
    B_conti = B_conti_max * 0.01
    counter = 0
    β_relative = 1e-4
    centers = Dict(:α => value.(imp_vars[:α]))
    tr_constraints = Dict(:continuous => nothing)
    
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        config.verbose && @info "  Inner Benders Iteration $iter"
        
        optimize!(imp_model)
        st = MOI.get(imp_model, MOI.TerminationStatus())
        
        α_sol = value.(imp_vars[:α])
        model_estimate = sum(value.(imp_vars[:t_1_l])) + sum(value.(imp_vars[:t_1_f]))
        
        # 각 시나리오별 ISP 풀기
        subprob_obj = 0.0
        dict_cut_info_l = Dict{Int, Dict}()
        dict_cut_info_f = Dict{Int, Dict}()
        status = true
        
        for s in 1:S
            U_s = Dict(
                :R => Dict(:1 => R[s]),
                :r_dict => Dict(:1 => r_dict[s]),
                :xi_bar => Dict(:1 => xi_bar[s]),
                :epsilon => epsilon
            )
            
            status_l, cut_info_l = _isp_leader_optimize!(
                isp_leader_instances[s][1], isp_leader_instances[s][2];
                isp_data=isp_data, uncertainty_set=U_s,
                λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol
            )
            
            status_f, cut_info_f = _isp_follower_optimize!(
                isp_follower_instances[s][1], isp_follower_instances[s][2];
                isp_data=isp_data, uncertainty_set=U_s,
                λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol
            )
            
            status = status && (status_l == :OptimalityCut) && (status_f == :OptimalityCut)
            dict_cut_info_l[s] = cut_info_l
            dict_cut_info_f[s] = cut_info_f
            subprob_obj += cut_info_l[:obj_val] + cut_info_f[:obj_val]
        end
        
        lower_bound = max(lower_bound, subprob_obj)
        gap = model_estimate - lower_bound
        
        # 종료 조건
        if gap <= config.tol
            config.verbose && @info "  Inner termination condition met"
            result[:past_obj] = past_obj
            result[:past_subprob_obj] = past_subprob_obj
            result[:α_sol] = α_sol
            result[:obj_val] = objective_value(imp_model)
            result[:past_lower_bound] = past_lower_bound
            result[:iter] = iter
            result[:tr_constraints] = tr_constraints[:continuous]
            return (:OptimalityCut, result)
        end
        
        # Inner loop에 대한 Serious step test (TRNestedBendersSolver인 경우)
        if solver isa TRNestedBendersSolver
            if iter == 1
                push!(past_major_subprob_obj, subprob_obj)
            end
            
            predicted_increase = model_estimate - past_major_subprob_obj[end]
            β_dynamic = max(1e-8, β_relative * predicted_increase)
            improvement = subprob_obj - past_major_subprob_obj[end]
            is_serious_step = (improvement >= β_dynamic)
            
            if is_serious_step
                distance = norm(α_sol - centers[:α], Inf)
                centers[:α] = α_sol
                push!(major_iter, iter)
                push!(past_major_subprob_obj, subprob_obj)
                
                if improvement >= 0.5 * β_dynamic && distance >= B_conti - 1e-6
                    B_conti = min(B_conti_max, B_conti * 2.0)
                end
                
                tr_constraints = _update_inner_trust_region_constraints!(
                    imp_model, imp_vars, centers, B_conti, tr_constraints, pd[:network]
                )
            else
                ρ = min(1, B_conti) * improvement / β_dynamic
                if ρ > 3.0
                    B_conti = B_conti / min(ρ, 4)
                    counter = 0
                    tr_constraints = _update_inner_trust_region_constraints!(
                        imp_model, imp_vars, centers, B_conti, tr_constraints, pd[:network]
                    )
                elseif 1.0 < ρ && counter >= 3
                    B_conti = B_conti / min(ρ, 4)
                    counter = 0
                    tr_constraints = _update_inner_trust_region_constraints!(
                        imp_model, imp_vars, centers, B_conti, tr_constraints, pd[:network]
                    )
                else
                    counter += 1
                end
            end
        end
        
        # Inner cut 추가
        subgradient_l = [dict_cut_info_l[s][:μtilde] for s in 1:S]
        subgradient_f = [dict_cut_info_f[s][:μtilde] for s in 1:S]
        intercept_l = [dict_cut_info_l[s][:intercept] for s in 1:S]
        intercept_f = [dict_cut_info_f[s][:intercept] for s in 1:S]
        
        if config.multi_cut
            cut_added_l = @constraint(imp_model, [s=1:S],
                imp_vars[:t_1_l][s] <= intercept_l[s] + imp_vars[:α]' * subgradient_l[s]
            )
            cut_added_f = @constraint(imp_model, [s=1:S],
                imp_vars[:t_1_f][s] <= intercept_f[s] + imp_vars[:α]' * subgradient_f[s]
            )
            result[:cuts]["opt_cut_$(iter)_l"] = cut_added_l
            result[:cuts]["opt_cut_$(iter)_f"] = cut_added_f
        else
            intercept_sum = sum(intercept_l) + sum(intercept_f)
            subgradient_sum = sum(subgradient_l) + sum(subgradient_f)
            t_1 = sum(imp_vars[:t_1_l]) + sum(imp_vars[:t_1_f])
            cut_added = @constraint(imp_model, t_1 <= intercept_sum + imp_vars[:α]' * subgradient_sum)
            result[:cuts]["opt_cut_$iter"] = cut_added
        end
        
        push!(past_obj, model_estimate)
        push!(past_subprob_obj, subprob_obj)
        push!(past_lower_bound, lower_bound)
    end
    
    return (:Error, result)
end

"""
ISP Leader 최적화 (Placeholder - 실제 구현 필요)
"""
function _isp_leader_optimize!(model::Model, vars::Dict; kwargs...)
    # 실제 구현은 기존 isp_leader_optimize! 함수를 사용
    # 여기서는 인터페이스만 정의
    error("_isp_leader_optimize! must be implemented or imported from external module")
end

"""
ISP Follower 최적화 (Placeholder - 실제 구현 필요)
"""
function _isp_follower_optimize!(model::Model, vars::Dict; kwargs...)
    # 실제 구현은 기존 isp_follower_optimize! 함수를 사용
    error("_isp_follower_optimize! must be implemented or imported from external module")
end

"""
Outer cut 추가
"""
function _add_outer_cut!(solver::AbstractNestedBendersSolver, cut_info::Dict, iter::Int)
    omp_model = solver.models.omp_model
    omp_vars = solver.models.omp_vars
    config = solver.config
    
    # Cut coefficients 계산 (evaluate_master_opt_cut 로직)
    cut_coeff = _evaluate_master_opt_cut(solver, cut_info, iter)
    
    Uhat1 = cut_coeff[:Uhat1]
    Utilde1 = cut_coeff[:Utilde1]
    Uhat3 = cut_coeff[:Uhat3]
    Utilde3 = cut_coeff[:Utilde3]
    Ztilde1_3 = cut_coeff[:Ztilde1_3]
    βtilde1_1 = cut_coeff[:βtilde1_1]
    βtilde1_3 = cut_coeff[:βtilde1_3]
    intercept = cut_coeff[:intercept]
    
    pd = solver.problem_data
    network = pd[:network]
    num_arcs = length(network.arcs) - 1
    S = pd[:S]
    v = pd[:v]
    uncertainty_set = pd[:uncertainty_set]
    xi_bar = uncertainty_set[:xi_bar]
    
    x, h, λ, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
    
    if config.multi_cut
        t_0_l, t_0_f = omp_vars[:t_0_l], omp_vars[:t_0_f]
        intercept_l, intercept_f = cut_coeff[:intercept_l], cut_coeff[:intercept_f]
        
        # Cut expressions 계산
        E = ones(num_arcs, num_arcs + 1)
        I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]
        d0 = zeros(num_arcs + 1)
        d0[end] = 1.0
        
        # Multi-cut 추가
        for s in 1:S
            D_s = Diagonal(xi_bar[s])
            
            # Leader cut
            cut_expr_l = sum(tr(Uhat1[s,:,:] * (D_s - Diagonal(x) * E)) for s in s:s)
            cut_expr_l += sum(tr(Uhat3[s,:,:] * (v * D_s - v * Diagonal(x) * E)) for s in s:s)
            cut_expr_l += intercept_l[s]
            
            # Follower cut  
            cut_expr_f = sum(tr(Utilde1[s,:,:] * (D_s - Diagonal(x) * E)) for s in s:s)
            cut_expr_f += sum(tr(Utilde3[s,:,:] * (v * D_s - v * Diagonal(x) * E)) for s in s:s)
            cut_expr_f += sum(tr(Ztilde1_3[s,:,:] * (Diagonal(λ * ones(num_arcs) - v .* ψ0))) for s in s:s)
            cut_expr_f += sum(βtilde1_1[s,:] .* (λ * d0 - v .* [ψ0; 0]) for s in s:s)
            cut_expr_f += sum(βtilde1_3[s,:] .* h for s in s:s)
            cut_expr_f += intercept_f[s]
            
            @constraint(omp_model, t_0_l >= cut_expr_l)
            @constraint(omp_model, t_0_f >= cut_expr_f)
        end
    else
        t_0 = omp_vars[:t_0]
        
        # Single cut 계산 및 추가
        E = ones(num_arcs, num_arcs + 1)
        d0 = zeros(num_arcs + 1)
        d0[end] = 1.0
        
        cut_expr = intercept
        for s in 1:S
            D_s = Diagonal(xi_bar[s])
            cut_expr += tr(Uhat1[s,:,:] * (D_s - Diagonal(x) * E))
            cut_expr += tr(Utilde1[s,:,:] * (D_s - Diagonal(x) * E))
            cut_expr += tr(Uhat3[s,:,:] * (v * D_s - v * Diagonal(x) * E))
            cut_expr += tr(Utilde3[s,:,:] * (v * D_s - v * Diagonal(x) * E))
            cut_expr += tr(Ztilde1_3[s,:,:] * Diagonal(λ * ones(num_arcs) - v .* ψ0))
            cut_expr += βtilde1_1[s,:]' * (λ * d0 - v .* [ψ0; 0])
            cut_expr += βtilde1_3[s,:]' * h
        end
        
        cut = @constraint(omp_model, t_0 >= cut_expr)
        set_name(cut, "opt_cut_$iter")
        solver.history.cuts["opt_cut_$iter"] = cut
    end
end

"""
Master optimality cut 계수 평가
"""
function _evaluate_master_opt_cut(solver::AbstractNestedBendersSolver, cut_info::Dict, iter::Int)
    models = solver.models
    pd = solver.problem_data
    S = pd[:S]
    config = solver.config
    
    isp_leader_instances = models.isp_leader_instances
    isp_follower_instances = models.isp_follower_instances
    
    # ISP 인스턴스들에서 cut coefficients 추출
    Uhat1 = cat([value.(isp_leader_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
    Utilde1 = cat([value.(isp_follower_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
    Uhat3 = cat([value.(isp_leader_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
    Utilde3 = cat([value.(isp_follower_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
    Ztilde1_3 = cat([value.(isp_follower_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
    βtilde1_1 = cat([value.(isp_follower_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
    βtilde1_3 = cat([value.(isp_follower_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)
    
    if config.multi_cut
        intercept_l = [value.(isp_leader_instances[s][2][:intercept]) for s in 1:S]
        intercept_f = [value.(isp_follower_instances[s][2][:intercept]) for s in 1:S]
        intercept = sum(intercept_l) + sum(intercept_f)
    else
        intercept = sum(value.(isp_leader_instances[s][2][:intercept]) for s in 1:S) +
                   sum(value.(isp_follower_instances[s][2][:intercept]) for s in 1:S)
        intercept_l, intercept_f = nothing, nothing
    end
    
    return Dict(
        :Uhat1 => Uhat1,
        :Utilde1 => Utilde1,
        :Uhat3 => Uhat3,
        :Utilde3 => Utilde3,
        :Ztilde1_3 => Ztilde1_3,
        :βtilde1_1 => βtilde1_1,
        :βtilde1_3 => βtilde1_3,
        :intercept => intercept,
        :intercept_l => intercept_l,
        :intercept_f => intercept_f
    )
end

# ============================================================================
# Trust Region Specific Functions
# ============================================================================
"""
Outer Trust Region 제약 업데이트
"""
function _update_outer_trust_region_constraints!(
    model::Model,
    vars::Dict,
    centers::Dict,
    B_bin::Float64,
    B_con::Union{Float64, Nothing},
    old_cons::Dict,
    network
)
    interdictable_arc_indices = findall(network.interdictable_arcs)
    x = vars[:x]
    xhat = centers[:x]
    
    # 기존 제약 제거
    if old_cons[:binary] !== nothing
        delete(model, old_cons[:binary])
    end
    if old_cons[:continuous] !== nothing
        delete(model, old_cons[:continuous])
    end
    
    # Binary Trust Region (L1-norm)
    tr_binary_expr = @expression(model,
        sum((1 - x[k]) for k in interdictable_arc_indices if abs(xhat[k] - 1.0) < 1e-6) +
        sum(x[k] for k in interdictable_arc_indices if abs(xhat[k]) < 1e-6)
    )
    new_tr_binary = @constraint(model, tr_binary_expr <= B_bin)
    set_name(new_tr_binary, "TR_binary")
    
    new_tr_continuous = nothing
    
    return Dict(
        :binary => new_tr_binary,
        :continuous => new_tr_continuous
    )
end

"""
Inner Trust Region 제약 업데이트
"""
function _update_inner_trust_region_constraints!(
    model::Model,
    vars::Dict,
    centers::Dict,
    B_conti::Float64,
    old_cons::Dict,
    network
)
    α = vars[:α]
    αhat = centers[:α]
    num_arcs = length(α)
    
    # 기존 제약 제거
    if old_cons[:continuous] !== nothing
        delete.(model, old_cons[:continuous])
    end
    
    # L∞ box constraints for α
    new_tr_cons = Vector{ConstraintRef}()
    for k in 1:num_arcs
        con_ub = @constraint(model, α[k] <= αhat[k] + B_conti)
        con_lb = @constraint(model, α[k] >= αhat[k] - B_conti)
        push!(new_tr_cons, con_ub)
        push!(new_tr_cons, con_lb)
    end
    
    return Dict(:continuous => new_tr_cons)
end

"""
Reverse Region 제약 추가
"""
function _add_reverse_region_constraint!(model::Model, x, xhat, B_old::Float64, network)
    interdictable_arc_indices = findall(network.interdictable_arcs)
    
    reverse_expr = @expression(model,
        sum((1 - x[k]) for k in interdictable_arc_indices if abs(xhat[k] - 1.0) < 1e-6) +
        sum(x[k] for k in interdictable_arc_indices if abs(xhat[k]) < 1e-6)
    )
    
    reverse_con = @constraint(model, reverse_expr >= B_old + 1)
    set_name(reverse_con, "reverse_region")
    
    @info "  Added reverse region constraint: ||x - x̂_old||₁ ≥ $(B_old + 1)"
    
    return reverse_con
end

# ============================================================================
# Utility Functions
# ============================================================================
"""
    get_result(solver::AbstractNestedBendersSolver)

Solver의 결과를 반환합니다.
"""
function get_result(solver::AbstractNestedBendersSolver)
    return solver.result
end

"""
    get_history(solver::AbstractNestedBendersSolver)

Solver의 히스토리를 반환합니다.
"""
function get_history(solver::AbstractNestedBendersSolver)
    return solver.history
end

end # module