"""
메모리 측정 스크립트: Polska S=20에서 ISP 모델이 얼마나 메모리를 쓰는지 확인.
solve는 하지 않으므로 프리즈 위험 없음.

사용법: julia measure_memory.jl   (스레드 불필요)
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Serialization

include("network_generator.jl")
include("build_uncertainty_set.jl")
include("parallel_utils.jl")
include("strict_benders.jl")
include("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_polska_network, print_realworld_network_summary, generate_capacity_scenarios_uniform_model

function measure_isp_memory(; S=20, seed=42, epsilon_hat=0.5, epsilon_tilde=0.5, γ_ratio=0.10, ρ=0.2, v=1.0)
    println("="^60)
    println("ISP 메모리 측정: Polska, S=$S")
    println("="^60)

    # --- 네트워크 생성 ---
    network = generate_polska_network()
    print_realworld_network_summary(network)
    num_arcs = length(network.arcs) - 1

    ϕU_hat = 1/epsilon_hat
    ϕU_tilde = 1/epsilon_tilde
    λU = ϕU_hat
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)

    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar, digits=4)

    capacity_scenarios_regular = capacities[1:end-1, :]
    R, r_dict_hat, r_dict_tilde, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon_hat, epsilon_tilde)
    uncertainty_set = Dict(:R => R, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde, :xi_bar => xi_bar, :epsilon_hat => epsilon_hat, :epsilon_tilde => epsilon_tilde)

    source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
    max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
    max_cap = maximum(capacity_scenarios_regular)
    πU_hat = ϕU_hat
    πU_tilde = ϕU_tilde
    yU = min(max_cap, ϕU_tilde)
    ytsU = min(max_flow_ub, ϕU_tilde)

    # --- 메모리 측정: ISP 생성 전 ---
    GC.gc(); GC.gc()
    sleep(1)
    mem_before = Sys.free_memory()
    rss_before = Sys.maxrss()  # peak RSS (일부 OS에서만 정확)
    println("\n[Before ISP]")
    println("  Free memory:  $(round(mem_before / 1024^3, digits=2)) GB")
    println("  Total memory: $(round(Sys.total_memory() / 1024^3, digits=2)) GB")

    # --- 더미 초기해 생성 (OMP 안 풀고 측정만 하므로) ---
    num_arcs_val = length(network.arcs) - 1
    λ_dummy = 1.0
    x_dummy = zeros(num_arcs_val)
    h_dummy = zeros(num_arcs_val)
    ψ0_dummy = zeros(num_arcs_val)
    α_dummy = zeros(num_arcs_val)

    # --- ISP 모델 생성 (시나리오별로 메모리 추적) ---
    println("\nISP 모델 생성 시작...")
    R_us = uncertainty_set[:R]
    r_dict_hat_us = uncertainty_set[:r_dict_hat]
    r_dict_tilde_us = uncertainty_set[:r_dict_tilde]
    xi_bar_us = uncertainty_set[:xi_bar]

    leader_instances = Dict{Int, Tuple{Model, Dict}}()
    follower_instances = Dict{Int, Tuple{Model, Dict}}()

    for s in 1:S
        U_s_hat = Dict(:R => Dict(:1=>R_us[s]), :r_dict => Dict(:1=>r_dict_hat_us[s]),
                    :xi_bar => Dict(:1=>xi_bar_us[s]), :epsilon => epsilon_hat)
        U_s_tilde = Dict(:R => Dict(:1=>R_us[s]), :r_dict => Dict(:1=>r_dict_tilde_us[s]),
                    :xi_bar => Dict(:1=>xi_bar_us[s]), :epsilon => epsilon_tilde)

        leader_instances[s] = build_isp_leader(network, 1, ϕU_hat, λU, γ, w, v, U_s_hat,
            Mosek.Optimizer, λ_dummy, x_dummy, h_dummy, ψ0_dummy, α_dummy, S; πU=πU_hat)
        follower_instances[s] = build_isp_follower(network, 1, ϕU_tilde, λU, γ, w, v, U_s_tilde,
            Mosek.Optimizer, λ_dummy, x_dummy, h_dummy, ψ0_dummy, α_dummy, S; πU=πU_tilde, yU=yU, ytsU=ytsU)

        GC.gc()
        mem_now = Sys.free_memory()
        used = (mem_before - mem_now) / 1024^2
        println("  s=$s: 누적 사용량 ≈ $(round(used, digits=1)) MB  (남은 free: $(round(mem_now/1024^3, digits=2)) GB)")

        # 안전 체크: free memory가 2GB 이하면 중단
        if mem_now < 2 * 1024^3
            println("\n⚠ FREE MEMORY < 2GB — 안전을 위해 중단!")
            println("  현재까지 $s개 시나리오에서 $(round(used, digits=1)) MB 사용")
            println("  S=20 전체 추정: $(round(used / s * 20, digits=1)) MB")
            # 메모리 해제
            leader_instances = nothing
            follower_instances = nothing
            GC.gc(); GC.gc()
            return
        end
    end

    # --- 메모리 측정: ISP 생성 후 ---
    GC.gc(); GC.gc()
    sleep(1)
    mem_after = Sys.free_memory()
    total_used_mb = (mem_before - mem_after) / 1024^2

    println("\n[After ISP — S=$S 모델 전부 생성 완료]")
    println("  Free memory:  $(round(mem_after / 1024^3, digits=2)) GB")
    println("  ISP 총 메모리: $(round(total_used_mb, digits=1)) MB ($(round(total_used_mb/1024, digits=2)) GB)")
    println("  모델당 평균:   $(round(total_used_mb / (2*S), digits=1)) MB")

    # --- OMP 모델도 측정 ---
    println("\nOMP 모델 생성...")
    GC.gc(); GC.gc()
    mem_pre_omp = Sys.free_memory()
    model_omp, vars_omp = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
    GC.gc()
    mem_post_omp = Sys.free_memory()
    omp_mb = (mem_pre_omp - mem_post_omp) / 1024^2
    println("  OMP 메모리: $(round(omp_mb, digits=1)) MB")

    # --- 최종 요약 ---
    println("\n" * "="^60)
    println("최종 요약")
    println("="^60)
    total_gb = (total_used_mb + omp_mb) / 1024
    println("  ISP ($(2*S) Mosek 모델): $(round(total_used_mb, digits=1)) MB")
    println("  OMP (Gurobi):            $(round(omp_mb, digits=1)) MB")
    println("  합계:                    $(round(total_gb, digits=2)) GB")
    println("  시스템 총 RAM:           $(round(Sys.total_memory()/1024^3, digits=1)) GB")
    println("  남은 free:               $(round(mem_post_omp/1024^3, digits=2)) GB")
    println()

    remaining = mem_post_omp / 1024^3
    if remaining < 3.0
        println("⚠ 경고: 남은 메모리 $(round(remaining, digits=1))GB — solve 시 페이징/프리즈 위험 높음!")
    elseif remaining < 5.0
        println("⚠ 주의: 남은 메모리 $(round(remaining, digits=1))GB — MW cut 시 메모리 부족 가능")
    else
        println("✓ 남은 메모리 충분: $(round(remaining, digits=1))GB")
    end

    # 정리
    leader_instances = nothing
    follower_instances = nothing
    model_omp = nothing
    GC.gc(); GC.gc()
end

# 실행
measure_isp_memory(S=20)
