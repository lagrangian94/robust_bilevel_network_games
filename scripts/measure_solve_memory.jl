"""
Solve 단계 메모리 측정: Polska S=20에서 ISP solve 시 메모리 사용량 추적.
안전장치: free memory < 3GB이면 자동 중단.

사용법:
  julia measure_solve_memory.jl              # 순차 solve만
  julia -t 4 measure_solve_memory.jl         # 순차 + 병렬 비교
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Infiltrator
using Serialization
using Revise

include("../network_generator.jl")
include("../build_uncertainty_set.jl")
include("../parallel_utils.jl")
include("../strict_benders.jl")
include("../nested_benders_trust_region.jl")

using .NetworkGenerator: generate_polska_network, print_realworld_network_summary, generate_capacity_scenarios_uniform_model

const SAFE_FREE_GB = 3.0  # 이 이하면 중단

function free_gb()
    return Sys.free_memory() / 1024^3
end

function check_safe(label)
    f = free_gb()
    if f < SAFE_FREE_GB
        println("\n⚠ [$label] FREE = $(round(f, digits=2)) GB < $(SAFE_FREE_GB) GB → 안전 중단!")
        return false
    end
    return true
end

const v = 1.0  # isp_follower_optimize!가 글로벌 v를 참조함

function main()
    S = 20
    seed = 42
    epsilon = 0.5
    γ_ratio = 0.10
    ρ = 0.2

    println("="^70)
    println("SOLVE 메모리 측정: Polska, S=$S, threads=$(Threads.nthreads())")
    println("="^70)

    # --- Setup ---
    network = generate_polska_network()
    print_realworld_network_summary(network)
    num_arcs = length(network.arcs) - 1

    ϕU = 1/epsilon
    λU = ϕU
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)

    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar, digits=4)

    capacity_scenarios_regular = capacities[1:end-1, :]
    R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
    uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

    source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
    max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
    max_cap = maximum(capacity_scenarios_regular)
    πU = ϕU
    yU = min(max_cap, ϕU)
    ytsU = min(max_flow_ub, ϕU)

    # --- OMP 먼저 풀어서 실제 초기해 확보 ---
    println("\n[Step 0] OMP 풀기 (초기해 확보)...")
    GC.gc(); GC.gc()
    mem0 = free_gb()
    model_omp, vars_omp = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
    optimize!(model_omp)
    λ_sol = value(vars_omp[:λ])
    x_sol = value.(vars_omp[:x])
    h_sol = value.(vars_omp[:h])
    ψ0_sol = value.(vars_omp[:ψ0])
    α_sol = zeros(num_arcs)  # IMP 안 풀었으니 0으로 시작
    println("  OMP solved. λ=$(round(λ_sol, digits=4)), sum(x)=$(sum(x_sol))")
    println("  Free: $(round(free_gb(), digits=2)) GB")

    # --- ISP 모델 생성 ---
    println("\n[Step 1] ISP 모델 생성 (S=$S)...")
    GC.gc(); GC.gc()
    mem1 = free_gb()

    leader_instances, follower_instances = initialize_isp(
        network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol,
        πU=πU, yU=yU, ytsU=ytsU, scaling_S=S)

    GC.gc(); GC.gc()
    mem2 = free_gb()
    println("  ISP 생성 완료. 메모리 사용: $(round((mem1-mem2)*1024, digits=1)) MB")
    println("  Free: $(round(mem2, digits=2)) GB")

    if !check_safe("ISP 생성 후")
        return
    end

    # --- isp_data 준비 ---
    E = ones(num_arcs, num_arcs+1)
    d0 = zeros(num_arcs + 1); d0[end] = 1.0
    isp_data = Dict(
        :E => E, :ϕU => ϕU, :d0 => d0, :S => S,
        :πU => πU, :yU => yU, :ytsU => ytsU,
        :w => w, :uncertainty_set => uncertainty_set,
        :scaling_S => S
    )

    # =====================================================
    # TEST A: 순차 solve — 1개씩 풀면서 메모리 추적
    # =====================================================
    println("\n" * "="^70)
    println("[TEST A] 순차 solve: 시나리오 1개씩 풀면서 메모리 추적")
    println("="^70)

    GC.gc(); GC.gc()
    mem_before_solve = free_gb()
    println("  Solve 전 Free: $(round(mem_before_solve, digits=2)) GB")

    solve_mems = Float64[]
    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]),
                    :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)

        mem_pre = free_gb()
        isp_leader_optimize!(leader_instances[s][1], leader_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)

        isp_follower_optimize!(follower_instances[s][1], follower_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)

        GC.gc()
        mem_post = free_gb()
        delta = (mem_pre - mem_post) * 1024  # MB
        push!(solve_mems, delta)
        cumul = (mem_before_solve - mem_post) * 1024
        println("  s=$s: Δ=$(round(delta, digits=1)) MB, 누적=$(round(cumul, digits=1)) MB, Free=$(round(mem_post, digits=2)) GB")

        if !check_safe("순차 solve s=$s")
            return
        end
    end

    GC.gc(); GC.gc()
    mem_after_seq = free_gb()
    total_seq = (mem_before_solve - mem_after_seq) * 1024
    println("\n  [순차 요약]")
    println("  총 메모리 변화: $(round(total_seq, digits=1)) MB")
    println("  시나리오당 평균 Δ: $(round(mean(solve_mems), digits=1)) MB")
    println("  시나리오당 최대 Δ: $(round(maximum(solve_mems), digits=1)) MB")
    println("  Free: $(round(mem_after_seq, digits=2)) GB")

    # =====================================================
    # TEST B: 병렬 solve (스레드 > 1일 때만)
    # =====================================================
    if Threads.nthreads() > 1
        println("\n" * "="^70)
        println("[TEST B] 병렬 solve: $(Threads.nthreads()) 스레드로 S=$S 동시 solve")
        println("="^70)

        # 점진적으로: 5 → 10 → 15 → 20
        for batch_S in [5, 10, 15, 20]
            if batch_S > S
                break
            end

            GC.gc(); GC.gc()
            sleep(1)
            mem_pre_par = free_gb()

            if !check_safe("병렬 solve batch=$batch_S 시작 전")
                return
            end

            println("\n  --- 병렬 batch: $batch_S 시나리오 ---")
            println("  Pre-solve Free: $(round(mem_pre_par, digits=2)) GB")

            # 실제 병렬 solve
            @threads for s in 1:batch_S
                U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]),
                            :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
                isp_leader_optimize!(leader_instances[s][1], leader_instances[s][2];
                    isp_data=isp_data, uncertainty_set=U_s,
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
                isp_follower_optimize!(follower_instances[s][1], follower_instances[s][2];
                    isp_data=isp_data, uncertainty_set=U_s,
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
            end

            GC.gc(); GC.gc()
            sleep(1)
            mem_post_par = free_gb()
            delta_par = (mem_pre_par - mem_post_par) * 1024
            println("  Post-solve Free: $(round(mem_post_par, digits=2)) GB")
            println("  메모리 변화: $(round(delta_par, digits=1)) MB")
            println("  ✓ batch=$batch_S 완료")

            if !check_safe("병렬 solve batch=$batch_S 완료 후")
                return
            end
        end
    else
        println("\n[TEST B 건너뜀] 단일 스레드 — 병렬 테스트하려면 julia -t 4 로 실행")
    end

    # =====================================================
    # TEST C: evaluate_master_opt_cut (MW cut 포함) 메모리
    # =====================================================
    println("\n" * "="^70)
    println("[TEST C] evaluate_master_opt_cut + MW cut 메모리 측정")
    println("="^70)

    GC.gc(); GC.gc()
    mem_pre_eval = free_gb()

    # evaluate_master_opt_cut에 필요한 cut_info 구성
    cut_info = Dict(:α_sol => α_sol)
    evaluate_master_opt_cut(leader_instances, follower_instances, isp_data, cut_info, 1; parallel=false)

    GC.gc(); GC.gc()
    mem_post_eval = free_gb()
    delta_eval = (mem_pre_eval - mem_post_eval) * 1024
    println("  evaluate_master_opt_cut 메모리: $(round(delta_eval, digits=1)) MB")
    println("  Free: $(round(mem_post_eval, digits=2)) GB")

    # MW cut
    println("\n  MW cut 측정...")
    GC.gc(); GC.gc()
    mem_pre_mw = free_gb()

    try
        evaluate_mw_opt_cut(leader_instances, follower_instances, isp_data, cut_info, 1;
            parallel=false, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

        GC.gc(); GC.gc()
        mem_post_mw = free_gb()
        delta_mw = (mem_pre_mw - mem_post_mw) * 1024
        println("  MW cut 메모리: $(round(delta_mw, digits=1)) MB")
        println("  Free: $(round(mem_post_mw, digits=2)) GB")
    catch e
        println("  MW cut 에러 (무시): $e")
        println("  Free: $(round(free_gb(), digits=2)) GB")
    end

    # =====================================================
    # 최종 요약
    # =====================================================
    println("\n" * "="^70)
    println("최종 요약")
    println("="^70)
    final_free = free_gb()
    total_used = (mem0 - final_free)
    println("  시스템 총 RAM:     $(round(Sys.total_memory()/1024^3, digits=1)) GB")
    println("  시작 시 Free:      $(round(mem0, digits=2)) GB")
    println("  현재 Free:         $(round(final_free, digits=2)) GB")
    println("  총 사용량:         $(round(total_used, digits=2)) GB")
    println("  스레드 수:         $(Threads.nthreads())")
    println()

    if final_free < 4.0
        println("⚠ 실제 실행 시 MW + 병렬 + 다수 iteration → 프리즈 위험!")
    else
        println("✓ 현재까지는 안전 범위")
    end
end

# mean 헬퍼
mean(x) = sum(x) / length(x)

main()
