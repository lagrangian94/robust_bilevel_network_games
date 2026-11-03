#=
diag_bound_compare_zs.jl — zero-sum (네트워크 차단, 결합 δ = 0.1) 에서 고정된 x̄ 의 V*(x̄) 상계를 세 방법으로 같은 시간·스레드에서 비교 (2026-10-22).
  KKT   : 결합판 monolithic KKT (evaluate_monolithic_coupled, BD-KKT 의 전역 단계) + Gurobi
  Ω-GRB : Ω (build_true_dro_subproblem, NonConvex) + Gurobi (BD-G 의 전역 단계)
  Ω-BnB : Ω + 우리 α-B&B (global_bilinear_solve, Ours 의 전역 단계)
x̄ 는 우리 방법 (menu) 의 최적해. 인스턴스 생성은 run_nig_coupled_bdkkt.jl 과 같음.
환경변수: DZ_T = 300  DZ_THREADS = 2  DZ_NETS = polska,abilene
실행: julia -t 4,1 diag_bound_compare_zs.jl
=#
using JuMP, Gurobi, Printf, Dates, LinearAlgebra, Random, Statistics
include("../../network_generator.jl"); NG = NetworkGenerator
include("../true_dro_data.jl"); include("../true_dro_build_omp.jl"); include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl"); include("../true_dro_build_isp_follower.jl"); include("../true_dro_benders.jl")
include("../true_dro_mincut_vi.jl"); include("../belief_menu_benders.jl"); include("../true_dro_structured_eval.jl"); include("../bd_kkt_benders.jl")

function make_network(name)
    name == "grid5x5" && (net = NG.generate_grid_network(5, 5; seed=42); return net, 2, net.interdictable_arcs)
    gen = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network,
               "nobel_us" => NG.generate_nobel_us_network, "sioux_falls" => NG.generate_sioux_falls_network)
    net = gen[name](); intd = fill(true, length(net.arcs))
    net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs, net.N, intd, net.arc_adjacency, net.node_arc_incidence)
    return net, 2, intd
end
const NETWORK_EPS = Dict("grid5x5" => 0.1, "polska" => 0.1, "sioux_falls" => 0.1, "abilene" => 0.2, "nobel_us" => 0.2)
T = parse(Float64, get(ENV, "DZ_T", "300")); NT = parse(Int, get(ENV, "DZ_THREADS", "2"))
nets = String.(strip.(split(get(ENV, "DZ_NETS", "polska,abilene"), ",")))
const GRB_ENV = Gurobi.Env()
GRBT() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o)
S, δ, β = 10, 0.1, 0.4
@printf("상계 비교 (zero-sum, 결합 δ=%.2f, S=%d): 방법당 %.0f s, 스레드 %d\n", δ, S, T, NT); flush(stdout)
for name in nets
    eps = NETWORK_EPS[name]; net, γ, intd = make_network(name); K = length(net.arcs) - 1
    caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S; interdictable_arcs=intd, seed=42, num_factors=5)
    w = round(0.5 * γ * median(caps[findall(intd[1:K]), :]); digits=4)
    Random.seed!(42); v = zeros(K, S)
    for k in 1:K, s in 1:S; v[k, s] = intd[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0; end
    td = make_true_dro_data(net, caps, fill(1.0/S, S), eps, eps; w=w, lambda_U=10.0, gamma=γ, beta=β, v_scenarios=v)
    td, δc = reduce_coupling(td, δ)
    line = first(filter(l -> startswith(l, "RESULT"), readlines(joinpath(@__DIR__, "logs", "nig_coupled", "S_10", "$(name)_menu.log"))))
    xo = parse.(Int, split(match(r"x=([0-9;]+)", line)[1], ";")); x̄ = zeros(K); x̄[xo] .= 1.0
    Zref = parse(Float64, match(r"Z0=([-0-9.]+)", line)[1])
    @printf("\n=== %s x=%s (Ours Z₀ = %.6f)\n", name, string(xo), Zref); flush(stdout)
    t = @elapsed (ab = global_bilinear_solve(td, x̄; nworkers=NT, time_limit=T, rel_gap=1e-3, verbose=false, delta_couple=δc))
    @printf("  Ω-BnB : 하계 %10.4f  상계 %10.4f  %6.1f s\n", ab[:LB], ab[:UB], t); flush(stdout)
    sub, sv = build_true_dro_subproblem(td, x̄; optimizer=GRBT, silent=true, delta_couple=δc)
    set_optimizer_attribute(sub, "NonConvex", 2); set_optimizer_attribute(sub, "MIPGap", 1e-3); set_time_limit_sec(sub, T)
    update_true_dro_subproblem_objective!(sub, sv, td, x̄)
    t = @elapsed optimize!(sub)
    @printf("  Ω-GRB : 하계 %10.4f  상계 %10.4f  %6.1f s  %s\n", has_values(sub) ? objective_value(sub) : -Inf, objective_bound(sub), t,
            string(termination_status(sub))); flush(stdout)
    sd = make_struct_eval_data(td, x̄; lp_optimizer=Gurobi.Optimizer)
    t = @elapsed (k = evaluate_monolithic_coupled(sd; optimizer=GRBT, delta_couple=δc, time_limit=T, mip_gap=1e-3, lp_optimizer=Gurobi.Optimizer))
    @printf("  KKT   : 하계 %10.4f  상계 %10.4f  %6.1f s  %s\n", k[:V], k[:V_bound], t, string(k[:status])); flush(stdout)
end
