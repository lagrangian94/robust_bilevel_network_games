# zs_vi_check.jl (2026-11-28 아침): min-cut VI 우변 (h = 0, q̂ 기대) 과 h = 0 최대흐름 LP 를 고정 x̄ 에서 직접 비교. Polska s4 {7,18}: VI 우변 28.07 > V 21.38 이라 VI 가 틀린 곳 찾기.
#   julia zs_vi_check.jl   환경 변수 NET polska, ZSEED 4, XCHK "7,18"
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra, Random, Statistics
import MathOptInterface as MOI
include(joinpath(root, "network_generator.jl")); NG = NetworkGenerator
for f in ("true_dro_data.jl", "true_dro_build_omp.jl", "true_dro_build_subproblem.jl", "true_dro_build_isp_leader.jl",
          "true_dro_build_isp_follower.jl", "true_dro_benders.jl", "true_dro_mincut_vi.jl", "belief_menu_benders.jl")
    include(joinpath(root, "true_dro", f))
end
const NET = get(ENV, "NET", "polska"); const ZSEED = parse(Int, get(ENV, "ZSEED", "4")); const S = 10
WALL = parse(Float64, get(ENV, "WALL", "900"))
if NET == "grid5x5"
    net = NG.generate_grid_network(5, 5; seed=42); intd = net.interdictable_arcs
else
    gen = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network,
               "nobel_us" => NG.generate_nobel_us_network, "sioux_falls" => NG.generate_sioux_falls_network)
    net0 = gen[NET](); intd = fill(true, length(net0.arcs))
    net = NG.RealWorldNetworkData(net0.name, net0.original_node_names, net0.nodes, net0.arcs, net0.N, intd, net0.arc_adjacency, net0.node_arc_incidence)
end
γ = 2; Kz = length(net.arcs) - 1
caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S; interdictable_arcs=intd, seed=ZSEED, num_factors=5)
w = round(0.5 * γ * median(caps[findall(intd[1:Kz]), :]); digits=4)
Random.seed!(ZSEED); v = zeros(Kz, S)
for k in 1:Kz, s in 1:S; v[k, s] = intd[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0; end
td = make_true_dro_data(net, caps, fill(1.0 / S, S), 0.3, 0.3; w=w, lambda_U=10.0, gamma=γ, beta=0.4, v_scenarios=v)
ss = nothing
if NET != "grid5x5"
    src = [i for i in 1:Kz if net.arcs[i][1] == "s"]; snk = [i for i in 1:Kz if net.arcs[i][2] == "t"]
    ss = Dict{Symbol,Vector{Int}}(); length(src) >= 2 && (ss[:source_arcs] = src); length(snk) >= 2 && (ss[:sink_arcs] = snk)
    isempty(ss) && (ss = nothing)
end
# min-cut VI 우변을 {XCHK} 에서: Phase 1 VI 와 같은 식 (q̂ 기대, h = 0, 용량 ξ̄(1 − v x)) 의 최소 cut LP
xc = zeros(td.num_arcs); for i in parse.(Int, split(get(ENV, "XCHK", "7,18"), ",")); xc[i] = 1.0; end
K = td.num_arcs; topo = extract_arc_topology(td.Ny, td.nv1)
@printf("K %d, nv1 %d, Nts 크기 %s, 호 이름 수 %d\n", K, td.nv1, string(size(td.Nts)), length(net.arcs))
for k in vcat(1:3, 7, 18, K)   # 위상 확인: 이름 (tail, head) 과 Ny 로 읽은 tail/head
    @printf("  호 %d %s → %s | Ny tail %d head %d | ξ̄[k,1] %.3f v[k,1] %.0f\n", k, net.arcs[k][1], net.arcs[k][2], topo.tail[k], topo.head[k], td.xi_bar[k, 1], td.v[k, 1])
end
flows = Float64[]; cuts = Float64[]; cutsR = Float64[]
for j in 1:td.S
    cap = [td.xi_bar[k, j] * (1 - td.v[k, j] * xc[k]) for k in 1:K]
    m = Model(Gurobi.Optimizer); set_silent(m)
    @variable(m, 0 <= y[k=1:K] <= cap[k]); @variable(m, yts >= 0)
    @constraint(m, td.Ny * y .+ vec(td.Nts) .* yts .== 0); @objective(m, Max, yts); optimize!(m)
    push!(flows, objective_value(m))
    for (dir, store) in ((1, cuts), (-1, cutsR))   # dir = 1: VI 와 같은 방향, −1: 반대 방향
        c = Model(Gurobi.Optimizer); set_silent(c)
        @variable(c, 0 <= δ[1:K] <= 1); @variable(c, 0 <= τ[1:td.nv1] <= 1); @constraint(c, τ[td.nv1] == 1)
        for k in 1:K
            th = topo.head[k] == 0 ? 0.0 : τ[topo.head[k]]; tt = topo.tail[k] == 0 ? 0.0 : τ[topo.tail[k]]
            @constraint(c, δ[k] >= dir * (th - tt))
        end
        @objective(c, Min, sum(cap[k] * δ[k] for k in 1:K)); optimize!(c); push!(store, objective_value(c))
    end
end
@printf("x {%s}: q̂ 기대 h = 0 최대흐름 %.6f | VI 방향 최소 cut 기대 %.6f | 반대 방향 %.6f\n", get(ENV, "XCHK", "7,18"), mean(flows), mean(cuts), mean(cutsR))
@printf("  시나리오별 최대흐름 %s\n  VI 방향 cut %s\n", string(round.(flows; digits=3)), string(round.(cuts; digits=3)))
