# corner_premise_check.jl (2026-11-28 새벽): nz_alpha_bnb_dr 의 꼭짓점 LP 쌍대 범위 (range_lp = true) 의 전제를 직접 검사.
#   주장: 상자 [l, u] 의 모든 α 에서 최적 쌍대면 위 σ_k (등식 쌍은 ν = σ⁺ − σ⁻) 는 [min_{면(u)}, max_{면(l)}] 안에 있다 (전제 = submodularity).
#   검사: 무작위 상자와 그 안의 무작위 α 에서 면(α) 의 최소·최대가 그 범위를 벗어나는지 (여유 1e-6, 범위 LP 와 같음). 벗어나면 그 상자에서 꼭짓점 범위가 틀림.
#   KIND = loc (GEO, XS, S) | zs (NET, ZSEED, XS)  NBOX 200  WFRAC 1.0 (상자 폭 상한 = WFRAC·hᵁ)  SEED 1
#   julia --project=. docs/nz_ccg_bnp_scripts/corner_premise_check.jl
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
FTOL = parse(Float64, get(ENV, "FTOL", "1e-7"))
KIND = get(ENV, "KIND", "loc"); NBOX = parse(Int, get(ENV, "NBOX", "200")); WFRAC = parse(Float64, get(ENV, "WFRAC", "1.0"))
rng = MersenneTwister(parse(Int, get(ENV, "SEED", "1")))
if KIND == "loc"
    nd = make_loc_tkp(; g=parse(Int, get(ENV, "GEO", "1")), S=parse(Int, get(ENV, "S", "10")), seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=1e4, NB=3)
    K, tiers, nA = nd.meta[:K], nd.meta[:tiers], nd.meta[:nA]; xb = zeros(nd.nx)
    for p in split(get(ENV, "XS", "3:300;10:200;15:300"), ";")
        i, tv = parse.(Int, split(p, ":")); xb[(i - nA - 1) * K + findfirst(==(Float64(tv)), tiers)] = 1.0
    end
    tag = "loc geo $(get(ENV, "GEO", "1")) x $(loc_tkp_xstr(nd, xb))"
else
    include(joinpath(root, "network_generator.jl")); using .NetworkGenerator; const NG = NetworkGenerator
    include(joinpath(root, "true_dro", "true_dro_data.jl"))
    NET = get(ENV, "NET", "polska"); ZSEED = parse(Int, get(ENV, "ZSEED", "42")); S = 10
    net0 = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network)[NET]()
    intd = fill(true, length(net0.arcs))
    net = NG.RealWorldNetworkData(net0.name, net0.original_node_names, net0.nodes, net0.arcs, net0.N, intd, net0.arc_adjacency, net0.node_arc_incidence)
    Kz = length(net.arcs) - 1
    caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S; interdictable_arcs=intd, seed=ZSEED, num_factors=5)
    w = round(0.5 * 2 * median(caps[1:Kz, :]); digits=4)
    Random.seed!(ZSEED); v = [rand() < 0.75 ? 1.0 : 0.0 for k in 1:Kz, s in 1:S]
    td = make_true_dro_data(net, caps, fill(1.0 / S, S), 0.3, 0.3; w=w, lambda_U=1e4, gamma=2, beta=0.4, v_scenarios=v)
    nd = nz_with_delta(nz_from_true_dro(td), 0.1)
    xb = zeros(nd.nx); for i in parse.(Int, split(get(ENV, "XS", "6,33"), ",")); xb[i] = 1.0; end
    tag = "zs $(NET) s$(ZSEED) x $(get(ENV, "XS", "6,33"))"
end
env = Gurobi.Env(); opt = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
m, nh, S = nz_m(nd), nz_nh(nd), nd.S
P = nz_eq_pairs(nd); inpair = Set(vcat(collect(keys(P)), collect(values(P))))
Hk = [k for k in 1:m if nd.hrow[k] > 0 && !(k in inpair)]
CL = [_nz_dr_build_corner(nd, xb, s; optimizer=opt) for s in 1:S]
"α 에서 시나리오 s 의 최적 쌍대면 위 식들의 (최소, 최대)"
function face_ranges(C, α, s, exprs)
    rr = nz_rhs(nd, xb, α, s)
    for k in 1:m; nd.hrow[k] > 0 && set_normalized_coefficient(C.opt, C.σ[k], rr[k]); end
    set_normalized_rhs(C.opt, 1e20)
    @objective(C.model, Min, sum(rr[k] * C.σ[k] for k in 1:m if rr[k] != 0)); optimize!(C.model)
    termination_status(C.model) == MOI.OPTIMAL || error("face_ranges: 최적값 LP $(termination_status(C.model))")
    z = objective_value(C.model); set_normalized_rhs(C.opt, z + FTOL * max(1.0, abs(z)))   # 최적면 여유 (범위 LP 기본 1e-7)
    out = Tuple{Float64,Float64}[]
    for e in exprs
        @objective(C.model, Min, e); optimize!(C.model); termination_status(C.model) == MOI.OPTIMAL || error("face_ranges: 최소 LP")
        a = objective_value(C.model)
        @objective(C.model, Max, e); optimize!(C.model); termination_status(C.model) == MOI.OPTIMAL || error("face_ranges: 최대 LP")
        push!(out, (a, objective_value(C.model)))
    end
    return out
end
@printf("%s | S %d, H 행 %d (등식 쌍 %d), 상자 %d, 폭 상한 %.2f·hᵁ\n", tag, S, length(Hk), length(P), NBOX, WFRAC); flush(stdout)
nviol = 0; worst = 0.0; nchk = 0; nbad = 0; vals = Float64[]; scl = Float64[]
for b in 1:NBOX
    l = [rand(rng) * nd.hU[i] for i in 1:nh]; u = [min(nd.hU[i], l[i] + rand(rng) * WFRAC * nd.hU[i]) for i in 1:nh]
    α = l .+ rand(rng, nh) .* (u .- l); s = rand(rng, 1:S); C = CL[s]
    exprs = vcat([C.σ[k] for k in Hk], [C.σ[kp] - C.σ[kn] for (kn, kp) in P])
    rl = face_ranges(C, l, s, exprs); ru = face_ranges(C, u, s, exprs); ra = face_ranges(C, α, s, exprs)
    bad = false
    for j in eachindex(exprs)
        global nchk += 1
        hiv = ra[j][2] - (rl[j][2] + 1e-6); lov = (ru[j][1] - 1e-6) - ra[j][1]
        vv = max(hiv, lov); push!(scl, max(abs(rl[j][2]), abs(ru[j][1])))
        if vv > 1e-7
            push!(vals, vv); global nviol += 1; bad = true; global worst = max(worst, vv)
        end
    end
    bad && (global nbad += 1)
end
@printf("결과 (FTOL %.0e): 상자 %d 중 %d 개에서 위반, 성분 검사 %d 중 위반 %d, 최대 위반 %.4g | 위반 > 1e-4: %d, > 1e-3: %d, > 1e-2: %d, > 0.1: %d | 꼭짓점 범위 끝값 크기 중앙값 %.3g\n",
        FTOL, NBOX, nbad, nchk, nviol, worst, count(>(1e-4), vals), count(>(1e-3), vals), count(>(1e-2), vals), count(>(0.1), vals), median(scl))
