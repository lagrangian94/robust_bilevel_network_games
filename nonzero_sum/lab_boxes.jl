"""
lab_boxes.jl — 저장된 B&B 전선 (lab_frontier.jl) 의 같은 상자 묶음에서 노드 완화의 변형을 비교.

변형마다: 상자별 노드 상한 (LB 대비 상대 gap), 가지치기되는 상자 수, 상자당 범위 MIP·완화 LP 시간, LP 행 수.
B&B 전체를 다시 돌리지 않고 "완화가 얼마나 조여지는가 / LP 가 얼마나 빨라지는가" 를 먼저 본다.

환경변수: LB_IN = (전선 파일)  LB_K = 48 (상자 수: 상한이 큰 순서로 절반, 나머지는 고르게)  LB_THREADS = 12
          LB_VARIANTS = "base,fwd+r30+tol7"  (아래 parse_variant 의 토큰을 + 로 이은 이름, 쉼표)  LB_PRUNE = 1.5e-3 (가지치기 기준 상대 gap)
실행: julia -t 12 nonzero_sum/lab_boxes.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization, Statistics
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("LB_IN", joinpath(tempdir(), "nz_frontier_seed3_S20.jls")))
K = parse(Int, envf("LB_K", "48")); NT = parse(Int, envf("LB_THREADS", "12")); PR = parse(Float64, envf("LB_PRUNE", "1.5e-3"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=fr.S, seed=fr.seed,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[fr.x] .= 1
hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp)
nh, S = nz_nh(nd), nd.S
LB = fr.LB; sc = max(1.0, abs(LB))
ord = sortperm([b.ub for b in fr.boxes]; rev=true)
live = [k for k in ord if fr.boxes[k].ub > LB + 1e-3 * sc]
sel = unique(vcat(live[1:min(K ÷ 2, end)], live[round.(Int, range(1, length(live); length=min(K, length(live))))]))
boxes = fr.boxes[sel]
@printf("lab_boxes: seed=%d S=%d x=%s LB=%.4f | 전선 %d 개 중 상한 > LB 인 %d 개에서 %d 개 선택 (부모 상한 gap %.2e ~ %.2e)\n",
        fr.seed, S, string(fr.x), LB, length(fr.boxes), length(live), length(boxes),
        minimum((b.ub - LB) / sc for b in boxes), maximum((b.ub - LB) / sc for b in boxes)); flush(stdout)

# 변형 이름 = "+" 로 이은 토큰:  fwd (follower 약쌍대) | kmc | dd | sticky | rN (분리 라운드 상한 N, 기본 1 = B&B 의 child_rounds) |
#   tol7·tol6 (NumericFocus 0 + Feasibility/OptimalityTol 1e-7·1e-6) | nf0 | primal | base (아무것도 안 바꿈)
function parse_variant(name)
    attach = Dict{Symbol,Any}(:fwd => false, :kmc => false, :dd => false); lp = Dict{Symbol,Any}(); rounds = 1; sticky = false   # 토큰이 없으면 끔
    for t in split(name, "+")
        t == "base" && continue
        if t == "fwd"; attach[:fwd] = true
        elseif t == "sticky"; sticky = true
        elseif t == "kmc"; attach[:kmc] = true
        elseif t == "dd"; attach[:dd] = true
        elseif startswith(t, "r") && all(isdigit, t[2:end]); rounds = parse(Int, t[2:end])
        elseif t == "tol7"; lp[:NumericFocus] = 0; lp[:FeasibilityTol] = 1e-7; lp[:OptimalityTol] = 1e-7
        elseif t == "tol6"; lp[:NumericFocus] = 0; lp[:FeasibilityTol] = 1e-6; lp[:OptimalityTol] = 1e-6
        elseif t == "nf0"; lp[:NumericFocus] = 0
        elseif t == "primal"; lp[:Method] = 0
        else error("모르는 변형 토큰: $t")
        end
    end
    return (attach=(; attach...), lp=(; lp...), rounds=rounds, sticky=sticky)
end
envs = _nz_envs(NT)
mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)

function run_variant(name)
    V = parse_variant(name)
    res = Vector{Any}(undef, length(boxes))
    nxt = Threads.Atomic{Int}(1)
    troot = zeros(NT); zroot = zeros(NT)
    t = @elapsed begin
        tasks = map(1:NT) do w
            Threads.@spawn begin
                P = _nz_build_sep(nd, x̄; optimizer=mk(envs[w]))
                DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk(envs[w]), V.attach...)
                for (k, val) in pairs(V.lp)
                    set_optimizer_attribute(P.O.model, string(k), val)
                    k == :Method && (P.O.model.ext[:lp_method] = val)
                end
                lo0, hi0 = _nz_dr_global(nd, DR)
                # 루트 (B&B 가 worker 마다 루트 분리 행을 넣어 주는 것과 같게)
                l = zeros(nh); u = copy(nd.hU)
                troot[w] = @elapsed begin
                    V.sticky ? _nz_sticky_set_box!(P, l, u) : _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                    lo, hi, _, phib = _nz_dr_ranges(DR, nd, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S))
                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib)
                    zroot[w] = (V.sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=30)
                end
                while true
                    n = Threads.atomic_add!(nxt, 1)
                    n > length(boxes) && break
                    b = boxes[n]
                    V.sticky ? _nz_sticky_set_box!(P, b.l, b.u) : _nz_sep_set_box!(P, b.l, b.u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                    tg = @elapsed (rg = _nz_dr_ranges(DR, nd, b.l, b.u, b.lo, b.hi, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), b.phib))
                    if rg === nothing
                        res[n] = (z=-Inf, tg=tg, tl=0.0, rows=0, it=0, nlp=0, unfixed=0); continue
                    end
                    lo, hi, wt, phib = rg
                    dl, dh = _nz_dr_dglobal(nd, DR)
                    isempty(DR.pairs) || (tg += @elapsed ((dl, dh) = _nz_dr_dranges!(DR, nd, b.l, b.u, lo, hi, dl, dh, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), wt)))
                    _nz_dr_set!(P, DR, nd, b.l, b.u, lo, hi, phib; dlo=dl, dhi=dh)
                    n0 = _NZ_LPN[]
                    tl = @elapsed (z = (V.sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=V.rounds))
                    z == -Inf && (z = _nz_dr_recheck_infeasible!(P.O.model))
                    it = 0
                    res[n] = (z=min(z, b.ub), tg=tg, tl=tl, rows=num_constraints(P.O.model; count_variable_in_set_constraints=false),
                              it=it, nlp=0, unfixed=count(hi .- lo .> 1e-4))
                end
            end
        end
        foreach(wait, tasks)
    end
    gaps = [isfinite(r.z) ? (r.z - LB) / sc : -1.0 for r in res]
    @printf("RESULT %-14s 루트 %.2f (%.0fs) | 평균 gap %.3e 중앙값 %.3e 최대 %.3e | 가지치기 (gap ≤ %.1e) %d / %d | 상자당 LP %.2fs (중앙값 %.2f) 범위 MIP %.2fs | 행 %d | 전체 %.0fs\n",
            name, mean(zroot), mean(troot), mean(gaps), median(gaps), maximum(gaps), PR, count(<=(PR), gaps), length(gaps),
            mean(r.tl for r in res), median(r.tl for r in res), mean(r.tg for r in res), round(Int, mean(r.rows for r in res)), t)
    flush(stdout)
    return gaps
end

G = Dict{String,Vector{Float64}}()
for name in String.(strip.(split(envf("LB_VARIANTS", "base,fwd"), ",")))
    G[name] = run_variant(name)
end
ref = envf("LB_REF", "base")
if haskey(G, ref)
    for name in sort(collect(keys(G)))
        name == ref && continue
        fin = [i for i in eachindex(G[ref]) if G[ref][i] > -0.5 && G[name][i] > -0.5]     # 둘 다 실행 가능한 상자만
        d = G[name][fin] .- G[ref][fin]
        @printf("  DIFF %-22s − %s: 둘 다 실행 가능 %d 개에서 gap 차이 평균 %.3e, 중앙값 %.3e, 최소 %.3e, 최대 %.3e (음수 = 더 조임) | 실행 불가능 %d vs %d
",
                name, ref, length(fin), mean(d), median(d), minimum(d), maximum(d), count(<=(-0.5), G[name]), count(<=(-0.5), G[ref]))
    end
end
