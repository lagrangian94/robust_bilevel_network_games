"""
lab_profile.jl — α-B&B 노드 하나를 처리하는 시간의 분해 (worker 한 개, 저장된 전선의 상자들을 차례로).

  상자 설정 (_nz_sep_set_box!) / 범위 MIP (_nz_dr_ranges) / 계수 갱신 (_nz_dr_set!) / LP (Gurobi 내부 시간과 벽시계) /
  위반 계산 (_nz_sep_violations) / 행 추가 (_nz_sep_add!) / 해 읽기 / α 고정 LP (nz_eval_alpha!)
환경변수: LB_IN  LB_K = 30  LP_TOL = 1e-7  LP_ROUNDS = 1  LM_KW = "NamedTuple()"
실행: julia -t 1 nonzero_sum/lab_profile.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization, Statistics
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("LB_IN", joinpath(tempdir(), "nz_frontier_seed3_S20.jls")))
K = parse(Int, envf("LB_K", "30")); tolv = parse(Float64, envf("LP_TOL", "1e-7")); R = parse(Int, envf("LP_ROUNDS", "1"))
kw = eval(Meta.parse(envf("LM_KW", "NamedTuple()")))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=fr.S, seed=fr.seed,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[fr.x] .= 1
hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp)
nh, S = nz_nh(nd), nd.S
env = Gurobi.Env()
mk() = (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
P = _nz_build_sep(nd, x̄; optimizer=mk)
DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk, kw...)
if tolv > 1e-9
    for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, tolv); end
    set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = tolv
end
F = nz_build_fixed_alpha(nd, x̄; optimizer=mk)
lo0, hi0 = _nz_dr_global(nd, DR)
v = P.O.v; Hr = v[:Hr]
T = Dict{Symbol,Float64}(k => 0.0 for k in (:box, :range, :set, :lpwall, :lpgrb, :viol, :add, :read, :eval))
nlp = Ref(0)
function node!(l, u, lop, hip, phibp; rounds=R, count=true)
    t = @elapsed (_nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)); count && (T[:box] += t)
    t = @elapsed (rg = _nz_dr_ranges(DR, nd, l, u, lop, hip, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), phibp)); count && (T[:range] += t)
    rg === nothing && return -Inf
    lo, hi, _, phib = rg
    t = @elapsed _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib); count && (T[:set] += t)
    z = NaN
    for rnd in 1:rounds+1
        g0 = _NZ_LPGRB[]
        t = @elapsed (z = _nz_solve_lp!(P.O.model)); count && (T[:lpwall] += t; T[:lpgrb] += _NZ_LPGRB[] - g0; nlp[] += 1)
        (isnan(z) || isinf(z)) && return z
        rnd == rounds + 1 && break
        t = @elapsed (V = _nz_sep_violations(P)); count && (T[:viol] += t)
        isempty(V) && break
        sort!(V; by=first)
        t = @elapsed for (_, i, j, islo) in V[1:min(5000, length(V))]; _nz_sep_add!(P, i, j, islo, islo ? P.l[i] : P.u[i]); end
        count && (T[:add] += t)
    end
    t = @elapsed begin
        α̂ = clamp.(value.(v[:α]), l, u); rv = value.(v[:r]); dv = value.(v[:d])
        ϖv = zeros(size(v[:ϖ])); for k in Hr, s in 1:S; ϖv[k, s] = value(v[:ϖ][k, s]); end
        ζLv = value.(v[:ζL]); ζFv = value.(v[:ζF]); ζWv = value.(v[:ζW])
    end
    count && (T[:read] += t)
    t = @elapsed nz_eval_alpha!(F, nd, α̂); count && (T[:eval] += t)
    return z
end
node!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S); rounds=30, count=false)
bx = sort(fr.boxes; by=b -> -b.ub)[1:min(K, end)]
node!(bx[1].l, bx[1].u, bx[1].lo, bx[1].hi, bx[1].phib; count=false)      # 컴파일
gc0 = Base.gc_num(); tall = @elapsed for b in bx; node!(b.l, b.u, b.lo, b.hi, b.phib); end
gct = (Base.gc_num().total_time - gc0.total_time) / 1e9
n = length(bx)
@printf("노드 %d 개, 노드당 %.3fs (LP %d 회/노드) | 변형 %s, 허용오차 %.0e, 분리 라운드 %d, 행 %d\n", n, tall / n, round(Int, nlp[] / n), string(kw), tolv, R,
        num_constraints(P.O.model; count_variable_in_set_constraints=false))
for k in (:box, :range, :set, :lpwall, :lpgrb, :viol, :add, :read, :eval)
    @printf("  %-7s %.3fs/노드 (%.0f%%)\n", k, T[k] / n, 100 * T[k] / tall)
end
@printf("  GC 시간 %.3fs/노드 (%.0f%%)\n", gct / n, 100 * gct / tall)
