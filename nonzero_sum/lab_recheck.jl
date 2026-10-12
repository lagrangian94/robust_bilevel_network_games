"""
lab_recheck.jl — 실행 불가능 판정 재확인 (_nz_dr_recheck_infeasible!) 의 비용과 대안 비교.

노드 LP 가 INFEASIBLE 이면 기저를 버리고 FeasibilityTol 1e-6 에서 처음부터 다시 푼다 (10.6 의 거짓 실행 불가능 대책).
새 기본 구성에서는 노드의 12~14% 가 이 재확인을 거치는데, 재확인 시간은 t_relax 에 잡히지 않는다.
저장된 전선의 상자 중 첫 판정이 INFEASIBLE 인 것을 골라, 재확인 방법별 시간과 판정을 잰다.
  A: 지금 (GRBreset + primal simplex, Presolve 0)      B: GRBreset + dual simplex + Presolve 켬 + DualReductions 0
  C: GRBreset + primal simplex + Presolve 켬           D: GRBreset + dual simplex, Presolve 0
환경변수: LB_IN  LR_N = 20 (재확인할 상자 수)  LB_THREADS = 4
실행: julia -t 4 nonzero_sum/lab_recheck.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization, Statistics
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("LB_IN", joinpath(tempdir(), "nz_frontier_seed3_S20.jls")))
N = parse(Int, envf("LR_N", "20")); NT = parse(Int, envf("LB_THREADS", "4"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=fr.S, seed=fr.seed,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[fr.x] .= 1
hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp)
nh, S = nz_nh(nd), nd.S
envs = _nz_envs(NT)
mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
function recheck!(model, method, presolve)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-6); set_optimizer_attribute(model, "Method", method)
    set_optimizer_attribute(model, "Presolve", presolve); set_optimizer_attribute(model, "DualReductions", presolve == 0 ? 1 : 0)
    Gurobi.GRBreset(JuMP.unsafe_backend(model), 0)
    t = @elapsed (z = _nz_solve_lp!(model))
    set_optimizer_attribute(model, "FeasibilityTol", 1e-7); set_optimizer_attribute(model, "Method", 1)
    set_optimizer_attribute(model, "Presolve", 0); set_optimizer_attribute(model, "DualReductions", 1)
    return z, t
end
variants = (("A", 0, 0), ("B", 1, -1), ("C", 0, -1), ("D", 1, 0))
out = Any[]; lk = ReentrantLock(); nxt = Threads.Atomic{Int}(1); nfound = Threads.Atomic{Int}(0)
bx = fr.boxes
tasks = map(1:NT) do w
    Threads.@spawn begin
        P = _nz_build_sep(nd, x̄; optimizer=mk(envs[w])); DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk(envs[w]))
        for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, 1e-7); end
        set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = 1e-7
        lo0, hi0 = _nz_dr_global(nd, DR); dl0, dh0 = _nz_dr_dglobal(nd, DR)
        function setbox!(l, u, lop, hip, phibp)
            _nz_sticky_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
            rg = _nz_dr_ranges(DR, nd, l, u, lop, hip, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), phibp)
            rg === nothing && return false
            lo, hi, wt, phib = rg
            dl, dh = _nz_dr_dranges!(DR, nd, l, u, lo, hi, dl0, dh0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), wt)
            _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; dlo=dl, dhi=dh)
            return true
        end
        setbox!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S)); _nz_sticky_solve!(P; maxrounds=30)
        while nfound[] < N
            n = Threads.atomic_add!(nxt, 1); n > length(bx) && break
            b = bx[n]
            setbox!(b.l, b.u, b.lo, b.hi, b.phib) || continue
            t0 = @elapsed (z = _nz_sticky_solve!(P; maxrounds=1))
            z == -Inf || continue
            Threads.atomic_add!(nfound, 1)
            r = [(name, recheck!(P.O.model, meth, pre)...) for (name, meth, pre) in variants]
            lock(lk) do; push!(out, (t0=t0, rows=num_constraints(P.O.model; count_variable_in_set_constraints=false), r=r)); end
        end
    end
end
foreach(wait, tasks)
@printf("첫 판정이 INFEASIBLE 인 상자 %d 개 (전선 %d 개 중 %d 개까지 봄), LP 행 평균 %d, 첫 LP 평균 %.2fs\n", length(out), length(bx), min(nxt[] - 1, length(bx)),
        round(Int, mean(o.rows for o in out)), mean(o.t0 for o in out))
for (k, (name, meth, pre)) in enumerate(variants)
    ts = [o.r[k][3] for o in out]; zs = [o.r[k][2] for o in out]
    @printf("  %s (Method %d, Presolve %2d): 평균 %.2fs 중앙값 %.2fs 최대 %.2fs | 실행 불가능 확정 %d, 되살아남 %d, 그 밖 %d\n", name, meth, pre,
            mean(ts), median(ts), maximum(ts), count(==(-Inf), zs), count(isfinite, zs), count(z -> !(isfinite(z) || z == -Inf), zs))
end
agree = count(o -> length(unique([(z == -Inf) for (_, z, _) in o.r])) == 1, out)
println("  네 방법의 판정 (실행 불가능 / 아님) 이 모두 같은 상자: ", agree, " / ", length(out))
for o in out
    length(unique([(z == -Inf) for (_, z, _) in o.r])) == 1 && continue
    println("    판정이 갈린 상자: ", join([@sprintf("%s: %s", name, z == -Inf ? "실행 불가능" : @sprintf("z − LB = %+.3f", z - fr.LB)) for (name, z, _) in o.r], ", "))
end
