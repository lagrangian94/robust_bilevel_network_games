"""
lab_branch.jl — 분기 좌표 선택 규칙의 진단. 저장된 전선의 상자마다, 활성 좌표를 하나씩 중점에서 나눠 본 두 자식의 노드 상한
(= strong branching) 과 지금 규칙 (곱 위반이 가장 큰 좌표) 이 고르는 좌표를 비교한다.

  개선량(i) = z(부모) − max(z(왼쪽 자식), z(오른쪽 자식))     (가지치기되는 자식은 LB 로 둠)
출력: 상자별로 좌표마다의 개선량·폭·위반 점수, 규칙이 고른 좌표의 개선량 / 최선의 개선량.
환경변수: LB_IN  LB_K = 16  LM_KW = "(fwd=true,)"  LM_TOL = 1e-7  LM_ROUNDS = 1  LB_THREADS = 2
실행: julia -t 2 nonzero_sum/lab_branch.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization, Statistics
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("LB_IN", joinpath(tempdir(), "nz_frontier_seed3_S20.jls")))
K = parse(Int, envf("LB_K", "16")); kw = eval(Meta.parse(envf("LM_KW", "(fwd=true,)"))); tolv = parse(Float64, envf("LM_TOL", "1e-7"))
R = parse(Int, envf("LM_ROUNDS", "1")); NT = parse(Int, envf("LB_THREADS", "2"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=fr.S, seed=fr.seed,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[fr.x] .= 1
hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp)
nh, S = nz_nh(nd), nd.S
LB = fr.LB; sc = max(1.0, abs(LB)); prune = LB + 1e-3 * sc
envs = _nz_envs(NT)
mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
bx = sort(fr.boxes; by=b -> -b.ub)
bx = bx[round.(Int, range(1, min(length(bx), 4K); length=K))]
out = Vector{Any}(undef, length(bx)); nxt = Threads.Atomic{Int}(1)
tasks = map(1:NT) do w
    Threads.@spawn begin
        P = _nz_build_sep(nd, x̄; optimizer=mk(envs[w])); DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk(envs[w]), kw...)
        for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, tolv); end
        set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = tolv
        v = P.O.v; Hr = v[:Hr]
        function solve!(l, u, lop, hip, phibp; rounds=R)
            _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
            rg = _nz_dr_ranges(DR, nd, l, u, lop, hip, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), phibp)
            rg === nothing && return -Inf, lop, hip, phibp
            lo, hi, _, phib = rg
            _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib)
            z = _nz_sep_solve!(P; maxrounds=rounds)
            z == -Inf && (z = _nz_dr_recheck_infeasible!(P.O.model))
            return z, lo, hi, phib
        end
        lo0, hi0 = _nz_dr_global(nd, DR)
        solve!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S); rounds=30)
        while true
            n = Threads.atomic_add!(nxt, 1); n > length(bx) && break
            b = bx[n]
            z, lo, hi, phib = solve!(b.l, b.u, b.lo, b.hi, b.phib)
            if !(isfinite(z) && z > prune); out[n] = nothing; continue; end
            α̂ = clamp.(value.(v[:α]), b.l, b.u); rv = value.(v[:r]); dv = value.(v[:d])
            vW = zeros(nh); vF = zeros(nh); vL = zeros(nh); vP = zeros(nh)
            for i in 1:nh
                vL[i] = sum(abs(value(v[:ζL][i, s]) - α̂[i] * rv[s]) for s in 1:S); vF[i] = sum(abs(value(v[:ζF][i, s]) - α̂[i] * dv[s]) for s in 1:S)
            end
            for (jj, k) in enumerate(Hr); vW[nd.hrow[k]] += sum(abs(value(v[:ζW][jj, s]) - α̂[nd.hrow[k]] * value(v[:ϖ][k, s])) for s in 1:S); end
            for (a, k) in enumerate(DR.Hk); vP[nd.hrow[k]] += sum(abs(value(DR.zP[a, s]) - α̂[nd.hrow[k]] * value(v[:π̃][k, s])) for s in 1:S); end
            imp = fill(NaN, nh)
            for i in 1:nh
                (b.u[i] - b.l[i] > 1e-3 && vL[i] + vF[i] + vW[i] + vP[i] > 1e-9) || continue
                t = 0.5 * (b.l[i] + b.u[i]); u1 = copy(b.u); u1[i] = t; l2 = copy(b.l); l2[i] = t
                z1 = solve!(b.l, u1, lo, hi, phib)[1]; z2 = solve!(l2, b.u, lo, hi, phib)[1]
                imp[i] = z - max(LB, min(z, max(z1, z2)))
            end
            out[n] = (z=z, w=b.u .- b.l, vL=vL, vF=vF, vW=vW, vP=vP, imp=imp)
        end
    end
end
foreach(wait, tasks)
act = findall(nd.hU .> 0)
rat1 = Float64[]; rat2 = Float64[]; ratw = Float64[]
println("좌표 (활성): ", act, " | 열: 폭 / 개선량 / 위반 (ζL+ζF+ζW) / ζΠ 위반")
for (n, o) in enumerate(out)
    o === nothing && continue
    s1 = o.vL .+ o.vF .+ o.vW; s2 = s1 .+ o.vP
    c = [i for i in act if !isnan(o.imp[i])]
    isempty(c) && continue
    best = maximum(o.imp[c]); i1 = c[argmax(s1[c])]; i2 = c[argmax(s2[c])]; iw = c[argmax(o.w[c])]
    push!(rat1, o.imp[i1] / max(best, 1e-9)); push!(rat2, o.imp[i2] / max(best, 1e-9)); push!(ratw, o.imp[iw] / max(best, 1e-9))
    @printf("상자 %2d z−LB=%6.2f | 최선 좌표 %d (개선 %.2f) | 지금 규칙 → %d (%.2f) | +ζΠ → %d (%.2f) | 가장 넓은 → %d (%.2f)\n",
            n, o.z - LB, c[argmax(o.imp[c])], best, i1, o.imp[i1], i2, o.imp[i2], iw, o.imp[iw])
    for i in c; @printf("        좌표 %d: 폭 %5.1f 개선 %6.2f 위반 %8.2f ζΠ %8.2f\n", i, o.w[i], o.imp[i], s1[i], o.vP[i]); end
end
@printf("\n요약 (상자 %d 개): 규칙이 얻는 개선량 / 최선 — 지금 규칙 평균 %.2f, +ζΠ 평균 %.2f, 가장 넓은 좌표 평균 %.2f\n", length(rat1), mean(rat1), mean(rat2), mean(ratw))
