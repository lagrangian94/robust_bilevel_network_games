"""
test_dr_validity_incumbent.jl — 큰 S 에서의 유효성 검사 (KKT 평가기가 끝나지 않는 크기용).

test_dr_validity_boxes.jl 은 KKT 최적 α* 를 품은 상자를 보는데, S ≥ 5 에서는 KKT 가 시간 안에 끝나는 x̄ 가 거의 없다.
여기서는 α-B&B 가 찾은 incumbent α° (α 고정 정확 LP 값 G(α°) = LB, 벌점 0 인 실제 bilevel 실행가능해) 를 쓴다:
α° 를 품은 상자의 참 최댓값은 G(α°) 이상이므로, 노드 완화는 (i) 실행 가능하고 (ii) 상한 ≥ G(α°) 여야 한다.
유효 부등식이 α° 를 (잘못) 잘라내면 α° 근처의 작은 상자에서 상한이 G(α°) 아래로 떨어지거나 실행 불가능이 된다.
쌍대 차이 범위 (DD) 처럼 작은 S 에서는 거의 걸리지 않는 부등식을 실제로 걸리는 크기에서 보기 위한 것.

입력: lab_frontier.jl 이 저장한 파일 (incumbent α°, LB).
환경변수: LB_IN  TI_N = 150  TI_MODES = "full,old" (full: 기본 구성, old: FWD·KMC·DD 없음)  TI_WMIN = 1e-3
실행: julia -t 1 nonzero_sum/test_dr_validity_incumbent.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("LB_IN", joinpath(tempdir(), "nz_frontier_seed3_S20.jls")))
N = parse(Int, envf("TI_N", "150")); WMIN = parse(Float64, envf("TI_WMIN", "1e-3"))
modes = String.(strip.(split(envf("TI_MODES", "full,old"), ",")))
mkw = Dict("full" => NamedTuple(), "old" => (fwd=false, kmc=false, dd=false))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=fr.S, seed=fr.seed,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[fr.x] .= 1
hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp)
nh, S = nz_nh(nd), nd.S
env = Gurobi.Env()
mk() = (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
αs = clamp.(fr.α, 0.0, nd.hU)
F = nz_build_fixed_alpha(nd, x̄; optimizer=mk)
V = nz_eval_alpha!(F, nd, αs)
@printf("seed=%d S=%d x=%s | incumbent α° (활성 좌표) = %s, G(α°) = %.5f (저장된 LB %.5f)\n", fr.seed, S, string(fr.x),
        string(round.(αs[nd.hU .> 0]; digits=2)), V, fr.LB)
tolV = 1e-6 * max(1.0, abs(V)); nbad = 0
for md in modes
    P = _nz_build_sep(nd, x̄; optimizer=mk); DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk, mkw[md]...)
    for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, 1e-7); end      # B&B 의 기본 (lp_tol) 과 같게
    set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = 1e-7
    lo0, hi0 = _nz_dr_global(nd, DR)
    rng = MersenneTwister(4321)
    ninf = 0; nlow = 0; worst = Inf; nrec = 0; ntight = 0
    for n in 1:N
        l = zeros(nh); u = copy(nd.hU)
        for i in 1:nh
            nd.hU[i] > 0 || continue
            mode = rand(rng, 1:6); lw = log10(WMIN)
            wl = 10.0^(rand(rng) * (log10(nd.hU[i]) - lw) + lw); wu = 10.0^(rand(rng) * (log10(nd.hU[i]) - lw) + lw)
            mode == 1 && (wl = 0.0); mode == 2 && (wu = 0.0); mode == 3 && (wl = 0.0; wu = 0.0); mode == 4 && (wl = Inf; wu = Inf)
            l[i] = max(0.0, αs[i] - wl); u[i] = min(nd.hU[i], αs[i] + wu)
        end
        _nz_sticky_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
        rg = _nz_dr_ranges(DR, nd, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S))
        if rg === nothing
            ninf += 1; println("    범위 MIP 가 실행 불가능하다고 함: l = ", l, " u = ", u); continue
        end
        lo, hi, wt, phib = rg
        dl, dh = _nz_dr_dglobal(nd, DR)
        isempty(DR.pairs) || ((dl, dh) = _nz_dr_dranges!(DR, nd, l, u, lo, hi, dl, dh, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), wt))
        _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; dlo=dl, dhi=dh)
        z = _nz_sticky_solve!(P; maxrounds=30)
        if z == -Inf
            z = _nz_dr_recheck_infeasible!(P.O.model); nrec += 1
        end
        if z == -Inf
            ninf += 1
            ninf <= 3 && println("    노드 LP 가 재확인에서도 실행 불가능: 폭 ", round.((u .- l)[nd.hU .> 0]; sigdigits=3))
        elseif isfinite(z)
            worst = min(worst, z - V)
            z - V < -tolV && (nlow += 1; nlow <= 3 && println("    상한 < G(α°): 차이 ", z - V, " 폭 ", round.((u .- l)[nd.hU .> 0]; sigdigits=3)))
            z - V <= 1e-4 * max(1.0, abs(V)) && (ntight += 1)
        end
    end
    global nbad += ninf + nlow
    @printf("  %-5s 상자 %d 개: 실행 불가능 %d (재확인 %d 회), 상한 < G(α°) %d, min(상한 − G(α°)) = %.3e, 정확 (≤ 1e-4 상대) %d 개  %s\n",
            md, N, ninf, nrec, nlow, worst, ntight, ninf + nlow == 0 ? "유효" : "** 무효 **")
    flush(stdout)
end
println(nbad == 0 ? "전체: incumbent 를 품은 모든 상자에서 실행 가능 + 상한 ≥ G(α°)" : "전체: 무효 $nbad 건")
