"""
test_dr_validity_boxes.jl — 쌍대 범위 조임 (nz_alpha_bnb_dr.jl 의 W·F·C·V·X 행) 의 유효성 직접 검사.

B&B 의 [LB, UB] 가 V* 를 품는지만 보면 약하다 (휴리스틱이 LB 를 V* 근처로 올려 두면, 최적 α* 를 품은 상자가 잘못 잘려도
UB ≥ LB 라 드러나지 않음). 여기서는 독립 KKT 평가기의 최적 반응 α* 를 **품는 무작위 상자** 마다 노드 완화를 풀어
  (i) 실행 가능한가, (ii) 상한 ≥ V*(x̄) 인가
를 본다. 상자는 좌표마다 무작위 폭 (로그 균등 1e-4 ~ 전체), α* 가 내부·경계·모서리에 오도록, 일부 좌표는 폭 0.

환경변수: TV_S = "3,5"  TV_X = "1,2;1;2;1,2,3"  TV_N = 200 (x̄ 마다 상자 수)  TV_EPS = 0.3  TV_DELTA = 0.1  TV_KKT_TL = 600  TV_THREADS = 2
          TV_MODES = "full,noX,noV,base" (full: 전부, noX: 상보성 RLT 없이, noV: 값 상한 없이, base: W·F·C 만,
                     fwd / xx / cc / lsd: full + 그 부등식 하나, new: full + 넷 모두)
실행: julia -t 2,1 nonzero_sum/test_dr_validity_boxes.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
const NTHR = parse(Int, get(ENV, "TV_THREADS", "2"))
GRB() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NTHR); MOI.set(o, MOI.Silent(), true); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
ε = parse(Float64, envf("TV_EPS", "0.3")); δ = pd(envf("TV_DELTA", "0.1")); N = parse(Int, envf("TV_N", "200"))
xs = [isempty(strip(t)) ? Int[] : parse.(Int, split(t, ",")) for t in split(envf("TV_X", "1,2;1;2;1,2,3"), ";")]
modes = String.(strip.(split(envf("TV_MODES", "full,noX,noV,base"), ",")))
mkw = Dict("full" => (vcut=true, cslift=true), "noX" => (vcut=true, cslift=false), "noV" => (vcut=false, cslift=true), "base" => (vcut=false, cslift=false),
          "fwd" => (vcut=true, cslift=true, fwd=true), "xx" => (vcut=true, cslift=true, xx=true), "cc" => (vcut=true, cslift=true, cc=true),
          "lsd" => (vcut=true, cslift=true, lsd=true), "new" => (vcut=true, cslift=true, fwd=true, xx=true, cc=true, lsd=true),   # 2026-10-13 추가 부등식
          "off" => nothing)                       # off: 기존 완화 (범위 행 없음) — 같은 상자에서 기존 완화의 수치 거동을 보기 위한 기준
WMIN = parse(Float64, envf("TV_WMIN", "1e-4"))    # 좌표 폭의 하한 (로그 균등의 아래 끝)
ZERO = envf("TV_ZERO", "1") == "1"                # 폭 0 좌표·경계에 붙은 α* 를 허용할지
nbad_total = 0
for S in parse.(Int, split(envf("TV_S", "3,5"), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    for xi in xs
        x̄ = zeros(ndf.nx); x̄[xi] .= 1
        k = nz_kkt_value(ndf, x̄; optimizer=GRB, time_limit=parse(Float64, envf("TV_KKT_TL", "600")))
        if k[:status] != MOI.OPTIMAL
            @printf("S=%d x=%s: KKT 가 최적까지 못 감 (%s) → 건너뜀\n", S, string(xi), string(k[:status])); continue
        end
        V = k[:value]
        hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp); nh = nz_nh(nd)
        αs = clamp.(k[:α], 0.0, nd.hU)
        F = nz_build_fixed_alpha(nd, x̄; optimizer=GRB)
        zα = nz_eval_alpha!(F, nd, αs)                       # α* 고정 정확 LP 값 (= V* 여야 함. 프리솔브로 0 이 된 좌표가 α* 에서 양수였다면 달라짐)
        tolV = 1e-6 * max(1.0, abs(V))
        @printf("S=%d x=%-9s KKT V*=%.5f, α* 고정 LP=%.5f%s\n", S, string(xi), V, zα, abs(zα - V) > 1e-4 * max(1, abs(V)) ? "  ** 불일치 **" : "")
        for md in modes
            P = _nz_build_sep(nd, x̄; optimizer=GRB)
            DR = md == "off" ? nothing : _nz_dr_attach!(P, nd, x̄; optimizer=GRB, mkw[md]...)
            lo0, hi0 = md == "off" ? (zeros(0, 0), zeros(0, 0)) : _nz_dr_global(nd, DR)
            rng = MersenneTwister(1234 + S)
            ninf = 0; nlow = 0; worst = Inf; ntight = 0; nrec = 0
            for n in 1:N
                l = zeros(nh); u = copy(nd.hU)
                for i in 1:nh
                    nd.hU[i] > 0 || continue
                    mode = rand(rng, 1:6)
                    lw = log10(WMIN)
                    wl = 10.0^(rand(rng) * (log10(nd.hU[i]) - lw) + lw); wu = 10.0^(rand(rng) * (log10(nd.hU[i]) - lw) + lw)
                    ZERO && mode == 1 && (wl = 0.0)           # α* 가 아래 경계
                    ZERO && mode == 2 && (wu = 0.0)           # α* 가 위 경계
                    ZERO && mode == 3 && (wl = 0.0; wu = 0.0) # 폭 0
                    mode == 4 && (wl = Inf; wu = Inf)         # 전체 범위
                    l[i] = max(0.0, αs[i] - wl); u[i] = min(nd.hU[i], αs[i] + wu)
                end
                _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                if md != "off"
                    rg = _nz_dr_ranges(DR, nd, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S))
                    if rg === nothing
                        ninf += 1
                        ninf <= 2 && println("    범위 MIP 가 실행 불가능하다고 함 (α* 를 품은 상자인데): l = ", l, " u = ", u)
                        continue
                    end
                    lo, hi, _, phib = rg
                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib)
                end
                z = _nz_sep_solve!(P)
                if z == -Inf
                    # nz_alpha_bnb_dr 와 같은 처리: 기저를 버리고 FeasibilityTol 1e-6 에서 처음부터 재확인
                    z = _nz_dr_recheck_infeasible!(P.O.model)
                    nrec += 1
                end
                if z == -Inf
                    ninf += 1
                    ninf <= 1 && println("    노드 LP 가 재확인에서도 실행 불가능 (α* 를 품은 상자인데): 최소 폭 ", minimum(u[i] - l[i] for i in 1:nh if nd.hU[i] > 0))
                elseif isfinite(z)
                    worst = min(worst, z - V)
                    z - V < -tolV && (nlow += 1; nlow <= 2 && println("    상한 < V*: z − V* = ", z - V, " l = ", l, " u = ", u))
                    z - V <= 1e-4 * max(1.0, abs(V)) && (ntight += 1)
                end
            end
            bad = ninf + nlow
            global nbad_total += bad
            @printf("  %-5s 상자 %d 개: 실행 불가능 %d (첫 판정이 실행 불가능이어서 재확인한 횟수 %d), 상한 < V* %d, min(상한 − V*) = %.3e, 정확 (≤ 1e-4 상대) %d 개  %s\n",
                    md, N, ninf, nrec, nlow, worst, ntight, bad == 0 ? "유효" : "** 무효 **")
            flush(stdout)
        end
    end
end
println(nbad_total == 0 ? "전체: α* 를 품은 모든 상자에서 실행 가능 + 상한 ≥ V*" : "전체: 무효 $nbad_total 건")
