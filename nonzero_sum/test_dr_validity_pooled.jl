"""
test_dr_validity_pooled.jl — pooled 예약 (retailer 모형) 에서 비제로섬 α-B&B (nz_alpha_bnb_dr) 의 유효성 검사.
지금까지의 검증 (test_dr_validity_boxes.jl 등) 은 SGB128 pair 에서만 했다. 쌍대 범위 조임은 location 구조의 가정 (예약 행 쌍대 ≤ p,
용량 쌍대 상계) 을 쓰므로 pooled 에서 따로 확인한다. θᵁ 는 정확한 circuit 값 (_nz_theta_circuit_exact_pooled).

  (A) 상자 검사 (test_dr_validity_boxes.jl 와 같은 방법): KKT 평가기의 최적 반응 α* 를 품는 무작위 상자마다 기본 구성의 노드 완화를 풀어
      (i) 실행 가능한가 (재확인 포함), (ii) 상한 ≥ V*(x̄) 인가
  (B) B&B 대조: 모든 x 에서 nz_alpha_bnb_dr 의 [LB, UB] 가 KKT 값 V*(x) 를 품는가 (허용오차 1e-4 상대: B&B 의 목표 gap)
  KKT 가 시간 안에 못 끝나는 x̄ (무작위 pooled 는 d = 8 부터 그렇다) 는 test_dr_validity_incumbent.jl 과 같은 방법으로 바꾼다:
  B&B 의 incumbent α° (α 고정 정확 LP 값 G(α°) = LB, 실제 bilevel 실행가능해) 를 품은 상자의 상한이 G(α°) 이상이어야 한다.
  이 경우 (B) 는 독립 대조가 아니므로 하지 않는다 (참조 = "incumbent" 로 표시). incumbent 는 허용오차 안에서만 최적인 근사점일 수 있어
  (diag_pooled_dr.jl), 그 belief 에서 follower LP 의 정확한 최적 꼭짓점으로 다듬은 점을 기준으로 쓴다 (참조 = "꼭짓점").

환경변수: TVP_INST = sgb | random   TVP_NLOC = 8 (random)   NZ_SEED = 1   TVP_S = "3"   TVP_N = 200 (x̄ 마다 상자 수)   TVP_WRES = 300
          TVP_EPS = 0.3   TVP_DELTA = 0.1   TVP_KKT_TL = 600   TVP_THREADS = 2   TVP_BNB = 1   TVP_BNB_TL = 120   TVP_WORKERS = 2
          TVP_POOL = total | store (예약 한도: 전체 합 Σ h ≤ TVP_WRES, 또는 거점별 h_i ≤ TVP_QUOTA)   TVP_QUOTA = 150 (sgb) / 300 (random)
          TVP_X = "" (비우면 모든 x, 예 "3;1,3")   TVP_MODES = "full" (상자 검사의 완화 구성, test_dr_validity_boxes.jl 과 같은 이름:
                     full, old (FWD·KMC·DD 없음), fwd / kmc / dd (old + 하나), base (W·F·C 만), off (범위 행 없음))
실행: julia -t 4,1 nonzero_sum/test_dr_validity_pooled.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
const NTHR = parse(Int, get(ENV, "TVP_THREADS", "2"))
GRB() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NTHR); MOI.set(o, MOI.Silent(), true); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
ε = parse(Float64, envf("TVP_EPS", "0.3")); δ = pd(envf("TVP_DELTA", "0.1")); N = parse(Int, envf("TVP_N", "200"))
inst = envf("TVP_INST", "sgb"); nloc = parse(Int, envf("TVP_NLOC", "8")); seed = parse(Int, envf("NZ_SEED", "1")); wres = parse(Float64, envf("TVP_WRES", "300"))
pool = Symbol(envf("TVP_POOL", "total")); quota = parse(Float64, envf("TVP_QUOTA", inst == "sgb" ? "150" : "300"))
WMIN = 1e-4
function mkinst(S)
    nd0 = if inst == "sgb"
        make_location_instance(; coords=:sgb128, reservation=:pooled, wres=wres, pool_limit=pool, quota=quota, S=S, seed=seed, eps_hat=ε, eps_tilde=ε, beta=0.4)
    else
        cap = 150.0 * (nloc - 5) / 2
        make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed, bA=cap, bB=cap, qB=cap,
                               dlo=50.0, dhi=150.0, reservation=:pooled, wres=wres, pool_limit=pool, quota=quota, eps_hat=ε, eps_tilde=ε, beta=0.4)
    end
    nd1 = nz_with_delta(nd0, δ)
    return nz_with_bounds(nd1; thetaU=nz_theta_circuit_exact(nd1)[1])
end
all_x(nd) = [Float64.(collect(b)) for b in Iterators.product(fill(0:1, nd.nx)...)
             if sum(b) <= nd.gamma && all(nd.x_allowed[i] || b[i] == 0 for i in 1:nd.nx)]
xsel = strip(envf("TVP_X", "")); xsel = isempty(xsel) ? nothing : [sort(parse.(Int, split(t, ","))) for t in split(xsel, ";")]
modes = String.(strip.(split(envf("TVP_MODES", "full"), ",")))
mkw = Dict("full" => NamedTuple(), "old" => (fwd=false, kmc=false, dd=false), "fwd" => (fwd=true, kmc=false, dd=false),
           "kmc" => (fwd=false, kmc=true, dd=false), "dd" => (fwd=false, kmc=false, dd=true), "base" => (vcut=false, cslift=false), "off" => nothing)

nbad = 0
for S in parse.(Int, split(envf("TVP_S", "3"), ","))
    ndf = mkinst(S)
    @printf("== %s pooled S=%d seed=%d n_h=%d θᵁ=%.3f λᵁ=%.1f 예약 한도 %s\n", inst == "sgb" ? "SGB128" : "무작위 d=$nloc", S, seed, nz_nh(ndf), ndf.thetaU, ndf.lambdaU,
            pool == :store ? "거점별 $quota" : "전체 $wres")
    for xv in all_x(ndf)
        xi = findall(xv .> 0.5)
        xsel === nothing || xi in xsel || continue
        bnb() = nz_alpha_bnb_dr(ndf, xv; nworkers=parse(Int, envf("TVP_WORKERS", "2")), time_limit=parse(Float64, envf("TVP_BNB_TL", "120")),
                                rel_gap=1e-4, verbose=false, ipopt_time=30.0)
        k = try
            nz_kkt_value(ndf, xv; optimizer=GRB, time_limit=parse(Float64, envf("TVP_KKT_TL", "600")))
        catch err
            # nz_kkt_value 는 시간 제한에서 해가 없으면 "KKT eval: …" 오류를 낸다. 그 경우만 incumbent 기준 검사로 바꾸고 나머지는 그대로 던진다
            (err isa ErrorException && startswith(err.msg, "KKT eval")) || rethrow()
            nothing
        end
        ref = (k !== nothing && k[:status] == MOI.OPTIMAL) ? "KKT" : "incumbent"
        rB = nothing
        if ref == "KKT"
            V = k[:value]; αref = k[:α]
        else
            # incumbent α° 는 α 고정 LP 의 허용오차 안에서만 최적인 근사점일 수 있다 (꼭짓점에서 1e-4 쯤 벗어남, diag_pooled_dr.jl).
            # 그런 점을 기준으로 폭 0 경계를 두면 진짜 최적 꼭짓점이 상자 밖으로 나가 거짓 무효가 나온다 → α° 의 belief d° 에서
            # follower LP 를 simplex 로 풀어 정확한 최적 꼭짓점 α* 로 다듬고, 값이 떨어지지 않으면 그 점을 기준으로 쓴다
            rB = bnb(); V = rB[:LB]; αref = rB[:α]
            hU0, _ = nz_presolve_hU(ndf, xv); nd0 = nz_with_hU(ndf, hU0)
            F0 = nz_build_fixed_alpha(nd0, xv; optimizer=GRB); G0 = nz_eval_alpha!(F0, nd0, clamp.(αref, 0.0, nd0.hU))
            d0 = [value(F0.v[:d][s]) for s in 1:S]
            fm = Model(GRB); set_optimizer_attribute(fm, "Method", 0); nh0 = nz_nh(nd0); ny0 = nz_ny(nd0)
            @variable(fm, 0 <= αf[i = 1:nh0] <= nd0.hU[i]); @variable(fm, yf[1:ny0, 1:S] >= 0); @constraint(fm, nd0.W * αf .<= nd0.wvec)
            for s in 1:S, kk in 1:nz_m(nd0)
                @constraint(fm, sum(nd0.A[kk, j] * yf[j, s] for j in 1:ny0 if nd0.A[kk, j] != 0) <=
                                nz_rhs(nd0, xv, zeros(nh0), s)[kk] + (nd0.hrow[kk] > 0 ? nd0.hcoef[kk] * αf[nd0.hrow[kk]] : 0.0))
            end
            @objective(fm, Max, dot(nd0.c0, αf) + sum(d0[s] * nd0.c[j] * yf[j, s] for j in 1:ny0, s in 1:S if nd0.c[j] != 0)); optimize!(fm)
            αsnap = clamp.(value.(αf), 0.0, nd0.hU); Gs = nz_eval_alpha!(F0, nd0, αsnap)
            if isfinite(Gs) && Gs >= G0 - 1e-6 * max(1.0, abs(G0))
                ref = "꼭짓점"; αref = αsnap; V = Gs
            end
        end
        # KKT 평가기 (MIPGap 1e-6, indicator 허용오차) 는 참값 0 인 x 에서도 ±2e-6 쯤을 낸다 (SGB128 x = ∅, {3,4}) → 판정 허용오차 1e-5 상대
        tolV = 1e-5 * max(1.0, abs(V))
        hUp, _ = nz_presolve_hU(ndf, xv); nd = nz_with_hU(ndf, hUp); nh = nz_nh(nd)
        αs = clamp.(αref, 0.0, nd.hU)
        F = nz_build_fixed_alpha(nd, xv; optimizer=GRB); zα = nz_eval_alpha!(F, nd, αs)
        mis = abs(zα - V) > 1e-4 * max(1, abs(V))
        global nbad += mis
        # (A) 상자 검사 (구성마다)
        for md in modes
        P = _nz_build_sep(nd, xv; optimizer=GRB)
        DR = md == "off" ? nothing : _nz_dr_attach!(P, nd, xv; optimizer=GRB, mkw[md]...)
        lo0, hi0 = md == "off" ? (zeros(0, 0), zeros(0, 0)) : _nz_dr_global(nd, DR)
        rng = MersenneTwister(1234 + S)
        ninf = 0; nlow = 0; worst = Inf; ntight = 0; nrec = 0
        for n in 1:N
            l = zeros(nh); u = copy(nd.hU)
            for i in 1:nh
                nd.hU[i] > 0 || continue
                mode = rand(rng, 1:6); lw = log10(WMIN)
                wl = 10.0^(rand(rng) * (log10(nd.hU[i]) - lw) + lw); wu = 10.0^(rand(rng) * (log10(nd.hU[i]) - lw) + lw)
                mode == 1 && (wl = 0.0); mode == 2 && (wu = 0.0); mode == 3 && (wl = 0.0; wu = 0.0); mode == 4 && (wl = Inf; wu = Inf)
                l[i] = max(0.0, αs[i] - wl); u[i] = min(nd.hU[i], αs[i] + wu)
            end
            _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
            if md != "off"
            rg = _nz_dr_ranges(DR, nd, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S))
            if rg === nothing
                ninf += 1; ninf <= 2 && println("    [$md] 범위 MIP 가 실행 불가능 (α* 를 품은 상자): l = ", l, " u = ", u); continue
            end
            lo, hi, wt, phib = rg
            dl, dh = _nz_dr_dglobal(nd, DR)
            isempty(DR.pairs) || ((dl, dh) = _nz_dr_dranges!(DR, nd, l, u, lo, hi, dl, dh, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), wt))
            _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; dlo=dl, dhi=dh)
            end
            z = _nz_sep_solve!(P)
            z == -Inf && (z = _nz_dr_recheck_infeasible!(P.O.model); nrec += 1)
            if z == -Inf
                ninf += 1; ninf <= 1 && println("    노드 LP 가 재확인에서도 실행 불가능 (α* 를 품은 상자)")
            elseif isfinite(z)
                worst = min(worst, z - V)
                z - V < -tolV && (nlow += 1; nlow <= 2 && println("    상한 < V*: z − V* = ", z - V, " l = ", l, " u = ", u))
                z - V <= 1e-4 * max(1.0, abs(V)) && (ntight += 1)
            end
        end
        global nbad += ninf + nlow
        @printf("  x=%-9s [%-4s] 상자 %d: 실행불가 %d (재확인 %d), 상한<V %d, min(상한−V)=%.2e, 정확 %d
", string(xi), md, N, ninf, nrec, nlow, worst, ntight)
        end
        line = @sprintf("  x=%-9s 참조=%-9s V=%11.4f α 고정 LP=%11.4f%s", string(xi), ref, V, zα, mis ? " **불일치**" : "")
        # (B) B&B 대조 (KKT 참조일 때만 독립 대조)
        if ref == "KKT" && envf("TVP_BNB", "1") == "1"
            r = bnb()
            tolB = 1e-4 * max(1.0, abs(V))
            inb = r[:LB] <= V + tolB && r[:UB] >= V - tolB
            global nbad += !inb
            line *= @sprintf(" | B&B [%.4f, %.4f] %s %s", r[:LB], r[:UB], r[:is_exact] ? "Optimal" : "TimeLim", inb ? "∋ V*" : @sprintf("** V* 밖 ** (V*=%.3e)", V))
        elseif rB !== nothing
            line *= @sprintf(" | B&B [%.4f, %.4f] %s", rB[:LB], rB[:UB], rB[:is_exact] ? "Optimal" : "TimeLim")
        end
        println(line); flush(stdout)
    end
end
@printf("RESULT 무효·불일치 합계 %d\n", nbad)
