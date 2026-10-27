"""
nz_ccg.jl — 벤치마크 CCG (column-and-constraint generation, Zeng & An 계열, 2026-10-22 사용자 결정).

V*(x) = max_{(p,p̃)∈D} max_{h∈Ψ(x,p̃)} R^p[φ(x,h,ξ)] 의 적대적 선택 (truth p, belief p̃) 을 "시나리오" 로 하나씩 모은다.
  master: min_x q_xᵀx + η  s.t. 모은 시나리오 j 마다
            · follower 2SP 의 KKT (belief p̃_j, 우변 r_s(x, h_j) — x 는 우변에만 있어 상보성 slack 에 선형, indicator)
            · 리더 평가 recourse 복사본 ŷ_j 의 KKT (우변 r_s(x, h_j))
            · η ≥ CVaR^{p_j}_β[ℓᵀŷ_j] = τ_j + Σ_s p_js u_js / (1−β), u_js ≥ ℓᵀŷ_js − τ_j, u ≥ 0
          → MILP (binary = x + 상보성 indicator). belief 를 하나 넣을 때마다 x 에 의존하는 KKT 블록이 하나씩 붙는다.
  subproblem: x̄ 에서 V*(x̄) 를 우리 spatial B&B (nz_alpha_bnb_dr, Ours 와 같은 설정·목표값 조기 종료) 로 → 상한 UB,
          incumbent response 를 고정한 LP (restricted ISP) 에서 최악 (p, p̃) 를 읽어 새 시나리오로.
master 는 각 시나리오 안에서 follower 의 최적해가 여럿일 때 (h 와 recourse 의 동률) 리더에게 유리한 것을 고른다 (낙관적).
pessimistic 동률 처리는 KKT 만으로는 master (min) 에 넣을 수 없으므로 (max 의 쌍대가 필요 = 우리 재정식) master 값은
pessimistic 값의 하한 → LB 는 유효, UB 는 subproblem 의 정확한 상한. 동률이 최적에 영향을 주면 같은 x̄ 에서 진전이 없을 수
있고, 그때는 :Stalled 로 끝낸다.
필요: nz_data.jl, nz_omega.jl, nz_kkt_eval.jl (_comp!), nz_benders.jl (_nz_omp, _rel, _remaining, _oracle_gap_eff), nz_alpha_bnb.jl,
      nz_alpha_bnb_dr.jl
"""

"master 에 시나리오 (p, p̃) 블록 추가. 반환: 블록의 h 변수"
function _nz_ccg_add_scenario!(omp, x, t0, nd::NZData, p::Vector{Float64}, p̃::Vector{Float64})
    A, c, ℓ = nd.A, nd.c, nd.ell
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S
    nW = size(nd.W, 1); β = nd.beta
    α = @variable(omp, [i=1:nh], lower_bound = 0.0, upper_bound = nd.hU[i])
    y = @variable(omp, [1:ny, 1:S], lower_bound = 0.0)
    π = @variable(omp, [1:m, 1:S], lower_bound = 0.0)      # p̃_s 로 스케일된 follower dual
    κ = @variable(omp, [1:nW], lower_bound = 0.0)
    R = [[nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) +
          (nd.xrow[k] > 0 ? nd.g[k, s] * x[nd.xrow[k]] : 0.0) for k in 1:m] for s in 1:S]
    slackP = [@expression(omp, R[s][k] - sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0)) for k in 1:m, s in 1:S]
    slackD = [@expression(omp, sum(A[k, j] * π[k, s] for k in 1:m if A[k, j] != 0) - c[j] * p̃[s]) for j in 1:ny, s in 1:S]
    @constraint(omp, [k=1:m, s=1:S], slackP[k, s] >= 0)
    @constraint(omp, [j=1:ny, s=1:S], slackD[j, s] >= 0)
    slackW = [@expression(omp, nd.wvec[l] - sum(nd.W[l, i] * α[i] for i in 1:nh)) for l in 1:nW]
    @constraint(omp, [l=1:nW], slackW[l] >= 0)
    slackA = [@expression(omp, sum(nd.W[l, i] * κ[l] for l in 1:nW) - nd.c0[i]
                               - sum(nd.hcoef[k] * π[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i)) for i in 1:nh]
    @constraint(omp, [i=1:nh], slackA[i] >= 0)
    for k in 1:m, s in 1:S; _comp!(omp, π[k, s], slackP[k, s]); end
    for j in 1:ny, s in 1:S; _comp!(omp, y[j, s], slackD[j, s]); end
    for l in 1:nW; _comp!(omp, κ[l], slackW[l]); end
    for i in 1:nh; _comp!(omp, α[i], slackA[i]); end
    # 리더 평가 recourse 복사본
    ŷ = @variable(omp, [1:ny, 1:S], lower_bound = 0.0)
    σ = @variable(omp, [1:m, 1:S], lower_bound = 0.0)
    sP = [@expression(omp, R[s][k] - sum(A[k, j] * ŷ[j, s] for j in 1:ny if A[k, j] != 0)) for k in 1:m, s in 1:S]
    sD = [@expression(omp, sum(A[k, j] * σ[k, s] for k in 1:m if A[k, j] != 0) - c[j]) for j in 1:ny, s in 1:S]
    @constraint(omp, [k=1:m, s=1:S], sP[k, s] >= 0)
    @constraint(omp, [j=1:ny, s=1:S], sD[j, s] >= 0)
    for k in 1:m, s in 1:S; _comp!(omp, σ[k, s], sP[k, s]); end
    for j in 1:ny, s in 1:S; _comp!(omp, ŷ[j, s], sD[j, s]); end
    # η ≥ CVaR^p_β[ℓᵀŷ]
    τ = @variable(omp); u = @variable(omp, [1:S], lower_bound = 0.0)
    @constraint(omp, [s=1:S], u[s] >= sum(ℓ[j] * ŷ[j, s] for j in 1:ny if ℓ[j] != 0) - τ)
    @constraint(omp, t0 >= τ + sum(p[s] * u[s] for s in 1:S) / (1 - β))
    return α
end

function nz_ccg(nd::NZData; optimizer, max_iter=500, tol=5e-3, oracle_gap=1e-3, time_limit=3600.0, nworkers=12,
                bnb_kw=NamedTuple(), verbose=true)
    verbose && (println("  [설정] CCG: master = 시나리오별 KKT 블록 (MILP, 동률 낙관), subproblem = 우리 spatial B&B (목표값 조기 종료)"); flush(stdout))
    omp, x, t0 = _nz_omp(nd; optimizer=optimizer)
    Fx = Main.nz_build_fixed_alpha(nd, zeros(nd.nx); optimizer=optimizer)
    scen = Tuple{Vector{Float64},Vector{Float64}}[]
    push!(scen, (copy(nd.q_hat), copy(nd.q_hat)))
    _nz_ccg_add_scenario!(omp, x, t0, nd, scen[1]...)
    LB, UB, best_x = -Inf, Inf, zeros(nd.nx)
    wall = time(); iter = 0; status = :MaxIter; t_master = 0.0; t_sub = 0.0; stall = 0; prev = nothing
    while iter < max_iter
        iter += 1
        rem = time_limit - (time() - wall)
        rem <= 0 && (status = :TimeLimit; break)
        set_time_limit_sec(omp, rem)
        t_m = @elapsed optimize!(omp); t_master += t_m
        stm = termination_status(omp)
        if stm != MOI.OPTIMAL
            stm == MOI.TIME_LIMIT || error("CCG master: $stm")
            # 시간 한도: master 의 하한 (bound) 만 LB 로
            LB = max(LB, objective_bound(omp)); status = :TimeLimit; break
        end
        x̄ = Float64.(value.(x) .> 0.5); η = value(t0)
        LB = objective_value(omp)
        _rel(LB, UB) <= tol && (status = :Optimal; break)
        rem = time_limit - (time() - wall)
        rem <= 0 && (status = :TimeLimit; break)
        target = η + 0.99 * tol * max(1.0, abs(LB)) / (1 + tol)
        gap_o = _oracle_gap_eff(oracle_gap, tol, LB, η)
        t_s = @elapsed (r = Main.nz_alpha_bnb_dr(nd, x̄; nworkers=nworkers, time_limit=rem, rel_gap=gap_o, verbose=false,
                                                target=target, bnb_kw...))
        t_sub += t_s
        nz_set_objective!(Fx, nd, x̄)
        z = Main.nz_eval_alpha!(Fx, nd, r[:α])
        bel = nz_read_belief(Fx, nd)
        zbd = max(r[:UB], z)
        ub_here = dot(nd.qx, x̄) + zbd
        ub_here < UB && (UB = ub_here; best_x = copy(x̄))
        newsc = z > η + 1e-6 * max(1.0, abs(η))
        p = max.(bel[:a], 0.0); p ./= sum(p); p̃ = max.(bel[:d], 0.0); p̃ ./= sum(p̃)
        if newsc
            push!(scen, (p, p̃)); _nz_ccg_add_scenario!(omp, x, t0, nd, p, p̃)
        end
        # 같은 x̄ 에서 새 시나리오가 master 값을 못 올리면 (동률 낙관) 정체
        key = (x̄, round(LB; digits=6))
        stall = (key == prev) ? stall + 1 : 0; prev = key
        verbose && @printf("  [ccg]  it %3d LB=%11.4f UB=%11.4f x=%-10s η=%11.4f V(x̄)≥%11.4f bd=%11.4f scen=%d%s (master %.1fs, sub %.1fs)\n",
                           iter, LB, UB, string(findall(x̄ .> 0.5)), η, z, zbd, length(scen), newsc ? " (+)" : "", t_m, t_s)
        verbose && flush(stdout)
        _rel(LB, UB) <= tol && (status = :Optimal; break)
        stall >= 3 && (status = :Stalled; break)
        (!newsc && zbd > target) && (status = :Stalled; break)
    end
    return Dict(:status => status, :LB => LB, :UB => UB, :x => best_x, :iters => iter, :scenarios => length(scen),
                :master_time => t_master, :sub_time => t_sub, :wall => time() - wall)
end
