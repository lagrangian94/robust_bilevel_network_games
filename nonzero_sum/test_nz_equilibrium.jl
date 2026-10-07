"""
test_nz_equilibrium.jl — follower 의 aggregate 2단계 LP 가 개별 고객 선주문의 가격 균형과 정확히 같은지 수치 확인.

aggregate (x̄, belief d 고정, reservation = :pair):
  max  Σ_ij c0_ij h_ij + Σ_s d_s cᵀyˢ
  s.t. 수요 Σ_i (y^R+y^S)_ijs ≥ ξ_js,  예약 y^R_ijs ≤ h_ij,  용량 Σ_j (y^R+y^S)_ijs ≤ cap_i(x̄)  [μ_is],
       할당량 Σ_j h_ij ≤ w_i  [κ_i],  h, y ≥ 0
결합 제약은 용량·할당량뿐. 그 dual (μ, κ) 을 가격으로 두면 고객 j 의 문제:
  max  Σ_i (c0_ij − κ_i) h_ij + Σ_s [ d_s c_jᵀ y_jˢ − Σ_i μ_is (y^R+y^S)_ijs ]   s.t. 수요_j, 예약_j
  (단위 rent ρ_is = μ_is / d_s: 시나리오 s 에서 점포 i 의 혼잡 가격 또는 대기 비용)

검사 (각 x̄, d):
  (1) aggregate 해 (h*, y*) 가 각 고객 문제의 최적해인가    : U_j(h*_j, y*_j) ≥ V_j − tol
  (2) 강쌍대                                               : Σ_j V_j + Σ μ cap + Σ κ w = Z
  (3) 역방향: 고객별로 따로 푼 해가 시장을 청산하는가          : 용량·할당량 위반, 청산 시 aggregate 목적 = Z
  (4) 가격 0 (ρ=κ=0) 이면 개별 최적해가 용량·할당량을 어기는가
  (5) pooled (점포별 예약 풀) 와 비교: Z_pooled ≥ Z_pair (pooled 는 완화)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))

all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...) if sum(bits) <= nd.gamma]

"follower aggregate 2단계 LP (일반 형식). 반환 (Z, h, y, μ[cap 행, s], κ[W 행], model 정보)"
function aggregate_lp(nd, x̄, d)
    m, ny, nh, S = size(nd.A, 1), size(nd.A, 2), length(nd.c0), nd.S
    mdl = Model(GRB); set_silent(mdl)
    set_optimizer_attribute(mdl, "FeasibilityTol", 1e-9); set_optimizer_attribute(mdl, "OptimalityTol", 1e-9)
    @variable(mdl, h[1:nh] >= 0); @variable(mdl, y[1:ny, 1:S] >= 0)
    rhs(k, s) = nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) +
                (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0)
    @constraint(mdl, con[k=1:m, s=1:S], sum(nd.A[k, j] * y[j, s] for j in 1:ny if nd.A[k, j] != 0) <= rhs(k, s))
    @constraint(mdl, quota[l=1:size(nd.W, 1)], sum(nd.W[l, i] * h[i] for i in 1:nh) <= nd.wvec[l])
    @objective(mdl, Max, dot(nd.c0, h) + sum(d[s] * dot(nd.c, y[:, s]) for s in 1:S))
    optimize!(mdl)
    termination_status(mdl) == MOI.OPTIMAL || error("aggregate LP: $(termination_status(mdl))")
    μ = [shadow_price(con[k, s]) for k in 1:m, s in 1:S]
    κ = [shadow_price(quota[l]) for l in 1:size(nd.W, 1)]
    return objective_value(mdl), value.(h), value.(y), μ, κ,
           [value(sum(nd.A[k, j] * y[j, s] for j in 1:ny if nd.A[k, j] != 0)) for k in 1:m, s in 1:S]
end

"고객 j 의 개별 문제 (가격 μ, κ). 반환 (최적값, h_j, y_j, 목적함수 평가 함수)"
function customer_problem(nd, x̄, d, j, μ, κ; avail=nothing)
    mt = nd.meta; nst = mt[:nst]; S = nd.S
    rcap, rres, rdem, hidx, iyR, iyS = mt[:rcap], mt[:rres], mt[:rdem], mt[:hidx], mt[:iyR], mt[:iyS]
    store_of_quota(i) = i                                   # pair: W 행 i = 점포 i
    function obj(hj, yR, yS)                                # hj[i], yR[i,s], yS[i,s]
        sum((nd.c0[hidx(i, j)] - κ[store_of_quota(i)]) * hj[i] for i in 1:nst) +
        sum(d[s] * (nd.c[iyR(i, j)] * yR[i, s] + nd.c[iyS(i, j)] * yS[i, s]) -
            μ[rcap(i), s] * (yR[i, s] + yS[i, s]) for i in 1:nst, s in 1:S)
    end
    mdl = Model(GRB); set_silent(mdl)
    set_optimizer_attribute(mdl, "FeasibilityTol", 1e-9); set_optimizer_attribute(mdl, "OptimalityTol", 1e-9)
    @variable(mdl, hj[1:nst] >= 0); @variable(mdl, yR[1:nst, 1:S] >= 0); @variable(mdl, yS[1:nst, 1:S] >= 0)
    ξj = [-nd.u[rdem(j), s] for s in 1:S]
    @constraint(mdl, [s=1:S], sum(yR[i, s] + yS[i, s] for i in 1:nst) >= ξj[s])
    @constraint(mdl, [i=1:nst, s=1:S], yR[i, s] <= hj[i])
    # 닫힌 점포 (용량 0) 는 고객의 선택지에 없음. 그 용량 dual 은 "없는 점포의 가치" 라 혼잡 가격이 아님
    if avail !== nothing
        for i in 1:nst, s in 1:S
            avail[i, s] || (fix(yR[i, s], 0.0; force=true); fix(yS[i, s], 0.0; force=true))
        end
    end
    @objective(mdl, Max, obj(hj, yR, yS))
    optimize!(mdl)
    termination_status(mdl) == MOI.OPTIMAL || error("customer $j: $(termination_status(mdl))")
    return objective_value(mdl), value.(hj), value.(yR), value.(yS), obj
end

function check(nd, x̄, d; verbose=false)
    mt = nd.meta; nst, ncu, S = mt[:nst], mt[:ncu], nd.S
    rcap, hidx, iyR, iyS = mt[:rcap], mt[:hidx], mt[:iyR], mt[:iyS]
    Z, h, y, μ, κ, usage = aggregate_lp(nd, x̄, d)
    capv(i, s) = nd.u[rcap(i), s] + (nd.xrow[rcap(i)] > 0 ? nd.g[rcap(i), s] * x̄[nd.xrow[rcap(i)]] : 0.0)
    avail = [capv(i, s) > 0 for i in 1:nst, s in 1:S]
    # (1), (2)
    worst_br = -Inf; sumV = 0.0
    sol = []
    for j in 1:ncu
        V, hj, yRj, ySj, obj = customer_problem(nd, x̄, d, j, μ, κ; avail=avail)
        hs = [h[hidx(i, j)] for i in 1:nst]
        yRs = [y[iyR(i, j), s] for i in 1:nst, s in 1:S]; ySs = [y[iyS(i, j), s] for i in 1:nst, s in 1:S]
        U = obj(hs, yRs, ySs)
        worst_br = max(worst_br, (V - U) / max(1, abs(V)))
        sumV += V
        push!(sol, (hj, yRj, ySj))
    end
    dual_rev = sum(μ[rcap(i), s] * capv(i, s) for i in 1:nst, s in 1:S) + dot(κ, nd.wvec)
    gap2 = abs(sumV + dual_rev - Z) / max(1, abs(Z))
    # (3) 역방향: 개별 해 묶음의 청산 여부
    over_cap = maximum(sum(sol[j][2][i, s] + sol[j][3][i, s] for j in 1:ncu) - capv(i, s) for i in 1:nst, s in 1:S)
    over_q = maximum(sum(sol[j][1][i] for j in 1:ncu) - nd.wvec[i] for i in 1:nst)
    # (4) 가격 0
    over0_cap = -Inf; over0_q = -Inf
    s0 = [customer_problem(nd, x̄, d, j, zero(μ), zero(κ); avail=avail)[2:4] for j in 1:ncu]
    over0_cap = maximum(sum(s0[j][2][i, s] + s0[j][3][i, s] for j in 1:ncu) - capv(i, s) for i in 1:nst, s in 1:S)
    over0_q = maximum(sum(s0[j][1][i] for j in 1:ncu) - nd.wvec[i] for i in 1:nst)
    nbind_cap = count(avail[i, s] && μ[rcap(i), s] > 1e-7 for i in 1:nst, s in 1:S)     # 열린 점포만
    nbind_q = count(κ .> 1e-7)
    return (Z=Z, br=worst_br, dual=gap2, over_cap=over_cap, over_q=over_q, over0_cap=over0_cap, over0_q=over0_q,
            nbind_cap=nbind_cap, nbind_q=nbind_q,
            μmax=maximum([μ[rcap(i), s] for i in 1:nst, s in 1:S if avail[i, s]]; init=0.0), κmax=maximum(κ))
end

function tv_samples(q, ε, n; rng)
    S = length(q); out = [copy(q)]
    while length(out) < n
        z = randn(rng, S); z .-= sum(z) / S                # 합 0 방향
        d = q .+ z .* (2ε * rand(rng) / sum(abs.(z)))
        all(d .>= 0) && push!(out, d)
    end
    return out
end

function main()
    cases = [
        ("기본 (B 용량 360, 할당량 100)", (bB=360.0, quota=100.0)),
        ("결합 제약 묶임 (B 용량 120, 할당량 40)", (bB=120.0, quota=40.0)),
        ("벤치마크 대조: 예약 없음 (할당량 0), B 용량 120", (bB=120.0, quota=0.0)),
    ]
    rng = MersenneTwister(0)
    for (title, kw) in cases
        nd = make_location_instance(; S=3, seed=4, reservation=:pair, kw...)
        ndp = make_location_instance(; S=3, seed=4, reservation=:pooled, wres=kw.quota * nd.meta[:nst], bB=kw.bB)
        # (3)·(4) 의 용량 위반은 열린 점포 기준 (닫힌 점포는 고객 선택지에서 제외)
        ds = tv_samples(nd.q_hat, nd.eps_tilde, 4; rng=rng)
        println("="^100, "\n$title — $(nd.name), x̄ 16 개 × belief d $(length(ds)) 개\n", "="^100)
        @printf("  %-12s %-5s %12s | %9s %9s | %9s %9s | %9s %9s | %7s %5s %9s | %12s\n", "x̄", "d#", "Z (pair)",
                "(1)BR", "(2)쌍대", "(3)용량", "(3)할당", "(4)용량", "(4)할당", "용량묶임", "할당", "max μ", "Z pooled")
        worst = Dict(:br => -Inf, :dual => 0.0, :ocap => -Inf, :oq => -Inf)
        n_bind = 0; n_total = 0; n_noprice_viol = 0; n_nonclear = 0
        for x̄ in all_x(nd), (di, d) in enumerate(ds)
            r = check(nd, x̄, d)
            Zp = aggregate_lp(ndp, x̄, d)[1]
            n_total += 1
            (r.nbind_cap + r.nbind_q > 0) && (n_bind += 1)
            (max(r.over0_cap, r.over0_q) > 1e-6) && (n_noprice_viol += 1)
            (max(r.over_cap, r.over_q) > 1e-6) && (n_nonclear += 1)
            worst[:br] = max(worst[:br], r.br); worst[:dual] = max(worst[:dual], r.dual)
            if di == 1 || r.nbind_cap + r.nbind_q > 0
                @printf("  %-12s %-5d %12.4f | %9.1e %9.1e | %9.2f %9.2f | %9.2f %9.2f | %7d %5d %9.4f | %12.4f\n",
                        string(findall(x̄ .> 0.5)), di, r.Z, r.br, r.dual, r.over_cap, r.over_q, r.over0_cap, r.over0_q,
                        r.nbind_cap, r.nbind_q, r.μmax, Zp)
            end
        end
        @printf("  요약: (x̄, d) %d 개 중 결합 제약 묶임 %d 개 | (1) 최대 상대 BR 차이 %.1e | (2) 최대 상대 쌍대 차이 %.1e\n",
                n_total, n_bind, worst[:br], worst[:dual])
        @printf("        (3) 개별 해 묶음이 시장 청산 못한 경우 %d 개 | (4) 가격 0 에서 용량·할당 위반 %d 개\n",
                n_nonclear, n_noprice_viol)
        flush(stdout)
    end
end
main()
