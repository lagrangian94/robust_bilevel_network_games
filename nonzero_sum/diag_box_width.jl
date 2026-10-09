"""
diag_box_width.jl — α-B&B 노드 완화 (RLT + 분리) 의 상한이 α 상자 폭에 따라 얼마나 빨리 V* 로 수렴하는지.

최적 α* (KKT 의 follower 반응) 를 중심으로 폭 w 인 상자에서 노드 LP 를 풀고 상한 − V* 를 기록.
완화 해에서 목적 F 의 성분을 α* 고정 정확 LP 의 성분과 비교해 어느 항이 부풀었는지도 출력:
  ŷ 항 Σ(ℓ + θc)ᵀŷ,  −θ uᵀϖ,  −θ hcoef ζW,  x 행 항 (π̂ᵁ ρ̂, λᵁ π_Fᵁ ρ̃, λᵁ ρ⁰)
환경변수: BW_X = "1,2"  BW_QUOTA = 150  BW_DELTA = 0.1  BW_EPS = 0.3  BW_THETA = circuit | 숫자
          BW_WIDTHS = "150,50,10,3,1,0.3,0.1,0.03,0.01"   BW_WA = "10,3,1"   BW_WW = "Inf,50,10,1" (2부: α 폭 × ϖ 폭)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
ε = parse(Float64, envf("BW_EPS", "0.3"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=parse(Float64, envf("BW_QUOTA", "150")),
        wres=300.0, S=3, seed=1, eps_hat=ε, eps_tilde=ε, beta=0.4), pd(envf("BW_DELTA", "0.1")))
θ = envf("BW_THETA", "circuit") == "circuit" ? nz_theta_circuit_exact(nd0)[1] : parse(Float64, envf("BW_THETA", "1"))
nd = nz_with_bounds(nd0; thetaU=θ)
x̄ = zeros(nd.nx); x̄[parse.(Int, split(envf("BW_X", "1,2"), ","))] .= 1
k = nz_kkt_value(nd, x̄; optimizer=GRB, time_limit=300.0)
V, αs = k[:value], k[:α]
@printf("%s x=%s θᵁ=%.3f  V*=%.4f  α* = %s\n", nd.name, string(findall(x̄ .> 0.5)), θ, V, string(round.(αs; digits=2)))

function parts(O)
    v = O.v; S, ny, m = nd.S, nz_ny(nd), nz_m(nd); Xr, Hr = v[:Xr], v[:Hr]
    y = sum((nd.ell[j] + θ * nd.c[j]) * value(v[:ŷ][j, s]) for j in 1:ny, s in 1:S)
    u = -θ * sum(nd.u[k, s] * value(v[:ϖ][k, s]) for k in 1:m, s in 1:S if nd.u[k, s] != 0)
    w = -θ * sum(nd.hcoef[k] * value(v[:ζW][jj, s]) for (jj, k) in enumerate(Hr), s in 1:S)
    gx = -θ * sum(nd.g[k, s] * x̄[nd.xrow[k]] * value(v[:ϖ][k, s]) for k in Xr, s in 1:S)
    tot = objective_value(O.model)
    return (y=y, u=u, w=w, gx=gx, rest=tot - y - u - w - gx, tot=tot)
end
F = nz_build_fixed_alpha(nd, x̄; optimizer=GRB)
zex = nz_eval_alpha!(F, nd, αs)
pe = parts(F)
@printf("α* 고정 정확 LP: F=%.2f | ŷ항 %.1f  −θuᵀϖ %.1f  −θhζW %.1f  x행ϖ %.1f  나머지(ρ) %.1f\n", zex, pe.y, pe.u, pe.w, pe.gx, pe.rest)
P = _nz_build_sep(nd, x̄; optimizer=GRB)
for w in pd.(split(envf("BW_WIDTHS", "150,50,10,3,1,0.3,0.1,0.03,0.01"), ","))
    l = max.(0.0, αs .- w / 2); u = min.(nd.hU, αs .+ w / 2)
    _nz_sep_set_box!(P, l, u)
    t = @elapsed z = _nz_sep_solve!(P)
    pr = parts(P.O)
    @printf("w=%7.3f  상한 %12.2f  상한−V* %11.2f | ŷ항 %10.1f  −θuᵀϖ %10.1f  −θhζW %10.1f  x행ϖ %8.1f  ρ %8.1f | RLT행 %d (%.1fs)\n",
            w, z, z - V, pr.y, pr.u, pr.w, pr.gx, pr.rest, count(P.active), t)
    flush(stdout)
end

# ---- 2. ϖ 에도 상자를 주면? (가설: α·ϖ 오차 ∝ α 폭 × ϖ 폭 → ϖ 분기가 필요) ----
# ϖ* = α* 고정 정확 LP 의 H 행 인증서. ϖ 상자 [ϖ* − wϖ/2, ϖ* + wϖ/2] ∩ [0, ϖᵁ rmax] 를 RLT 행 (ϖ − L ≥ 0, U − ϖ ≥ 0) × 자기 α 로.
Hr = F.v[:Hr]
ϖs = Dict((jj, s) => value(F.v[:ϖ][Hr[jj], s]) for jj in eachindex(Hr), s in 1:nd.S)
println("\nα 폭 × ϖ 폭 (ϖ* 범위: ", round(minimum(values(ϖs)); digits=2), " ~ ", round(maximum(values(ϖs)); digits=2), ")")
for wα in pd.(split(envf("BW_WA", "10,3,1"), ",")), wϖ in pd.(split(envf("BW_WW", "Inf,50,10,1"), ","))
    Pq = _nz_build_sep(nd, x̄; optimizer=GRB)
    if isfinite(wϖ)
        for (idx, (jj, s, i)) in enumerate(Pq.wpos)
            ub0 = upper_bound(F.v[:ϖ][Hr[jj], s])
            L = max(0.0, ϖs[(jj, s)] - wϖ / 2); U = min(isfinite(ub0) ? ub0 : Inf, ϖs[(jj, s)] + wϖ / 2)
            push!(Pq.rows, _NZRow(-L, [(:w, idx, 1.0)], i)); push!(Pq.rows, _NZRow(U, [(:w, idx, -1.0)], i))
            set_lower_bound(Pq.Y[:w][idx], L); isfinite(U) && set_upper_bound(Pq.Y[:w][idx], U)
        end
    end
    _nz_sep_set_box!(Pq, max.(0.0, αs .- wα / 2), min.(nd.hU, αs .+ wα / 2))
    z = _nz_sep_solve!(Pq)
    @printf("  wα=%5.1f  wϖ=%5s  상한−V* %10.2f\n", wα, string(wϖ), z - V); flush(stdout)
end

# ---- 3. 약쌍대 부등식 (WD): 시나리오마다 cᵀŷˢ ≤ (uˢ + Σ_X gˢ x̄)ᵀϖˢ + Σ_H hcoef ζWˢ ----
# 정확한 점에서는 약쌍대성으로 성립 (θ 항 ≤ 0). 완화에서는 ζL·ζW 가 따로 완화돼 깨질 수 있음 → 추가하면 벌점이 보상으로 바뀌는 것을 막음.
println("\n약쌍대 부등식 (WD) 추가 효과 (α 상자만, ϖ 상자 없음)")
function add_wd!(Pq)
    v = Pq.O.v; m = Pq.O.model; S, ny, mm = nd.S, nz_ny(nd), nz_m(nd); Hr = v[:Hr]
    for s in 1:S
        @constraint(m, sum(nd.c[j] * v[:ŷ][j, s] for j in 1:ny if nd.c[j] != 0) <=
            sum((nd.u[k, s] + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0)) * v[:ϖ][k, s] for k in 1:mm) +
            sum(nd.hcoef[k] * v[:ζW][jj, s] for (jj, k) in enumerate(Hr)))
    end
end
for w in pd.(split(envf("BW_WD_WIDTHS", "150,50,10,3,1,0.3"), ","))
    for wd in (false, true)
        Pq = _nz_build_sep(nd, x̄; optimizer=GRB)
        wd && add_wd!(Pq)
        _nz_sep_set_box!(Pq, max.(0.0, αs .- w / 2), min.(nd.hU, αs .+ w / 2)); _nz_sep_set_wbox!(Pq, Pq.Lw, Pq.Uw)
        z = _nz_sep_solve!(Pq)
        @printf("  w=%6.1f  WD=%-5s  상한−V* %11.2f\n", w, string(wd), z - V); flush(stdout)
    end
end
