"""
diag_full_rlt.jl — α-B&B 노드 완화에 full RLT 를 더하면 상자 폭에 대한 상한 수렴이 얼마나 빨라지는지 (프로토타입).

현재 노드 완화: belief 행 × α, 인증서 (ϖ ≥ 0, ϖ ≤ ϖᵁ r) × 자기 α, WD. 여기에 곱 변수
  Ξ_iks = α_i ϖ_ks (모든 행 k),  ψ_ijs = α_i ŷ_js
와 (α_i − l_i), (u_i − α_i) 를 곱한 RLT 행을 추가:
  쌍대 실행가능성 Σ_k A_kj ϖ_ks ≥ c_j r_s,  ϖ_ks ≥ 0,  ŷ_js ≥ 0,  Wcap ϖ_ks ≤ ϖᵁ_k r_s (H 행),
  우변에 α 가 없는 primal 행 Σ_j A_kj ŷ_js (+ ρ̂ 항 제외) ≤ u_ks r_s (H 행 제외, x 행 제외 — ρ̂ 와의 곱이 필요),
  H 행의 Ξ_{h(k),k,s} = ζW (같은 곱).
최적 α* 중심 폭 w 상자에서 (현재 완화) vs (+ full RLT) 의 상한 − V*.
환경변수: FR_X = "1,2"  FR_WIDTHS = "150,10,3,1,0.3"
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=3, seed=1,
        eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(nd.nx); x̄[parse.(Int, split(envf("FR_X", "1,2"), ","))] .= 1
k = nz_kkt_value(nd, x̄; optimizer=GRB, time_limit=300.0); V, αs = k[:value], k[:α]
@printf("V* = %.4f\n", V)

function add_full_rlt!(P, l, u)
    v = P.O.v; m = P.O.model; S, ny, mm, nh = nd.S, nz_ny(nd), nz_m(nd), nz_nh(nd)
    A, c = nd.A, nd.c; Hr = v[:Hr]; Xr = v[:Xr]
    α, r, ϖ, ŷ, ζL, ζW = v[:α], v[:r], v[:ϖ], v[:ŷ], v[:ζL], v[:ζW]
    Ξ = @variable(m, [1:nh, 1:mm, 1:S])
    ψ = @variable(m, [1:nh, 1:ny, 1:S])
    for (jj, kk) in enumerate(Hr), s in 1:S
        @constraint(m, Ξ[nd.hrow[kk], kk, s] == ζW[jj, s])
    end
    # 인수 (α_i − l_i) ≥ 0 과 (u_i − α_i) ≥ 0. g(·) ≥ 0 이 (변수 v, 곱 변수 Z, 상수 c0) 형태일 때
    #  (α − l) g = Σ coef·Z_i − l Σ coef·v + c0 (α − l)  ≥ 0,   (u − α) g = u Σ coef·v − Σ coef·Z_i + c0 (u − α) ≥ 0
    for i in 1:nh
        li, ui = l[i], u[i]
        fac(pairs, rpairs) = begin   # pairs: (coef, 원변수, 곱변수), rpairs: (coef, r_s, ζL_is) 는 우변 r 항
            e1 = sum(cf * (Z - li * x) for (cf, x, Z) in pairs; init=AffExpr(0.0)) +
                 sum(cf * (Z - li * x) for (cf, x, Z) in rpairs; init=AffExpr(0.0))
            e2 = sum(cf * (ui * x - Z) for (cf, x, Z) in pairs; init=AffExpr(0.0)) +
                 sum(cf * (ui * x - Z) for (cf, x, Z) in rpairs; init=AffExpr(0.0))
            @constraint(m, e1 >= 0); @constraint(m, e2 >= 0)
        end
        for s in 1:S
            for kk in 1:mm; fac([(1.0, ϖ[kk, s], Ξ[i, kk, s])], []); end                          # ϖ ≥ 0
            for j in 1:ny; fac([(1.0, ŷ[j, s], ψ[i, j, s])], []); end                             # ŷ ≥ 0
            for j in 1:ny                                                                          # Aᵀϖ − c r ≥ 0
                fac([(A[kk, j], ϖ[kk, s], Ξ[i, kk, s]) for kk in 1:mm if A[kk, j] != 0], [(-c[j], r[s], ζL[i, s])])
            end
            for kk in Hr                                                                           # ϖᵁ r − ϖ ≥ 0
                isfinite(nd.varpiU[kk]) || continue
                fac([(-1.0, ϖ[kk, s], Ξ[i, kk, s])], [(nd.varpiU[kk], r[s], ζL[i, s])])
            end
            for kk in 1:mm                                                                         # u r − Aŷ ≥ 0 (H, X 행 제외)
                (nd.hrow[kk] > 0 || nd.xrow[kk] > 0) && continue
                fac([(-A[kk, j], ŷ[j, s], ψ[i, j, s]) for j in 1:ny if A[kk, j] != 0], [(nd.u[kk, s], r[s], ζL[i, s])])
            end
        end
    end
end
for w in pd.(split(envf("FR_WIDTHS", "150,10,3,1,0.3"), ","))
    l = max.(0.0, αs .- w / 2); u = min.(nd.hU, αs .+ w / 2)
    out = Float64[]
    for full in (false, true)
        P = _nz_build_sep(nd, x̄; optimizer=GRB)
        _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
        full && add_full_rlt!(P, l, u)
        t = @elapsed z = _nz_sep_solve!(P)
        push!(out, z - V)
        @printf("  w=%6.1f  full=%-5s  상한−V* %10.3f  (%.1fs, 행 %d)\n", w, string(full), z - V, t,
                num_constraints(P.O.model; count_variable_in_set_constraints=false)); flush(stdout)
    end
end
