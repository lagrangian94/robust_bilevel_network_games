"""
diag_vex_envelope.jl — 노드 완화에 "follower recourse 값 Q_s 의 상자 꼭짓점 볼록 포락" 을 넣으면 상한이 얼마나 조여지는지.

배경: 리더 블록은 r_s [P_s(α) − θ Q_s(α)] (P_s, Q_s 모두 α 에 오목). −θ Q_s 는 볼록이라 완화가 필요하고, 현재는 인증서 곱
ζW = α ϖ 의 McCormick/RLT + 약쌍대 (WD) 로 완화한다. 여기서는 쌍대목적 D_s = uˢᵀϖˢ + Σ_H hcoef ζWˢ + Σ_X gˢ x̄ ϖˢ 에
  D_s ≥ Σ_k λ_sk Q_s(α^k),   Σ_k λ_sk = r_s,   Σ_k λ_sk α^k = ζL_s,   λ ≥ 0      (α^k: 노드 상자의 꼭짓점)
를 더한다. 유효성: 정확한 점에서 D_s = r_s ρ_s(α)ᵀσ_s ≥ r_s Q_s(α) (σ_s 쌍대 실행가능), Q_s 오목이라 α = Σ μ_k α^k 이면
Q_s(α) ≥ Σ μ_k Q_s(α^k), λ_sk = r_s μ_k 로 두면 위 식을 만족. (오목함수의 상자 위 볼록 포락은 꼭짓점 값으로 생성됨)

비교: 최적 α* 중심 폭 w 상자에서 (cur) 현재 완화, (vex) 포락 추가. 상한 − V* 를 출력.
환경변수: DV_S = 3  DV_X = "1,2"  DV_EPS = 0.3  DV_DELTA = 0.1  DV_WIDTHS = "150,50,10,3,1,0.3"
          DV_ALPHA = kkt | abb (α* 와 기준값을 KKT 로 / 짧은 α-B&B 의 incumbent 로)  DV_ABB_TL = 60
실행: julia -t 14,1 nonzero_sum/diag_vex_envelope.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
S = parse(Int, envf("DV_S", "3")); ε = parse(Float64, envf("DV_EPS", "0.3")); δ = pd(envf("DV_DELTA", "0.1"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                           eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[parse.(Int, split(envf("DV_X", "1,2"), ","))] .= 1
hUp, nfix = nz_presolve_hU(ndf, x̄)
nd = nz_with_hU(ndf, hUp)                       # 닫힌 점포의 예약 좌표를 0 으로 (α-B&B 의 프리솔브와 같음)
nh = nz_nh(nd)

if envf("DV_ALPHA", "kkt") == "kkt"
    k = nz_kkt_value(ndf, x̄; optimizer=GRB, time_limit=600.0)
    V, αs = k[:value], min.(k[:α], nd.hU)
    @printf("%s x=%s  KKT V*=%.4f\n", nd.name, string(findall(x̄ .> 0.5)), V)
else
    r = nz_alpha_bnb(ndf, x̄; nworkers=12, time_limit=parse(Float64, envf("DV_ABB_TL", "60")), rel_gap=1e-3, verbose=false)
    V, αs = r[:LB], min.(r[:α], nd.hU)
    @printf("%s x=%s  α-B&B incumbent LB=%.4f (UB %.4f) — 아래의 '상한−V*' 는 이 LB 기준\n", nd.name, string(findall(x̄ .> 0.5)), V, r[:UB])
end
println("α* = ", round.(αs; digits=2)); flush(stdout)

# 시나리오 s 의 follower recourse LP (rhs 만 바꿔 재사용)
function build_Q(s)
    mdl = Model(GRB); set_silent(mdl); set_optimizer_attribute(mdl, "Threads", 1)
    @variable(mdl, y[1:nz_ny(nd)] >= 0)
    con = @constraint(mdl, [k=1:nz_m(nd)], sum(nd.A[k, j] * y[j] for j in 1:nz_ny(nd) if nd.A[k, j] != 0) <= 0.0)
    @objective(mdl, Max, dot(nd.c, y))
    return mdl, con
end
const QM = [build_Q(s) for s in 1:S]
function Qval(s, α)
    mdl, con = QM[s]
    rr = nz_rhs(nd, x̄, α, s)
    for k in eachindex(rr); set_normalized_rhs(con[k], rr[k]); end
    optimize!(mdl)
    termination_status(mdl) == MOI.OPTIMAL || error("Q LP s=$s: $(termination_status(mdl))")
    return objective_value(mdl)
end

"상자 [l, u] 의 꼭짓점 포락 행을 P 에 추가. 반환: (꼭짓점 수, Q 계산 시간)"
function add_vex!(P, l, u)
    v = P.O.v; m = P.O.model; Hr, Xr = v[:Hr], v[:Xr]
    I = findall(u .- l .> 1e-9); n = length(I)
    verts = Vector{Vector{Float64}}(undef, 2^n)
    for mask in 0:(2^n - 1)
        a = copy(l)
        for (b, i) in enumerate(I); (mask >> (b - 1)) & 1 == 1 && (a[i] = u[i]); end
        verts[mask + 1] = a
    end
    tq = @elapsed Qv = [Qval(s, a) for s in 1:S, a in verts]
    nv = length(verts)
    @variable(m, λ[1:S, 1:nv] >= 0)
    @constraint(m, [s=1:S], sum(λ[s, k] for k in 1:nv) == v[:r][s])
    @constraint(m, [s=1:S, i=1:nh], sum(verts[k][i] * λ[s, k] for k in 1:nv) == v[:ζL][i, s])
    @constraint(m, [s=1:S],
        sum(nd.u[k, s] * v[:ϖ][k, s] for k in 1:nz_m(nd) if nd.u[k, s] != 0)
        + sum(nd.hcoef[k] * v[:ζW][jj, s] for (jj, k) in enumerate(Hr))
        + sum(nd.g[k, s] * x̄[nd.xrow[k]] * v[:ϖ][k, s] for k in Xr)
        >= sum(Qv[s, k] * λ[s, k] for k in 1:nv))
    return nv, tq
end

for w in pd.(split(envf("DV_WIDTHS", "150,50,10,3,1,0.3"), ","))
    l = max.(0.0, αs .- w / 2); u = min.(nd.hU, αs .+ w / 2)
    res = Dict{Symbol,Any}()
    for mode in (:cur, :vex)
        P = _nz_build_sep(nd, x̄; optimizer=GRB)
        _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
        nv, tq = mode == :vex ? add_vex!(P, l, u) : (0, 0.0)
        t = @elapsed z = _nz_sep_solve!(P)
        res[mode] = (z=z, t=t, nv=nv, tq=tq, rows=count(P.active))
    end
    c, x = res[:cur], res[:vex]
    @printf("w=%7.2f | cur 상한−V* %10.3f (RLT %5d, %.1fs) | vex 상한−V* %10.3f (RLT %5d, 꼭짓점 %d, Q %.1fs, LP %.1fs) | 비율 %.3f\n",
            w, c.z - V, c.rows, c.t, x.z - V, x.rows, x.nv, x.tq, x.t, (x.z - V) / max(c.z - V, 1e-12))
    flush(stdout)
end
