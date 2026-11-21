# polymatroid (Lovász 확장) 부등식이 높은점 완화의 루트 하한을 얼마나 올리는가 (2026-11-20, 효과 측정용 — 유효성은 V_F submodular 가정).
#   원소 = 증분 (거점 i, 단계 k): u_ik(x) = Σ_{k′ ≥ k} x_ik′ ∈ [0,1], 같은 거점 안에서 u_i1 ≥ u_i2 ≥ u_i3.
#   V_F (follower 이득, 최대화) 가 증분 격자에서 submodular 면  conv{(x, F): F ≤ ... } 가 아니라 follower 최적성 F ≥ V_F(x) 의 볼록 껍질은
#   F ≥ V_F(∅) + Σ_j [V_F(S_j) − V_F(S_{j−1})] u_{e_j}(x)   (u 내림차순 사슬 S_0 ⊂ S_1 ⊂ … 마다, greedy 분리)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(@__DIR__, "bgen_core.jl"))
envf(k, d) = get(ENV, k, d)
nd = make_loc_tkp(; g=parse(Int, envf("GEO", "1")), S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
_BXROWS[] = unique(e.k for e in nz_xentries(nd)); K, nB, S = nd.meta[:K], nd.meta[:nB], nd.S; qh = nd.q_hat
m, ny, nh, nx = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.nx; A = nd.A; XE = nz_xentries(nd)
rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]; ents_of_row = [[e for e in XE if e.k == k] for k in 1:m]
d = qh; r = qh
ev = BEv(nd, GRB_ENV); cache = Dict{Vector{Int},Float64}()
function VF(lev::Vector{Int})
    get!(cache, copy(lev)) do
        xv = zeros(nx); for i in 1:nB; lev[i] > 0 && (xv[(i - 1) * K + lev[i]] = 1.0); end
        bev_setx!(ev, nd, xv); bev_resp!(ev, nd, d) === nothing && error("follower LP 실패"); objective_value(ev.R)
    end
end
V0 = VF(zeros(Int, nB))
M = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 8); o))
@variable(M, 0 <= x[1:nx] <= 1); @variable(M, t0 >= -1e5)
@constraint(M, sum(x) <= nd.gamma)
for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(x[i] for i in idx) <= rhs); end
@variable(M, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(M, nd.W * h .<= nd.wvec); @variable(M, y[1:ny, 1:S] >= 0)
@constraint(M, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0)
                               - sum(e.g[s] * x[e.i] for e in ents_of_row[k]; init=0.0) <= nd.u[k, s])
@constraint(M, t0 >= sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0))
Fexpr = @expression(M, dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
@constraint(M, Fexpr >= V0)
@objective(M, Min, dot(nd.qx, x) + t0)
