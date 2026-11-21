# μ 벌점 하한  LB_μ(x) = min_y [μ·cost_F(y) + cost_L(y)] − μ·V_F(x)   (cost_F = follower 비용, cost_L = 리더 비용 = −수익)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Statistics
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
MM = nd.meta; nA, nB, S = MM[:nA], MM[:nB], nd.S
m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); A = nd.A; XE = nz_xentries(nd); d = nd.q_hat; r = nd.q_hat
rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]
F = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); o))
@variable(F, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(F, nd.W * h .<= nd.wvec); @variable(F, y[1:ny, 1:S] >= 0)
@constraint(F, con[k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s])
costF = @expression(F, -(dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0)))
costL = @expression(F, sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0))
xrows = unique(e.k for e in XE)
function setx!(lev)
    for k in xrows, s in 1:S; set_normalized_rhs(con[k, s], nd.u[k, s]); end
    for (st, l) in enumerate(lev); l == 0 && continue; k = MM[:rcap](nA + st); for s in 1:S; set_normalized_rhs(con[k, s], nd.u[k, s] + MM[:tiers][l]); end; end
end
rows = []
for line in eachline(joinpath(@__DIR__, "vtable_g1_nominal.tsv"))
    startswith(line, "set") && continue
    p_ = split(line, '\t'); isempty(p_[1]) && continue
    lev = zeros(Int, nB); for tk in split(p_[1], ';'); a, b = parse.(Int, split(tk, ':')); lev[a] = b; end
    push!(rows, (lev, parse(Float64, p_[2]), parse(Float64, p_[3]), -parse(Float64, p_[4])))     # (lev, q, V, V_F 비용)
end
best = minimum(q + V for (_, q, V, _) in rows)
@printf("최적 %.2f\n", best)
for μ in (1 / 120, 2 / 120, 5 / 120, 10 / 120, 30 / 120, 1.0)
    @objective(F, Min, μ * costF + costL)
    lb = Float64[]; viol = 0; t = @elapsed for (lev, q, V, VFc) in rows
        setx!(lev); optimize!(F); b = objective_value(F) - μ * VFc
        b > V + 1e-6 * max(1.0, abs(V)) && (viol += 1); push!(lb, (V - b) / abs(V))
    end
    surv = count(i -> rows[i][2] + (rows[i][3] - lb[i] * abs(rows[i][3])) < best + 0.005 * abs(best), eachindex(rows))
    @printf("MU μ·κ = %5.1f: 위반 %d | (V − LB)/|V| 중앙 %.3f, 90%% %.3f, 최대 %.3f | 남는 x %d / %d (%.0fs)\n", μ * 120, viol, median(lb), quantile(lb, 0.9), maximum(lb), surv, length(rows), t)
end
