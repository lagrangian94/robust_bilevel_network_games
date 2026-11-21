# 작은 무작위 인스턴스에서 "집합 부분가법성" R(T∪S) ≤ R(T) + R(S) 와 거점별 대체재 성질이 일반적으로 성립하는지 반례 탐색 (2026-11-21).
#   follower: min f Σ h + Σ_s d_s Σ_ij (dist_ij (yR+yS)_ijs + p yS_ijs)
#     B 거점: Σ_j yR_ijs + sl_is = h_i (약정은 쓰든 안 쓰든), h_i + Σ_j yS_ijs ≤ cap_i,  A 거점: Σ_j yR ≤ h_i, Σ_j (yR+yS) ≤ capA
#     h_i ≤ w,  수요 Σ_i (yR+yS)_ijs ≥ ξ_js
#   리더 수입 (B): Σ_i [f h_i + p Σ_s r_s Σ_j yS_ijs]  (비관적: follower 최적 중 최소)
using JuMP, Gurobi, Random, Printf
const ENVG = Gurobi.Env()
envf(k, d) = get(ENV, k, d)
NTR = parse(Int, envf("NTR", "300")); SEED = parse(Int, envf("SEED", "1")); STAGE1 = envf("STAGE1", "1") == "1"; WA = envf("WA", "same"); WB = envf("WB", "same")
rng = MersenneTwister(SEED)
f, p = 600.0, 800.0
function solveR(cap, inst)
    (; nA, nB, nc, S, dist, ξ, d, r, w, capA) = inst
    n = nA + nB
    M = Model(() -> (o = Gurobi.Optimizer(ENVG); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o))
    set_optimizer_attribute(M, "FeasibilityTol", 1e-9); set_optimizer_attribute(M, "OptimalityTol", 1e-9)
    @variable(M, 0 <= h[i=1:n] <= (i <= nA ? (WA == "same" ? w : parse(Float64, WA)) : (WB == "same" ? w : parse(Float64, WB)))); @variable(M, yR[1:n, 1:nc, 1:S] >= 0); @variable(M, yS[1:n, 1:nc, 1:S] >= 0); @variable(M, sl[1:n, 1:S] >= 0)
    if !STAGE1                                          # 대조: h 를 시나리오별로 고를 수 있으면 (단일 단계 네트워크)
        @variable(M, 0 <= hs[1:n, 1:S] <= w)
    end
    H(i, s) = STAGE1 ? h[i] : hs[i, s]
    for s in 1:S
        for i in 1:nA
            @constraint(M, sum(yR[i, j, s] for j in 1:nc) <= H(i, s)); @constraint(M, sum(yR[i, j, s] + yS[i, j, s] for j in 1:nc) <= capA)
        end
        for b in 1:nB
            i = nA + b
            @constraint(M, sum(yR[i, j, s] for j in 1:nc) + sl[i, s] == H(i, s)); @constraint(M, H(i, s) + sum(yS[i, j, s] for j in 1:nc) <= cap[b])
        end
        for j in 1:nc; @constraint(M, sum(yR[i, j, s] + yS[i, j, s] for i in 1:n) >= ξ[j, s]); end
    end
    hcost = STAGE1 ? f * sum(h) : f * sum(d[s] * hs[i, s] for i in 1:n, s in 1:S)
    CF = @expression(M, hcost + sum(d[s] * (dist[i, j] * (yR[i, j, s] + yS[i, j, s]) + p * yS[i, j, s]) for i in 1:n, j in 1:nc, s in 1:S))
    hrev(i) = STAGE1 ? f * h[i] : f * sum(r[s] * hs[i, s] for s in 1:S)
    rev = [@expression(M, hrev(nA + b) + p * sum(r[s] * yS[nA + b, j, s] for j in 1:nc, s in 1:S)) for b in 1:nB]
    @objective(M, Min, CF); optimize!(M); termination_status(M) == MOI.OPTIMAL || error("LP1 $(termination_status(M))")
    v = objective_value(M); @constraint(M, CF <= v + 1e-9 * abs(v) + 1e-9)
    @objective(M, Min, sum(rev)); optimize!(M); termination_status(M) == MOI.OPTIMAL || error("LP2 $(termination_status(M))")
    return value.(rev), objective_value(M)
end
nviol_set = 0; nviol_site = 0; worst_set = 0.0; worst_site = 0.0; ntest = 0; ex = nothing
for t in 1:NTR
    nA, nB, nc, S = rand(rng, 1:2), rand(rng, 2:4), rand(rng, 2:4), rand(rng, 1:3)
    n = nA + nB
    pts = rand(rng, n + nc, 2) .* 100
    dist = [round(sqrt(sum((pts[i, :] .- pts[n + j, :]) .^ 2))) for i in 1:n, j in 1:nc] .* 10
    ξ = round.(rand(rng, nc, S) .* 100 .+ 20)
    d = rand(rng, S); d ./= sum(d); r = rand(rng, S); r ./= sum(r)
    w = Float64(rand(rng, [60, 100, 200])); capA = maximum(sum(ξ; dims=1)) + 1
    inst = (; nA, nB, nc, S, dist, ξ, d, r, w, capA)
    capfull = Float64.(rand(rng, [50, 100, 150, 200], nB))
    # 두 집합으로 무작위 분할 (빈 집합 아님)
    for rep in 1:3
        lab = rand(rng, 1:2, nB); (all(==(1), lab) || all(==(2), lab)) && continue
        cT = capfull .* (lab .== 1); cS = capfull .* (lab .== 2)
        rT, RT = solveR(cT, inst); rS, RS = solveR(cS, inst); rU, RU = solveR(capfull, inst)
        global ntest += 1
        gap = (RU - RT - RS) / max(1.0, RT + RS)
        if gap > 1e-6
            global nviol_set += 1; global worst_set = max(worst_set, gap); ex === nothing && (global ex = (t, inst, capfull, lab, RT, RS, RU))
        end
        for b in 1:nB                                   # 거점별: 다른 쪽을 열면 이 거점 수입이 늘지 않나
            own = lab[b] == 1 ? rT[b] : rS[b]
            gs = (rU[b] - own) / max(1.0, own)
            gs > 1e-6 && (global nviol_site += 1; global worst_site = max(worst_site, gs))
        end
    end
end
@printf("WA=%s WB=%s STAGE1=%s 시험 %d: 집합 부분가법 위반 %d (최대 %.2e), 거점별 대체재 위반 %d (최대 %.2e)\n", WA, WB, STAGE1, ntest, nviol_set, worst_set, nviol_site, worst_site)
if ex !== nothing
    (t, inst, capfull, lab, RT, RS, RU) = ex
    @printf("  첫 반례: 시험 %d, nA=%d nB=%d nc=%d S=%d, cap=%s, 분할=%s, R(T)=%.2f R(S)=%.2f R(T∪S)=%.2f\n", t, inst.nA, inst.nB, inst.nc, inst.S, string(capfull), string(lab), RT, RS, RU)
end
