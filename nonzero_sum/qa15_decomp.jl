# A 한도 스윕 (sgb15_qasweep.jl) 결과의 정확한 증폭 분해 (2026-11-08). LP 만 (단일 스레드).
# 결정·한도 설정마다 Data-DR 의 예약 h_D (belief q̂) 와 결합의 최악점 (a_C, d_C, h_C) (CSV 의 incumbent α·exact_ad) 를 놓고
#   증폭 V_C − V_D = [CVaR^{a_C} φ(h_C) − CVaR^{a_C} φ(h_D)]  (예약 반응 몫, ≥ 증폭)
#                  + [CVaR^{a_C} φ(h_D) − V_D]                (진짜 분포 몫, ≤ 0: a_C 는 Data-DR 에서도 가능)
# 그리고 CVaR 꼬리 시나리오 (a_C 아래 가중 r > 0) 에서 B 가 잃은 판매가 어디로 갔는지 (A 예약분, A 현물, B 현물·예약) 를 센다.
# recourse 는 h 고정, 시나리오마다 follower 최적 (비용 최소) 중 B 판매 최소 (pessimistic, nz_recourse_values 와 같은 규칙).
# CSV 의 h 는 반올림 (소수 2~3 자리) 이라 값이 1e-2 수준으로 다를 수 있음 → 재계산 값과 CSV 값을 같이 출력.
# 인자: QD_CSV (기본 qa15/qa_s10_p600.csv, 없으면 .part), QD_SET ("SL:SJ" 목록, 기본 전부), QD_X (결정 목록, 기본 전부), QD_DETAIL (1 이면 시나리오 표)
const SPD = joinpath(@__DIR__, "logs", "sgb15_fac") * "/"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Distributions
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(@__DIR__, "nz_data.jl")); include(joinpath(@__DIR__, "fac_builder.jl"))
fn = get(ENV, "QD_CSV", SPD * "qa15/qa_s10_p600.csv"); isfile(fn) || (fn *= ".part")
wantset = get(ENV, "QD_SET", ""); wantx = get(ENV, "QD_X", ""); DETAIL = get(ENV, "QD_DETAIL", "0") == "1"

"h 고정, 시나리오 s 의 pessimistic recourse: (비용, B 판매, 흐름 분해 (A 예약, A 현물, B 예약, B 현물))"
function recourse_flows(nd, x̄, h, s)
    meta = nd.meta; nst, ncu, nA = meta[:nst], meta[:ncu], meta[:nA]; iyR, iyS = meta[:iyR], meta[:iyS]
    m, ny = nz_m(nd), nz_ny(nd)
    M = Model(GRB1); @variable(M, y[1:ny] >= 0)
    for k in 1:m
        rhs = nd.u[k, s] + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0) + (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0)
        @constraint(M, sum(nd.A[k, j] * y[j] for j in 1:ny if nd.A[k, j] != 0) <= rhs)
    end
    @objective(M, Max, dot(nd.c, y)); optimize!(M); termination_status(M) == MOI.OPTIMAL || error("recourse LP: $(termination_status(M))")
    F = objective_value(M)
    @constraint(M, dot(nd.c, y) >= F - 1e-7 * (1 + abs(F))); @objective(M, Max, dot(nd.ell, y)); optimize!(M)
    termination_status(M) == MOI.OPTIMAL || error("pessimistic recourse LP: $(termination_status(M))")
    yv = value.(y)
    fl(rng, f) = sum(yv[f(i, j)] for i in rng, j in 1:ncu)
    D = meta[:dist]; dA = [minimum(D[1:nA, j]) for j in 1:ncu]
    # B 판매 중 A 쪽이 더 가까운 고객 몫 (nearA) 과 B 쪽이 더 가깝거나 같은 고객 몫
    nearA = sum(yv[iyR(i, j)] + yv[iyS(i, j)] for i in nA+1:nst, j in 1:ncu if dA[j] < D[i, j]; init=0.0)
    return (cost=-F, φ=dot(nd.ell, yv), AR=fl(1:nA, iyR), AS=fl(1:nA, iyS), BR=fl(nA+1:nst, iyR), BS=fl(nA+1:nst, iyS),
            nearA=nearA, SL=sum(yv[iyR(1, j)] + yv[iyS(1, j)] for j in 1:ncu), SJ=sum(yv[iyR(2, j)] + yv[iyS(2, j)] for j in 1:ncu))
end
"CVaR_β^a[φ] = max {Σ r φ : r ≥ 0, Σ r = 1, (1−β) r ≤ a} (나쁜 φ 부터 채움). (값, r)"
function cvar(φ, a, β)
    r = zeros(length(φ)); left = 1.0
    for s in sortperm(φ; rev=true)
        t = min(a[s] / (1 - β), left); r[s] = t; left -= t; left <= 1e-12 && break
    end
    return dot(r, φ), r
end
"Data-DR 의 최악 a: max_a CVaR^a[φ], TV(a, q̂) ≤ ε̂ (LP)"
function worst_a(φ, q, ε, β)
    S = length(φ); M = Model(GRB1)
    @variable(M, 0 <= a[1:S] <= 1); @variable(M, b[1:S] >= 0); @variable(M, r[1:S] >= 0)
    @constraint(M, sum(a) == 1); @constraint(M, sum(r) == 1); @constraint(M, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(M, [s = 1:S], a[s] + b[s] >= q[s])
    @constraint(M, sum(b) <= 2ε); @constraint(M, [s = 1:S], (1 - β) * r[s] <= a[s]); @objective(M, Max, dot(φ, r)); optimize!(M)
    termination_status(M) == MOI.OPTIMAL || error("worst_a LP: $(termination_status(M))")
    return objective_value(M), value.(a)
end

rows = [split(l, ",") for l in readlines(fn)[2:end]]
H = Dict(k => i for (i, k) in enumerate(split(readline(fn), ",")))
col(r, k) = r[H[k]]
keys_ = unique([(col(r, "qSL"), col(r, "qSJ"), col(r, "x")) for r in rows])
nd_cache = Dict{Tuple{String,String},Any}()
@printf("%-13s %-5s %8s | %8s %8s %8s | %8s %8s | B판매 꼬리 Δ (→A예약, →A현물, B현물 Δ)\n", "SL:SJ", "x", "증폭", "예약몫", "진짜몫", "재계산", "D값", "C값")
for (qSL, qSJ, xs) in keys_
    (wantset == "" || "$(qSL):$(qSJ)" in split(wantset, ",") || "$(round(Int, parse(Float64, qSL))):$(round(Int, parse(Float64, qSJ)))" in split(wantset, ",")) || continue
    (wantx == "" || xs in split(wantx, ",")) || continue
    rD = findfirst(r -> (col(r, "qSL"), col(r, "qSJ"), col(r, "x"), col(r, "model")) == (qSL, qSJ, xs, "D"), rows)
    rC = findfirst(r -> (col(r, "qSL"), col(r, "qSJ"), col(r, "x"), col(r, "model")) == (qSL, qSJ, xs, "C"), rows)
    (rD === nothing || rC === nothing) && continue
    RD, RC = rows[rD], rows[rC]
    nd = get!(nd_cache, (qSL, qSJ)) do
        b = make_sgb15_fac(; S=10, seed=parse(Int, col(RD, "seed")), dem=:region, p=parse(Float64, col(RD, "p")), f=parse(Float64, col(RD, "f")), eps_hat=0.3, eps_tilde=0.3, beta=0.4)
        qv = fill(300.0, b.meta[:nst]); qv[1] = parse(Float64, qSL); qv[2] = parse(Float64, qSJ)
        b.meta[:cities][1:2] == ["Saint Louis", "San Jose"] || error("A 점포 순서가 다름")
        make_sgb15_fac(; S=10, seed=parse(Int, col(RD, "seed")), dem=:region, p=parse(Float64, col(RD, "p")), f=parse(Float64, col(RD, "f")), eps_hat=0.3, eps_tilde=0.3, beta=0.4, quota=qv)
    end
    nA = nd.meta[:nA]; ix = parse.(Int, split(strip(xs, ['{', '}']), ";")); x̄ = zeros(nd.nx); x̄[ix] .= 1.0; qx = dot(nd.qx, x̄)
    mkh(R) = (h = zeros(nz_nh(nd)); h[1] = parse(Float64, col(R, "hSL")); h[2] = parse(Float64, col(R, "hSJ"));
              hb = parse.(Float64, split(col(R, "hB"), ";")); for (k, i) in enumerate(ix); h[i+nA] = hb[k]; end; h)
    hD, hC = mkh(RD), mkh(RC)
    aC = parse.(Float64, split(col(RC, "a"), ";")); aC ./= sum(aC)
    S = nd.S; β = nd.beta
    fD = [recourse_flows(nd, x̄, hD, s) for s in 1:S]; fC = [recourse_flows(nd, x̄, hC, s) for s in 1:S]
    φD = [f.φ for f in fD]; φC = [f.φ for f in fC]
    VD, aD = worst_a(φD, nd.q_hat, nd.eps_hat, β)
    cC, rC_ = cvar(φC, aC, β); cDC, rDC = cvar(φD, aC, β)
    amp_csv = parse(Float64, col(RC, "witness")) - parse(Float64, col(RD, "witness"))
    # 꼬리 시나리오 (a_C, h_C 기준 가중 r) 에서 B 판매 감소와 그 행방 (가중 평균)
    w = rC_
    dB = sum(w[s] * ((fD[s].BR + fD[s].BS) - (fC[s].BR + fC[s].BS)) for s in 1:S)
    dAR = sum(w[s] * (fC[s].AR - fD[s].AR) for s in 1:S); dAS = sum(w[s] * (fC[s].AS - fD[s].AS) for s in 1:S)
    dBS = sum(w[s] * (fC[s].BS - fD[s].BS) for s in 1:S)
    nAD = sum(w[s] * fD[s].nearA for s in 1:S); dnA = sum(w[s] * (fD[s].nearA - fC[s].nearA) for s in 1:S)
    dSL = sum(w[s] * (fC[s].SL - fD[s].SL) for s in 1:S); dSJ = sum(w[s] * (fC[s].SJ - fD[s].SJ) for s in 1:S)
    @printf("%-13s %-5s %8.2f | %8.2f %8.2f %8.2f | %8.2f %8.2f | %6.1f (%+6.1f, %+6.1f, %+6.1f)\n", "$(round(Int, parse(Float64, qSL))):$(round(Int, parse(Float64, qSJ)))", xs,
            amp_csv, cC - cDC, cDC - VD, cC - VD, qx + VD, qx + cC, dB, dAR, dAS, dBS)
    @printf("    꼬리 B 판매 중 A-근접 고객: D %.1f, 감소 %.1f (B 판매 감소 %.1f 중) | A 점포 판매 증가 SL %+.1f SJ %+.1f
", nAD, dnA, dB, dSL, dSJ)
    if DETAIL
        @printf("    h_D: SL %.1f SJ %.1f B %s | h_C: SL %.1f SJ %.1f B %s\n", hD[1], hD[2], col(RD, "hB"), hC[1], hC[2], col(RC, "hB"))
        @printf("    %2s %5s %5s %5s %5s | %7s %7s | %6s %6s %6s %6s | %6s %6s %6s %6s\n", "s", "q̂", "a_D", "a_C", "d_C", "B판(D)", "B판(C)", "AR D", "AS D", "BR D", "BS D", "AR C", "AS C", "BR C", "BS C")
        dC = parse.(Float64, split(col(RC, "d"), ";"))
        for s in sortperm(φC; rev=true)
            @printf("    %2d %5.3f %5.3f %5.3f %5.3f | %7.1f %7.1f | %6.1f %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f %6.1f %s\n", s, nd.q_hat[s], aD[s], aC[s], dC[s],
                    fD[s].BR + fD[s].BS, fC[s].BR + fC[s].BS, fD[s].AR, fD[s].AS, fD[s].BR, fD[s].BS, fC[s].AR, fC[s].AS, fC[s].BR, fC[s].BS, rC_[s] > 1e-9 ? "꼬리 r=$(round(rC_[s]; digits=3))" : "")
        end
    end
end
