# A 점포 (Saint Louis·San Jose) 선판매 한도만 바꿔 메커니즘 조건 2 (A 예약이 한도에 걸려 belief 로 같이 줄지 않음) 를 직접 확인 (2026-11-07).
# 지역 요인 수요 SGB15 (S = 10) s10, p = 600, f = p/4. B 후보 13 곳의 한도는 300 그대로. 평가는 sgb15_psweep.jl 과 같음 (nz_alpha_bnb_dr,
# θᵁ = 회로 정확값, λᵁ = 40θᵁ, witness = incumbent α 에서 nz_exact_value). 증폭 = 결합 (C) − Data-DR (D) 이라 N 은 기본에서 뺌.
# 한도 결속 확인: incumbent α (= 그 모형의 최악에서 소매상 예약 h) 의 A 두 곳·B 점포 값을 기록 (D: belief q̂, C: 최악 belief d).
# 인자: QA_SEED (10), QA_SET ("SL:SJ|..." 숫자 또는 cap = A 용량 bA, 기본 6 설정), QA_P (600), QA_FRATIO (0.25), QA_XS ("|" 구분, 기본 단일 출점 13 개),
#       QA_MODELS ("D,C"), QA_TL (1200), QA_NW (4), QA_OUT, QA_TAG
root = dirname(@__DIR__)
const SPD = joinpath(@__DIR__, "logs", "sgb15_fac") * "/"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Distributions, Statistics
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(@__DIR__, "fac_builder.jl"))
const SEED = parse(Int, get(ENV, "QA_SEED", "10"))
P = parse(Float64, get(ENV, "QA_P", "600")); FR = parse(Float64, get(ENV, "QA_FRATIO", "0.25"))
TL = parse(Float64, get(ENV, "QA_TL", "1200")); nw = parse(Int, get(ENV, "QA_NW", "4"))
outdir = get(ENV, "QA_OUT", SPD * "qa15"); TAG = get(ENV, "QA_TAG", "")
MODELS = split(get(ENV, "QA_MODELS", "D,C"), ",")
XS = [parse.(Int, split(strip(t, ['{', '}']), r"[,;]")) for t in split(get(ENV, "QA_XS", join(["{$i}" for i in 1:13], "|")), "|")]
SETS = split(get(ENV, "QA_SET", "300:300|cap:cap|cap:300|300:cap|300:200|200:200"), "|")
const MPAR = Dict("N" => (0.0, 0.0, Inf), "D" => (0.3, 0.0, Inf), "C" => (0.3, 0.3, 0.1))

nd0 = make_sgb15_fac(; S=10, seed=SEED, dem=:region, p=P, f=FR * P, eps_hat=0.3, eps_tilde=0.3, beta=0.4)
const CS = nd0.meta[:cities]; const NA = nd0.meta[:nA]; const NST = nd0.meta[:nst]; const BA = nd0.meta[:bA]
const iSL = findfirst(==("Saint Louis"), CS[1:NA]); const iSJ = findfirst(==("San Jose"), CS[1:NA])
(iSL === nothing || iSJ === nothing) && error("A 점포 이름이 Saint Louis·San Jose 가 아님: $(CS[1:NA])")
@assert NST == NA + nd0.nx && nz_nh(nd0) == NST "pool_limit = :store 가정 (h 가 점포별)"
qval(t) = t == "cap" ? BA : parse(Float64, t)
function quota_vec(qSL, qSJ)
    qv = fill(300.0, NST); qv[iSL] = qSL; qv[iSJ] = qSJ; return qv
end
function mk(nm, qv)
    eh, et, δ = MPAR[nm]
    nd = nz_with_delta(make_sgb15_fac(; S=10, seed=SEED, dem=:region, p=P, f=FR * P, eps_hat=eh, eps_tilde=et, beta=0.4, quota=qv), δ)
    nd.wvec == qv || error("quota 벡터가 반영되지 않음")
    θ = nz_theta_circuit_exact(nd)[1]; return nz_with_bounds(nd; thetaU=θ, lambdaU=40θ)
end
"경합 고객 비중 (μ 가중): |가장 가까운 열린 B − 가장 가까운 A| < p (sgb15_psweep.jl 과 같음)"
function contested(nd, xs, p)
    D = nd.meta[:dist]; nA = nd.meta[:nA]; ncu = nd.meta[:ncu]; μ = sgb15_mu(sgb15_kuhn().pop)
    dA = [minimum(D[1:nA, j]) for j in 1:ncu]
    dB = [isempty(xs) ? Inf : minimum(D[i+nA, j] for i in xs) for j in 1:ncu]
    return sum(μ[j] for j in 1:ncu if abs(dB[j] - dA[j]) < p; init=0.0) / sum(μ)
end
"nz_exact_value 의 LP 로 결합의 최악 (a, d) (sgb15_psweep.jl 의 exact_ad 와 같음)"
function exact_ad(nd, x̄, α; tolF=1e-7)
    S, m, ny, nh = nd.S, nz_m(nd), nz_ny(nd), nz_nh(nd); nW = size(nd.W, 1)
    Q, φ = nz_recourse_values(nd, x̄, α); q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    M = Model(GRB1)
    @variable(M, 0 <= a[1:S] <= 1); @variable(M, b[1:S] >= 0); @variable(M, 0 <= r[1:S] <= 1)
    @constraint(M, sum(a) == 1); @constraint(M, sum(r) == 1)
    @constraint(M, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(M, [s = 1:S], a[s] + b[s] >= q[s])
    @constraint(M, sum(b) <= 2ε̂); @constraint(M, [s = 1:S], (1 - β) * r[s] <= a[s])
    @variable(M, 0 <= d[1:S] <= 1); @variable(M, e[1:S] >= 0); @constraint(M, sum(d) == 1)
    @constraint(M, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(M, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(M, sum(e) <= 2ε̃)
    if nz_coupled(nd)
        @variable(M, cpl[1:S] >= 0)
        @constraint(M, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(M, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(M, sum(cpl) <= 2nz_delta(nd))
    end
    @variable(M, π̃[1:m, 1:S] >= 0); @variable(M, κ[1:nW] >= 0)
    @constraint(M, [j = 1:ny, s = 1:S], sum(nd.A[k, j] * π̃[k, s] for k in 1:m if nd.A[k, j] != 0) >= nd.c[j] * d[s])
    @constraint(M, [i = 1:nh], sum(nd.hcoef[k] * π̃[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + nd.c0[i] <= sum(nd.W[l, i] * κ[l] for l in 1:nW))
    B = [nz_rhs(nd, x̄, zeros(nh), s) for s in 1:S]; lhs = dot(nd.c0, α)
    @constraint(M, lhs + sum(Q[s] * d[s] for s in 1:S) >= sum(B[s][k] * π̃[k, s] for k in 1:m, s in 1:S if B[s][k] != 0) + dot(nd.wvec, κ)
                                                          - tolF * (1 + abs(lhs) + maximum(abs, Q)))
    @objective(M, Max, sum(φ[s] * r[s] for s in 1:S)); optimize!(M)
    st = termination_status(M)
    for (nf, meth) in ((3, 1), (0, 2))
        st == MOI.NUMERICAL_ERROR || break
        set_optimizer_attribute(M, "NumericFocus", nf); set_optimizer_attribute(M, "Method", meth); optimize!(M)
        st = termination_status(M)
    end
    st == MOI.OPTIMAL && return value.(a), value.(d), tolF
    (st == MOI.INFEASIBLE && tolF < 1e-5) && return exact_ad(nd, x̄, α; tolF=10tolF)
    error("exact LP: $st (tolF=$tolF)")
end
mkpath(outdir); fn = joinpath(outdir, "qa_s$(SEED)_p$(Int(P))$(TAG).csv"); io = open(fn * ".part", "w")
println(io, "seed,p,f,qSL,qSJ,x,contested,model,LB,UB,exact,time,witness,hSL,hSJ,hB,a,d,tolF")
@printf("QA seed=%d p=%.0f f=%.1f 설정 %s, 결정 %d 개, 모형 %s, TL=%.0fs, workers=%d\n", SEED, P, FR * P, join(SETS, " "), length(XS), join(MODELS, ","), TL, nw)
@printf("A 점포: %d=%s, %d=%s, 용량 bA=%.1f; B 점포 = %s\n", iSL, CS[iSL], iSJ, CS[iSJ], BA, join(["{$i}=$(CS[i+NA])" for i in 1:nd0.nx], " ")); flush(stdout)
for st in SETS
    tSL, tSJ = split(st, ":"); qSL, qSJ = qval(tSL), qval(tSJ); qv = quota_vec(qSL, qSJ)
    models = [(nm, mk(nm, qv)) for nm in MODELS]
    @printf("한도 SL=%.1f SJ=%.1f  θᵁ=%s\n", qSL, qSJ, join([@sprintf("%s %.3f", nm, nd.thetaU) for (nm, nd) in models], ", ")); flush(stdout)
    for xs in XS
        x̄ = zeros(nd0.nx); x̄[xs] .= 1.0; xstr = "{" * join(xs, ";") * "}"
        cs = contested(models[1][2], xs, P); vals = Dict{String,Float64}()
        for (nm, nd) in models
            t0 = time()
            r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=1e-3, verbose=false, ipopt_time=120.0)
            qx = dot(nd.qx, x̄); vals[nm] = qx + r[:LB]
            α = r[:α]
            wit = α === nothing ? -Inf : qx + _ev(nd, x̄, α)     # nz_exact_value 는 (값, d) 또는 −Inf
            a, d, tF = nm == "C" && α !== nothing ? exact_ad(nd, x̄, α) : (Float64[], Float64[], NaN)
            hSL, hSJ = α === nothing ? (NaN, NaN) : (α[iSL], α[iSJ])
            hB = α === nothing ? "" : join([@sprintf("%.2f", α[i+NA]) for i in xs], ";")
            @printf(io, "%d,%.0f,%.1f,%.1f,%.1f,%s,%.4f,%s,%.6f,%.6f,%s,%.1f,%.6f,%.3f,%.3f,%s,%s,%s,%g\n", SEED, P, FR * P, qSL, qSJ, xstr, cs, nm,
                    qx + r[:LB], qx + r[:UB], string(r[:is_exact]), time() - t0, wit, hSL, hSJ, hB,
                    join(round.(a; digits=4), ";"), join(round.(d; digits=4), ";"), tF); flush(io)
            @printf("  SL=%.0f SJ=%.0f %-5s %s: [%.2f, %.2f] exact=%s witness=%.2f  h SL=%.1f SJ=%.1f B=%s  %.0fs\n", qSL, qSJ, xstr, nm,
                    qx + r[:LB], qx + r[:UB], string(r[:is_exact]), wit, hSL, hSJ, hB, time() - t0)
            flush(stdout)
        end
        haskey(vals, "C") && haskey(vals, "D") && @printf("  SL=%.0f SJ=%.0f %-5s 경합 %.0f%%  증폭 (C−D) %+.2f\n", qSL, qSJ, xstr, 100cs, vals["C"] - vals["D"])
        flush(stdout)
    end
end
close(io); mv(fn * ".part", fn; force=true)
println("QA_DONE seed=$SEED p=$(Int(P))")
