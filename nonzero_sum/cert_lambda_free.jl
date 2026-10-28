"""
cert_lambda_free.jl — 본 실험 결과의 λ 무관 최적성 인증 (사후 처리, 2026-10-17). 솔버 파일은 고치지 않는다.

근거 (원고 Corollary conservative-bounds, Proposition certificate):
  UB : Benders 가 보고한 상한 = c_Lᵀx̂ + (α-B&B 상한 ≥ V*_λ(x̂)). 어떤 λᵁ 로 풀었든 V*_λ ≥ V* 라 원래 문제의 유효한 상한.
  w(x): x 에서 정확히 최적인 반응 (witness) 의 리더 평가값 (nz_exact_value, θ·λ 없음) ≤ V*(x).
  LB_free = min_{x ∈ X} [c_Lᵀx + w(x)] 는 원래 문제의 유효한 하한 (λ 무관).
  (UB − LB_free) / max(1, |UB|) ≤ tol 이면 그 인스턴스의 최적성이 λ 와 무관하게 인증된다.
witness 는 α-B&B 의 incumbent 를 정확히 최적인 반응으로 다듬어 (exact_lb) 얻는다. x 마다 짧은 제한부터 시작해
c_Lᵀx + w(x) ≥ UB − tol·max(1,|UB|) 가 될 때까지 제한을 늘린다 (CF_TLS). 끝까지 안 되는 x 가 있으면 그 x 를 보고하고 인증 실패.

입력: 로그의 RESULT 줄 (pilot_pooled_store.jl 형식). 인스턴스는 pilot_pooled_store.jl 의 mk 와 같은 방식으로 다시 만든다
  (N_B = nB, 거점별 한도 300, f 0.1, p 0.3, ε̂ = ε̃ = 0.3, δ 0.1, β 0.4, θᵁ = circuit 정확값, λᵁ 는 RESULT 의 값).
  같은 (로그, seed, S) 의 CERT 줄이 출력 파일에 이미 있으면 건너뛴다 (중간에 끊어도 이어서).
환경변수: CF_LOGS = "main_list_L*_S*.log,main_pooled_store_A*_S*.log" (logs/ 안의 glob, 쉼표)   CF_OUT = logs/cert_lambda_free.log
          CF_TOL = 5e-3   CF_TLS = "10,120,600" (x 하나에 쓰는 α-B&B 제한, 차례로)   CF_WORKERS = 12   CF_SKIP = "contended"
실행: julia -t 14,1 nonzero_sum/cert_lambda_free.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))
strs(k, d) = String.(strip.(split(envf(k, d), ",")))

# CF_LOGDIR = "" (기본: nonzero_sum/logs) | logs 아래 하위 폴더 (예 "l6diag")   CF_BNBKW = "" | 예 "range_lp=true,dd=false" (α-B&B 에 더 넘길 인자) — 2026-10-28 L6 인증용, 기본 실행은 그대로
logdir = joinpath(root, "nonzero_sum", "logs", envf("CF_LOGDIR", ""))
function _cf_kw(str)
    kv = Pair{Symbol,Any}[]
    for t in split(str, ","; keepempty=false)
        a, b = strip.(split(t, "="))
        push!(kv, Symbol(a) => (b == "true" ? true : b == "false" ? false : occursin(r"^[0-9]+$", b) ? parse(Int, b) : parse(Float64, b)))
    end
    return (; kv...)
end
bnbkw = _cf_kw(envf("CF_BNBKW", ""))
out = envf("CF_OUT", joinpath(logdir, "cert_lambda_free.log"))
tol = pf("CF_TOL", "5e-3"); tls = parse.(Float64, strs("CF_TLS", "10,120,600")); nw = pint("CF_WORKERS", "12")
skip = strs("CF_SKIP", "contended")
Threads.nthreads() >= nw + 2 || error("Julia 스레드 $(Threads.nthreads()) < CF_WORKERS+2 = $(nw + 2) (julia -t $(nw + 2),1)")

globre(p) = Regex("^" * replace(replace(p, "." => "\\."), "*" => ".*") * "\$")
files = sort(unique(vcat([filter(f -> occursin(globre(p), f), readdir(logdir)) for p in strs("CF_LOGS", "main_list_L*_S*.log,main_pooled_store_A*_S*.log")]...)))
files = filter(f -> !any(occursin(s, f) for s in skip if !isempty(s)), files)

field(line, k) = (m = match(Regex("(?:^|\\s)" * k * "=(\\S+)"), line); m === nothing ? error("RESULT 에 $k 없음: $line") : m.captures[1])
done = Set{Tuple{String,Int,Int}}()
if isfile(out)
    for l in eachline(out)
        startswith(l, "CERT ") || continue
        push!(done, (field(l, "file"), parse(Int, field(l, "seed")), parse(Int, field(l, "S"))))
    end
end

function mk(nloc, nA, nB, NB, S, seed, qB, λ)
    capv = 150.0 * (nloc - nA - nB) / nA
    nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=nA, nB=nB, NB=NB, S=S, seed=seed,
        bA=capv, bB=capv, qB=qB, dlo=50.0, dhi=150.0, p=0.3, f=0.1, reservation=:pooled, pool_limit=:store, quota=300.0,
        eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=λ)
end
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"

open(out, "a") do io
    @printf(io, "# cert_lambda_free 시작 %s | tol=%.0e TLs=%s workers=%d | 파일 %s\n", string(Base.Libc.strftime(time())), tol, string(tls), nw, join(files, ","))
    flush(io)
    for f in files, line in eachline(joinpath(logdir, f))
        startswith(line, "RESULT pooled-store") || continue
        seed = parse(Int, field(line, "seed")); S = parse(Int, field(line, "S"))
        (f, seed, S) in done && continue
        nloc = parse(Int, field(line, "d")); nA = parse(Int, field(line, "nA")); nB = parse(Int, field(line, "nB"))
        λ = parse(Float64, field(line, "λ")); qB = parse(Float64, field(line, "qB"))
        UB = parse(Float64, field(line, "UB")); st = field(line, "status"); xhat = field(line, "x")
        isfinite(UB) || (@printf(io, "CERT file=%s seed=%d S=%d status=%s certified=no reason=UB_not_finite\n", f, seed, S, st); flush(io); continue)
        t0 = time()
        nd = mk(nloc, nA, nB, nB, S, seed, qB, λ)
        thr = UB - tol * max(1.0, abs(UB))
        xs = [Float64.(collect(bb)) for bb in Iterators.product(fill(0:1, nd.nx)...)
              if sum(bb) <= nd.gamma && all(nd.x_allowed[i] || bb[i] == 0 for i in 1:nd.nx)]
        LBf = Inf; argx = ""; bad = String[]; detail = String[]
        for x in xs
            w = -Inf; used = 0.0
            for tl in tls
                r = nz_alpha_bnb_dr(nd, x; nworkers=nw, time_limit=tl, rel_gap=tol / 4, verbose=false, ipopt_time=min(30.0, tl), bnbkw...)
                (r[:α] === nothing || isempty(r[:α])) || (w = max(w, exact_lb(nd, x, r[:α])[1]))
                used = tl
                dot(nd.qx, x) + w >= thr && break
            end
            v = dot(nd.qx, x) + w
            v < LBf && (LBf = v; argx = xstr(x))
            v < thr && push!(bad, xstr(x))
            push!(detail, @sprintf("%s:%.4f@%.0f", xstr(x), v, used))
        end
        gapf = (UB - LBf) / max(1.0, abs(UB))
        @printf(io, "CERT file=%s seed=%d S=%d status=%s x=%s UB=%.6f LBfree=%.6f gapfree=%.2e certified=%s argminLB=%s uncert=%s time=%.1f xs=%s\n",
                f, seed, S, st, xhat, UB, LBf, gapf, gapf <= tol ? "yes" : "no", argx, isempty(bad) ? "-" : join(bad, ";"),
                time() - t0, join(detail, ";"))
        flush(io)
    end
    @printf(io, "# cert_lambda_free 끝 %s\n", string(Base.Libc.strftime(time())))
end
println("CERT_DONE")
