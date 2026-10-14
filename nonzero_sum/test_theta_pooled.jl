"""
test_theta_pooled.jl — pooled 예약 (retailer 모형) 의 정확한 circuit θ̄ (nz_theta_circuit_exact → _nz_theta_circuit_exact_pooled) 검증.

  (1) 인스턴스별 θ̄, 열거한 경로 수, 시간, 최대 비율을 주는 사이클, 기존 일반 상계 v·D (:circuit) 와 비교
  (2) 직접 확인: 무작위 (x, α ∈ H, 시나리오 s) 에서 nz_theta_diag 로 그 점에 필요한 최소 θ (리더 평가 LP 의 최적성 제약 dual) 를 구해
      θ̄ 이상인 점이 없는지, θᵁ = θ̄ 에서 벌점 완화의 초과 (L(θ̄) − φ) 가 0 인지 본다. 필요 θ 의 최댓값이 θ̄ 에 닿으면 상계가 빡빡하다.
      α 는 상자 안 균등 (예약 한도를 넘으면 비례 축소) 과, 수요 원자 근처의 꺾임을 노린 격자점 (0, ξ 의 부분합) 을 섞는다.
  (3) pair 판 회귀: SGB128 pair 의 θ̄ 가 이전 값 1 그대로인지

환경변수: TP_N = 400 (인스턴스당 표본)  TP_SEEDS = "1,2,3"  TP_NLOC = "8,10,12,15"
실행: julia nonzero_sum/test_theta_pooled.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
envf(k, d) = get(ENV, k, d)
N = parse(Int, envf("TP_N", "400"))
seeds = parse.(Int, split(envf("TP_SEEDS", "1,2,3"), ",")); nlocs = parse.(Int, split(envf("TP_NLOC", "8,10,12,15"), ","))

function check(nd, label; rng=MersenneTwister(1))
    t = @elapsed (θ̄, info) = nz_theta_circuit_exact(nd)
    vD = nd.meta[:v] * nd.meta[:D]
    @printf("%-34s θ̄ = %8.4f (v·D = %5.1f)  경로 %9d  %.1fs\n    최대 사이클: %s\n", label, θ̄, vD, info[:npaths], t, info[:circuit])
    ndθ = nz_with_bounds(nd; thetaU=θ̄)
    xs = [Float64.(collect(b)) for b in Iterators.product(fill(0:1, nd.nx)...) if sum(b) <= nd.gamma]
    ξ = nd.meta[:xi]; nh = nz_nh(nd)
    maxneed = 0.0; maxexc = -Inf; nviol = 0; nhigh = 0
    for k in 1:N
        x = xs[rand(rng, eachindex(xs))]; s = rand(rng, 1:nd.S)
        if isodd(k)
            α = nd.hU .* rand(rng, nh)
        else                                                     # 꺾임 근처: 수요 원자의 무작위 부분합 (+ 작은 흔들림)
            α = [min(nd.hU[i], sum(ξ[j, rand(rng, 1:nd.S)] for j in 1:size(ξ, 1) if rand(rng) < 0.5; init=0.0) + rand(rng, (0.0, 0.0, 1e-3))) for i in 1:nh]
        end
        for l in 1:size(nd.W, 1)                                 # 예약 한도 W α ≤ w
            tot = dot(nd.W[l, :], α); tot > nd.wvec[l] && (α .*= nd.wvec[l] / tot)
        end
        r = nz_theta_diag(ndθ, x, α, s; lp_optimizer=GRB1)
        maxneed = max(maxneed, r[:theta_need]); maxexc = max(maxexc, r[:penalty_excess])
        # 판정은 벌점 초과로 한다: 최적성 제약의 dual 은 퇴화 시 유일하지 않아 θ_need 가 필요 이상으로 나올 수 있다 (그때는 따로 센다)
        tol = 1e-6 * max(1.0, abs(r[:phi]))
        r[:theta_need] > θ̄ + 1e-6 && (nhigh += 1)
        r[:penalty_excess] > tol && (nviol += 1;
            @printf("    위반: x=%s s=%d θ_need=%.6f excess=%.3e\n", string(findall(x .> 0.5)), s, r[:theta_need], r[:penalty_excess]))
    end
    @printf("    표본 %d: 필요 θ (dual) 최대 %.4f (θ̄ 대비 %.0f%%, θ̄ 초과 dual %d 건), 벌점 초과 최대 %.2e, 위반 %d\n", N, maxneed,
            100maxneed / max(θ̄, 1e-12), nhigh, maxexc, nviol)
    flush(stdout)
    return nviol
end

tot = 0
tot += check(make_location_instance(; coords=:sgb128, reservation=:pooled, wres=300.0, S=3, seed=1), "SGB128 pooled S=3")
for nloc in nlocs, seed in seeds
    cap = 150.0 * (nloc - 5) / 2
    nd = make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=3, seed=seed, bA=cap, bB=cap, qB=cap,
                                dlo=50.0, dhi=150.0, reservation=:pooled, wres=300.0)
    global tot += check(nd, "무작위 pooled d=$nloc seed=$seed")
end
θp, ip = nz_theta_circuit_exact(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=3, seed=1))
@printf("회귀: SGB128 pair θ̄ = %.4f (이전 값 1)\n", θp)
@printf("RESULT 위반 합계 %d\n", tot)
