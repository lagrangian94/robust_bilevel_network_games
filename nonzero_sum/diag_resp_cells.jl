"""
diag_resp_cells.jl — follower 의 최적 반응 α*(d) 가 belief d 에 따라 몇 가지나 나오는지 센다 (분기 차원이 큰 인스턴스의 진단, 문서 §10.15).

follower 의 2 단계 LP 는 d 가 목적에만 들어가므로 α*(d) 는 d 의 조각별 상수 (LP 꼭짓점) 이고, V*(x̄) 는 그 꼭짓점들 (과 경계의 면) 위의 최댓값이다.
S 가 작으면 d 의 자유도가 S − 1 뿐이라 서로 다른 반응의 수가 적을 수 있다 → α 공간 (n_h 차원) 대신 belief 공간에서 세는 쪽이 싸다.

  1. TV 볼의 d 를 훑는다: S = 3 이면 격자 (간격 DC_STEP), 그 밖에는 무작위 DC_N 개 + 볼의 꼭짓점
  2. d 마다 follower LP 를 풀어 α*(d) 를 얻고 (기저해 하나), 서로 다른 α* 를 모은다
  3. 서로 다른 α* 마다 α 고정 LP 로 G(α*) 를 구해 최댓값을 α-B&B 의 incumbent 와 비교
주의: 표본이라 놓친 반응이 있을 수 있고, 셀 경계 (최적 반응이 면 전체) 의 내부 점은 보지 않는다 — 최적값의 증명이 아니라 규모 확인이다.

환경변수: NZ_SEED = 1  DC_S = 3  DC_X = "1"  DC_NLOC = 15  DC_QUOTA = 300  DC_STEP = 0.01  DC_N = 20000
          DC_INST = random (기본) | sgb (기본 인스턴스 SGB128, 할당량 150, 정확한 circuit θᵁ. DC_X 는 "1,2" 로)
          DC_GRID = 1 (S = 3 이면 격자. 0 이면 S = 3 도 무작위 표본)
실행: julia -t 2 nonzero_sum/diag_resp_cells.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env()
GRB() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pf(s) = parse(Float64, s)
seed = parse(Int, envf("NZ_SEED", "1")); S = parse(Int, envf("DC_S", "3")); nloc = parse(Int, envf("DC_NLOC", "15"))
xi = parse.(Int, split(envf("DC_X", "1"), ","))
cap = 150.0 * (nloc - 5) / 2
inst = envf("DC_INST", "random"); inst in ("random", "sgb") || error("DC_INST = random | sgb")
if inst == "sgb"
    ndb = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=seed,
                                               eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    nd0 = nz_with_bounds(ndb; thetaU=nz_theta_circuit_exact(ndb)[1])
else
    nd0 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed,
        bA=cap, bB=cap, qB=cap, dlo=50.0, dhi=150.0, reservation=:pair, quota=pf(envf("DC_QUOTA", "300")), wres=300.0,
        eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
end
x̄ = zeros(nd0.nx); x̄[xi] .= 1
hUp, _ = nz_presolve_hU(nd0, x̄); nd = nz_with_hU(nd0, hUp)
nh, m, ny = nz_nh(nd), nz_m(nd), nz_ny(nd); q = nd.q_hat; ε = nd.eps_tilde

# follower 의 2 단계 LP (목적 계수만 d 마다 갱신)
fm = Model(GRB); @variable(fm, 0 <= α[i = 1:nh] <= nd.hU[i]); @variable(fm, y[1:ny, 1:S] >= 0)
@constraint(fm, nd.W * α .<= nd.wvec)
for s in 1:S, k in 1:m
    @constraint(fm, sum(nd.A[k, j] * y[j, s] for j in 1:ny if nd.A[k, j] != 0) <=
                    nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0))
end
function resp(d)
    @objective(fm, Max, dot(nd.c0, α) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0)); optimize!(fm)
    termination_status(fm) == MOI.OPTIMAL || error("follower LP: $(termination_status(fm))")
    return value.(α)
end
inball(d) = all(>=(-1e-12), d) && sum(abs, d .- q) <= 2ε + 1e-9
ds = Vector{Float64}[]
if S == 3 && envf("DC_GRID", "1") == "1"
    h = pf(envf("DC_STEP", "0.01"))
    for d1 in 0:h:1, d2 in 0:h:(1-d1)
        d = [d1, d2, 1 - d1 - d2]; inball(d) && push!(ds, d)
    end
else
    rng = MersenneTwister(1)
    for _ in 1:parse(Int, envf("DC_N", "20000"))
        δ = randn(rng, S); δ .-= mean(δ); δ .*= (2ε * rand(rng)^(1 / (S - 1))) / sum(abs, δ)
        d = q .+ δ
        any(<(0), d) && (d = max.(d, 0.0); d ./= sum(d))
        inball(d) && push!(ds, d)
    end
    for sp in 1:S, sm in 1:S                      # 볼의 꼭짓점 쪽: 한 시나리오로 질량 ε 을 옮김 (받는 쪽 sp, 주는 쪽은 sm 부터 차례로)
        sp == sm && continue
        d = copy(q); left = ε
        for t in vcat(sm, setdiff(1:S, (sp, sm)))
            mv = min(left, d[t]); d[t] -= mv; d[sp] += mv; left -= mv
            left <= 1e-12 && break
        end
        inball(d) && push!(ds, d)
    end
end
@printf("diag_resp_cells: %s seed=%d S=%d x=%s n_h=%d (활성 %d) | belief 표본 %d 개\n", inst == "sgb" ? "SGB128" : "무작위 d=$nloc", seed, S, string(xi), nh, count(nd.hU .> 0), length(ds)); flush(stdout)
cells = Dict{Vector{Float64},Tuple{Vector{Float64},Int,Vector{Float64}}}()      # 반올림한 α → (α, 표본 수, 처음 본 d)
t1 = @elapsed for d in ds
    a = resp(d); key = round.(a; digits=3)
    c = get(cells, key, nothing)
    cells[key] = c === nothing ? (a, 1, d) : (c[1], c[2] + 1, c[3])
end
@printf("서로 다른 최적 반응 %d 개 (follower LP %d 회, %.1fs)\n", length(cells), length(ds), t1); flush(stdout)
F = nz_build_fixed_alpha(nd, x̄; optimizer=GRB)
vals = Tuple{Float64,Int,Vector{Float64},Vector{Float64}}[]
best_t = Tuple{Float64,Float64}[]                 # (경과 시간, 그때까지의 최댓값): 표본 순서대로 평가했을 때 incumbent 가 언제 나오는지
t2 = @elapsed let t0 = time(), cur = -Inf
    for (_, (a, cnt, d)) in cells
        z = nz_eval_alpha!(F, nd, a); push!(vals, (z, cnt, a, d))
        isfinite(z) && z > cur && (cur = z; push!(best_t, (time() - t0, z)))
    end
end
println("평가 순서대로의 최댓값: ", join([@sprintf("%.1fs %.3f", t, z) for (t, z) in best_t[max(1, end - 5):end]], " → "))
sort!(vals; by=first, rev=true)
fin = [v for v in vals if isfinite(v[1])]
best = isempty(fin) ? NaN : fin[1][1]
@printf("α 고정 LP %d 회 (%.1fs): 실행 가능 %d 개, 최댓값 %.4f | 최댓값의 1%% 안 %d 개, 5%% 안 %d 개\n", length(vals), t2, length(fin), best,
        count(v -> v[1] >= best - 0.01 * abs(best), fin), count(v -> v[1] >= best - 0.05 * abs(best), fin))
for (z, cnt, a, d) in fin[1:min(8, end)]
    @printf("  G = %10.4f  표본 %5d 개  d = %s  Σα = %.1f, α > 0 인 좌표 %d\n", z, cnt, string(round.(d; digits=3)), sum(a), count(>(1e-6), a))
end
cnts = sort([v[2] for v in vals]; rev=true)
@printf("셀 크기 (표본 수): 최대 %d, 중앙값 %d, 표본 1 개뿐인 셀 %d 개\n", cnts[1], cnts[(end+1)÷2], count(==(1), cnts))
