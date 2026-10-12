"""
diag_kink_spread.jl — 좌표 하나로 나눠서 꺾임을 떼어낼 수 있는가.

nz_alpha_bnb_dr 을 TL 초 돌린 뒤 상한이 큰 열린 노드 K 개에서, 쌍대가 고정되지 않은 성분 (예약 행 k, 시나리오 s) 마다
그 행의 좌표 i = hrow(k) 방향으로 "두 쌍대값이 모두 가능한 구간" 을 범위 MIP 로 잰다.
  p⁺ = max{α_i : 상자 안에서 σ_k 가 최댓값 hi 일 수 있음},  p⁻ = min{α_i : σ_k 가 최솟값 lo 일 수 있음}
  (σ_k 는 α_i 에 비증가. α_i > p⁺ 면 hi 가 불가능, α_i < p⁻ 면 lo 가 불가능)
  미정 구간 비율 ω = |[p⁻, p⁺] ∩ [l_i, u_i]| / (u_i − l_i)
ω ≈ 1 이면 다른 좌표가 상자 안에서 움직이는 것만으로 꺾임이 좌표 i 의 전 구간을 쓸고 지나간다는 뜻이라
좌표 i 를 어디서 나눠도 그 성분은 고정되지 않는다 (꺾임 면이 좌표축에 비스듬함). ω 가 작으면 꺾임 위치가 좌표 i 로 잘 잡힌다.
환경변수: OS_S = 20  NZ_SEED = 3  OS_X = "1,2"  OS_TL = 300  OS_GAP = 1e-3  OS_BOXES = 30  OS_WORKERS = 12
실행: julia -t 14,1 nonzero_sum/diag_kink_spread.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Statistics
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
S = parse(Int, envf("OS_S", "20")); seed = parse(Int, envf("NZ_SEED", "3"))
ε = parse(Float64, envf("OS_EPS", "0.3")); δ = pd(envf("OS_DELTA", "0.1"))
TL = parse(Float64, envf("OS_TL", "300")); GAP = parse(Float64, envf("OS_GAP", "1e-3")); nw = parse(Int, envf("OS_WORKERS", "12"))
K = parse(Int, envf("OS_BOXES", "30"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=seed,
                                           eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(nd.nx); x̄[parse.(Int, split(envf("OS_X", "1,2"), ","))] .= 1
@printf("꺾임 퍼짐 진단: seed=%d S=%d x=%s TL=%.0fs workers=%d\n", seed, S, string(findall(x̄ .> 0.5)), TL, nw); flush(stdout)

r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=true, log_every=60.0, ipopt_time=120.0)
LB = r[:LB]; sc = max(1.0, abs(LB)); open = r[:state].open
@printf("\n결과: LB=%.4f UB=%.4f gap=%.3e, 노드 %d, 열린 노드 %d\n", LB, r[:UB], (r[:UB] - LB) / sc, r[:nodes], length(open))
isempty(open) && (println("열린 노드 없음"); exit())

ndp = nz_with_hU(nd, nz_presolve_hU(nd, x̄)[1])
nh = nz_nh(ndp)
Hk = [k for k in findall(ndp.hrow .> 0) if ndp.hU[ndp.hrow[k]] > 0 && isfinite(ndp.varpiU[k])]   # _nz_dr_attach! 와 같은 순서
size(open[1][4]) == (length(Hk), S) || error("쌍대 범위 행렬 크기 불일치: $(size(open[1][4])) vs $((length(Hk), S))")
mips = [_nz_dr_build_mip(ndp, x̄, s; optimizer=GRB) for s in 1:S]

ord = sortperm([b[3] for b in open]; rev=true)[1:min(K, end)]
ωs = Float64[]; bycoord = Dict{Int,Vector{Float64}}(); nun_node = Int[]; nfull_node = Int[]; wrel = Float64[]
for b in open[ord]
    l, u, lo, hi = b[1], b[2], b[4], b[5]
    nun = 0; nfull = 0
    for (a, k) in enumerate(Hk), s in 1:S
        gap = hi[a, s] - lo[a, s]
        gap > 1e-4 || continue
        i = ndp.hrow[k]; w = u[i] - l[i]
        w > 1e-6 || continue
        M = mips[s]
        for j in 1:nh; set_lower_bound(M.α[j], l[j]); set_upper_bound(M.α[j], u[j]); end
        τ = min(1e-3, 0.01 * gap) + 2e-6
        set_lower_bound(M.σ[k], hi[a, s] - τ); @objective(M.model, Max, M.α[i]); optimize!(M.model)
        pp = termination_status(M.model) == MOI.OPTIMAL ? objective_value(M.model) : l[i]     # hi 가 상자에서 불가능하면 구간 없음
        set_lower_bound(M.σ[k], 0.0)
        set_upper_bound(M.σ[k], lo[a, s] + τ); @objective(M.model, Min, M.α[i]); optimize!(M.model)
        pm = termination_status(M.model) == MOI.OPTIMAL ? objective_value(M.model) : u[i]
        set_upper_bound(M.σ[k], ndp.varpiU[k])
        ω = clamp((min(pp, u[i]) - max(pm, l[i])) / w, 0.0, 1.0)
        push!(ωs, ω); push!(get!(bycoord, i, Float64[]), ω); push!(wrel, w / ndp.hU[i])
        nun += 1; ω >= 0.99 && (nfull += 1)
    end
    push!(nun_node, nun); push!(nfull_node, nfull)
end
println("\n상한이 큰 열린 노드 ", length(ord), " 개, 고정 안 된 성분 ", length(ωs), " 개 (노드당 중앙값 ", median(nun_node), ")")
if !isempty(ωs)
    @printf("미정 구간 비율 ω: 중앙값 %.2f, 평균 %.2f | ω ≥ 0.99 (좌표 하나로는 못 뗌) %.0f%% | ω ≤ 0.5 %.0f%% | ω ≤ 0.1 (꺾임 위치가 잘 잡힘) %.0f%%\n",
            median(ωs), mean(ωs), 100 * mean(ωs .>= 0.99), 100 * mean(ωs .<= 0.5), 100 * mean(ωs .<= 0.1))
    @printf("그 성분의 좌표 폭 / 원래 상한: 중앙값 %.3f\n", median(wrel))
    @printf("노드당 ω ≥ 0.99 인 성분 수: 중앙값 %.0f (고정 안 된 성분의 %.0f%%)\n", median(nfull_node), 100 * sum(nfull_node) / max(sum(nun_node), 1))
    println("좌표별 (성분 수, ω 중앙값, ω ≥ 0.99 비율):")
    for i in sort(collect(keys(bycoord)))
        v = bycoord[i]; @printf("  α[%d]: %4d, %.2f, %.0f%%\n", i, length(v), median(v), 100 * mean(v .>= 0.99))
    end
end
