"""
test_lshaped_node.jl — 단일 노드에서 L-shaped master (nz_lshaped.jl) 와 원래 노드 완화 LP 의 값 비교 + 시간·cut 수.
같은 상자에서 둘 다 분리 수렴까지. 상자: 루트 (전체) 와 최적 α* 중심 폭 TL_W.
환경변수: TL_S = "3,20"  TL_X = "1,2"  TL_EPS = 0.3  TL_DELTA = 0.1  TL_W = 10  TL_INOUT = 0.5  TL_PEN = 1e4
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_lshaped.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
ε = parse(Float64, envf("TL_EPS", "0.3")); δ = pd(envf("TL_DELTA", "0.1")); W = parse(Float64, envf("TL_W", "10"))
inout = parse(Float64, envf("TL_INOUT", "0.5")); pen = parse(Float64, envf("TL_PEN", "1e4"))
xi = parse.(Int, split(envf("TL_X", "1,2"), ","))
mk1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
for S in parse.(Int, split(envf("TL_S", "3,20"), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    x̄ = zeros(nd.nx); x̄[xi] .= 1
    αs = S <= 3 ? nz_kkt_value(nd, x̄; optimizer=GRB, time_limit=300.0)[:α] : fill(20.0, nz_nh(nd))   # S>3 은 KKT 가 느려 고정 중심
    boxes = [("root", zeros(nz_nh(nd)), copy(nd.hU)), ("α*±$(W/2)", max.(0.0, αs .- W / 2), min.(nd.hU, αs .+ W / 2))]
    for (name, l, u) in boxes
        P1 = _nz_build_sep(nd, x̄; optimizer=mk1)
        _nz_sep_set_box!(P1, l, u)
        t1 = @elapsed z1 = _nz_sep_solve!(P1; maxrounds=500, stall_rounds=500)
        P2 = _nz_build_sep(nd, x̄; optimizer=mk1)
        tb = @elapsed M = nz_lshaped_master!(P2, nd, x̄; block_optimizer=mk1, pen=pen)
        _nz_sep_set_box!(P2, l, u)
        t2 = @elapsed z2 = nz_lshaped_solve!(P2, M; maxrounds=500, stall_rounds=500, inout=inout)
        @printf("S=%3d %-10s full %12.4f (%.1fs) | L-shaped %12.4f (빌드 %.1fs + %.1fs, cut %d) | 차 %.2e\n",
                S, name, z1, t1, z2, tb, t2, M.ncuts, (z2 - z1) / max(1.0, abs(z1)))
        flush(stdout)
    end
end

# ---- 2 부: 노드 연속 (cut 재사용) — 같은 모델로 무작위 하위 상자 TL_NODES 개를 차례로, full 과 L-shaped 의 노드당 시간·값 ----
using Random
nnodes = parse(Int, envf("TL_NODES", "10"))
for S in parse.(Int, split(envf("TL_S2", envf("TL_S", "3,20")), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    x̄ = zeros(nd.nx); x̄[xi] .= 1
    nd = nz_with_hU(nd, nz_presolve_hU(nd, x̄)[1])
    P1 = _nz_build_sep(nd, x̄; optimizer=mk1)
    P2 = _nz_build_sep(nd, x̄; optimizer=mk1); M = nz_lshaped_master!(P2, nd, x̄; block_optimizer=mk1, pen=pen)
    rng = MersenneTwister(1); nh = nz_nh(nd)
    t1s = Float64[]; t2s = Float64[]; dmax = 0.0
    for t in 0:nnodes
        l = zeros(nh); u = copy(nd.hU)
        if t > 0                                   # 무작위 상자: 각 좌표를 무작위 깊이만큼 중점 분할 (W l ≤ w 인 실행가능 상자만)
            while true
                l .= 0.0; u .= nd.hU
                for i in findall(nd.hU .> 0), _ in 1:rand(rng, 0:3)
                    mid = 0.5 * (l[i] + u[i]); rand(rng) < 0.5 ? (u[i] = mid) : (l[i] = mid)
                end
                all(nd.W * l .<= nd.wvec .+ 1e-9) && break
            end
        end
        _nz_sep_set_box!(P1, l, u); a1 = @elapsed z1 = _nz_sep_solve!(P1; maxrounds=500, stall_rounds=500)
        c0 = M.ncuts
        _nz_sep_set_box!(P2, l, u); a2 = @elapsed z2 = nz_lshaped_solve!(P2, M; maxrounds=500, stall_rounds=500, inout=inout)
        dmax = max(dmax, abs(z2 - z1) / max(1.0, abs(z1)))
        t > 0 && (push!(t1s, a1); push!(t2s, a2))
        @printf("  S=%3d 노드 %2d: full %.2fs | L-shaped %.2fs (새 cut %d)\n", S, t, a1, a2, M.ncuts - c0)
        flush(stdout)
    end
    @printf("S=%3d 노드 연속 %d 개: 평균 full %.2fs, L-shaped %.2fs, 최대 상대차 %.1e, 총 cut %d\n",
            S, nnodes, sum(t1s) / nnodes, sum(t2s) / nnodes, dmax, M.ncuts)
end
