"""
lab_belief_cells.jl — belief 공간 셀 열거 oracle (nz_belief_cells.jl, S = 3 전용) 의 시험.

  BC_MODE = one   : 인스턴스 하나의 x̄ 에서 값과 시간 (BC_KKT = 1 이면 KKT 값과 대조)
  BC_MODE = check : 모든 x 에서 KKT 값 (nz_kkt_value_reduced) 과 대조 (작은 인스턴스용)
  BC_MODE = all   : 모든 x 를 이 oracle 로 평가해 리더 문제 전체를 열거로 푼다
환경변수: BC_INST = random | sgb   NZ_SEED = 1   BC_NLOC = 15   BC_X = "1"   BC_THREADS = 12   BC_KKT = 0   BC_HU = demand | quota (무작위 인스턴스의 예약 상한)
실행: julia -t 14 nonzero_sum/lab_belief_cells.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_cells.jl"))
envf(k, d) = get(ENV, k, d)
seed = parse(Int, envf("NZ_SEED", "1")); inst = envf("BC_INST", "random"); nloc = parse(Int, envf("BC_NLOC", "15")); mode = envf("BC_MODE", "one")
nt = parse(Int, envf("BC_THREADS", "12")); S = 3
if inst == "sgb"
    ndb = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=seed, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    nd = nz_with_bounds(ndb; thetaU=nz_theta_circuit_exact(ndb)[1])
else
    cap = 150.0 * (nloc - 5) / 2
    nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed,
        bA=cap, bB=cap, qB=cap, dlo=50.0, dhi=150.0, reservation=:pair, quota=300.0, wres=300.0, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    if envf("BC_HU", "demand") == "demand"
        mt = nd.meta; dm = [maximum(mt[:xi][j, :]) for j in 1:mt[:ncu]]
        global nd = nz_with_hU(nd, [min(nd.hU[mt[:hidx](i, j)], dm[j]) for i in 1:mt[:nst] for j in 1:mt[:ncu]])
    end
end
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
@printf("lab_belief_cells: %s seed=%d S=%d n_h=%d mode=%s\n", inst == "sgb" ? "SGB128" : "무작위 d=$nloc", seed, S, nz_nh(nd), mode); flush(stdout)
function one(x; kkt=false)
    r = nz_belief_cell_oracle(nd, x; nthreads=nt, verbose=false)
    line = @sprintf("  x=%-9s V* ∈ [%12.4f, %12.4f] gap %.1e | 셀 %3d 변 %3d 점 %3d (조각 %d) 최댓값의 층 %-5s | %.1fs (분할 %.1fs, follower LP %d)", xstr(x), r[:LB], r[:UB],
                    (r[:UB] - r[:LB]) / max(1.0, abs(r[:LB])), r[:cells], r[:edges], r[:points], r[:fuzzy], string(r[:best_kind]), r[:time], r[:t_split], r[:follower_lps])
    ok = true
    if kkt
        k = try nz_kkt_value_reduced(nd, x; optimizer=GRB, time_limit=120.0) catch e; nothing end
        if k === nothing || k[:status] != MOI.OPTIMAL
            line *= " | KKT 미해결"
        else
            tolv = 2e-4 * max(1.0, abs(k[:value])); ok = r[:LB] <= k[:value] + tolv && k[:value] <= r[:UB] + tolv
            line *= @sprintf(" | KKT %12.4f %s", k[:value], ok ? "" : " ← 불일치")
        end
    end
    println(line); flush(stdout)
    return ok
end
if mode == "check"
    nbad = count(!one(x; kkt=true) for x in all_x(nd))
    @printf("RESULT check %s seed=%d 불일치 %d 건\n", inst, seed, nbad)
elseif mode == "all"            # 모든 x 를 정확한 oracle 로 평가해 리더 문제 전체를 푼다 (열거)
    tall = @elapsed begin
        rs = [(x, nz_belief_cell_oracle(nd, x; nthreads=nt, verbose=false)) for x in all_x(nd)]
    end
    for (x, r) in rs
        @printf("  x=%-9s q·x + V* ∈ [%12.4f, %12.4f] | 셀 %3d 변 %3d 점 %3d 최댓값의 층 %-5s | %.1fs\n", xstr(x), dot(nd.qx, x) + r[:LB], dot(nd.qx, x) + r[:UB],
                r[:cells], r[:edges], r[:points], string(r[:best_kind]), r[:time])
    end
    ub = [dot(nd.qx, x) + r[:UB] for (x, r) in rs]; ib = argmin(ub)
    @printf("RESULT all %s d=%d seed=%d n_h=%d: 최적 x=%s 값 %.4f (둘째 %.4f) | x %d 개, 전체 %.1fs\n", inst, nloc, seed, nz_nh(nd), xstr(rs[ib][1]), ub[ib],
            sort(ub)[min(2, end)], length(rs), tall)
else
    x = zeros(nd.nx); x[parse.(Int, split(envf("BC_X", inst == "sgb" ? "1,2" : "1"), ","))] .= 1
    one(x; kkt=envf("BC_KKT", "0") == "1")
end
