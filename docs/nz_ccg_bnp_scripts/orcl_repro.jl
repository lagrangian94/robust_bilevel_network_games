# oracle (α-B&B) 이 큰 λᵁ 에서 α 고정 LP 비유계로 멈추는 것 재현 (2026-11-23)
#  GEO, XSTR (예 "{8:200;14:400}"), LAMU, TLIM. NZ_NUMERR_DUMP 를 주면 실패 LP 를 .mps/.bas 로 저장
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl"))
envf(k, d) = get(ENV, k, d)
g = parse(Int, envf("GEO", "12")); lam = parse(Float64, envf("LAMU", "1e4")); xs = envf("XSTR", "{8:200;14:400}"); tl = parse(Float64, envf("TLIM", "600"))
nd = make_loc_tkp(; g=g, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=lam)
K = nd.meta[:K]; nx = nd.nx
site(i) = (i - 1) ÷ K + 1; al = findall(nd.x_allowed); sites = unique(site.(al))
x̄ = nothing
for n in 0:nd.gamma, comb in Iterators.filter(c -> issorted(c) && allunique(c), Iterators.product(ntuple(_ -> sites, n)...)), lv in Iterators.product(ntuple(_ -> 1:K, n)...)
    z = zeros(nx); for (st, k) in zip(comb, lv); z[(st - 1) * K + k] = 1.0; end
    loc_tkp_xstr(nd, z) == xs && (global x̄ = z; break)
end
x̄ === nothing && error("x 를 못 찾음: $xs")
@printf("--- oracle 재현 g=%d x=%s λᵁ=%.0f (nd.lambdaU=%.0f) θᵁ=%.3f, 스레드 %d\n", g, xs, lam, nd.lambdaU, nd.thetaU, Threads.nthreads()); flush(stdout)
t = @elapsed res = try
    nz_alpha_bnb_dr(nd, x̄; nworkers=max(1, Threads.nthreads() - 2), time_limit=tl, rel_gap=1e-4, verbose=true)
catch e
    println("오류: ", sprint(showerror, e)[1:min(end, 400)]); nothing
end
res === nothing || @printf("RESULT λᵁ=%.0f V in [%.4f, %.4f] (%.0fs)\n", lam, res[:LB], res[:UB], t)
