# 진단 (2026-11-01): L1 S50 seed 7 의 λ 무관 인증에서 exact value LP NUMERICAL_ERROR 재현, 실패한 LP 를 MPS 로 덤프. 솔버 파일은 안 고침.
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
for f in ("nz_data.jl", "nz_omega.jl", "nz_kkt_eval.jl", "nz_alpha_bnb.jl", "nz_alpha_bnb_dr.jl"); include(joinpath(root, "nonzero_sum", f)); end
const DUMP = joinpath(root, "nonzero_sum", "logs", "certdiag", "exact_numerr.mps")
src = read(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"), String)
src = replace(src, "error(\"exact value LP: \$st\")" => "(write_to_file(M, DUMP); println(\"DUMPED \", st, \" x=\", x̄, \" α=\", α); error(\"exact value LP: \$st\"))")
include_string(Main, src)
capv = 150.0 * (15 - 5) / 2
nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=15, nA=2, nB=3, NB=3, S=50, seed=7,
    bA=capv, bB=capv, qB=750.0, dlo=50.0, dhi=150.0, p=0.3, f=0.1, reservation=:pooled, pool_limit=:store, quota=300.0,
    eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
nd = nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=2000.0)
xs = [Float64.(collect(bb)) for bb in Iterators.product(fill(0:1, nd.nx)...) if sum(bb) <= nd.gamma && all(nd.x_allowed[i] || bb[i] == 0 for i in 1:nd.nx)]
for x in xs, tl in (10.0, 120.0, 600.0)
    r = nz_alpha_bnb_dr(nd, x; nworkers=12, time_limit=tl, rel_gap=5e-3 / 4, verbose=false, ipopt_time=min(30.0, tl))
    (r[:α] === nothing || isempty(r[:α])) && continue
    try
        w = exact_lb(nd, x, r[:α])[1]; @printf("x=%s tl=%.0f w=%.4f\n", string(findall(x .> 0.5)), tl, dot(nd.qx, x) + w)
    catch e
        println("FAIL x=", findall(x .> 0.5), " tl=", tl, " : ", e); exit(0)
    end
end
println("재현 안 됨")
