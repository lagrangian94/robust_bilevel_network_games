"""
check_theta_validity.jl — θᵁ 를 바꾼 Ω 가 V* 를 과대평가하지 않는지 (θ 부족이면 Ω 해 값 > V*) 모든 x 에서 KKT 와 비교.
환경변수: CV_THETA = circuit | 숫자   CV_QUOTA = 150   CV_DELTA = 0.1   CV_EPS = 0.3   CV_TL = 60 (Gurobi Ω, x 하나당)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
ε = parse(Float64, envf("CV_EPS", "0.3")); TL = parse(Float64, envf("CV_TL", "60"))
nd = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=parse(Float64, envf("CV_QUOTA", "150")),
        wres=300.0, S=3, seed=1, eps_hat=ε, eps_tilde=ε, beta=0.4), pd(envf("CV_DELTA", "0.1")))
θ = envf("CV_THETA", "circuit") == "circuit" ? nz_theta_circuit_exact(nd)[1] : parse(Float64, envf("CV_THETA", "1"))
ndθ = nz_with_bounds(nd; thetaU=θ)
@printf("%s θᵁ=%.4f λᵁ=%.1f  (위반 = Ω 해 값 > V* + 1e-6·max(1,|V*|))\n", nd.name, θ, nd.lambdaU)
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
O = build_nz_omega(ndθ; optimizer=GRB)
nviol = 0; maxgap = 0.0
for x in all_x(nd)
    V = nz_kkt_value(nd, x; optimizer=GRB, time_limit=300.0)[:value]
    r = nz_solve!(O, ndθ, x; time_limit=TL)
    viol = r[:Fval] > V + 1e-6 * max(1, abs(V))
    global nviol += viol; global maxgap = max(maxgap, (r[:bound] - V) / max(1, abs(V)))
    @printf("  x=%-10s V*=%11.4f  Ω inc=%11.4f  Ω bound=%11.4f  %s %s\n", string(findall(x .> 0.5)), V, r[:Fval], r[:bound],
            string(r[:status]), viol ? "← 위반" : "")
    flush(stdout)
end
@printf("위반 %d 개, 최대 (bound − V*)/|V*| = %.2e\n", nviol, maxgap)
