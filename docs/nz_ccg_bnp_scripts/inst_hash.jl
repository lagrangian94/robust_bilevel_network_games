root = length(ARGS) > 0 ? ARGS[1] : "C:/Users/user/robust_bilevel_network_games"
using JuMP, Printf, LinearAlgebra, Random
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
println("julia ", VERSION)
for S in (10, 20)
    nd = make_loc_tkp(; g=1, S=S, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=1e4)
    @printf("S=%d: nx %d nh %d | sum u %.6f | sum q %.6f | sum hU %.3f | first u %s | keys %s\n", S, nd.nx, length(nd.hU), sum(nd.u), sum(nd.q_hat), sum(nd.hU),
            string(round.(nd.u[1:min(3, end), 1]; digits=4)), join(sort(string.(collect(keys(nd.meta))))[1:min(6, end)], ","))
    haskey(nd.meta, :demand) && @printf("   demand sum %.4f\n", sum(nd.meta[:demand]))
end
