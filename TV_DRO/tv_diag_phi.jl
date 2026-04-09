include("tv_verify.jl")

# S=2: 정상 작동 instance에서 φ range 확인
for S_test in [2, 3, 20]
    network, tv = setup_instance(; m=3, n=3, S=S_test, seed=42, eps_hat=0.15, eps_tilde=0.15)
    K = tv.num_arcs

    full_model, full_vars = build_full_tv_model(tv; optimizer=Gurobi.Optimizer)
    optimize!(full_model)

    φh = [value(full_vars[:φ_hat][k,s]) for k in 1:K, s in 1:S_test]
    φt = [value(full_vars[:φ_tilde][k,s]) for k in 1:K, s in 1:S_test]
    λv = value(full_vars[:λ])

    @printf("S=%2d → obj=%10.6f  λ*=%6.4f  max(φ̂)=%8.5f  max(φ̃)=%8.5f  phi_U=%5.2f\n",
            S_test, objective_value(full_model), λv, maximum(φh), maximum(φt), tv.phi_U)
end

# 이제 Benders에서 ISP-F unbounded 재현: x=0, λ=10 에서 ISP-F 풀기
println("\n--- ISP-F unbounded 재현 테스트 ---")
network, tv = setup_instance(; m=3, n=3, S=20, seed=42, eps_hat=0.15, eps_tilde=0.15)
K = tv.num_arcs

x_sol = zeros(K)
λ_sol = 10.0
h_sol = fill(tv.w / K, K)
ψ0_sol = zeros(K)  # x=0 → ψ⁰=0
α_sol = fill(tv.w / K, K)

println("Testing ISP-F at x=0, λ=10, phi_U=$(tv.phi_U)...")
try
    isp_f, isp_f_v = build_tv_isp_follower(tv, x_sol, h_sol, λ_sol, ψ0_sol;
                                              optimizer=Gurobi.Optimizer)
    _, fc = tv_isp_follower_optimize!(isp_f, isp_f_v, tv, α_sol)
    @printf("ISP-F obj = %.6f\n", fc[:obj_val])
catch e
    println("ISP-F error: $e")
end

# 같은 조건에서 phi_U 키워서 테스트
for test_phi in [1.0, 5.0, 10.0, 11.0]
    tv2 = TVData(tv.Ny, tv.Nts, tv.nv1, tv.num_arcs, tv.S, tv.xi_bar, tv.q_hat,
                 tv.eps_hat, tv.eps_tilde, tv.v, tv.gamma, tv.w, tv.lambda_U,
                 tv.interdictable_arcs, test_phi)
    try
        isp_f, isp_f_v = build_tv_isp_follower(tv2, x_sol, h_sol, λ_sol, ψ0_sol;
                                                  optimizer=Gurobi.Optimizer)
        _, fc = tv_isp_follower_optimize!(isp_f, isp_f_v, tv2, α_sol)
        @printf("phi_U=%5.1f → ISP-F obj = %.6f\n", test_phi, fc[:obj_val])
    catch e
        @printf("phi_U=%5.1f → ISP-F error: %s\n", test_phi, e)
    end
end
