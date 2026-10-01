"""
compute_vor.jl — Value of Robustness (VoR) + Value of Follower Ambiguity (VoF) 계산
  각 (eps, beta) 폴더에서:
    VoR: nominal log → x_nom, double log → Z*
         WC(x_nom) via bilinear subproblem, VoR = (WC(x_nom) - Z*) / Z* × 100
    VoF: single_l log → x_sl, double log → Z*
         WC(x_sl) via bilinear subproblem, VoF = (WC(x_sl) - Z*) / Z* × 100

  bilinear subproblem은 true_dro_build_subproblem.jl의 구조를
  직접 이식 (VI/rho_bound 없이 순수 모델만).

Usage:
  julia compute_vor.jl
"""

using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics

include("../../network_generator.jl")
NG = NetworkGenerator

include("../true_dro_data.jl")

# ── network definitions (same as run_baseline_batch) ──
function make_network(name)
    if name == "grid5x5"
        net = NG.generate_grid_network(5, 5; seed=42)
        intd_arcs = net.interdictable_arcs
        γ = 2
        return net, γ, intd_arcs
    end
    gen = Dict(
        "polska"      => NG.generate_polska_network,
        "abilene"     => NG.generate_abilene_network,
        "nobel_us"    => NG.generate_nobel_us_network,
        "sioux_falls" => NG.generate_sioux_falls_network,
    )
    net = gen[name]()
    intd_arcs = fill(true, length(net.arcs))
    net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
    γ = 2
    return net, γ, intd_arcs
end

function fmt_val(v)
    return replace(@sprintf("%.2g", v), "." => "p")
end

# ── log parsing ──
function parse_x_arcs(log_path)
    lines = readlines(log_path)
    for i in length(lines):-1:1
        m = match(r"x arcs = \[([^\]]+)\]", lines[i])
        if m !== nothing
            return parse.(Int, split(m.captures[1], r",\s*"))
        end
    end
    return nothing
end

function parse_Z0(log_path)
    lines = readlines(log_path)
    for i in length(lines):-1:1
        m = match(r"Z₀=([\d.]+)", lines[i])
        if m !== nothing
            return parse(Float64, m.captures[1])
        end
    end
    return nothing
end

# ── 순수 bilinear subproblem 빌드 (true_dro_build_subproblem.jl 이식, VI 없음) ──
function build_vor_subproblem(td::TrueDROData, x_bar::Vector{Float64})
    S = td.S
    K = td.num_arcs
    m = td.nv1
    Ny = td.Ny
    Nts = td.Nts
    q = td.q_hat
    ε̂ = td.eps_hat
    ε̃ = td.eps_tilde
    ξ = td.xi_bar
    v = td.v
    w = td.w
    φ̂U = td.phi_hat_U
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "TimeLimit", 3600.0)

    # ── TV ball bounds (§10.1) ──
    a_min = [max(0.0, q[s] - 2 * ε̂) for s in 1:S]
    a_max = [min(1.0, q[s] + 2 * ε̂) for s in 1:S]
    d_min = [max(0.0, q[s] - 2 * ε̃) for s in 1:S]
    d_max = [min(1.0, q[s] + 2 * ε̃) for s in 1:S]

    # ── α: Lagrangian multiplier ──
    @variable(model, 0 <= α[1:K] <= w)
    @constraint(model, sum(α[k] for k in 1:K) <= w)

    # ── CVaR ──
    _use_cvar = (td.beta !== nothing)
    β_risk = _use_cvar ? td.beta : 0.0

    # ================================================================
    # ISP-L variables
    # ================================================================
    @variable(model, σ_hat[1:S] >= 0)
    @variable(model, u_hat[1:K, 1:S] >= 0)
    @variable(model, a_min[s] <= a[s=1:S] <= a_max[s])
    @variable(model, 0 <= b[1:S] <= 2 * ε̂)
    @variable(model, ρ_hat_1[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_2[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_3[1:K, 1:S] >= 0)

    # DL-1
    @constraint(model, [j=1:m, s=1:S],
        sum(Ny[j, k] * u_hat[k, s] for k in 1:K) + Nts[j] * σ_hat[s] == 0)

    if _use_cvar
        r_max = [a_max[s] / (1.0 - β_risk) for s in 1:S]
        @variable(model, 0 <= r[s=1:S] <= r_max[s])
        @variable(model, 0 <= ζL[k=1:K, s=1:S] <= w * r_max[s])
        @constraint(model, [k=1:K, s=1:S], ζL[k, s] == α[k] * r[s])
        @constraint(model, [s=1:S], r[s] <= a[s] / (1.0 - β_risk))
        @constraint(model, sum(r[s] for s in 1:S) == 1)

        # DL-2, DL-3 (CVaR path: r instead of a)
        @constraint(model, [k=1:K, s=1:S],
            -ξ[k, s] * r[s] - ζL[k, s]
            + u_hat[k, s] + ρ_hat_2[k, s] - ρ_hat_3[k, s] <= 0)
        @constraint(model, [k=1:K, s=1:S],
            v[k, s] * ξ[k, s] * r[s]
            - ρ_hat_1[k, s] - ρ_hat_2[k, s] + ρ_hat_3[k, s] <= 0)
    else
        @variable(model, 0 <= ζL[k=1:K, s=1:S] <= w * a_max[s])
        @constraint(model, [k=1:K, s=1:S], ζL[k, s] == α[k] * a[s])

        # DL-2, DL-3 (expectation path)
        @constraint(model, [k=1:K, s=1:S],
            -ξ[k, s] * a[s] - ζL[k, s]
            + u_hat[k, s] + ρ_hat_2[k, s] - ρ_hat_3[k, s] <= 0)
        @constraint(model, [k=1:K, s=1:S],
            v[k, s] * ξ[k, s] * a[s]
            - ρ_hat_1[k, s] - ρ_hat_2[k, s] + ρ_hat_3[k, s] <= 0)
    end

    # DL-4 ~ DL-7
    @constraint(model, [s=1:S], a[s] - b[s] <= q[s])
    @constraint(model, [s=1:S], a[s] + b[s] >= q[s])
    @constraint(model, sum(b[s] for s in 1:S) <= 2 * ε̂)
    @constraint(model, sum(a[s] for s in 1:S) == 1)

    # ================================================================
    # ISP-F variables
    # ================================================================
    @variable(model, d_min[s] <= d[s=1:S] <= d_max[s])
    @variable(model, 0 <= e[1:S] <= 2 * ε̃)
    @variable(model, u_tilde[1:K, 1:S] >= 0)
    @variable(model, σ_tilde[1:S] >= 0)
    @variable(model, ω[1:m, 1:S])
    @variable(model, β_var[1:K, 1:S] >= 0)   # β in original (renamed to avoid conflict)
    @variable(model, δ >= 0)
    @variable(model, ρ_tilde_1[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_2[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_3[1:K, 1:S] >= 0)
    @variable(model, ρ_psi0_1[1:K] >= 0)
    @variable(model, ρ_psi0_2[1:K] >= 0)
    @variable(model, ρ_psi0_3[1:K] >= 0)

    # ζF = α·d bilinear
    @variable(model, 0 <= ζF[k=1:K, s=1:S] <= w * d_max[s])
    @constraint(model, [k=1:K, s=1:S], ζF[k, s] == α[k] * d[s])

    # DF-1 ~ DF-4
    @constraint(model, [s=1:S], d[s] - e[s] <= q[s])
    @constraint(model, [s=1:S], d[s] + e[s] >= q[s])
    @constraint(model, sum(e[s] for s in 1:S) <= 2 * ε̃)
    @constraint(model, sum(d[s] for s in 1:S) == 1)

    # DF-5
    @constraint(model, [j=1:m, s=1:S],
        sum(Ny[j, k] * u_tilde[k, s] for k in 1:K) + Nts[j] * σ_tilde[s] == 0)

    # DF-6
    @constraint(model, [k=1:K, s=1:S],
        -ξ[k, s] * d[s] - ζF[k, s]
        + u_tilde[k, s] + ρ_tilde_2[k, s] - ρ_tilde_3[k, s] <= 0)

    # DF-7
    @constraint(model, [k=1:K, s=1:S],
        v[k, s] * ξ[k, s] * d[s]
        - ρ_tilde_1[k, s] - ρ_tilde_2[k, s] + ρ_tilde_3[k, s] <= 0)

    # DF-8
    @constraint(model, [k=1:K, s=1:S],
        sum(Ny[j, k] * ω[j, s] for j in 1:m) - β_var[k, s] <= 0)

    # DF-9
    @constraint(model, [s=1:S],
        d[s] + sum(Nts[j] * ω[j, s] for j in 1:m) <= 0)

    # DF-h
    @constraint(model, [k=1:K], sum(β_var[k, s] for s in 1:S) <= δ)

    # DF-λ
    @constraint(model,
        sum(σ_tilde[s] for s in 1:S)
        >= sum(ξ[k, s] * β_var[k, s] for k in 1:K, s in 1:S)
           + w * δ
           + sum(ρ_psi0_2[k] for k in 1:K)
           - sum(ρ_psi0_3[k] for k in 1:K))

    # DF-ψ
    @constraint(model, [k=1:K],
        sum(v[k, s] * ξ[k, s] * β_var[k, s] for s in 1:S)
        + ρ_psi0_1[k] + ρ_psi0_2[k] >= ρ_psi0_3[k])

    # ================================================================
    # Objective: max obj_L(x̄) + obj_F(x̄)
    # ================================================================
    obj_L = sum(σ_hat[s] for s in 1:S) -
            φ̂U * sum(x_bar[k] * ρ_hat_1[k, s] for k in 1:K, s in 1:S) -
            φ̂U * sum((1.0 - x_bar[k]) * ρ_hat_3[k, s] for k in 1:K, s in 1:S)

    obj_F = -φ̃U * sum(x_bar[k] * ρ_tilde_1[k, s] for k in 1:K, s in 1:S) -
             φ̃U * sum((1.0 - x_bar[k]) * ρ_tilde_3[k, s] for k in 1:K, s in 1:S) -
             λU * sum(x_bar[k] * ρ_psi0_1[k] for k in 1:K) -
             λU * sum((1.0 - x_bar[k]) * ρ_psi0_3[k] for k in 1:K)

    @objective(model, Max, obj_L + obj_F)

    return model
end

# ── 공통: subproblem solve + 결과 반환 ──
function solve_wc(td, x_arcs, num_arcs)
    x_bar = zeros(num_arcs)
    for a in x_arcs
        x_bar[a] = 1.0
    end
    model = build_vor_subproblem(td, x_bar)
    optimize!(model)
    status = termination_status(model)
    if status == MOI.OPTIMAL || status == MOI.TIME_LIMIT
        return objective_value(model), status
    else
        return NaN, status
    end
end

# ── settings ──
networks = ["grid5x5", "polska", "abilene", "nobel_us", "sioux_falls"]
S = 10
λU = 10.0
β_list   = [0.0, 0.05, 0.4, 0.7]
eps_list = [0.1, 0.2, 0.5]

# metrics: (label, log_suffix_for_x)
#   VoR: x from nominal, VoF: x from single_l
metrics = [
    # ("VoR", "nominal"),   # 이미 계산됨 — logs/compute_vor.log 참조
    ("VoF", "single_l"),
]

println("=" ^ 80)
println("Value of Robustness (VoR) + Value of Follower Ambiguity (VoF)")
println("=" ^ 80)
flush(stdout)

# ── results ──
results = []   # (metric, network, eps, beta, x_arcs, Z_star, wc, value, time, status)

for β_risk in β_list
    for eps in eps_list
        beta_str = fmt_val(β_risk)
        eps_str  = fmt_val(eps)
        log_dir  = joinpath(@__DIR__, "logs", "factor", "eps_$(eps_str)_beta_$(beta_str)")

        if !isdir(log_dir)
            @printf("  [skip] eps_%s_beta_%s — directory not found\n", eps_str, beta_str)
            continue
        end

        for network_name in networks
            dbl_log = joinpath(log_dir, "$(network_name)_double.log")
            if !isfile(dbl_log)
                @printf("  [skip] %s ε=%s β=%s — double log missing\n", network_name, eps_str, beta_str)
                continue
            end
            Z_star = parse_Z0(dbl_log)
            if Z_star === nothing
                @printf("  [skip] %s ε=%s β=%s — cannot parse Z* from double\n", network_name, eps_str, beta_str)
                continue
            end

            # Build network + data (shared across VoR/VoF)
            net, γ, intd_arcs = make_network(network_name)
            num_arcs = length(net.arcs) - 1

            caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
                interdictable_arcs=intd_arcs, seed=42, num_factors=5)
            intd_idx = findall(intd_arcs[1:num_arcs])
            w = round(0.5 * γ * median(caps[intd_idx, :]); digits=4)
            q_hat = fill(1.0/S, S)

            Random.seed!(42)
            v_rand = zeros(num_arcs, S)
            for k in 1:num_arcs, s in 1:S
                v_rand[k, s] = intd_arcs[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
            end

            td = make_true_dro_data(net, caps, q_hat, eps, eps;
                w=w, lambda_U=λU, gamma=γ, beta=β_risk, v_scenarios=v_rand)

            for (metric_name, log_suffix) in metrics
                src_log = joinpath(log_dir, "$(network_name)_$(log_suffix).log")
                if !isfile(src_log)
                    @printf("  [skip] %s %s ε=%s β=%s — %s log missing\n",
                            metric_name, network_name, eps_str, beta_str, log_suffix)
                    continue
                end

                x_arcs = parse_x_arcs(src_log)
                if x_arcs === nothing
                    @printf("  [skip] %s %s ε=%s β=%s — cannot parse x\n",
                            metric_name, network_name, eps_str, beta_str)
                    continue
                end

                # Skip if already in previous VoR log (check existing results)
                already = any(r -> r.metric == metric_name && r.network == network_name &&
                              r.eps == eps && r.beta == β_risk, results)
                if already
                    @printf("  [skip] %s %s ε=%s β=%s — already computed\n",
                            metric_name, network_name, eps_str, beta_str)
                    continue
                end

                @printf("  %s %s ε=%s β=%s: x=%s, Z*=%.6f ... ",
                        metric_name, network_name, eps_str, beta_str, x_arcs, Z_star)
                flush(stdout)

                t0 = time()
                wc, status = solve_wc(td, x_arcs, num_arcs)
                wt = time() - t0

                if !isnan(wc)
                    val = (wc - Z_star) / abs(Z_star) * 100.0
                    @printf("WC=%.6f, %s=%.2f%%, time=%.1fs (%s)\n", wc, metric_name, val, wt, status)
                    push!(results, (metric=metric_name, network=network_name, eps=eps, beta=β_risk,
                                    x_arcs=x_arcs, Z_star=Z_star, wc=wc,
                                    value=val, time=wt, status=string(status)))
                else
                    @printf("FAILED (%s), time=%.1fs\n", status, wt)
                    push!(results, (metric=metric_name, network=network_name, eps=eps, beta=β_risk,
                                    x_arcs=x_arcs, Z_star=Z_star, wc=NaN,
                                    value=NaN, time=wt, status=string(status)))
                end
                flush(stdout)
            end
        end
    end
end

# ── Summary tables ──
for metric_name in ["VoR", "VoF"]
    sub = filter(r -> r.metric == metric_name, results)
    isempty(sub) && continue

    println("\n" * "=" ^ 80)
    @printf("%s Summary Table\n", metric_name)
    println("=" ^ 80)
    src_label = metric_name == "VoR" ? "x_nom" : "x_sl"
    @printf("%-12s  %5s  %5s  %-20s  %10s  %10s  %8s\n",
            "Network", "ε", "β", src_label, "Z*", "WC", "$(metric_name)(%)")
    println("-" ^ 80)
    for r in sub
        x_str = "[" * join(r.x_arcs, ",") * "]"
        if isnan(r.value)
            @printf("%-12s  %5.2f  %5.2f  %-20s  %10.4f  %10s  %8s\n",
                    r.network, r.eps, r.beta, x_str, r.Z_star, "FAIL", "N/A")
        else
            @printf("%-12s  %5.2f  %5.2f  %-20s  %10.4f  %10.4f  %8.2f\n",
                    r.network, r.eps, r.beta, x_str, r.Z_star, r.wc, r.value)
        end
    end
end

println("\n" * "=" ^ 80)
println("Done!")
println("=" ^ 80)
