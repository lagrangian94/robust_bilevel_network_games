"""
run_baseline_batch.jl — Baseline batch: nominal / single_l / double
  networks: grid5x5, polska, abilene, nobel_us, sioux_falls
  β ∈ {0.05, 0.4, 0.7}, ε ∈ {0.1, 0.2, 0.5}
  q̂=uniform, factor_additive scenario (k=5)
  logs → logs/factor/eps_Xp_beta_Yp/

Usage:
  julia run_baseline_batch.jl
"""

using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra, Random

include("../../network_generator.jl")
NG = NetworkGenerator

include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")
include("../true_dro_mincut_vi.jl")

# ── tee helper ──
function run_with_tee(f, log_path)
    log_io = open(log_path, "w")
    original_stdout = stdout
    rd, wr = redirect_stdout()
    reader_task = @async begin
        try
            while !eof(rd)
                line = readline(rd; keep=false)
                println(original_stdout, line)
                println(log_io, line)
                flush(original_stdout)
                flush(log_io)
            end
        catch e
            e isa InterruptException || e isa Base.IOError || rethrow()
        end
    end
    try
        f()
    finally
        redirect_stdout(original_stdout)
        close(wr)
        wait(reader_task)
        close(rd)
        close(log_io)
    end
end

# ── network definitions ──
function make_network(name)
    if name == "grid5x5"
        net = NG.generate_grid_network(5, 5; seed=42)
        intd_arcs = net.interdictable_arcs
        num_arcs = length(net.arcs) - 1
        # γ = ceil(Int, num_arcs * 0.1)
        γ  = 2
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
    num_arcs = length(net.arcs) - 1
    # γ = ceil(Int, num_arcs * 0.1)
    γ = 2
    return net, γ, intd_arcs
end

function make_ss_cut(net, num_arcs, network_name)
    network_name == "grid5x5" && return nothing
    source_arcs = [i for i in 1:num_arcs if net.arcs[i][1] == "s"]
    sink_arcs   = [i for i in 1:num_arcs if net.arcs[i][2] == "t"]
    ss = Dict{Symbol, Vector{Int}}()
    length(source_arcs) >= 2 && (ss[:source_arcs] = source_arcs)
    length(sink_arcs) >= 2   && (ss[:sink_arcs]   = sink_arcs)
    return isempty(ss) ? nothing : ss
end

# ── settings ──
networks = ["grid5x5", "polska", "abilene", "nobel_us", "sioux_falls"]
S = 10
λU = 10.0

# sweep parameters
β_list   = [0.0, 0.05, 0.4, 0.7]
eps_list = [0.1, 0.2, 0.5]

# (label, ε̂_mult, ε̃_mult) — multiplied by eps to get actual ε̂, ε̃
#   nominal: ε̂=0, ε̃=0 (eps-independent)
#   single_l: ε̂=eps, ε̃=0
#   double: ε̂=eps, ε̃=eps
settings = [
    ("nominal",  0.0, 0.0),
    ("single_l", 1.0, 0.0),
    ("double",   1.0, 1.0),
]

function fmt_val(v)
    return replace(@sprintf("%.2g", v), "." => "p")
end

println("=" ^ 70)
@printf("Baseline batch: S=%d, λU=%.1f, q̂=uniform, scenario=factor_additive(k=5)\n", S, λU)
@printf("Networks: %s\n", join(networks, ", "))
@printf("β sweep:   %s\n", join(β_list, ", "))
@printf("ε sweep:   %s\n", join(eps_list, ", "))
@printf("Settings:  %s\n", join([s[1] for s in settings], ", "))
println("=" ^ 70)
flush(stdout)

for β_risk in β_list
    for eps in eps_list
        beta_str = fmt_val(β_risk)
        eps_str  = fmt_val(eps)
        log_dir  = joinpath(@__DIR__, "logs", "factor", "eps_$(eps_str)_beta_$(beta_str)")
        mkpath(log_dir)

        for network_name in networks
            net, γ, intd_arcs = make_network(network_name)
            num_arcs = length(net.arcs) - 1

            caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
                interdictable_arcs=intd_arcs, seed=42, num_factors=5)
            intd_idx = findall(intd_arcs[1:num_arcs])
            w = round(0.5 * γ * median(caps[intd_idx, :]); digits=4)
            q_hat = fill(1.0/S, S)
            ss_cut = make_ss_cut(net, num_arcs, network_name)

            # v_scenarios
            Random.seed!(42)
            v_rand = zeros(num_arcs, S)
            for k in 1:num_arcs, s in 1:S
                v_rand[k, s] = intd_arcs[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
            end

            for (label, ε_hat_mult, ε_tilde_mult) in settings
                ε_hat   = eps * ε_hat_mult
                ε_tilde = eps * ε_tilde_mult

                log_name = "$(network_name)_$(label).log"
                log_path = joinpath(log_dir, log_name)

                # nominal은 eps-independent → 첫 eps에서 풀고 나머지는 복사
                if label == "nominal" && eps != eps_list[1]
                    if !isfile(log_path)
                        first_eps_str = fmt_val(eps_list[1])
                        first_dir = joinpath(@__DIR__, "logs", "factor", "eps_$(first_eps_str)_beta_$(beta_str)")
                        first_log = joinpath(first_dir, "$(network_name)_nominal.log")
                        if isfile(first_log)
                            cp(first_log, log_path)
                            @printf("  [copy] nominal β=%s → eps_%s\n", beta_str, eps_str)
                        else
                            @printf("  [warn] nominal source not found: %s\n", first_log)
                        end
                    else
                        @printf("  [skip] %s already exists\n", log_path)
                    end
                    continue
                end

                if isfile(log_path)
                    @printf("  [skip] %s already exists\n", log_path)
                    continue
                end

                @printf("\n%s: arcs=%d, intd=%d, γ=%d, w=%.4f\n", network_name, num_arcs, length(intd_idx), γ, w)
                flush(stdout)

                run_with_tee(log_path) do
                    println("=" ^ 60)
                    @printf("%s — %s: ε̂=%.2f, ε̃=%.2f, β=%.2f\n", network_name, label, ε_hat, ε_tilde, β_risk)
                    println("=" ^ 60)
                    flush(stdout)

                    td = make_true_dro_data(net, caps, q_hat, ε_hat, ε_tilde;
                        w=w, lambda_U=λU, gamma=γ, beta=β_risk, v_scenarios=v_rand)
                    t0 = time()
                    res = true_dro_benders_optimize!(td;
                        mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
                        max_iter=1000, tol=5e-3, verbose=true, sub_time_limit=15.0,
                        mini_benders=true, max_mini_benders_iter=5,
                        strengthen_cuts=:mw, valid_inequality=:mincut,
                        inexact=true, nonconvex_attr=("NonConvex" => 2),
                        source_sink_cut=ss_cut)
                    wt = time() - t0
                    x_sol = round.(Int, res[:x])

                    @printf("\nResult (%s): Z₀=%.6f, iters=%d, time=%.1fs\n", label, res[:Z0], res[:iters], wt)
                    println("x arcs = $(findall(x_sol .> 0))")
                    flush(stdout)
                end

                @printf("  [done] %s\n", log_name)
                flush(stdout)
            end
        end
    end
end

println("\n" * "=" ^ 70)
println("All baseline batch done!")
println("=" ^ 70)
