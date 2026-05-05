"""
run_single_f_batch.jl — Single-F (ε̂=0, ε̃>0) Benders batch for Polska
  q̂: uniform, sample1, sample3, sample8
  ε̃: 0.1, 0.3
  γ=2, S=20, k=5 factor_additive, λU=10.0

Usage:
  julia run_single_f_batch.jl
"""

using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra, Logging

include("../../network_generator.jl")
NG = NetworkGenerator

include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")
include("../true_dro_mincut_vi.jl")
include("qhat_samples.jl")

# ── network setup ──
net = NG.generate_polska_network()
γ = 2
intd_arcs = fill(true, length(net.arcs))
net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)

num_arcs = length(net.arcs) - 1
S = 20
λU = 10.0

caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=intd_arcs, seed=42, num_factors=5)
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)

# source/sink cut
source_arcs = [i for i in 1:num_arcs if net.arcs[i][1] == "s"]
sink_arcs   = [i for i in 1:num_arcs if net.arcs[i][2] == "t"]
ss_cut = Dict{Symbol, Vector{Int}}()
if length(source_arcs) >= 2; ss_cut[:source_arcs] = source_arcs; end
if length(sink_arcs) >= 2;   ss_cut[:sink_arcs]   = sink_arcs;   end
if isempty(ss_cut); ss_cut = nothing; end

@printf("polska k=5: arcs=%d, intd=%d, γ=%d, λU=%.1f, w=%.4f\n\n", num_arcs, length(intd_idx), γ, λU, w)

# ── q̂ configurations ──
qhat_configs = [
    ("uniform",  fill(1.0/S, S)),
    ("sample1",  QHAT_SAMPLES["sample1"] ./ sum(QHAT_SAMPLES["sample1"])),
    ("sample3",  QHAT_SAMPLES["sample3"] ./ sum(QHAT_SAMPLES["sample3"])),
    ("sample8",  QHAT_SAMPLES["sample8"] ./ sum(QHAT_SAMPLES["sample8"])),
]

epsilons = [0.1, 0.3]

# ── tee helper: redirect stdout to both console and file ──
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

# ── batch loop ──
for (qname, q_hat) in qhat_configs
    for ε_tilde in epsilons
        eps_str = replace(@sprintf("%.1f", ε_tilde), "." => "p")
        suffix = qname == "uniform" ? "" : "_$(qname)"
        log_name = "polska_eps$(eps_str)_single_f_factor$(suffix).log"
        log_path = joinpath(@__DIR__, "logs", log_name)
        mkpath(dirname(log_path))

        if isfile(log_path)
            @printf("[skip] %s already exists\n", log_name)
            continue
        end

        run_with_tee(log_path) do
            println("=" ^ 60)
            @printf("Single-F: q̂=%s, ε̂=0.0, ε̃=%.1f\n", qname, ε_tilde)
            println("=" ^ 60)
            flush(stdout)

            td = make_true_dro_data(net, caps, q_hat, 0.0, ε_tilde; w=w, lambda_U=λU, gamma=γ)
            t0 = time()
            res = true_dro_benders_optimize!(td;
                mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
                max_iter=500, tol=1e-4, verbose=true, sub_time_limit=30.0,
                mini_benders=true, max_mini_benders_iter=5,
                strengthen_cuts=:mw, valid_inequality=:mincut,
                inexact=true, nonconvex_attr=("NonConvex" => 2),
                source_sink_cut=ss_cut)
            wt = time() - t0
            x_sol = round.(Int, res[:x])

            @printf("\nResult: Z₀=%.6f, iters=%d, time=%.1fs\n", res[:Z0], res[:iters], wt)
            println("x arcs = $(findall(x_sol .> 0))")
            flush(stdout)
        end

        @printf("[main] Done: %s\n\n", log_name)
    end
end

println("\nAll single-F batch done!")
