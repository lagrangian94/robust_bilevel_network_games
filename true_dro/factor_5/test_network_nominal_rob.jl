"""
test_network_nominal_rob.jl — Nominal (ε=0) + Robust (ε=1) sequential comparison
  S=20, λU=10.0

Usage:
  julia test_network_nominal_rob.jl <network_name> [scenario] [qhat]
  network_name: grid5x5, abilene, nobel_us, sioux_falls, polska
  scenario: uniform (default) or factor (factor_additive, k=5)
  qhat: uniform (default), sample1, sample3, sample8 (non-uniform q̂ from qhat_samples.jl)
"""

using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra

include("../../network_generator.jl")
NG = NetworkGenerator

include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")
include("../true_dro_mincut_vi.jl")

# ── parse argument ──
if length(ARGS) >= 1
    network_name = lowercase(ARGS[1])
    scenario = length(ARGS) >= 2 ? lowercase(ARGS[2]) : "uniform"
    qhat_name = length(ARGS) >= 3 ? lowercase(ARGS[3]) : "uniform"
else
    print("network (grid5x5/abilene/nobel_us/sioux_falls/polska): ")
    network_name = lowercase(strip(readline()))
    print("scenario (uniform/factor) [uniform]: ")
    scen_input = strip(readline())
    scenario = isempty(scen_input) ? "uniform" : lowercase(scen_input)
    print("qhat (uniform/sample1/sample3/sample8) [uniform]: ")
    qhat_input = strip(readline())
    qhat_name = isempty(qhat_input) ? "uniform" : lowercase(qhat_input)
end
scenario in ("uniform", "factor") || error("Unknown scenario: $scenario (uniform or factor)")
if qhat_name != "uniform"
    include(joinpath(@__DIR__, "qhat_samples.jl"))
    haskey(QHAT_SAMPLES, qhat_name) || error("Unknown qhat: $qhat_name. Available: $(keys(QHAT_SAMPLES))")
end

# ── log file ──
scen_suffix = scenario == "factor" ? "_factor" : ""
qhat_suffix = qhat_name == "uniform" ? "" : "_$(qhat_name)"
log_path = joinpath(@__DIR__, "logs", "$(network_name)_nominal_rob$(scen_suffix)$(qhat_suffix).log")
mkpath(dirname(log_path))
log_io = open(log_path, "w")
original_stdout = stdout
tee_rd, tee_wr = redirect_stdout()
tee_done = Channel{Nothing}(1)
@async begin
    try
        while isopen(tee_rd)
            data = readavailable(tee_rd)
            isempty(data) && break
            write(original_stdout, data)
            flush(original_stdout)
            write(log_io, data)
            flush(log_io)
        end
    catch e
        e isa EOFError || rethrow()
    end
    put!(tee_done, nothing)
end

# ── generate network ──
if network_name == "grid5x5"
    net = NG.generate_grid_network(5, 5; seed=42)
    γ = 2
    intd_arcs = net.interdictable_arcs
elseif network_name == "abilene"
    net = NG.generate_abilene_network()
    γ = 1
    intd_arcs = fill(true, length(net.arcs))
    net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
elseif network_name == "nobel_us"
    net = NG.generate_nobel_us_network()
    γ = 1
    intd_arcs = fill(true, length(net.arcs))
    net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
elseif network_name == "sioux_falls"
    net = NG.generate_sioux_falls_network()
    γ = 1
    intd_arcs = fill(true, length(net.arcs))
    net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
elseif network_name == "polska"
    net = NG.generate_polska_network()
    γ = 2
    intd_arcs = fill(true, length(net.arcs))
    net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
else
    error("Unknown network: $network_name\n  Supported: grid5x5, abilene, nobel_us, sioux_falls, polska")
end

num_arcs = length(net.arcs) - 1
S = 20
λU = 10.0

if scenario == "factor"
    caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
        interdictable_arcs=intd_arcs, seed=42, num_factors=5)
else
    caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
        interdictable_arcs=intd_arcs, seed=42)
end
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
if qhat_name == "uniform"
    q_hat = fill(1.0/S, S)
else
    q_hat = copy(QHAT_SAMPLES[qhat_name])
    length(q_hat) == S || error("q̂ length mismatch: $(length(q_hat)) vs S=$S")
    q_hat ./= sum(q_hat)  # ensure exact sum=1
end

@printf("%s %s (q̂=%s): arcs=%d, intd=%d, γ=%d, λU=%.1f, w=%.4f\n", network_name, scenario, qhat_name, num_arcs, length(intd_idx), γ, λU, w)
flush(stdout)

# ── source/sink connectivity cut (real networks only) ──
ss_cut = nothing
if network_name != "grid5x5"
    source_arcs = [i for i in 1:num_arcs if net.arcs[i][1] == "s"]
    sink_arcs   = [i for i in 1:num_arcs if net.arcs[i][2] == "t"]
    ss_cut = Dict{Symbol, Vector{Int}}()
    if length(source_arcs) >= 2
        ss_cut[:source_arcs] = source_arcs
        @printf("  source arcs (%d): %s\n", length(source_arcs), source_arcs)
    end
    if length(sink_arcs) >= 2
        ss_cut[:sink_arcs] = sink_arcs
        @printf("  sink arcs (%d): %s\n", length(sink_arcs), sink_arcs)
    end
    if isempty(ss_cut)
        ss_cut = nothing
    end
    flush(stdout)
end

# ── Nominal (Benders ε=0) ──
println("\n" * "="^60)
println("Nominal (Benders ε=0, λU=$λU)")
println("="^60)
flush(stdout)

td_0 = make_true_dro_data(net, caps, q_hat, 0.0, 0.0; w=w, lambda_U=λU, gamma=γ)
t0 = time()
res_nom = true_dro_benders_optimize!(td_0;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=500, tol=1e-4, verbose=true, sub_time_limit=30.0,
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw, valid_inequality=:mincut,
    inexact=true, nonconvex_attr=("NonConvex" => 2),
    source_sink_cut=ss_cut)
wt_nom = time() - t0

x_nom_int = round.(Int, res_nom[:x])
Z0_nom = res_nom[:Z0]

@printf("\nNominal: Z₀=%.6f, iters=%d, time=%.2fs\n", Z0_nom, res_nom[:iters], wt_nom)
println("x_nom arcs = $(findall(x_nom_int .> 0))")
flush(stdout)

# ── Robust DRO (ε=1.0) ──
println("\n" * "="^60)
println("Robust DRO (ε=1.0, λU=$λU)")
println("="^60)
flush(stdout)

td_1 = make_true_dro_data(net, caps, q_hat, 1.0, 1.0; w=w, lambda_U=λU, gamma=γ)
t0 = time()
res_rob = true_dro_benders_optimize!(td_1;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=500, tol=1e-4, verbose=true, sub_time_limit=30.0,
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw, valid_inequality=:mincut,
    inexact=true, nonconvex_attr=("NonConvex" => 2),
    source_sink_cut=ss_cut)
wt_rob = time() - t0
x_rob = round.(Int, res_rob[:x])

@printf("\nRobust: Z₀=%.6f, iters=%d, time=%.1fs\n", res_rob[:Z0], res_rob[:iters], wt_rob)
println("x_rob arcs = $(findall(x_rob .> 0))")
flush(stdout)

# ── comparison ──
println("\n" * "="^60)
println("x_nom arcs: $(findall(x_nom_int .> 0))")
println("x_rob arcs: $(findall(x_rob .> 0))")
println("SAME? $(x_nom_int == x_rob)")
println("="^60)
flush(stdout)

# ── close: 파이프 먼저 닫고 async task가 남은 버퍼 다 읽을 때까지 대기 ──
close(tee_wr)
take!(tee_done)
redirect_stdout(original_stdout)
close(log_io)
