"""
test_network.jl — unified test script for all networks
  k=5 (additive factor), S=20, λU=10.0
  grid5x5: γ=2, original interdictable arcs
  real-world (abilene, nobel_us, sioux_falls, polska): γ=1, all arcs interdictable

Usage:
  julia test_network.jl <network_name> [eps]
  network_name: grid5x5, abilene, nobel_us, sioux_falls, polska
  eps: robust ε value (default 1.0)
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
    ε_rob = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
else
    print("network (grid5x5/abilene/nobel_us/sioux_falls/polska): ")
    network_name = lowercase(strip(readline()))
    print("eps [1.0]: ")
    eps_input = strip(readline())
    ε_rob = isempty(eps_input) ? 1.0 : parse(Float64, eps_input)
end

# ── log file (redirect stdout so Benders verbose output goes to log too) ──
eps_str = replace(string(ε_rob), "." => "p")
log_path = joinpath(@__DIR__, "logs", "$(network_name)_eps$(eps_str).log")
mkpath(dirname(log_path))
log_io = open(log_path, "w")
original_stdout = stdout
tee_rd, tee_wr = redirect_stdout()
tee_done = Channel{Nothing}(1)
@async begin
    try
        while isopen(tee_rd)
            line = readline(tee_rd)
            println(original_stdout, line)
            println(log_io, line)
            flush(original_stdout); flush(log_io)
        end
    catch
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
    γ = 1
    intd_arcs = fill(true, length(net.arcs))
    net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
else
    error("Unknown network: $network_name\n  Supported: grid5x5, abilene, nobel_us, sioux_falls, polska")
end

num_arcs = length(net.arcs) - 1
S = 20
λU = 10.0

caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=intd_arcs, seed=42, num_factors=5)
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

@printf("%s k=5: arcs=%d, intd=%d, γ=%d, λU=%.1f, w=%.4f\n", network_name, num_arcs, length(intd_idx), γ, λU, w)
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

# ── Robust DRO (ε=ε_rob) ──
println("\n" * "="^60)
println("Robust DRO (ε=$ε_rob, λU=$λU)")
println("="^60)
flush(stdout)

td = make_true_dro_data(net, caps, q_hat, ε_rob, ε_rob; w=w, lambda_U=λU, gamma=γ)
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

# ── close: 파이프 먼저 닫고 async task가 남은 버퍼 다 읽을 때까지 대기 ──
close(tee_wr)
take!(tee_done)
redirect_stdout(original_stdout)
close(log_io)
