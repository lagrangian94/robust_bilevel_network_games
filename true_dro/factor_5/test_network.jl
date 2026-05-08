"""
test_network.jl — unified test script for all networks
  S=20, λU=10.0
  grid5x5: γ=2, original interdictable arcs
  real-world (abilene, nobel_us, sioux_falls, polska): γ=2(polska)/1(others), all arcs interdictable

Usage:
  julia test_network.jl <network_name> [eps] [scenario] [mode] [qhat] [key=value...]
  network_name: grid5x5, abilene, nobel_us, sioux_falls, polska
  eps: robust ε value (default 1.0)
  scenario: uniform (default) or factor (factor_additive, k=5)
  mode: double (default), single_l (ε̂=eps, ε̃=0), single_f (ε̂=0, ε̃=eps)
  qhat: uniform (default), sample1, sample3, sample8 (non-uniform q̂ from qhat_samples.jl)
  beta=<float>: CVaR risk level β ∈ [0,1). 미지정 시 기존 expectation (r 없음).

Examples:
  julia test_network.jl grid5x5 0.3 uniform single_l          # expectation (기존, r 없음)
  julia test_network.jl grid5x5 0.3 uniform single_l beta=0.0 # CVaR β=0 (r=a 강제, 동일 결과)
  julia test_network.jl grid5x5 0.3 uniform single_l beta=0.3 # CVaR β=0.3
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
# key=value 형태 arg 파싱 (예: beta=0.3). 나머지는 positional.
_kw_args = Dict{String,String}()
_pos_args = String[]
for arg in ARGS
    if occursin("=", arg)
        k, v = split(arg, "="; limit=2)
        _kw_args[lowercase(k)] = v
    else
        push!(_pos_args, arg)
    end
end

if length(_pos_args) >= 1
    network_name = lowercase(_pos_args[1])
    ε_rob = length(_pos_args) >= 2 ? parse(Float64, _pos_args[2]) : 1.0
    scenario = length(_pos_args) >= 3 ? lowercase(_pos_args[3]) : "uniform"
    mode = length(_pos_args) >= 4 ? lowercase(_pos_args[4]) : "double"
    qhat_name = length(_pos_args) >= 5 ? lowercase(_pos_args[5]) : "uniform"
else
    print("network (grid5x5/abilene/nobel_us/sioux_falls/polska): ")
    network_name = lowercase(strip(readline()))
    print("eps [1.0]: ")
    eps_input = strip(readline())
    ε_rob = isempty(eps_input) ? 1.0 : parse(Float64, eps_input)
    print("scenario (uniform/factor) [uniform]: ")
    scen_input = strip(readline())
    scenario = isempty(scen_input) ? "uniform" : lowercase(scen_input)
    print("mode (double/single_l/single_f) [double]: ")
    mode_input = strip(readline())
    mode = isempty(mode_input) ? "double" : lowercase(mode_input)
    print("qhat (uniform/sample1/sample3/sample8) [uniform]: ")
    qhat_input = strip(readline())
    qhat_name = isempty(qhat_input) ? "uniform" : lowercase(qhat_input)
end
β_risk = haskey(_kw_args, "beta") ? parse(Float64, _kw_args["beta"]) : nothing
scenario in ("uniform", "factor") || error("Unknown scenario: $scenario (uniform or factor)")
mode == "single" && (mode = "single_l")  # backward compat
mode in ("double", "single_l", "single_f") || error("Unknown mode: $mode (double/single_l/single_f)")
if qhat_name != "uniform"
    include(joinpath(@__DIR__, "qhat_samples.jl"))
    haskey(QHAT_SAMPLES, qhat_name) || error("Unknown qhat: $qhat_name. Available: $(keys(QHAT_SAMPLES))")
end

# ── log file (redirect stdout so Benders verbose output goes to log too) ──
eps_str = replace(string(ε_rob), "." => "p")
mode_suffix = mode == "single_l" ? "_single" : mode == "single_f" ? "_single_f" : ""
scen_suffix = scenario == "factor" ? "_factor" : ""
qhat_suffix = qhat_name == "uniform" ? "" : "_$(qhat_name)"
beta_suffix = (β_risk === nothing || β_risk == 0.0) ? "" : "_beta" * replace(string(β_risk), "." => "p")
log_path = joinpath(@__DIR__, "logs", "$(network_name)_eps$(eps_str)$(mode_suffix)$(scen_suffix)$(qhat_suffix)$(beta_suffix).log")
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

# ── solve ──
ε_hat   = mode == "single_f" ? 0.0 : ε_rob
ε_tilde = mode == "single_l" ? 0.0 : ε_rob
mode_label = mode == "single_l" ? "Single-layer DRO (leader only)" :
             mode == "single_f" ? "Single-layer DRO (follower only)" : "Double-layer DRO"

println("\n" * "="^60)
beta_label = (β_risk !== nothing) ? "CVaR (β=$β_risk)" : "Expectation"
println("$mode_label — $beta_label (ε̂=$ε_hat, ε̃=$ε_tilde, λU=$λU)")
println("="^60)
flush(stdout)

td = make_true_dro_data(net, caps, q_hat, ε_hat, ε_tilde; w=w, lambda_U=λU, gamma=γ, beta=β_risk)
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

β_str = β_risk === nothing ? "none" : @sprintf("%.2f", β_risk)
@printf("\nResult (%s, β=%s): Z₀=%.6f, iters=%d, time=%.1fs\n", mode, β_str, res[:Z0], res[:iters], wt)
println("x arcs = $(findall(x_sol .> 0))")
flush(stdout)

# ── close: 파이프 먼저 닫고 async task가 남은 버퍼 다 읽을 때까지 대기 ──
close(tee_wr)
take!(tee_done)
redirect_stdout(original_stdout)
close(log_io)
