"""
test_benders_boost.jl — Benders boost 단계 solver 비교: :gurobi (기존) vs :alpha_bnb (global_bilinear_solver).
설정은 factor_5/run_baseline_batch.jl 과 동일 (tol=5e-3, sub_time_limit=15, mini-Benders, mw, mincut, inexact, ss cut).

실행: julia -t 14 test_benders_boost.jl [network] [S] [solver...]
"""

include("test_structured_eval.jl")          # make_batch_instance, NetworkGenerator 등
include("global_bilinear_solver.jl")

function make_ss_cut_for(name)
    gen = Dict("polska" => generate_polska_network, "abilene" => generate_abilene_network,
               "nobel_us" => generate_nobel_us_network, "sioux_falls" => generate_sioux_falls_network)
    haskey(gen, name) || return nothing
    net = gen[name](); num_arcs = length(net.arcs) - 1
    src = [i for i in 1:num_arcs if net.arcs[i][1] == "s"]
    snk = [i for i in 1:num_arcs if net.arcs[i][2] == "t"]
    ss = Dict{Symbol,Vector{Int}}()
    length(src) >= 2 && (ss[:source_arcs] = src)
    length(snk) >= 2 && (ss[:sink_arcs] = snk)
    return isempty(ss) ? nothing : ss
end

function run_benders(name, S, solver; beta=0.4, eps=0.2, boost_time_limit=3600.0, belief_menu=false,
                     wall_time_limit=7200.0)
    td = make_batch_instance(name; beta=beta, eps=eps, S=S)
    t0 = time()
    res = true_dro_benders_optimize!(td;
        mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
        max_iter=1000, tol=5e-3, verbose=true, sub_time_limit=15.0,
        mini_benders=true, max_mini_benders_iter=5,
        strengthen_cuts=:mw, valid_inequality=:mincut,
        inexact=true, nonconvex_attr=("NonConvex" => 2),
        source_sink_cut=make_ss_cut_for(name),
        boost_solver=solver, boost_time_limit=boost_time_limit, boost_nworkers=12,
        belief_menu=belief_menu, wall_time_limit=wall_time_limit)
    wt = time() - t0
    h = res[:history]
    @printf("BENDERS %s S=%d solver=%-9s menu=%-5s status=%s Z0=%.6f LB=%.6f UB=%.6f iters=%d wall=%.1fs sub_total=%.1fs sub_max=%.1fs x=%s\n",
            name, S, solver, belief_menu, res[:status], res[:Z0], res[:lower_bound], res[:upper_bound], res[:iters], wt,
            sum(h[:sub_times]), maximum(h[:sub_times]), string(findall(res[:x] .> 0.5)))
    flush(stdout)
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    name = isempty(ARGS) ? "abilene" : ARGS[1]
    S = length(ARGS) < 2 ? 10 : parse(Int, ARGS[2])
    solvers = length(ARGS) < 3 ? [:alpha_bnb, :gurobi] : Symbol.(ARGS[3:end])
    for sv in solvers
        run_benders(name, S, sv)
    end
end
