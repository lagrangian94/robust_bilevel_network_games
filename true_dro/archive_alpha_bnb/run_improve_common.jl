include(raw"C:\Users\user\robust_bilevel_network_games\true_dro\test_benders_boost.jl")
include(raw"C:\Users\user\robust_bilevel_network_games\true_dro\belief_menu_benders.jl")
function runA(netname, S; RB=true, LF=true, tag="A2", verbose=false)
    td = make_batch_instance(netname; beta=0.4, eps=0.2, S=S)
    r = belief_menu_benders_optimize!(td; mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer,
            lp_optimizer=Gurobi.Optimizer, oracle=:alpha_bnb, oracle_time_limit=600.0,
            tol=5e-3, verbose=verbose, source_sink_cut=make_ss_cut_for(netname), target_stop=true,
            repeat_boost=RB, local_first=LF)
    @printf("FIN %-11s S=%3d %-8s status=%-13s LB=%.6f UB=%.6f gap=%.2e wall=%7.1fs iters=%d oracle=%d local=%d/%.0fs bnb=%d/%.0fs menu=%d x=%s\n",
            netname, S, tag, r[:status], r[:lower_bound], r[:upper_bound],
            abs(r[:upper_bound] - r[:lower_bound]) / max(abs(r[:upper_bound]), 1e-10), r[:wall_time],
            r[:iters], r[:oracle_calls], r[:local_hits], r[:local_time], r[:bnb_calls], r[:bnb_time],
            r[:menu_size], string(findall(r[:x] .> 0.5))); flush(stdout)
end
function runB(netname, S)
    r = run_benders(netname, S, :alpha_bnb)
    h = r[:history]
    @printf("FIN %-11s S=%3d %-8s status=%-13s LB=%.6f UB=%.6f gap=%.2e wall=%7.1fs iters=%d sub_max=%.1fs x=%s\n",
            netname, S, "standard", r[:status], r[:lower_bound], r[:upper_bound],
            abs(r[:upper_bound] - r[:lower_bound]) / max(abs(r[:upper_bound]), 1e-10), r[:wall_time],
            r[:iters], maximum(h[:sub_times]), string(findall(r[:x] .> 0.5))); flush(stdout)
end
safe(f, args...; kw...) = try f(args...; kw...) catch e; println("FINERR $(args): ", sprint(showerror, e)[1:min(end,600)]); flush(stdout) end
