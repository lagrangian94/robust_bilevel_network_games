"""
test_global_bilinear_solver.jl — global_bilinear_solve (α-공간 B&B) vs Gurobi global (Ω, NonConvex).

실행: julia -t 14 test_global_bilinear_solver.jl [S] [time_limit]
"""

include("test_structured_eval.jl")          # make_batch_instance, random_xs, grb
include("global_bilinear_solver.jl")

gap(lb, ub) = (ub - lb) / max(1.0, abs(lb))

function compare(; S=50, time_limit=300.0, nworkers=12, run_gurobi=true)
    td = make_batch_instance("abilene"; S=S)
    for x̄ in random_xs(td, 2)
        xs = string(findall(x̄ .> 0.5))
        r = global_bilinear_solve(td, x̄; nworkers=nworkers, time_limit=time_limit, rel_gap=1e-4, verbose=false)
        @printf("S=%3d x̄=%-8s α-B&B : LB=%.6f UB=%.6f gap=%.2e nodes=%d ipopt=%d %.1fs\n", S, xs,
                r[:LB], r[:UB], gap(r[:LB], r[:UB]), r[:nodes], r[:ipopt_calls], r[:time]); flush(stdout)
        if run_gurobi
            V, B, opt, t = global_eval(td, x̄; time_limit=time_limit)
            @printf("S=%3d x̄=%-8s Gurobi: LB=%.6f UB=%.6f gap=%.2e %.1fs\n", S, xs, V, B, gap(V, B), t); flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    S = isempty(ARGS) ? 50 : parse(Int, ARGS[1])
    T = length(ARGS) < 2 ? 300.0 : parse(Float64, ARGS[2])
    compare(; S=S, time_limit=T)
end
