using Gurobi, JuMP
env = Gurobi.Env()
for (nm, kv) in [("기본", []), ("NF3+M1", [("NumericFocus",3),("Method",1)]), ("M0", [("Method",0)]), ("M2 barrier", [("Method",2)]),
                 ("NF3+M2+Crossover0", [("NumericFocus",3),("Method",2),("Crossover",0)]), ("ScaleFlag0", [("ScaleFlag",0)]), ("Presolve0+M1", [("Presolve",0),("Method",1)]),
                 ("Quad1+M1", [("Quad",1),("Method",1)])]
    m = read_from_file(joinpath(@__DIR__, "exact_numerr.mps")); set_optimizer(m, () -> Gurobi.Optimizer(env)); set_silent(m)
    set_optimizer_attribute(m, "Threads", 1)
    for (k, v) in kv; set_optimizer_attribute(m, k, v); end
    optimize!(m); st = termination_status(m)
    println(rpad(nm, 22), st, "  ", st == MOI.OPTIMAL ? objective_value(m) : "")
end
m = read_from_file(joinpath(@__DIR__, "exact_numerr.mps")); set_optimizer(m, () -> Gurobi.Optimizer(env)); set_optimizer_attribute(m, "Threads", 1); optimize!(m)
