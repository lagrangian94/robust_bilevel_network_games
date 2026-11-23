# 덤프한 실패 LP (.mps) 의 계수 범위와 새로 풀 때의 상태 (2026-11-23)
using JuMP, Gurobi, Printf
import MathOptInterface as MOI
env = Gurobi.Env()
for f in ARGS
    println("=== ", basename(f))
    m = read_from_file(f)
    set_optimizer(m, () -> Gurobi.Optimizer(env))
    set_optimizer_attribute(m, "OutputFlag", 1); set_optimizer_attribute(m, "TimeLimit", 120.0)
    optimize!(m)
    @printf("상태 %s\n", termination_status(m))
    for (nf, meth) in ((3, 1), (3, 2))
        set_optimizer_attribute(m, "NumericFocus", nf); set_optimizer_attribute(m, "Method", meth); set_optimizer_attribute(m, "OutputFlag", 0)
        optimize!(m); @printf("  NumericFocus %d Method %d → %s\n", nf, meth, termination_status(m))
    end
    # 큰 계수: 변수 상한과 목적·행렬 계수의 최대
    vs = all_variables(m)
    ubs = [has_upper_bound(v) ? upper_bound(v) : 0.0 for v in vs]
    @printf("  변수 상한 최대 %.3e (상한 있는 변수 %d / %d)\n", maximum(ubs), count(v -> has_upper_bound(v), vs), length(vs))
end
