"""
run_oos_experiment.jl — Phase A + Phase B OOS evaluation wrapper.

Phase A: Symmetric Dirichlet, tail metrics (p5, p95, min, max, win_rate)
Phase B: Asymmetric Dirichlet, OOS mean gap (E[q_true] ≠ uniform)

Usage:
    include("true_dro/run_oos_experiment.jl")
"""

using Printf
using Statistics
using Serialization
using Dates
using LinearAlgebra
using Random
using Distributions

# ---- Load modules ----
if !@isdefined(NetworkGenerator)
    include(joinpath(@__DIR__, "..", "network_generator.jl"))
end
using .NetworkGenerator

include(joinpath(@__DIR__, "oos_dirichlet.jl"))
include(joinpath(@__DIR__, "..", "oos_evaluation.jl"))    # build_maxflow_template, solve_deterministic_maxflow!
include(joinpath(@__DIR__, "oos_evaluate.jl"))             # oos_evaluate, oos_evaluate_phase_b, compute_win_rate

# Phase A & B runners
include(joinpath(@__DIR__, "run_oos_phase_a.jl"))
include(joinpath(@__DIR__, "run_oos_phase_b.jl"))


# ============================================================
# Main
# ============================================================

println("=" ^ 80)
println("  OOS EXPERIMENT (Phase A + Phase B)")
println("  $(now())")
println("=" ^ 80)

# ---- Phase A ----
println("\n" * "▶" ^ 40)
println("  Phase A: Symmetric Dirichlet (tail metrics)")
println("▶" ^ 40)
t0 = time()
try
    run_oos_phase_a()
catch e
    @error "Phase A failed" exception=(e, catch_backtrace())
end
@printf("Phase A elapsed: %.1f sec\n", time() - t0)

# ---- Phase B ----
println("\n" * "▶" ^ 40)
println("  Phase B: Asymmetric Dirichlet (OOS mean gap)")
println("▶" ^ 40)
t0 = time()
try
    run_oos_phase_b()
catch e
    @error "Phase B failed" exception=(e, catch_backtrace())
end
@printf("Phase B elapsed: %.1f sec\n", time() - t0)

println("\n" * "=" ^ 80)
println("  OOS EXPERIMENT COMPLETE — $(now())")
println("=" ^ 80)
