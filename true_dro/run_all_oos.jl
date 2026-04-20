"""
run_all_oos.jl — OOS experiment wrapper.

Runs Phase A (symmetric Dirichlet, tail metrics) + Phase B (asymmetric Dirichlet, mean gap).

실행:
  julia true_dro/run_all_oos.jl
  또는 REPL에서:
  include("true_dro/run_all_oos.jl")
"""

cd(@__DIR__)
include("run_oos_experiment.jl")
