"""
run_all_oos.jl — 3개 OOS 실험 순차 실행 wrapper.

  1) Basic OOS experiment: calibrated ε로 3 models 비교
  2) VOR sensitivity: ε/ε* sweep (inverted-U curve)
  3) Phase 2: Information asymmetry scenarios (L, F)

실행:
  julia -t 8 true_dro/run_all_oos.jl
  또는 REPL에서:
  include("true_dro/run_all_oos.jl")
"""

cd(@__DIR__)

# ===== 공통 코드 로드 =====
include("run_oos_experiment.jl")
include("run_vor_sensitivity.jl")
include("run_phase2_asymmetry.jl")

# ===== 순차 실행 =====
println("=" ^ 80)
println("  ALL OOS EXPERIMENTS")
println("=" ^ 80)

# 1) Basic OOS
println("\n" * "▶" ^ 40)
println("  EXPERIMENT 1: Basic OOS (calibrated ε)")
println("▶" ^ 40)
try
    run_oos_experiment(generalize=false)
catch e
    @error "Basic OOS experiment failed" exception=(e, catch_backtrace())
end

# 2) VOR sensitivity
println("\n" * "▶" ^ 40)
println("  EXPERIMENT 2: VOR Sensitivity (ε/ε* sweep)")
println("▶" ^ 40)
try
    run_vor_sensitivity(generalize=false)
catch e
    @error "VOR sensitivity failed" exception=(e, catch_backtrace())
end

# 3) Phase 2: Information asymmetry
println("\n" * "▶" ^ 40)
println("  EXPERIMENT 3: Phase 2 (Information Asymmetry)")
println("▶" ^ 40)
try
    run_phase2_experiment(generalize=false)
catch e
    @error "Phase 2 experiment failed" exception=(e, catch_backtrace())
end

println("\n" * "=" ^ 80)
println("  ALL OOS EXPERIMENTS COMPLETE")
println("=" ^ 80)
