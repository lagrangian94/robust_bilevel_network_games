"""
Experiment 1 (VFM) + Experiment 2 (Source Decomposition) 순차 실행 wrapper.
6개 네트워크에 대해 순차적으로 Exp1 → Exp2 실행.

출력 폴더 구조:
  output/S{S}_e{ε}_g{γ_ratio}_r{ρ}/
    exp1_vfm_{network}.jls
    exp2_source_{network}.jls
    exp2_delta_{network}.png

실행:
  julia -t 8 run_all_experiments.jl
"""

# ===== 파라미터 상수 =====
const ALL_S = 20
const ALL_EPSILON = 0.5
const ALL_GAMMA_RATIO = 0.1
const ALL_RHO = 0.2
const ALL_K_TEST = 100
const ALL_K_POOL = 200

const ALL_NETWORKS = [:grid_4x4, :grid_5x5, :abilene, :nobel_us, :polska]
# grid_3x3 완료 → skip. 전체 재실행 시 복원:
# const ALL_NETWORKS = [:grid_3x3, :grid_4x4, :grid_5x5, :abilene, :nobel_us, :polska]

# Prefix 폴더
const OUTPUT_PREFIX = "output/S$(ALL_S)_e$(ALL_EPSILON)_g$(ALL_GAMMA_RATIO)_r$(ALL_RHO)"
mkpath(OUTPUT_PREFIX)

# 메모리 모니터 — 별도 CMD 창에서 Available MB + CPU% 표시 (5초 간격, 창 닫기로 종료)
let monitor_bat = joinpath(@__DIR__, "monitor_memory.bat")
    if isfile(monitor_bat)
        run(`cmd /c start "" $monitor_bat`; wait=false)
    end
end

println("="^80)
println("  ALL EXPERIMENTS")
println("  Networks: $(ALL_NETWORKS)")
println("  S=$(ALL_S), ε=$(ALL_EPSILON), γ_ratio=$(ALL_GAMMA_RATIO), ρ=$(ALL_RHO)")
println("  Output: $(OUTPUT_PREFIX)/")
println("="^80)

# ===== Experiment 1 include =====
include("run_experiment1_vfm.jl")

# ===== Experiment 2 include =====
include("run_experiment2_source.jl")

# grid_4x4 추가 (양쪽 network_configs에 없음 → include 후 추가)
network_configs[:grid_4x4] = Dict(:type => :grid, :m => 4, :n => 4)

# ===== 네트워크별 순차 실행 =====
for net_key in ALL_NETWORKS
    GC.gc()  # 이전 네트워크 모델 객체 해제

    println("\n" * "▶"^60)
    println("  NETWORK: $(net_key)")
    println("▶"^60)

    # ── Experiment 1: VFM ──
    global EXP_NETWORKS = [net_key]
    global EXP_S = ALL_S
    global EXP_S_FOLLOWER = ALL_S
    global EXP_GAMMA_RATIOS = [ALL_GAMMA_RATIO]
    global EXP_EPSILONS = [ALL_EPSILON]
    global EXP_K_TEST = ALL_K_TEST
    global EXP_RHO = ALL_RHO
    global CHECKPOINT_FILE = joinpath(OUTPUT_PREFIX, "exp1_vfm_$(net_key).jls")

    println("\n" * "▶"^40)
    println("  [$(net_key)] EXPERIMENT 1: VFM")
    println("▶"^40)

    try
        run_experiment()
    catch e
        @error "Exp1 failed for $(net_key)" exception=(e, catch_backtrace())
    end

    # ── Experiment 2: Source Decomposition (OOS re-eval only) ──
    global EXP_NETWORK = net_key
    global EXP_S = ALL_S
    global EXP_S_FOLLOWER = ALL_S
    global EXP_GAMMA_RATIO = ALL_GAMMA_RATIO
    global EXP_EPSILONS = [ALL_EPSILON]
    global EXP_K_POOL = ALL_K_POOL
    global EXP_RHO = ALL_RHO
    global EXP1_RESULTS_FILE = joinpath(OUTPUT_PREFIX, "exp1_vfm_$(net_key).jls")
    global CHECKPOINT_FILE = joinpath(OUTPUT_PREFIX, "exp2_source_$(net_key).jls")

    println("\n" * "▶"^40)
    println("  [$(net_key)] EXPERIMENT 2: SOURCE DECOMPOSITION")
    println("▶"^40)

    try
        run_experiment2()
    catch e
        @error "Exp2 failed for $(net_key)" exception=(e, catch_backtrace())
    end
end

println("\n" * "="^80)
println("  ALL EXPERIMENTS COMPLETE")
println("  Results in: $(OUTPUT_PREFIX)/")
println("="^80)
