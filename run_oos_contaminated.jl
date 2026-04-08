"""
Experiment: Contaminated OOS Re-evaluation

기존 S20_e0.5_g0.1_r0.2 실험의 x*를 재사용하여,
contaminated 분포로 OOS만 재평가.

DGP(δ) = (1-δ)·Uniform{1,...,10} + δ·Uniform{0,1,2}
- δ_F = 0.15 (follower belief)
- δ_T = 0.30 (OOS test)

실행:
  julia run_oos_contaminated.jl
"""

using JuMP
using HiGHS
using LinearAlgebra
using Statistics
using Serialization
using Printf
using Dates

include("network_generator.jl")
include("oos_evaluation.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model,
    generate_capacity_scenarios_contaminated, generate_capacity_scenarios_trunc_normal
using .NetworkGenerator: generate_nobel_us_network, generate_abilene_network,
    generate_polska_network

# ===== Parameters =====
const SEED = 42          # in-sample (w 재계산용)
const S = 20             # in-sample scenario 수
const S_F = 10           # follower belief scenarios
const K_TEST = 100       # OOS test scenarios
const SEED_FOLLOWER = 43
const SEED_TEST = 44
const γ_RATIO = 0.1
const ρ = 0.2
const V_PARAM = 1.0
const ε = 0.5

# Contamination levels
# Usage:  julia run_oos_contaminated.jl <δ_F> <δ_T>              — contaminated follower
#         julia run_oos_contaminated.jl --tn <μ> <σ> <δ_T>       — truncated normal follower
const FOLLOWER_MODE = length(ARGS) >= 1 && ARGS[1] == "--tn" ? :trunc_normal : :contaminated
const δ_F = FOLLOWER_MODE == :contaminated ? (length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.15) : 0.0
const δ_T = FOLLOWER_MODE == :contaminated ? (length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.30) :
            (length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.70)
const TN_μ = FOLLOWER_MODE == :trunc_normal ? parse(Float64, ARGS[2]) : 5.0
const TN_σ = FOLLOWER_MODE == :trunc_normal ? parse(Float64, ARGS[3]) : 2.0

const JLS_DIR = "output/S20_e0.5_g0.1_r0.2"
const OUTPUT_FILE = "output/S20_e0.5_g0.1_r0.2/oos_contaminated.jls"

# ===== Network configs =====
network_configs = Dict(
    :grid_3x3 => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_4x4 => Dict(:type => :grid, :m => 4, :n => 4),
    :grid_5x5 => Dict(:type => :grid, :m => 5, :n => 5),
    :abilene  => Dict(:type => :real_world, :generator => generate_abilene_network),
    :nobel_us => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :polska   => Dict(:type => :real_world, :generator => generate_polska_network),
)

# Valid (network, variant) combinations from plan
const VALID_COMBOS = Dict(
    :grid_3x3 => [:N, :FO, :TO, :FM],
    :grid_4x4 => [:N, :FO, :TO, :FM],
    :grid_5x5 => [:N, :FO, :TO, :FM],
    :abilene  => [:N, :FO, :TO, :FM],
    :nobel_us => [:N, :FO, :FM],          # TO = ERROR
    :polska   => [:N, :FO],               # TO, FM 없음
)

function make_network(config_key::Symbol)
    config = network_configs[config_key]
    if config[:type] == :grid
        return generate_grid_network(config[:m], config[:n], seed=SEED)
    else
        return config[:generator]()
    end
end

function compute_w(network, capacities)
    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_RATIO * num_interdictable)
    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar, digits=4)
    return w
end

function result_key(net_key, variant)
    return "$(net_key)_γ$(γ_RATIO)_ε$(ε)_$(variant)"
end

# ===== Main =====
function run_contaminated_oos()
    all_results = Dict{String, Any}()

    println("="^80)
    println("CONTAMINATED OOS RE-EVALUATION")
    println("="^80)
    if FOLLOWER_MODE == :trunc_normal
        println("  Follower: TruncNormal(μ=$(TN_μ), σ=$(TN_σ), [0,10])")
    else
        println("  Follower: Contaminated(δ_F=$(δ_F))")
    end
    println("  Test: Contaminated(δ_T=$(δ_T))")
    println("  S_f=$(S_F), K_test=$(K_TEST)")
    println("  seed_follower=$(SEED_FOLLOWER), seed_test=$(SEED_TEST)")
    println("="^80)

    for (net_key, variants) in sort(collect(VALID_COMBOS), by=first)
        # Load jls
        jls_file = joinpath(JLS_DIR, "exp1_vfm_$(net_key).jls")
        if !isfile(jls_file)
            @warn "JLS not found: $jls_file, skipping $net_key"
            continue
        end
        saved_results = deserialize(jls_file)

        # Regenerate network + w
        network = make_network(net_key)
        num_arcs_total = length(network.arcs)

        # In-sample caps for w computation (same seed as original experiment)
        insample_caps, _ = generate_capacity_scenarios_uniform_model(num_arcs_total, S, seed=SEED)
        w = compute_w(network, insample_caps)

        # Follower belief scenarios
        follower_caps, _ = if FOLLOWER_MODE == :trunc_normal
            generate_capacity_scenarios_trunc_normal(
                num_arcs_total, S_F, TN_μ, TN_σ, seed=SEED_FOLLOWER)
        else
            generate_capacity_scenarios_contaminated(
                num_arcs_total, S_F, δ_F, seed=SEED_FOLLOWER)
        end

        # Contaminated test scenarios
        test_caps, _ = generate_capacity_scenarios_contaminated(
            num_arcs_total, K_TEST, δ_T, seed=SEED_TEST)

        println("\n── $(net_key) (w=$(w), |A|=$(num_arcs_total-1)) ──")

        for variant in variants
            rkey = result_key(net_key, variant)
            if !haskey(saved_results, rkey) || haskey(saved_results[rkey], :error)
                println("  $(variant): SKIP (no valid x*)")
                continue
            end

            x_star = saved_results[rkey][:x_star]
            # Round near-zero values
            x_star = [abs(x) < 1e-6 ? 0.0 : round(x) for x in x_star]
            oos_original = saved_results[rkey][:oos_mean]

            print("  $(variant): evaluating... ")
            mean_flow, std_flow, h_star = evaluate_oos(
                network, x_star, V_PARAM, w, follower_caps, test_caps)

            Δ = mean_flow - oos_original
            println("  OOS_contam=$(round(mean_flow, digits=4)) ± $(round(std_flow, digits=4)) " *
                    "(original=$(round(oos_original, digits=4)), Δ=$(round(Δ, digits=4)))")

            all_results["$(net_key)_$(variant)"] = Dict(
                :network => net_key,
                :variant => variant,
                :x_star => x_star,
                :h_star => h_star,
                :oos_mean_contam => mean_flow,
                :oos_std_contam => std_flow,
                :oos_mean_original => oos_original,
                :delta_F => δ_F,
                :delta_T => δ_T,
                :w => w,
            )
        end
    end

    # Save results
    serialize(OUTPUT_FILE, all_results)
    println("\nResults saved to $(OUTPUT_FILE)")

    # Print summary table
    print_summary(all_results)

    return all_results
end

function print_summary(results)
    println("\n" * "="^80)
    println("SUMMARY: Contaminated OOS (δ_F=$(δ_F), δ_T=$(δ_T))")
    println("="^80)
    println("┌────────────┬─────────┬────────────┬────────────┬──────────┐")
    println("│  Network   │ Variant │ OOS_orig   │ OOS_contam │    Δ     │")
    println("├────────────┼─────────┼────────────┼────────────┼──────────┤")

    for (net_key, variants) in sort(collect(VALID_COMBOS), by=first)
        for variant in variants
            key = "$(net_key)_$(variant)"
            if !haskey(results, key)
                continue
            end
            r = results[key]
            orig = r[:oos_mean_original]
            contam = r[:oos_mean_contam]
            Δ = contam - orig
            println(@sprintf("│ %-10s │   %-5s │ %10.4f │ %10.4f │ %+8.4f │",
                string(net_key), string(variant), orig, contam, Δ))
        end
    end
    println("└────────────┴─────────┴────────────┴────────────┴──────────┘")
end

# ===== Sanity Check Mode =====
"""
δ_F=0, δ_T=0 → 기존 Uniform{1,...,10}과 동일해야 함.
단, seed가 다르므로 (43/44 vs 43/44) 완전 일치는 기대 불가.
seed를 기존과 맞춰서 검증.
"""
function run_sanity_check()
    println("="^60)
    println("SANITY CHECK: δ=0 should match Uniform{1,...,10}")
    println("="^60)

    net_key = :grid_3x3
    network = make_network(net_key)
    num_arcs_total = length(network.arcs)

    # 기존 experiment와 동일한 seed 사용
    caps_uniform, _ = generate_capacity_scenarios_uniform_model(num_arcs_total, 10, seed=43)
    caps_contam0, _ = generate_capacity_scenarios_contaminated(num_arcs_total, 10, 0.0, seed=43)

    println("Uniform caps[:,1] = $(caps_uniform[1:end-1, 1])")
    println("Contam0 caps[:,1] = $(caps_contam0[1:end-1, 1])")
    println("Match: $(caps_uniform ≈ caps_contam0)")

    # δ=0 → rand() < 0 is always false → all from Uniform{1,...,10}
    # But Random.seed! ordering differs (rand() call before rand(1:10))
    # So exact match requires δ=0 path to skip rand() call
    # Our implementation: if rand() < 0.0 → false → always nominal branch
    # BUT the extra rand() call shifts the RNG state!

    println("\n⚠ Note: δ=0 still calls rand() for the Bernoulli check,")
    println("  so RNG state differs from pure Uniform generator.")
    println("  To verify correctness, compare distributional properties instead.")

    # Distributional check
    caps_contam0_large, _ = generate_capacity_scenarios_contaminated(num_arcs_total, 10000, 0.0, seed=99)
    vals = vec(caps_contam0_large[1:end-1, :])
    println("\nδ=0, 10000 scenarios: min=$(minimum(vals)), max=$(maximum(vals)), mean=$(round(mean(vals), digits=2))")
    println("Expected Uniform{1,...,10}: min=1, max=10, mean=5.5")
end

# ===== Entry =====
if abspath(PROGRAM_FILE) == @__FILE__
    if "--sanity" in ARGS
        run_sanity_check()
    else
        run_contaminated_oos()
    end
end
