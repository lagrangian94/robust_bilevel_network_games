using Random, Printf, Statistics

include("network_generator.jl")
using .NetworkGenerator

net = generate_nobel_us_network()
num_arcs = length(net.arcs) - 1
intd_idx = findall(net.interdictable_arcs[1:num_arcs])

caps, F_mat = generate_capacity_scenarios_factor_sparse(length(net.arcs), 20;
    interdictable_arcs=net.interdictable_arcs, seed=42, num_factors=5)

println("=== Nobel_us additive factor k=5 scenario diversity ===")
println("c_bar=10, a=4, F~U(-4,4), ξ~Exp(1)\n")

# 각 시나리오별 interdictable arc capacity
println("--- Interdictable arc capacities (S=20) ---")
@printf("%6s", "arc")
for s in 1:20; @printf("  S%-2d  ", s); end
println()
for (idx, e) in enumerate(intd_idx)
    @printf("%6d", e)
    for s in 1:20
        @printf("  %5.2f", caps[e, s])
    end
    println()
end

# 각 시나리오별 min capacity arc (bottleneck)
println("\n--- Bottleneck arc per scenario ---")
for s in 1:20
    vals = caps[intd_idx, s]
    min_idx = argmin(vals)
    @printf("S%2d: arc %2d (c=%.2f)\n", s, intd_idx[min_idx], vals[min_idx])
end

# unique bottleneck arcs
bottlenecks = [intd_idx[argmin(caps[intd_idx, s])] for s in 1:20]
println("\nUnique bottleneck arcs: $(sort(unique(bottlenecks)))")
println("Bottleneck counts: ")
for a in sort(unique(bottlenecks))
    @printf("  arc %d: %d times\n", a, count(==(a), bottlenecks))
end

# capacity range per arc
println("\n--- Capacity range per interdictable arc ---")
@printf("%6s  %8s  %8s  %8s  %8s\n", "arc", "min", "max", "mean", "std")
for (idx, e) in enumerate(intd_idx)
    v = caps[e, :]
    @printf("%6d  %8.3f  %8.3f  %8.3f  %8.3f\n", e, minimum(v), maximum(v), mean(v), std(v))
end

# ranking diversity: 각 시나리오에서 capacity 오름차순 rank
println("\n--- Top-1 (lowest cap) arc per scenario ---")
top1 = [intd_idx[argmin(caps[intd_idx, s])] for s in 1:20]
println("Top-1 arcs: $top1")
println("Unique: $(sort(unique(top1)))")
