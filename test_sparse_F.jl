using Random, Statistics, Printf

Random.seed!(42)

include("network_generator.jl")
using .NetworkGenerator
net = generate_nobel_us_network()
num_arcs = length(net.arcs) - 1
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
num_intd = length(intd_idx)
println("num_intd = $num_intd")

# 1) 1000개 sparse column 생성 (70% zero, 30% rand(1:10))
Random.seed!(42)
N_pool = 1000
k = 5
zero_prob = 0.7
cols = zeros(Int, num_intd, N_pool)
for j in 1:N_pool
    for i in 1:num_intd
        if rand() > zero_prob
            cols[i, j] = rand(1:10)
        end
    end
end

nnz_per_col = [count(x -> x > 0, cols[:, j]) for j in 1:N_pool]
@printf("Column sparsity: mean=%.1f/%d nonzero, min=%d, max=%d\n",
        mean(nnz_per_col), num_intd, minimum(nnz_per_col), maximum(nnz_per_col))

# 2) k-means (k=5)
centroids_f = Float64.(cols[:, round.(Int, range(1, N_pool; length=k))])
assignments = zeros(Int, N_pool)
for iter in 1:100
    changed = false
    for j in 1:N_pool
        best_c, best_d = 1, Inf
        for c in 1:k
            d = sum((cols[i, j] - centroids_f[i, c])^2 for i in 1:num_intd)
            if d < best_d; best_d = d; best_c = c; end
        end
        if assignments[j] != best_c; assignments[j] = best_c; changed = true; end
    end
    for c in 1:k
        members = findall(==(c), assignments)
        if !isempty(members)
            for i in 1:num_intd
                centroids_f[i, c] = mean(cols[i, j] for j in members)
            end
        end
    end
    if !changed; println("k-means converged at iter $iter"); break; end
end

for c in 1:k
    @printf("Cluster %d: %d members\n", c, count(==(c), assignments))
end

# 3) Medoids
medoids = Int[]
for c in 1:k
    members = findall(==(c), assignments)
    best_j, best_d = members[1], Inf
    for j in members
        d = sum((cols[i, j] - centroids_f[i, c])^2 for i in 1:num_intd)
        if d < best_d; best_d = d; best_j = j; end
    end
    push!(medoids, best_j)
end

println("\n--- Medoid F columns (integer) ---")
@printf("%6s", "arc")
for c in 1:k; @printf("  F_%d", c); end
println()
for i in 1:num_intd
    @printf("%6d", intd_idx[i])
    for c in 1:k; @printf("  %3d", cols[i, medoids[c]]); end
    println()
end

println("\n--- Medoid nonzero arcs ---")
for c in 1:k
    nz = findall(x -> x > 0, cols[:, medoids[c]])
    @printf("F_%d: %d/%d nonzero, arcs = %s\n", c, length(nz), num_intd, string(intd_idx[nz]))
end

# overlap check
println("\n--- Pairwise overlap (shared nonzero arcs) ---")
for c1 in 1:k
    nz1 = Set(findall(x -> x > 0, cols[:, medoids[c1]]))
    for c2 in (c1+1):k
        nz2 = Set(findall(x -> x > 0, cols[:, medoids[c2]]))
        shared = length(intersect(nz1, nz2))
        @printf("F_%d ∩ F_%d: %d shared\n", c1, c2, shared)
    end
end
