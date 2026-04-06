"""
check_mhat_sparsity.jl — Mhat PSD cone의 aggregate sparsity pattern 분석
COSMO chordal decomposition 가능 여부 확인용.

Mhat ∈ S^{na+1} (na=num_arcs=76, dim=77).
Aggregate sparsity = Mhat[i,j]가 PSD 제약 외의 다른 제약에 참여하는지 여부.
"""

using Statistics
using Revise
includet("network_generator.jl")
includet("compact_ldr_utils.jl")
using .NetworkGenerator: generate_sioux_falls_network

network = generate_sioux_falls_network()
num_arcs = length(network.arcs) - 1  # 76
na1 = num_arcs + 1  # 77

println("Mhat dimension: $na1 x $na1 = $(na1^2) entries")
println("Unique (symmetric): $(div(na1*(na1+1),2)) entries")

# Aggregate sparsity matrix: S[i,j]=true if Mhat[i,j] appears in some constraint
agg_sparsity = falses(na1, na1)

# 1. cons_dual_constant: Mhat[end,end] <= 1/S
agg_sparsity[na1, na1] = true

# 2. trace constraint: tr(Mhat[1:na,1:na]) - Mhat[end,end]*eps^2 <= 0
for i in 1:num_arcs
    agg_sparsity[i, i] = true  # diagonal
end
agg_sparsity[na1, na1] = true

# 3. Phi_hat_L: Adj_L_Mhat_11 = -D_s * Mhat_11 (D_s = diag(xi_bar))
#    제약은 arc_adj[i,j]=true인 (i,j)에만 존재
#    Adj_L_Mhat_11[i,j] = -xi_bar[i]*Mhat[i,j] (D_s가 diagonal이므로)
#    Adj_L_Mhat_12[i] = -Mhat[i,end]*xi_bar' -> 제약 lhs_L[i,j]에 Mhat[i,end] 기여
arc_adj = get_ldr_adjacency(network; ldr_mode=:self)
for i in 1:num_arcs, j in 1:num_arcs
    if arc_adj[i,j]
        agg_sparsity[i, j] = true
        agg_sparsity[i, na1] = true
        agg_sparsity[na1, i] = true  # symmetric
    end
end

# 4. Phi_hat_0: -D_s*Mhat[1:na,end] and -xi_bar*Mhat[end,end]
#    all i -> Mhat[i,end] constrained
for i in 1:num_arcs
    agg_sparsity[i, na1] = true
    agg_sparsity[na1, i] = true
end
agg_sparsity[na1, na1] = true

# 5-6: Psi_hat_L / Psi_hat_0: same Mhat entries as Phi (already covered)

# Report
nnz = sum(agg_sparsity)
total = na1 * na1
println("\n=== Aggregate Sparsity Pattern ===")
println("Non-zero entries: $nnz / $total ($(round(nnz/total*100, digits=1))%)")
println("Zero entries:     $(total-nnz) / $total ($(round((total-nnz)/total*100, digits=1))%)")

println("\n=== Block Analysis ===")
nnz_11 = sum(agg_sparsity[1:num_arcs, 1:num_arcs])
total_11 = num_arcs^2
println("Mhat_11 ($num_arcs x $num_arcs): $nnz_11 / $total_11 ($(round(nnz_11/total_11*100, digits=1))%)")

nnz_12 = sum(agg_sparsity[1:num_arcs, na1])
println("Mhat_12 ($num_arcs x 1): $nnz_12 / $num_arcs ($(round(nnz_12/num_arcs*100, digits=1))%)")
println("Mhat_22 (1x1): $(agg_sparsity[na1,na1]) (always constrained)")

arc_adj_nnz = sum(arc_adj[1:num_arcs, 1:num_arcs])
println("\narc_adj (ldr_mode=:self): $arc_adj_nnz / $(num_arcs^2) ($(round(arc_adj_nnz/num_arcs^2*100, digits=1))%)")

diag_constrained = sum(agg_sparsity[i,i] for i in 1:na1)
println("Diagonal constrained: $diag_constrained / $na1")

println("\n=== Row density (Mhat_11 block) ===")
row_nnz = [sum(agg_sparsity[i, 1:num_arcs]) for i in 1:num_arcs]
println("Min row nnz: $(minimum(row_nnz))")
println("Max row nnz: $(maximum(row_nnz))")
println("Mean row nnz: $(round(mean(row_nnz), digits=1))")
println("Median row nnz: $(median(row_nnz))")

# Key insight: Mhat_12 column (last column) is FULLY DENSE
# because Phi_hat_0 constrains ALL Mhat[i,end]
# This connects every row to the last column -> fill-in during chordal completion
println("\n=== Chordal Decomposition Potential ===")
println("Mhat_12 (last column) density: $nnz_12 / $num_arcs = $(round(nnz_12/num_arcs*100,digits=1))%")
if nnz_12 == num_arcs
    println("!! Last column FULLY DENSE -> chordal completion makes entire matrix dense")
    println("   (every node connected to node $na1 -> single maximal clique)")
end
if nnz / total > 0.5
    println("DENSE ($(round(nnz/total*100,digits=1))%) -> chordal decomposition will NOT help")
elseif nnz / total > 0.2
    println("Moderate sparsity -> some decomposition possible")
else
    println("Sparse ($(round(nnz/total*100,digits=1))%) -> chordal decomposition should help")
end
