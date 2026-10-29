ENV["LA_NOMAIN"] = "1"
include(raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpstruct\lp_ablate.jl")
L = readlp(raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\infdump\S20b\infeas_1.mps"); cc = colclass(L)
g = [rowgroup(L, cc, i) for i in 1:L.nr]
for x in sort(unique(g)); @printf("%-44s %6d\n", x, count(==(x), g)); end
