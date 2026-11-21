# belief 블록 하나의 min_x q·x + P(x) 를 Benders 로: McCormick cut vs perspective cut. 반복 수 (= 방문 x 수) 와 시간
include(joinpath(@__DIR__, "persp_defs.jl"))
qloc = [nd.qx[(SITES[a] - 1) * K + k] for a in 1:nS for k in 1:K]
TOLB = 5e-3
function bend(fn, name)
    O = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); o))
    @variable(O, xv[1:nx], Bin); @variable(O, t0 >= -1e7)
    for a in 1:nS; @constraint(O, sum(xv[xe(a, k)] for k in 1:K) <= 1); end
    @constraint(O, sum(xv) <= 3)
    @objective(O, Min, dot(qloc, xv) + t0)
    UB = Inf; LB = -Inf; it = 0; seen = Set{Vector{Float64}}(); tcut = 0.0; wall = time(); hist = Float64[]
    while it < 200
        it += 1; optimize!(O); x̂ = Float64.(value.(xv) .> 0.5); LB = objective_value(O)
        push!(hist, LB)
        Pv = solveW(x̂)[1]; UB = min(UB, dot(qloc, x̂) + Pv)
        abs(UB - LB) <= TOLB * max(1.0, abs(UB)) && break
        x̂ in seen && (println("  같은 x 반복 — 멈춤"); break); push!(seen, x̂)
        tcut += @elapsed ((v, gr) = cutat(fn, x̂))
        @constraint(O, t0 >= v + dot(gr, xv .- x̂))
    end
    @printf("BEND %-11s 반복 %3d (방문 x %d / %d), LB %.1f UB %.1f, cut 시간 %.0fs, 전체 %.0fs | LB 추이 (5 회마다): %s\n", name, it, length(seen) + 1, length(allX),
            LB, UB, tcut, time() - wall, join(round.(Int, hist[1:5:end]), " "))
    flush(stdout)
end
allX = Vector{Vector{Tuple{Int,Int}}}(); push!(allX, Tuple{Int,Int}[])
for n in 1:min(3, nS), comb in Iterators.filter(c -> issorted(c) && allunique(c), Iterators.product(ntuple(_ -> 1:nS, n)...)), lv in Iterators.product(ntuple(_ -> 1:K, n)...)
    push!(allX, collect(zip(comb, lv)))
end
bend(solveW, "McCormick")
bend(solveSA, "perspective")
