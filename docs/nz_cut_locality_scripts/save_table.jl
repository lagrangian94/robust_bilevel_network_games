include(joinpath(@__DIR__, "lovasz_defs.jl"))
# x 고정 값 + 그 해의 거점별 판매량 (약정·현장·버림, 시나리오 평균) 과 follower 이득 V_F 를 함께 저장
MM = nd.meta; nA, ncu = MM[:nA], MM[:ncu]
function value_at(xv)
    for i in 1:nx; fix(x[i], xv[i]; force=true); end
    lev = zeros(Int, nB); for i in 1:nB, k in 1:K; xv[(i - 1) * K + k] > 0.5 && (lev[i] = k); end
    vf = VF(lev)
    c = @constraint(M, Fexpr >= vf); optimize!(M); v = objective_value(M) - dot(nd.qx, xv)
    sold = Dict{Int,NTuple{3,Float64}}()
    for i in 1:nB
        lev[i] == 0 && continue
        ib = nA + i
        R = sum(r[s] * sum(value(y[MM[:iyR](ib, j), s]) for j in 1:ncu) for s in 1:S)
        Sp = sum(r[s] * sum(value(y[MM[:iyS](ib, j), s]) for j in 1:ncu) for s in 1:S)
        W = sum(r[s] * value(y[MM[:ys](ib), s]) for s in 1:S)
        sold[i] = (R, Sp, W)
    end
    delete(M, c); for i in 1:nx; unfix(x[i]); set_lower_bound(x[i], 0.0); set_upper_bound(x[i], 1.0); end
    return v, vf, sold
end
xof(sel) = (xv = zeros(nx); for (i, k) in sel; xv[(i - 1) * K + k] = 1; end; xv)
sets = Vector{Vector{Tuple{Int,Int}}}(); push!(sets, Tuple{Int,Int}[])
for i in 1:nB, k in 1:K; push!(sets, [(i, k)]); end
for a in 1:nB, b in a+1:nB, ka in 1:K, kb in 1:K; push!(sets, [(a, ka), (b, kb)]); end
for a in 1:nB, b in a+1:nB, c in b+1:nB, ka in 1:K, kb in 1:K, kc in 1:K; push!(sets, [(a, ka), (b, kb), (c, kc)]); end
open(joinpath(@__DIR__, "vtable_g1_nominal.tsv"), "w") do io
    println(io, "set\tq\tV\tVF\tsold(i:R,Sp,W)")
    for sel in sets
        v, vf, sold = isempty(sel) ? (0.0, V0, Dict{Int,NTuple{3,Float64}}()) : value_at(xof(sel))
        println(io, join(["$(i):$(k)" for (i, k) in sel], ";"), "\t", sum(nd.qx[(i - 1) * K + k] for (i, k) in sel; init=0.0), "\t", v, "\t", vf, "\t",
                join(["$(i):" * join(round.(collect(t); digits=3), ",") for (i, t) in sort(collect(sold))], ";"))
    end
end
# 지리 정보
open(joinpath(@__DIR__, "geo_g1.tsv"), "w") do io
    cities = MM[:cities]; println(io, "site\tcity\tlon\tregion")
    for i in 1:nB; c = cities[nA + i]; println(io, i, "\t", c, "\t", _loc_lon(c), "\t", _loc_region(c)); end
    println(io, "# A: ", join(cities[1:nA], ", "), "\n# customers (region): ", join(["$(c)($(_loc_region(c)))" for c in cities[nA+nB+1:end]], ", "))
    D = MM[:dist]; println(io, "# dist B-site to customers (rows site, cols customer)")
    for i in 1:nB; println(io, join(Int.(round.(D[nA + i, :]; digits=0)), " ")); end
end
println("saved")
