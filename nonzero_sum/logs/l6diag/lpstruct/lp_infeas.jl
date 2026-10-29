#= lp_infeas.jl — 저장된 실행 불가능 노드 LP (NZ_INFEAS_DUMP) : 풀이 방법별 증명 비용 + IIS 의 행 종류 =#
ENV["LA_NOMAIN"] = "1"
include(raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpstruct\lp_ablate.jl")
setp!(m, k, v::Int) = C.GRBsetintparam(C.GRBgetenv(m), k, v)
setp!(m, k, v::Float64) = C.GRBsetdblparam(C.GRBgetenv(m), k, v)
function trial(path, lbl; method=1, presolve=0, warm=false)
    m = load(path); prm!(m); setp!(m, "Method", method); setp!(m, "Presolve", presolve); setp!(m, "TimeLimit", 300.0)
    rc = warm ? C.GRBread(m, replace(path, ".mps" => "_start.bas")) : 0
    t = @elapsed C.GRBoptimize(m)
    it = Int(getd(m, "IterCount")); bit = geti(m, "BarIterCount")
    @printf("  %-26s %-4s %7d simplex %4d barrier  %7.2f s%s\n", lbl, st(m), it, bit, t, rc == 0 ? "" : "  (basis 읽기 실패 $rc)")
    C.GRBfreemodel(m); flush(stdout)
end
function iis(path)
    L = readlp(path); cc = colclass(L); g = [rowgroup(L, cc, i) for i in 1:L.nr]
    m = load(path); prm!(m); setp!(m, "TimeLimit", 300.0)
    t = @elapsed (rc = C.GRBcomputeIIS(m))
    if rc != 0
        @printf("  IIS 실패 rc=%d (%.1f s)\n", rc, t); C.GRBfreemodel(m); return
    end
    ic = iarr(m, "IISConstr", L.nr); il = iarr(m, "IISLB", L.nc); iu = iarr(m, "IISUB", L.nc)
    rows = findall(!=(0), ic); cnt = Dict{String,Int}(); tot = Dict{String,Int}()
    for x in g; tot[x] = get(tot, x, 0) + 1; end
    for i in rows; cnt[g[i]] = get(cnt, g[i], 0) + 1; end
    @printf("  IIS (%.1f s, 최소=%d): 행 %d / %d, 열 상하한 %d\n", t, geti(m, "IISMinimal"), length(rows), L.nr, count(!=(0), il) + count(!=(0), iu))
    for (k, v) in sort(collect(cnt); by=x -> -x[2]); @printf("     %-44s %5d / %5d\n", k, v, tot[k]); end
    bf = Dict{String,Int}()
    for j in 1:L.nc
        (il[j] != 0 || iu[j] != 0) || continue
        f = anon(L.cn[j]) ? cc[j] : family(L.cn[j]); bf[f] = get(bf, f, 0) + 1
    end
    @printf("     열 상하한: %s\n", join(["$k $v" for (k, v) in sort(collect(bf); by=x -> -x[2])], ", "))
    C.GRBfreemodel(m); flush(stdout)
end
dir = raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\infdump\S20b"
ks = parse.(Int, get(ENV, "LI_K", join(1:12, ",")) |> s -> split(s, ","))
for k in ks
    p = joinpath(dir, "infeas_$k.mps")
    @printf("\n#### infeas_%d  (B&B 안: %s)\n", k, strip(read(joinpath(dir, "infeas_$k.txt"), String)))
    trial(p, "dual, 시작 basis"; warm=true)
    trial(p, "dual, cold")
    trial(p, "dual, cold, Presolve 켬"; presolve=-1)
    trial(p, "barrier, Presolve 켬"; method=2, presolve=-1)
    get(ENV, "LI_IIS", "1") == "1" && iis(p)
end
