"""
summarize_coupling_batch.jl — run_coupling_batch.jl 로그의 RESULT 줄을 모아 표 두 개와 CSV 를 만든다.

  (A) δ 효과: 네트워크·ρ 별로 x* 가 rect (ρ=1) 와 다른 구성 수, Z₀ 상대 감소 (평균·최대)
  (B) 계산 성능: 네트워크·ρ 별 시간·반복 기하평균 (tab:computational 과 같은 집계), Optimal 이 아닌 실행 수

ρ = 1 로그가 없으면 run_baseline_batch.jl 의 logs/factor/eps_*_beta_*/<net>_double.log 를 rect 로 씀 (시간은 다른 실행이라 (B) 에선 표시).
Benders 허용오차가 상대 5e-3 이므로 |ΔZ₀| / Z₀ < 5e-3 인 차이는 수렴 오차와 구별되지 않음 → (A) 에서 따로 셈.

Usage: julia summarize_coupling_batch.jl [logroot=logs/coupling]
"""

using Printf, Statistics

root = joinpath(@__DIR__, length(ARGS) >= 1 ? ARGS[1] : joinpath("logs", "coupling"))
const TOL = parse(Float64, get(ENV, "CPL_TOL", "5e-3"))   # run_coupling_batch.jl 과 같게
nets = ["grid5x5", "polska", "abilene", "nobel_us", "sioux_falls"]

parsekv(line) = Dict(m[1] => m[2] for m in eachmatch(r"(\w+)=(\S*)", line))

rows = Dict{Tuple{String,Float64,Float64,Float64},Dict{String,Any}}()   # (net, β, ε, ρ)
for (dir, _, files) in walkdir(root), f in files
    endswith(f, ".log") || continue
    for l in eachline(joinpath(dir, f))
        startswith(l, "RESULT ") || continue
        d = parsekv(l)
        key = (d["net"], parse(Float64, d["beta"]), parse(Float64, d["eps"]), parse(Float64, d["rho"]))
        rows[key] = Dict{String,Any}("status" => d["status"], "Z0" => parse(Float64, d["Z0"]),
            "iters" => parse(Int, d["iters"]), "time" => parse(Float64, d["time"]),
            "x" => isempty(d["x"]) ? Int[] : parse.(Int, split(d["x"], ";")), "src" => "batch")
    end
end

# rect 대체: baseline double 로그
fmt_val(v) = replace(@sprintf("%.2g", v), "." => "p")
base_dir = joinpath(@__DIR__, "logs", "factor")
for (net, β, ε) in unique((k[1], k[2], k[3]) for k in keys(rows))
    haskey(rows, (net, β, ε, 1.0)) && continue
    p = joinpath(base_dir, "eps_$(fmt_val(ε))_beta_$(fmt_val(β))", "$(net)_double.log")
    isfile(p) || continue
    txt = read(p, String)
    m = match(r"Result \(double\): Z₀=([-\d.eE+]+), iters=(\d+), time=([\d.]+)s", txt)
    mx = match(r"x arcs = \[([^\]]*)\]", txt)
    (m === nothing || mx === nothing) && continue
    rows[(net, β, ε, 1.0)] = Dict{String,Any}("status" => "baseline", "Z0" => parse(Float64, m[1]),
        "iters" => parse(Int, m[2]), "time" => parse(Float64, m[3]),
        "x" => isempty(strip(mx[1])) ? Int[] : parse.(Int, split(mx[1], ",")), "src" => "baseline")
end

isempty(rows) && (println("RESULT 줄이 없음: ", root); exit())

# CSV
csv = joinpath(root, "coupling_summary.csv")
open(csv, "w") do io
    println(io, "net,beta,eps,rho,delta,status,Z0,iters,time,x,src,Z0_rect,x_rect,rel_drop,x_changed")
    for key in sort(collect(keys(rows)))
        (net, β, ε, ρ) = key; r = rows[key]
        rr = get(rows, (net, β, ε, 1.0), nothing)
        zr = rr === nothing ? NaN : rr["Z0"]
        xr = rr === nothing ? "" : join(rr["x"], ";")
        drop = (zr - r["Z0"]) / abs(zr)
        chg = rr === nothing ? "" : string(r["x"] != rr["x"])
        δ = ρ >= 1 ? "Inf" : string(ρ * 2ε)
        @printf(io, "%s,%.2f,%.2f,%.2f,%s,%s,%.8f,%d,%.2f,%s,%s,%.8f,%s,%.6e,%s\n", net, β, ε, ρ, δ, r["status"],
                r["Z0"], r["iters"], r["time"], join(r["x"], ";"), r["src"], zr, xr, drop, chg)
    end
end
println("CSV: ", csv)

ρs = sort(unique(k[4] for k in keys(rows)); rev=true)
ρc = filter(<(1.0), ρs)
gm(v) = isempty(v) ? NaN : exp(mean(log.(max.(v, 1e-9))))

println("\n(A) δ 효과 (rect 대비).  x≠ = x* 가 바뀐 구성 수 / 비교 가능한 구성 수,  drop = (Z_rect − Z_δ)/Z_rect [%],",
        "  sig = drop > tol(", TOL * 100, "%) 인 구성 수")
@printf("%-12s", "network"); for ρ in ρc; @printf(" | ρ=%-4.2f x≠   mean%%   max%%  sig", ρ); end; println()
for net in nets
    @printf("%-12s", net)
    for ρ in ρc
        ks = [k for k in keys(rows) if k[1] == net && k[4] == ρ && haskey(rows, (k[1], k[2], k[3], 1.0))]
        if isempty(ks); @printf(" | %31s", "-"); continue; end
        ch = count(k -> rows[k]["x"] != rows[(k[1], k[2], k[3], 1.0)]["x"], ks)
        dr = [(rows[(k[1], k[2], k[3], 1.0)]["Z0"] - rows[k]["Z0"]) / abs(rows[(k[1], k[2], k[3], 1.0)]["Z0"]) for k in ks]
        @printf(" | %8s %6.2f %6.2f %4d", "$ch/$(length(ks))", 100mean(dr), 100maximum(dr), count(>(TOL), dr))
    end
    println()
end

println("\n(B) 계산 성능: 시간 (s) / 반복의 기하평균 (β, ε 구성에 대해),  [비Optimal 수].  * = rect 가 baseline 로그 (다른 실행)")
@printf("%-12s", "network"); for ρ in ρs; @printf(" | ρ=%-4.2f  time   iter  ", ρ); end; println()
for net in nets
    @printf("%-12s", net)
    for ρ in ρs
        ks = [k for k in keys(rows) if k[1] == net && k[4] == ρ]
        if isempty(ks); @printf(" | %22s", "-"); continue; end
        t = gm([rows[k]["time"] for k in ks]); it = gm([Float64(rows[k]["iters"]) for k in ks])
        bad = count(k -> rows[k]["status"] ∉ ("Optimal", "baseline"), ks)
        star = any(rows[k]["src"] == "baseline" for k in ks) ? "*" : " "
        @printf(" | %7.0f%s %5.0f [%d]", t, star, it, bad)
    end
    println()
end
