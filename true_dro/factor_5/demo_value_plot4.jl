#=
demo_value_plot4.jl — 발표용 비제로섬 예시: retailer 믿음의 수요 편향 (κ) 에 따른 실현 비용 (2026-10-21 사용자 요청).
데이터: nonzero_sum/logs/demo_value/<DP_NZ_TAG>/eval_bias.csv (demo_value_nz.jl DN_STAGE=bias), 추가 계산 없음.
  x 축 κ: 믿음 p̃_s ∝ p_s exp(κ z_s), z = 시나리오 총수요 표준화 (κ > 0 낙관, κ < 0 비관, 0 = 믿음이 맞음). 위에 평균 d_TV(p, p̃).
  y 축: 실현 비용 (낮을수록 좋음) 의 평균 (왼쪽) 과 q95 (오른쪽). 결정이 같은 모형은 한 선으로 묶어 범례에 같이 적음.
결과: 같은 폴더의 fig_nz_bias_g<γ>.png, 표 (stdout)
환경변수: DP_NZ_TAG = sgb128_S10_eps0p10_q150_s5   DP_NZ_EPS = 0.1   DP_GAMMAS = 5,1,0.1
실행: julia demo_value_plot4.jl
=#
using Plots, Printf, Statistics
gr()
root = dirname(dirname(@__DIR__))
tag = get(ENV, "DP_NZ_TAG", "sgb128_S10_eps0p10_q150_s5"); eps_ = get(ENV, "DP_NZ_EPS", "0.1")
dir = joinpath(root, "nonzero_sum", "logs", "demo_value", tag)
gammas = parse.(Float64, split(get(ENV, "DP_GAMMAS", "5,1,0.1"), ","))
site = Dict("{1;2}" => "{Youngstown, Yankton}", "{2}" => "{Yankton}", "{1}" => "{Youngstown}")
lab(m) = m == "nominal" ? "Nominal" : m == "partial" ? "Partial DR (ε̃ = 0)" : m == "product" ? "Product (δ = ∞)" :
         "Ours (ε = $eps_, δ = " * replace(replace(m, "coupled_d" => ""), "p" => ".") * ")"
order = ["nominal", "partial", "product", "coupled_d0p1"]

ls = readlines(joinpath(dir, "eval_bias.csv")); hdr = split(ls[1], ","); col(n) = findfirst(==(n), hdr)
rows = [split(l, ",") for l in ls[2:end] if !isempty(l)]
xof = Dict{String,String}()
for r in rows; xof[r[col("model")]] = r[col("x")]; end
models = [m for m in order if haskey(xof, m)]
groups = Dict{String,Vector{String}}()                   # 결정 → 그 결정을 고른 모형들
for m in models; push!(get!(groups, xof[m], String[]), m); end
gx = sort(collect(keys(groups)); by=k -> findfirst(m -> m in groups[k], models))
gcol = Dict(k => (any(startswith(m, "coupled") for m in groups[k]) ? :darkorange : :gray40) for k in gx)

for γ in gammas
    rg = [r for r in rows if parse(Float64, r[col("gamma")]) == γ]
    κs = sort(unique(parse(Float64, r[col("kappa")]) for r in rg))
    tvm = [mean(parse(Float64, r[col("tv")]) for r in rg if parse(Float64, r[col("kappa")]) == κ && r[col("model")] == models[1]) for κ in κs]
    stat = Dict{Tuple{String,Float64},Vector{Float64}}()
    for r in rg
        m = r[col("model")]; m == groups[xof[m]][1] || continue          # 결정마다 한 번만
        push!(get!(stat, (xof[m], parse(Float64, r[col("kappa")])), Float64[]), parse(Float64, r[col("cost")]))
    end
    @printf("\n[γ = %.1f] κ / 평균 TV / 결정별 평균, q95\n", γ)
    for (i, κ) in enumerate(κs)
        @printf("  κ=%5.2f TV=%.3f | %s\n", κ, tvm[i], join([@sprintf("%s 평균 %.1f q95 %.1f", k, mean(stat[(k, κ)]), quantile(stat[(k, κ)], 0.95)) for k in gx], " | "))
    end
    xt = (κs, [@sprintf("%g\n(%.2f)", κ, t) for (κ, t) in zip(κs, tvm)])
    ps = []
    for (fn, nm) in ((mean, "mean"), (v -> quantile(v, 0.95), "95% quantile"))
        p = plot(; xlabel="retailer's demand bias κ   (mean d_TV(p, p̃))", ylabel="$nm of realized cost of B", xticks=xt,
                 legend=(nm == "mean" ? :topleft : false), left_margin=8Plots.mm, bottom_margin=10Plots.mm, xtickfontsize=7)
        vals = [fn(stat[(k, κ)]) for k in gx for κ in κs]; pad = 0.15 * (maximum(vals) - minimum(vals))
        plot!(p; ylims=(minimum(vals) - pad, maximum(vals) + 2pad))
        plot!(p, [0.0, 0.0], [minimum(vals) - pad, maximum(vals) + 2pad]; color=:black, linestyle=:dot, label="")
        for k in gx
            l = join(lab.(groups[k]), " = ") * "\nopens " * site[k]
            plot!(p, κs, [fn(stat[(k, κ)]) for κ in κs]; color=gcol[k], linewidth=3, marker=:circle, label=l)
        end
        push!(ps, p)
    end
    plt = plot(ps...; layout=(1, 2), size=(1150, 500), legendfontsize=8,
               plot_title=@sprintf("Non-zero-sum: retailer optimism (κ > 0) vs pessimism (κ < 0), truth p ~ Dir(%g·S·q̂)", γ),
               plot_titlefontsize=12)
    f = joinpath(dir, @sprintf("fig_nz_bias_g%s.png", replace(string(γ), "." => "p"))); savefig(plt, f); println("saved $f")
end
