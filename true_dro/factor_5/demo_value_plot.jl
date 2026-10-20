#=
demo_value_plot.jl — demo_value_zs.jl / demo_value_nz.jl 의 eval_summary.csv 로 발표용 그림 2 장.
  fig_tail.png : x 축 belief 오차 τ, y 축 실현 손실의 95% 분위수 (모델별 선)
  fig_mean.png : 같은 축, 평균
모델 이름: nominal, partial (belief 를 안다고 가정), product (결합 없음), coupled_d* (우리).
실행: julia demo_value_plot.jl <결과 폴더> [제목]
=#
using Plots, Printf
gr()

dir = length(ARGS) >= 1 ? ARGS[1] : error("결과 폴더를 인자로")
ttl = length(ARGS) >= 2 ? ARGS[2] : basename(dir)
lines_ = readlines(joinpath(dir, "eval_summary.csv"))
hdr = split(lines_[1], ","); col(n) = findfirst(==(n), hdr)
rows = [split(l, ",") for l in lines_[2:end] if !isempty(l)]
models = unique([r[col("model")] for r in rows])
label(m) = m == "nominal" ? "Nominal" : m == "partial" ? "Belief-known (ε̃ = 0)" : m == "product" ? "Product (no coupling)" :
           "Coupled, δ = " * replace(replace(m, "coupled_d" => ""), "p" => ".")
style(m) = startswith(m, "coupled") ? (:solid, 3) : (:dash, 2)

for (stat, fname, ylab) in (("q95", "fig_tail.png", "95% quantile of realized loss"), ("mean", "fig_mean.png", "mean realized loss"))
    plt = plot(; xlabel="belief error τ  (d_TV(p, p̃) ≤ τ)", ylabel=ylab, title=ttl, legend=:topleft, size=(720, 460))
    for m in models
        rs = [r for r in rows if r[col("model")] == m]
        τ = [parse(Float64, r[col("tau")]) for r in rs]; v = [parse(Float64, r[col(stat)]) for r in rs]
        ls, lw = style(m)
        plot!(plt, τ, v; label=label(m), linestyle=ls, linewidth=lw, marker=:circle, markersize=3)
    end
    savefig(plt, joinpath(dir, fname))
    @printf("saved %s\n", joinpath(dir, fname))
end
