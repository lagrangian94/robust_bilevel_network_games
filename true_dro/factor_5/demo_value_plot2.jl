#=
demo_value_plot2.jl — 발표용 예시 그림 (2026-10-21 사용자 요청: τ 축 대신 결정·분포 중심으로).
  비제로섬 (SGB128 S=10 ε=0.2 한도 150 seed 5):
    fig_nz_delta.png : 모델별 보장 이익 (−자기 최악값) 과 고른 결정. δ 가 커지면 Product 와 같은 보수적 결정으로 바뀜
    fig_nz_dist.png  : 두 결정의 실현 이익 (= −실현 비용) 분포 (상자: 25~75%, 수염: 5~95%, 점: 1%·최저)
  zero-sum (Polska S=10):
    fig_zs_dist.png  : 두 결정의 실현 손실 (흐름) 분포 + 오른쪽 꼬리 (넘침 확률 P(loss > t))
데이터: 각 폴더의 eval_draws.csv (모든 τ 를 합침), solve_*.log 의 RESULT, x_*.txt. 추가 계산 없음.
실행: julia demo_value_plot2.jl
=#
using Plots, Printf, Statistics
gr()
root = dirname(dirname(@__DIR__))
nzdir = joinpath(root, "nonzero_sum", "logs", "demo_value", "sgb128_S10_eps0p20_q150_s5")
zsdir = joinpath(root, "true_dro", "factor_5", "logs", "demo_value", "polska")

function draws(dir, model)
    ls = readlines(joinpath(dir, "eval_draws.csv")); hdr = split(ls[1], ",")
    im, iv = findfirst(==("model"), hdr), length(hdr)
    return [parse(Float64, split(l, ",")[iv]) for l in ls[2:end] if !isempty(l) && split(l, ",")[im] == model]
end
function ub(dir, model)
    for l in readlines(joinpath(dir, "solve_$(model).log"))
        startswith(l, "RESULT") || continue
        m = match(r"UB=([-\d.eE+]+)", l); return parse(Float64, m.captures[1])
    end
    error("RESULT 없음: $model")
end
xof(dir, model) = strip(read(joinpath(dir, "x_$(model).txt"), String))

"상자 그림 하나 (위치 pos): 25~75% 상자, 5~95% 수염, 중앙값 선, 1%/99% 와 극값 점"
function box!(plt, pos, v; color, label="", tail=:upper)
    q(p) = quantile(v, p)
    w = 0.28
    plot!(plt, Shape([pos - w, pos + w, pos + w, pos - w], [q(0.25), q(0.25), q(0.75), q(0.75)]);
          fillcolor=color, fillalpha=0.35, linecolor=color, label=label)
    plot!(plt, [pos - w, pos + w], [q(0.5), q(0.5)]; color=color, linewidth=3, label="")
    plot!(plt, [pos, pos], [q(0.05), q(0.25)]; color=color, label=""); plot!(plt, [pos, pos], [q(0.75), q(0.95)]; color=color, label="")
    ext = tail == :upper ? (q(0.99), maximum(v)) : (q(0.01), minimum(v))
    scatter!(plt, [pos, pos], collect(ext); color=color, markershape=[:diamond, :star5], markersize=[6, 9], label="")
end

# ---------------- 비제로섬 ----------------
nz_models = [("nominal", "Nominal"), ("partial", "Belief-known\n(ε̃ = 0)"), ("coupled_d0p05", "Coupled\nδ = 0.05"),
             ("coupled_d0p1", "Coupled\nδ = 0.1"), ("coupled_d0p2", "Coupled\nδ = 0.2"), ("product", "Product\n(δ = ∞)")]
site = Dict("1,2" => "{Youngstown, Yankton}", "2" => "{Yankton}")
col_nz = Dict("1,2" => :royalblue, "2" => :firebrick)
plt = plot(; ylabel="guaranteed profit of B", left_margin=8Plots.mm, title="Non-zero-sum: coupling avoids over-conservatism",
           legend=:topright, size=(820, 470), xticks=(1:length(nz_models), last.(nz_models)), bottom_margin=6Plots.mm)
for (i, (m, _)) in enumerate(nz_models)
    x = xof(nzdir, m); v = -ub(nzdir, m)
    bar!(plt, [i], [v]; color=col_nz[x], alpha=0.75, label="", bar_width=0.6)
    annotate!(plt, i, v + 25, text(site[x], 7, :center))
end
for (x, c) in col_nz
    plot!(plt, [NaN], [NaN]; seriestype=:bar, color=c, alpha=0.75, label="opens " * site[x])
end
ylims!(plt, (0, 1200))
savefig(plt, joinpath(nzdir, "fig_nz_delta.png")); println("saved fig_nz_delta.png")

a = -draws(nzdir, "coupled_d0p1"); b = -draws(nzdir, "product")     # 실현 이익 (높을수록 좋음)
plt = plot(; ylabel="realized profit of B", left_margin=8Plots.mm,
           title="Non-zero-sum: realized profit of the two decisions", legend=:bottomright, size=(720, 470),
           xticks=([1, 2], ["Coupled δ ≤ 0.1\nopens {Youngstown, Yankton}", "Product (δ = ∞)\nopens {Yankton}"]), xlims=(0.4, 2.6),
           bottom_margin=6Plots.mm)
box!(plt, 1, a; color=:royalblue, tail=:lower); box!(plt, 2, b; color=:firebrick, tail=:lower)
scatter!(plt, [NaN], [NaN]; markershape=:diamond, color=:gray, label="1% quantile"); scatter!(plt, [NaN], [NaN]; markershape=:star5, color=:gray, label="minimum")
@printf("NZ realized profit: coupled mean %.1f, q05 %.1f, min %.1f | product mean %.1f, q05 %.1f, min %.1f (N=%d)\n",
        mean(a), quantile(a, 0.05), minimum(a), mean(b), quantile(b, 0.05), minimum(b), length(a))
savefig(plt, joinpath(nzdir, "fig_nz_dist.png")); println("saved fig_nz_dist.png")

# ---------------- zero-sum ----------------
n_ = draws(zsdir, "nominal"); r_ = draws(zsdir, "coupled_d0p1")
lab_n = "Nominal / Belief-known\ncuts Bialystok→t, s→Kolobrzeg"; lab_r = "Product / Coupled (ε̃ > 0)\ncuts Kolobrzeg→Bydgoszcz, s→Poznan"
p1 = plot(; ylabel="realized loss (CVaR of max flow)", title="Zero-sum (Polska): realized loss", left_margin=10Plots.mm,
          legend=false, xticks=([1, 2], [lab_n, lab_r]), xlims=(0.4, 2.6), bottom_margin=8Plots.mm, xtickfontsize=7)
box!(p1, 1, n_; color=:gray40); box!(p1, 2, r_; color=:darkorange)
tgrid = range(min(minimum(n_), minimum(r_)), max(maximum(n_), maximum(r_)); length=300)
exc(v, t) = mean(v .> t)
p2 = plot(; xlabel="loss level t", ylabel="P(realized loss > t)", yscale=:log10, title="right tail",
          legend=:bottomleft, ylims=(1e-3, 1.2))
plot!(p2, tgrid, [max(exc(n_, t), 1e-4) for t in tgrid]; color=:gray40, linewidth=2, label="Nominal / Belief-known")
plot!(p2, tgrid, [max(exc(r_, t), 1e-4) for t in tgrid]; color=:darkorange, linewidth=3, label="Product / Coupled")
plt = plot(p1, p2; layout=(1, 2), size=(1100, 470))
@printf("ZS realized loss: nominal mean %.3f, q95 %.3f, q99 %.3f, max %.3f | robust mean %.3f, q95 %.3f, q99 %.3f, max %.3f (N=%d)\n",
        mean(n_), quantile(n_, 0.95), quantile(n_, 0.99), maximum(n_), mean(r_), quantile(r_, 0.95), quantile(r_, 0.99), maximum(r_), length(n_))
savefig(plt, joinpath(zsdir, "fig_zs_dist.png")); println("saved fig_zs_dist.png")
