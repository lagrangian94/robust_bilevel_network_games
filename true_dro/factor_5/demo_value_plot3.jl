#=
demo_value_plot3.jl — 발표용 예시 그림, 원고 §6.2 OOS 그림 (fig:oos-boxplot) 과 같은 구성 (2026-10-21 사용자 요청).
  Nominal → Partial (리더 쪽 모호성만) → 우리 모형 (Full DR) 의 실현 CVaR 비용 상자 그림. 최소화 문제라 낮을수록 좋음.
  각 패널: Dirichlet 집중도 γ 하나, x 축 묶음 = Phase A (ζ = 0) 와 Phase B (ζ ∈ {0.5, 1, 2, 5}).
  상자 25~75%, 수염 5~95%, 굵은 선 중앙값, ◆ 평균.
데이터 (추가 계산 없음):
  zero-sum  : true_dro/factor_5/plots/eps_0p1_beta_0p4/polska_factor_phaseAB.jls (원고 OOS, plot_phaseAB_3way.jl)
  비제로섬 : nonzero_sum/logs/demo_value/<DP_NZ_TAG>/eval_phaseab.csv (demo_value_nz.jl DN_STAGE=phaseab)
결과: 각 데이터 폴더의 fig_*_oos_g<γ>.png, 그리고 γ·ζ 별 q95·평균 표 (stdout)
환경변수: DP_NZ_TAG = sgb128_S10_eps0p20_q100_s3   DP_NZ_EPS = 0.2   DP_NZ_OURS = coupled_d0p1   DP_NZ_MID = partial | product   DP_GAMMAS = 0.1,1.0
실행: julia demo_value_plot3.jl
=#
using Plots, Printf, Statistics, Serialization
gr()
root = dirname(dirname(@__DIR__))
nztag = get(ENV, "DP_NZ_TAG", "sgb128_S10_eps0p20_q100_s3")
nzdir = joinpath(root, "nonzero_sum", "logs", "demo_value", nztag)
zsjls = joinpath(root, "true_dro", "factor_5", "plots", "eps_0p1_beta_0p4", "polska_factor_phaseAB.jls")
zsdir = joinpath(root, "true_dro", "factor_5", "logs", "demo_value", "polska")
gammas = parse.(Float64, split(get(ENV, "DP_GAMMAS", "0.1,1.0"), ","))
zetas = [0.0, 0.5, 1.0, 2.0, 5.0]
colors = [:gray45, :steelblue, :darkorange]

"상자 하나: 25~75% 상자, 5~95% 수염, 중앙값 선, 평균 ◆"
function box!(plt, pos, v; color, w=0.11)
    q(p) = quantile(v, p)
    plot!(plt, Shape([pos - w, pos + w, pos + w, pos - w], [q(0.25), q(0.25), q(0.75), q(0.75)]);
          fillcolor=color, fillalpha=0.45, linecolor=color, label="")
    plot!(plt, [pos - w, pos + w], [q(0.5), q(0.5)]; color=color, linewidth=3, label="")
    plot!(plt, [pos, pos], [q(0.05), q(0.25)]; color=color, label=""); plot!(plt, [pos, pos], [q(0.75), q(0.95)]; color=color, label="")
    plot!(plt, [pos - w / 2, pos + w / 2], [q(0.95), q(0.95)]; color=color, linewidth=2, label="")
    plot!(plt, [pos - w / 2, pos + w / 2], [q(0.05), q(0.05)]; color=color, label="")
    scatter!(plt, [pos], [mean(v)]; color=:white, markerstrokecolor=color, markershape=:diamond, markersize=5, label="")
end

"data[(γ, ζ)] = [3 개 해의 비용 벡터] 로 패널 하나"
function panel(data, γ, labels; title, ylabel)
    plt = plot(; title=title, ylabel=ylabel, legend=:outertop, legendcolumns=3, size=(900, 480), left_margin=8Plots.mm,
               xticks=(1:length(zetas), ["Phase A\n(ζ = 0)"; ["ζ = $(z)" for z in zetas[2:end]]]), xlims=(0.4, length(zetas) + 0.6),
               bottom_margin=6Plots.mm)
    for (j, ζ) in enumerate(zetas), (i, v) in enumerate(data[(γ, ζ)])
        box!(plt, j + (i - 2) * 0.27, v; color=colors[i])
    end
    for i in 1:3
        plot!(plt, [NaN], [NaN]; seriestype=:shape, fillcolor=colors[i], fillalpha=0.45, linecolor=colors[i], label=labels[i])
    end
    return plt
end

function table(data, name, labels)
    println("\n[$name] q95 / 평균 (" * join(labels, " | ") * ")")
    for γ in sort(unique(first.(keys(data)))), ζ in zetas
        v = data[(γ, ζ)]
        @printf("  γ=%.1f ζ=%.1f  q95 %s   평균 %s\n", γ, ζ, join([@sprintf("%7.2f", quantile(x, 0.95)) for x in v], " "),
                join([@sprintf("%7.2f", mean(x)) for x in v], " "))
    end
end

# ---------------- zero-sum (Polska) ----------------
d = deserialize(zsjls); r = d[:results]
zs = Dict{Tuple{Float64,Float64},Vector{Vector{Float64}}}()
for γ in keys(r), (ph, v) in r[γ]
    ζ = startswith(ph, "β") ? 0.0 : parse(Float64, split(ph, "=")[2])
    zs[(γ, ζ)] = [v["nominal"], v["single_l"], v["double"]]
end
# 우리 모형 = Full DR 해 x = {13, 34}: Coupled δ = 0.1 도 같은 결정 (demo_value_zs.jl, x_coupled_d0p1.txt)
zs_labels = ["Nominal", "Partial DR (ε̃ = 0)", "Ours (ε = 0.1, δ = 0.1)"]
table(zs, "zero-sum Polska, x: nominal $(d[:x_arcs][1]), partial $(d[:x_arcs][2]), ours $(d[:x_arcs][3])", zs_labels)
for γ in filter(g -> haskey(zs, (g, 0.0)), gammas)     # 데이터에 있는 γ 만
    plt = panel(zs, γ, zs_labels; title=@sprintf("Zero-sum (Polska, ε = 0.1, β = 0.4), Dirichlet γ = %.1f", γ),
                ylabel="out-of-sample CVaR of max flow (lower is better)")
    f = joinpath(zsdir, @sprintf("fig_zs_oos_g%s.png", replace(string(γ), "." => "p"))); savefig(plt, f); println("saved $f")
end

# ---------------- 비제로섬 ----------------
ls = readlines(joinpath(nzdir, "eval_phaseab.csv")); hdr = split(ls[1], ",")
col(n) = findfirst(==(n), hdr)
nzeps = get(ENV, "DP_NZ_EPS", "0.2"); nzours = get(ENV, "DP_NZ_OURS", "coupled_d0p1")
nzmid = get(ENV, "DP_NZ_MID", "partial")      # 가운데 상자: partial (ε̃ = 0) 또는 product (결합 없음, δ = ∞)
nzmodels = ["nominal", nzmid, nzours]
midlabel = nzmid == "partial" ? "Partial DR (ε̃ = 0)" : nzmid == "product" ? "Product (ε = $nzeps, δ = ∞)" : error("DP_NZ_MID = partial | product")
nz_labels = ["Nominal", midlabel, "Ours (ε = $nzeps, δ = " * replace(replace(nzours, "coupled_d" => ""), "p" => ".") * ")"]
nz = Dict{Tuple{Float64,Float64},Vector{Vector{Float64}}}()
xs_nz = Dict{String,String}()
for l in ls[2:end]
    isempty(l) && continue
    f = split(l, ","); m = f[col("model")]; m in nzmodels || continue
    key = (parse(Float64, f[col("gamma")]), parse(Float64, f[col("zeta")]))
    v = get!(() -> [Float64[] for _ in nzmodels], nz, key)
    push!(v[findfirst(==(m), nzmodels)], parse(Float64, f[col("cost")])); xs_nz[m] = f[col("x")]
end
table(nz, "비제로섬 $nztag, x: " * join(["$m $(xs_nz[m])" for m in nzmodels], ", "), nz_labels)
for γ in filter(g -> haskey(nz, (g, 0.0)), gammas)
    plt = panel(nz, γ, nz_labels; title=@sprintf("Non-zero-sum (location, ε = %s, β = 0.4), Dirichlet γ = %.1f", nzeps, γ),
                ylabel="out-of-sample cost of B (lower is better)")
    f = joinpath(nzdir, @sprintf("fig_nz_oos_%s_g%s.png", nzmid, replace(string(γ), "." => "p"))); savefig(plt, f); println("saved $f")
end
