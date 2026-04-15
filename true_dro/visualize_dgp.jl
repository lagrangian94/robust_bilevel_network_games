"""
visualize_dgp.jl — DGP 파라미터 검증 시각화.

질문 1: β_H=50, β_L=0.3, κ=50이 세 scenario의 information structure를 의도대로 만드는가?
질문 2: 95% calibrated ε이 적절한가?

실행:
  include("true_dro/visualize_dgp.jl")
  visualize_dgp()
"""

using Distributions
using LinearAlgebra
using Statistics
using Printf
using Plots

function visualize_dgp(; S::Int=10, β::Float64=0.3, β_H::Float64=50.0, β_L::Float64=0.3,
                         κ::Float64=50.0, n_draws::Int=10000)

    q_hat = fill(1.0 / S, S)

    # ── 데이터 수집 ──
    scenarios = [:S, :L, :F]
    dist_data = Dict{Symbol, Dict{Symbol, Vector{Float64}}}()

    for sc in scenarios
        d_hat_true = zeros(n_draws)
        d_hat_tilde = zeros(n_draws)
        d_tilde_true = zeros(n_draws)

        for i in 1:n_draws
            if sc == :S
                q_true = rand(Dirichlet(S, β))
                q_tilde = rand(Dirichlet(S, β))
            elseif sc == :L
                q_true = rand(Dirichlet(S, β_H))
                q_tilde = rand(Dirichlet(S, β_L))
            elseif sc == :F
                q_true = rand(Dirichlet(S, β_L))
                q_tilde = rand(Dirichlet(κ * q_true))
            end

            d_hat_true[i] = norm(q_true - q_hat, 1) / 2   # TV distance
            d_hat_tilde[i] = norm(q_tilde - q_hat, 1) / 2
            d_tilde_true[i] = norm(q_tilde - q_true, 1) / 2
        end

        dist_data[sc] = Dict(
            :d_hat_true => d_hat_true,
            :d_hat_tilde => d_hat_tilde,
            :d_tilde_true => d_tilde_true,
        )
    end

    # ── ε calibration ──
    ε_β  = quantile(dist_data[:S][:d_hat_true], 0.95)   # 이미 TV scale
    ε_βL = quantile(dist_data[:L][:d_hat_tilde], 0.95)
    ε_βH = quantile(dist_data[:L][:d_hat_true], 0.95)

    # ── 콘솔 출력 ──
    println("=" ^ 70)
    println("DGP 검증 (S=$S, β=$β, β_H=$β_H, β_L=$β_L, κ=$κ)")
    println("=" ^ 70)
    println()
    println("TV distances (= L1/2):")
    @printf("  %-10s | %-18s | %-18s | %-18s\n",
            "Scenario", "TV(q̂, q_true)", "TV(q̂, q̃)", "TV(q̃, q_true)")
    println("  " * "-" ^ 62)
    for sc in scenarios
        dd = dist_data[sc]
        @printf("  %-10s | mean=%.3f p95=%.3f | mean=%.3f p95=%.3f | mean=%.3f p95=%.3f\n",
                sc,
                mean(dd[:d_hat_true]), quantile(dd[:d_hat_true], 0.95),
                mean(dd[:d_hat_tilde]), quantile(dd[:d_hat_tilde], 0.95),
                mean(dd[:d_tilde_true]), quantile(dd[:d_tilde_true], 0.95))
    end
    println()
    @printf("  calibrated ε(β=%.1f)   = %.4f\n", β, ε_β)
    @printf("  calibrated ε(β_H=%.0f) = %.4f\n", β_H, ε_βH)
    @printf("  calibrated ε(β_L=%.1f) = %.4f\n", β_L, ε_βL)

    # ================================================================
    # 그림 1: "누가 누구에게 가까운가?" — 세 scenario 비교
    # ================================================================
    #
    # 각 scenario에서 세 TV distance의 분포를 겹쳐 그림.
    # 한 scenario 한 패널. 총 3패널.
    #
    # 읽는 법:
    #   - 파란색(q̂↔q*) 이 왼쪽에 몰려있으면 → leader가 truth에 가깝다
    #   - 빨간색(q̂↔q̃) 이 왼쪽에 몰려있으면 → follower가 q̂에 가깝다
    #   - 초록색(q̃↔q*) 이 왼쪽에 몰려있으면 → follower가 truth에 가깝다
    #
    # 기대 패턴:
    #   S: 세 분포 비슷 (symmetric)
    #   L: 파란색만 왼쪽 (leader가 truth에 가까움)
    #   F: 초록색만 왼쪽 (follower가 truth에 가까움)

    scenario_titles = Dict(
        :S => "Scenario S: Symmetric\n(both uncertain)",
        :L => "Scenario L: Leader Advantage\n(leader close to truth)",
        :F => "Scenario F: Follower Advantage\n(follower close to truth)",
    )

    p1_panels = []
    for sc in scenarios
        dd = dist_data[sc]
        p = histogram(dd[:d_hat_true], bins=60, normalize=:pdf, alpha=0.55,
                      color=:royalblue, label="TV(qhat,q*): leader-truth",
                      xlabel="TV distance", ylabel="density",
                      title=scenario_titles[sc], titlefontsize=9,
                      legendfontsize=7, legend=:topright)
        histogram!(p, dd[:d_hat_tilde], bins=60, normalize=:pdf, alpha=0.55,
                   color=:crimson, label="TV(qhat,qt): leader-follower")
        histogram!(p, dd[:d_tilde_true], bins=60, normalize=:pdf, alpha=0.55,
                   color=:seagreen, label="TV(qt,q*): follower-truth")
        push!(p1_panels, p)
    end

    fig1 = plot(p1_panels..., layout=(1, 3), size=(1500, 450),
                plot_title="Q1: Who is close to whom in each scenario?",
                plot_titlefontsize=12, margin=5Plots.mm)

    savefig(fig1, "true_dro/dgp_fig1_who_is_close.png")
    println("\n  Saved: true_dro/dgp_fig1_who_is_close.png")

    # ================================================================
    # 그림 2: "calibrated ε이 적절한가?" — ε과 실제 거리 비교
    # ================================================================
    #
    # Leader가 calibrate하는 ε은 "d(q̂, q) ≤ ε 이 95% 확률로 성립"하도록 설정.
    # 이 ε이 실제 TV distance 분포를 잘 커버하는지 확인.
    #
    # 각 scenario에서 leader가 사용하는 ε₁(=ε̂), ε₂(=ε̃) 를
    # 실제 d(q̂, q_true), d(q̂, q̃) 히스토그램 위에 수직선으로 표시.
    #
    # 읽는 법:
    #   - 빨간 수직선(ε) 오른쪽에 5% 정도만 있으면 → 적절
    #   - 수직선이 너무 오른쪽이면 → ε이 과대 (overconservative)
    #   - 수직선이 너무 왼쪽이면 → ε이 과소 (coverage 부족)

    # Scenario별 (ε₁ 적용 대상, ε₂ 적용 대상)
    ε_configs = Dict(
        :S => [(ε_β,  :d_hat_true,  "e1 for d(qhat,q*)", "d(qhat, q*) — leader model error"),
               (ε_β,  :d_hat_tilde, "e2 for d(qhat,qt)", "d(qhat, qt) — follower deviation")],
        :L => [(ε_βH, :d_hat_true,  "e1(bH) for d(qhat,q*)", "d(qhat, q*) — leader model error"),
               (ε_βL, :d_hat_tilde, "e2(bL) for d(qhat,qt)", "d(qhat, qt) — follower deviation")],
        :F => [(ε_βL, :d_hat_true,  "e1(bL) for d(qhat,q*)", "d(qhat, q*) — leader model error"),
               (ε_βL, :d_hat_tilde, "e2(bL) for d(qhat,qt)", "d(qhat, qt) — follower deviation")],
    )

    p2_panels = []
    for sc in scenarios
        for (ε_val, dist_key, ε_label, dist_label) in ε_configs[sc]
            vals = dist_data[sc][dist_key]
            actual_cov = mean(vals .<= ε_val) * 100

            p = histogram(vals, bins=60, normalize=:pdf, alpha=0.7,
                          color=:slategray, label=nothing,
                          xlabel="TV distance", ylabel="density",
                          title="$(sc): $(dist_label)",
                          titlefontsize=8, legendfontsize=7)
            vline!(p, [ε_val], color=:red, linewidth=2.5, linestyle=:dash,
                   label="$(ε_label) = $(round(ε_val, digits=3)) (cov=$(round(actual_cov, digits=1))%)")
            push!(p2_panels, p)
        end
    end

    fig2 = plot(p2_panels..., layout=(3, 2), size=(1100, 900),
                plot_title="Q2: Does calibrated epsilon cover the actual distances?",
                plot_titlefontsize=12, margin=5Plots.mm)

    savefig(fig2, "true_dro/dgp_fig2_epsilon_coverage.png")
    println("  Saved: true_dro/dgp_fig2_epsilon_coverage.png")

    # ================================================================
    # 그림 3: 요약 — 세 scenario의 "핵심 거리" 비교
    # ================================================================
    #
    # 각 scenario에서 가장 중요한 거리 하나씩:
    #   S: d(q̃, q*) — follower↔truth (기본 misalignment)
    #   L: d(q̃, q*) — follower↔truth (follower가 멀어야 함)
    #   F: d(q̃, q*) — follower↔truth (follower가 가까워야 함)
    #
    # → 세 scenario의 d(q̃,q*) 분포를 하나의 패널에 겹침.
    # L에서 크고 F에서 작으면 파라미터가 적절.

    p3 = histogram(dist_data[:S][:d_tilde_true], bins=60, normalize=:pdf, alpha=0.5,
                   color=:gray, label="S: TV(qt,q*) — symmetric baseline",
                   xlabel="TV distance (follower - truth)", ylabel="density",
                   title="Follower calibration quality vs truth",
                   titlefontsize=11, legendfontsize=8, legend=:topright,
                   size=(800, 400))
    histogram!(p3, dist_data[:L][:d_tilde_true], bins=60, normalize=:pdf, alpha=0.5,
               color=:crimson, label="L: TV(qt,q*) — follower inaccurate (should be large)")
    histogram!(p3, dist_data[:F][:d_tilde_true], bins=60, normalize=:pdf, alpha=0.5,
               color=:seagreen, label="F: TV(qt,q*) — follower accurate (should be small)")

    # Mean annotations
    for (sc, col) in [(:S, :gray), (:L, :crimson), (:F, :seagreen)]
        m = mean(dist_data[sc][:d_tilde_true])
        vline!(p3, [m], color=col, linewidth=2, linestyle=:dot, label=nothing)
    end

    savefig(p3, "true_dro/dgp_fig3_follower_calibration.png")
    println("  Saved: true_dro/dgp_fig3_follower_calibration.png")

    println("\n  Done. Check 3 figures. Adjust beta_H, beta_L, kappa if patterns don't match.")
    return dist_data
end

# ===== Entry =====
if abspath(PROGRAM_FILE) == @__FILE__
    visualize_dgp()
end
