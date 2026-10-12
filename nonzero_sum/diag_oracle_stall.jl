"""
diag_oracle_stall.jl — 전역 α-B&B (nz_alpha_bnb_dr) 가 한 x̄ 의 상한 증명에서 정체할 때 어디가 병목인지.

같은 oracle 문제를 TL 초 돌린 뒤 남은 B&B 전선 (열린 노드) 을 분석한다.
  1. 처리량: 노드 수, 노드당 시간, 시간 분해 (완화 LP / 범위 MIP / α 고정 LP)
  2. 남은 gap 이 어디에 있는가: 열린 노드의 상한 분포, incumbent α 와의 거리, 상자 폭
  3. 남은 노드에서 쌍대가 고정되지 않은 성분 (꺾임을 품은 시나리오) 이 얼마나 되는가
환경변수: OS_S = 20  NZ_SEED = 3  OS_X = "1,2"  OS_TL = 600  OS_GAP = 1e-3  OS_EPS = 0.3  OS_DELTA = 0.1  OS_WORKERS = 12
          OS_NEED = (없으면 OS_GAP)  — Benders 가 요구하는 oracle 상대 gap (예: seed 3, S=20 은 약 1.5e-3)
실행: julia -t 14,1 nonzero_sum/diag_oracle_stall.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Statistics
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
S = parse(Int, envf("OS_S", "20")); seed = parse(Int, envf("NZ_SEED", "3"))
ε = parse(Float64, envf("OS_EPS", "0.3")); δ = pd(envf("OS_DELTA", "0.1"))
TL = parse(Float64, envf("OS_TL", "600")); GAP = parse(Float64, envf("OS_GAP", "1e-3")); nw = parse(Int, envf("OS_WORKERS", "12"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=seed,
                                           eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(nd.nx); x̄[parse.(Int, split(envf("OS_X", "1,2"), ","))] .= 1
@printf("oracle 정체 진단: seed=%d S=%d x=%s TL=%.0fs 목표 gap=%.1e workers=%d\n", seed, S, string(findall(x̄ .> 0.5)), TL, GAP, nw); flush(stdout)

t = @elapsed r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=true, log_every=60.0, ipopt_time=120.0)
LB, UB = r[:LB], r[:UB]; sc = max(1.0, abs(LB))
@printf("\n결과: LB=%.4f UB=%.4f gap=%.3e (%.0fs) | 루트 상한 %.2f (루트 gap %.1f%%)\n", LB, UB, (UB - LB) / sc, t, r[:root_UB], 100 * (r[:root_UB] - LB) / sc)

# ---- 1. 처리량 ----
n = r[:nodes]
@printf("\n[1] 처리량: 노드 %d 개, %.2f 노드/s (전체), worker 시간 합 = 완화 LP %.0fs + 범위 MIP %.0fs + α 고정 LP %.0fs (wall × worker = %.0fs)\n",
        n, n / t, r[:t_relax], r[:t_range], r[:t_eval], t * nw)
@printf("    노드당: 완화 LP %.2fs, 범위 MIP %.2fs (%d 개/노드), 고정 비율 평균 %.2f | 실행불가능 노드 %d (되살아남 %d), LP 수치오류 %d\n",
        r[:t_relax] / max(n, 1), r[:t_range] / max(n, 1), round(Int, r[:range_mips] / max(n, 1)), r[:fixfrac], r[:infeas], r[:infeas_recovered], r[:numerr])
@printf("    worker 유휴 비율 ≈ %.0f%% (열린 노드가 worker 수보다 적으면 큼)\n", 100 * (1 - (r[:t_relax] + r[:t_range] + r[:t_eval]) / (t * nw)))

# ---- 2. 남은 전선 ----
open = r[:state].open; αs = r[:α]
hU = nz_presolve_hU(nd, x̄)[1]
act = findall(hU .> 0)
println("\n[2] 남은 전선: 열린 노드 ", length(open), " 개, 활성 좌표 ", length(act), " 개 (", act, ")")
println("    incumbent α* (활성 좌표) = ", round.(αs[act]; digits=1))
if !isempty(open)
    gaps = [(nd_[3] - LB) / sc for nd_ in open]
    need = parse(Float64, envf("OS_NEED", string(GAP)))
    @printf("    노드 상한의 상대 gap: 최대 %.3e, 중앙값 %.3e, 최소 %.3e | 목표 %.1e 초과 %d 개, Benders 요구 %.1e 초과 %d 개\n",
            maximum(gaps), median(gaps), minimum(gaps), GAP, count(>(GAP), gaps), need, count(>(need), gaps))
    inbox(nd_) = all(nd_[1][i] - 1e-9 <= αs[i] <= nd_[2][i] + 1e-9 for i in act)
    dist(nd_) = maximum(max(nd_[1][i] - αs[i], αs[i] - nd_[2][i], 0.0) for i in act)
    widths(nd_) = [nd_[2][i] - nd_[1][i] for i in act]
    nin = count(inbox, open)
    @printf("    α* 를 품은 노드 %d 개, α* 에서 거리 (∞-노름) 중앙값 %.2f, 최대 %.1f\n", nin, median(dist.(open)), maximum(dist.(open)))
    @printf("    상자 폭 (활성 좌표): 최대 폭의 중앙값 %.2f, 최소 폭의 중앙값 %.3f\n", median(maximum.(widths.(open))), median(minimum.(widths.(open))))
    ord = sortperm(gaps; rev=true)
    println("    상한이 가장 큰 노드 10 개:  gap | α* 포함 | α* 거리 | 좌표별 폭 | 고정 안 된 쌍대 성분 (전체 ", length(open[1][4]), ")")
    for k in ord[1:min(10, end)]
        nd_ = open[k]; w = widths(nd_)
        nun = count(nd_[5] .- nd_[4] .> 1e-4)
        @printf("      %.3e | %-5s | %6.2f | %s | %d\n", gaps[k], string(inbox(nd_)), dist(nd_), string(round.(w; digits=1)), nun)
    end
    # ---- 3. 남은 노드의 gap 은 완화가 느슨해서인가, 그 상자에 정말 최적에 가까운 점이 있어서인가 ----
    # 상한이 큰 노드 몇 개에서 Ω 를 그 상자로 제한해 Gurobi 전역으로 짧게 풀어 상자 안의 실제 최선값 (incumbent) 을 본다.
    #   (상자 최선값 − LB) 가 0 에 가까우면: 최적에 가까운 점이 여러 곳에 있다 (평평한 지형) → 완화가 정확해도 넓은 영역을 잘게 봐야 함
    #   크게 음수면: 그 상자는 실제로는 나쁜데 완화 상한만 높다 (완화가 느슨) → 완화를 조이면 바로 가지치기됨
    K = parse(Int, envf("OS_BOXES", "8")); tb = parse(Float64, envf("OS_BOX_TL", "40"))
    if K > 0
        ndp = nz_with_hU(nd, hU)
        Ob = build_nz_omega(ndp; optimizer=GRB); set_optimizer_attribute(Ob.model, "Threads", nw)
        nz_set_objective!(Ob, ndp, x̄)
        println("\n[3] 상한이 큰 노드 ", min(K, length(ord)), " 개를 상자로 제한해 Gurobi 전역 ", tb, " s: 노드 상한 gap | 상자 안 최선값 − LB (상대) | Gurobi 상한 gap")
        for k in ord[1:min(K, end)]
            nd_ = open[k]
            for i in 1:nz_nh(ndp); set_lower_bound(Ob.v[:α][i], nd_[1][i]); set_upper_bound(Ob.v[:α][i], nd_[2][i]); end
            set_time_limit_sec(Ob.model, tb); set_optimizer_attribute(Ob.model, "MIPGap", 1e-4)
            optimize!(Ob.model)
            st = termination_status(Ob.model)
            inc = has_values(Ob.model) ? objective_value(Ob.model) : NaN
            bdg = st == MOI.INFEASIBLE ? -Inf : objective_bound(Ob.model)
            @printf("      %.3e | %s | %s  (%s, α* 거리 %.1f)\n", gaps[k], isnan(inc) ? "해 없음" : @sprintf("%+.3e", (inc - LB) / sc),
                    isfinite(bdg) ? @sprintf("%.3e", (bdg - LB) / sc) : "실행 불가능", string(st), dist(nd_))
            flush(stdout)
        end
    end
    # 어느 좌표가 아직 넓은가 (위 gap 초과 노드 평균)
    big = [open[k] for k in ord if gaps[k] > need]
    if !isempty(big)
        println("    요구 gap 초과 노드의 좌표별 평균 폭: ", round.(mean(reduce(hcat, widths.(big)); dims=2)[:]; digits=1))
        println("    요구 gap 초과 노드의 고정 안 된 쌍대 성분 수: 중앙값 ", median([count(b[5] .- b[4] .> 1e-4) for b in big]),
                ", 시나리오별 (고정 안 된 행 수의 평균): ", round.(mean(reduce(hcat, [vec(sum(b[5] .- b[4] .> 1e-4; dims=1)) for b in big]); dims=2)[:]; digits=1))
    end
end
