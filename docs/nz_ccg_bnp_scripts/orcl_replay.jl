# oracle (α-B&B) 한 번을 B&P 와 같은 인자로 따로 다시 부름 (2026-11-24, S=20 병목 분석)
#  환경 변수: ABB_PATCH (α-B&B 재정의 파일), HUCAP 0 (1 이면 연 B 거점 약정 상한 = 용량), GEO 1, S 10, XSTR (예 "{9:200;10:300;15:200}"), TAU (목표, 없으면 끔), GAP 1e-3, TLIM 3600, LAMU 1e4, LOG 60, KW (α-B&B 추가 인자), TAG (전선 파일 꼬리표)
#  ccgbp.jl 의 oracle_at! 과 같은 호출: nworkers = 스레드 − 1, rel_gap = GAP, target = TAU. verbose 로 하한·상한 경과를 찍음
#  2026-11-28 밤 (사용자 "수제 알파비앤비 말고 구로비 솔바로 푼 거랑도 비교"): SOLVER = abb (기본, 위 호출) | grb.
#    grb = 같은 x̄·정의역 (HUCAP + nz_presolve_hU 의 닫힌 거점 고정, α-B&B 가 안에서 하는 것과 같음) 의 Ω 를 Gurobi NonConvex 로 푼다.
#    BD-G 의 전역 단계 (nz_bd_kkt.jl 의 _nz_omega_global!) 그대로: MIPGap = GAP, BestObjStop = BestBdStop = TAU, Threads = 스레드 수. Gurobi 로그는 GRB_DISP 초마다
#    (기본 60). 전선 저장은 없음
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl"))
haskey(ENV, "ABB_PATCH") && include(ENV["ABB_PATCH"])   # 실험용 α-B&B 재정의 (예 abb_dd_patch.jl, 2026-11-25)
envf(k, d) = get(ENV, k, d)
g = parse(Int, envf("GEO", "1")); S = parse(Int, envf("S", "10")); lam = parse(Float64, envf("LAMU", "1e4"))
xs = envf("XSTR", "{9:200;10:300;15:200}"); tl = parse(Float64, envf("TLIM", "3600")); gap = parse(Float64, envf("GAP", "1e-3"))
tau = haskey(ENV, "TAU") ? parse(Float64, ENV["TAU"]) : nothing; le = parse(Float64, envf("LOG", "60"))
kw = eval(Meta.parse(envf("KW", "NamedTuple()")))   # α-B&B 추가 인자 (예 "(lp_method=2,)"), 기본 = 기본값 그대로
tag = envf("TAG", "")
nd = make_loc_tkp(; g=g, S=S, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=lam)
K = nd.meta[:K]; tiers = nd.meta[:tiers]; nA = nd.meta[:nA]
# "{i:t;...}" → x (loc_tkp_xstr 의 역)
x̄ = zeros(nd.nx)
for p in split(strip(xs, ['{', '}']), ";"; keepempty=false)
    i, t = parse.(Int, split(p, ":")); k = findfirst(==(Float64(t)), tiers)
    k === nothing && error("등급 $t 없음"); x̄[(i - nA - 1) * K + k] = 1.0
end
loc_tkp_xstr(nd, x̄) == xs || error("x 복원 실패: $(loc_tkp_xstr(nd, x̄)) ≠ $xs")
# HUCAP=1: 연 B 거점의 약정 상한을 그 거점 용량으로 (h_i ≤ b_i(x̄)). 약정 등식 Σ_j yR_ij + s_i = h_i 와 용량 행 Σ_j (yR_ij + yS_ij) + s_i ≤ b_i 에서
#   h_i > b_i 면 recourse 가 실행 불가능이라 그런 α 는 어떤 belief 의 최적 반응도 아님 → 정의역을 줄여도 V*(x̄) 그대로 (nz_presolve_hU 의 0 고정과 같은 논증)
if get(ENV, "HUCAP", "0") == "1"
    hU = copy(nd.hU); hidx = nd.meta[:hidx]; sx = nd.meta[:site_x]
    for i in nA+1:nd.meta[:nst]
        b = sum(tiers[(t - 1) % K + 1] * x̄[t] for t in sx[i]; init=0.0)
        k = hidx(i, 1); b > 0.5 && b < hU[k] && (@printf("  [HUCAP] 거점 %d: hᵁ %.0f → %.0f\n", i, hU[k], b); hU[k] = b)
    end
    global nd = nz_with_hU(nd, hU)
end
solver = envf("SOLVER", "abb"); solver in ("abb", "grb") || error("SOLVER = abb | grb")
@printf("--- oracle 재현 g=%d S=%d x=%s λᵁ=%.0f 목표 %s gap %.1e 상한 %.0fs, 스레드 %d, %s\n", g, S, xs, lam, string(tau), gap, tl, Threads.nthreads(),
        solver == "grb" ? "풀이기 Gurobi Ω (NonConvex)" : "추가 인자 " * string(kw)); flush(stdout)
if solver == "grb"
    tau === nothing && error("orcl_replay: SOLVER = grb 는 TAU 가 필요 (BestObjStop·BestBdStop)")
    include(joinpath(root, "nonzero_sum", "nz_bd_kkt.jl"))          # _nz_omega_global!
    hUp, npre = nz_presolve_hU(nd, x̄); ndg = nz_with_hU(nd, hUp)
    @printf("  [Gurobi] 정의역: 닫힌 거점 고정 %d 개, 0 아닌 hᵁ %d / %d\n", npre, count(hUp .> 0), length(hUp)); flush(stdout)
    genv = Gurobi.Env(); nthr = Threads.nthreads(); disp = parse(Int, envf("GRB_DISP", "60"))
    grbo() = (o = Gurobi.Optimizer(genv); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), nthr); MOI.set(o, MOI.RawOptimizerAttribute("DisplayInterval"), disp); o)
    tb = @elapsed (global O = build_nz_omega(ndg; optimizer=grbo, silent=false))
    t = @elapsed (global kg = _nz_omega_global!(O, ndg, x̄; time_limit=tl, gap=gap, target=tau))
    out = kg[:bound] <= tau ? "인증" : kg[:value] > tau ? "막음" : "미결"
    @printf("RESULT g=%d S=%d x=%s V in [%.4f, %.4f] (%.0fs, 구성 %.0fs) Gurobi %s, %s, 노드 %.0f, 단체법 반복 %.0f\n", g, S, xs, kg[:value], kg[:bound], t, tb,
            string(kg[:status]), out, MOI.get(O.model, MOI.NodeCount()), MOI.get(O.model, MOI.SimplexIterations()))
    exit(0)
end
tsplit = parse(Float64, envf("TSPLIT", "0"))   # 2026-11-26: > 0 이면 TSPLIT 초에서 끊고 이어 풀기 (resume) 로 나머지를 — 패치의 이어 풀기 점검용
if tsplit > 0
    t1 = @elapsed (r1 = nz_alpha_bnb_dr(nd, x̄; nworkers=max(1, Threads.nthreads() - 1), time_limit=tsplit, rel_gap=gap, target=tau, verbose=true, log_every=le, kw...))
    @printf("  [이어 풀기 점검] 1 차 %.0fs: V in [%.4f, %.4f], 노드 %d, 열린 %d%s\n", t1, r1[:LB], r1[:UB], r1[:nodes], length(r1[:state].open),
            haskey(r1, :dd_screen) ? ", DD $(r1[:dd_screen])" : ""); flush(stdout)
    t = t1 + @elapsed (global res = nz_alpha_bnb_dr(nd, x̄; nworkers=max(1, Threads.nthreads() - 1), time_limit=tl - tsplit, rel_gap=gap, target=tau, verbose=true, log_every=le, resume=r1[:state], kw...))
    res[:resumed] || error("orcl_replay: 이어 풀기가 안 됨")
else
    t = @elapsed (global res = nz_alpha_bnb_dr(nd, x̄; nworkers=max(1, Threads.nthreads() - 1), time_limit=tl, rel_gap=gap, target=tau, verbose=true, log_every=le, kw...))
end
@printf("RESULT g=%d S=%d x=%s V in [%.4f, %.4f] (%.0fs) 노드 %d, 이완 %.0fs, 평가 %.0fs, 범위 %.0fs (MIP %d, 실패 %d), 재확인 %.0fs, ipopt %d, 고정 비율 %.2f, 수치 오류 %d, 루트 UB %.4f\n",
        g, S, xs, res[:LB], res[:UB], t, res[:nodes], res[:t_relax], res[:t_eval], res[:t_range], res[:range_mips], res[:range_fail], res[:t_recheck],
        res[:ipopt_calls], res[:fixfrac], res[:numerr], res[:root_UB])
haskey(res, :dd_screen) && println("DD 선별 (패치): ", res[:dd_screen])
println("느린 노드 (상위 10): ", first(res[:slow_nodes], 10))
# 남은 전선 저장 (nonzero_sum/lab_frontier.jl 과 같은 형식 + loc_tkp 인자): 완화 오차 진단용
boxes = [(l=n[1], u=n[2], ub=n[3], lo=n[4], hi=n[5], phib=n[7], eq=collect(n[8].eq)) for n in res[:state].open]
fo = @sprintf("orep_front_g%d_S%d%s.jls", g, S, tag)
serialize(fo, (g=g, S=S, xs=xs, x=x̄, lam=lam, LB=res[:LB], UB=res[:UB], α=res[:α], root_UB=res[:root_UB], boxes=boxes))
@printf("전선 저장: %s (열린 노드 %d)\n", fo, length(boxes))
if tau !== nothing && !isempty(boxes)
    ubs = [b.ub for b in boxes]
    @printf("전선: 열린 노드 %d, 상한 ≤ 목표 %d (%.0f%%), 상한 범위 [%.2f, %.2f], 목표 %.2f, incumbent %.2f\n", length(ubs), count(<=(tau), ubs),
            100 * count(<=(tau), ubs) / length(ubs), minimum(ubs), maximum(ubs), tau, res[:LB])
end
