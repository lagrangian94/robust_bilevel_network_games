# oracle (α-B&B) 한 번을 B&P 와 같은 인자로 따로 다시 부름 (2026-11-24, S=20 병목 분석)
#  환경 변수: GEO 1, S 10, XSTR (예 "{9:200;10:300;15:200}"), TAU (목표, 없으면 끔), GAP 1e-3, TLIM 3600, LAMU 1e4, LOG 60, KW (α-B&B 추가 인자), TAG (전선 파일 꼬리표)
#  ccgbp.jl 의 oracle_at! 과 같은 호출: nworkers = 스레드 − 1, rel_gap = GAP, target = TAU. verbose 로 하한·상한 경과를 찍음
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl"))
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
@printf("--- oracle 재현 g=%d S=%d x=%s λᵁ=%.0f 목표 %s gap %.1e 상한 %.0fs, 스레드 %d, 추가 인자 %s\n", g, S, xs, lam, string(tau), gap, tl, Threads.nthreads(), string(kw)); flush(stdout)
t = @elapsed res = nz_alpha_bnb_dr(nd, x̄; nworkers=max(1, Threads.nthreads() - 1), time_limit=tl, rel_gap=gap, target=tau, verbose=true, log_every=le, kw...)
@printf("RESULT g=%d S=%d x=%s V in [%.4f, %.4f] (%.0fs) 노드 %d, 이완 %.0fs, 평가 %.0fs, 범위 %.0fs (MIP %d, 실패 %d), 재확인 %.0fs, ipopt %d, 고정 비율 %.2f, 수치 오류 %d, 루트 UB %.4f\n",
        g, S, xs, res[:LB], res[:UB], t, res[:nodes], res[:t_relax], res[:t_eval], res[:t_range], res[:range_mips], res[:range_fail], res[:t_recheck],
        res[:ipopt_calls], res[:fixfrac], res[:numerr], res[:root_UB])
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
