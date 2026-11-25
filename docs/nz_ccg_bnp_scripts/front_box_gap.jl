# 저장된 oracle 전선 (orcl_replay.jl 의 orep_front_*.jls) 에서 상한이 큰 상자마다 Ω 를 그 상자로 제한해 Gurobi 전역으로 짧게 풂 (2026-11-25 밤, α-B&B 튜닝 진단).
#   (상자 안 최선값 − incumbent) 가 0 근처면 그 상자에 정말 좋은 점이 있다 (평평한 지형, 완화가 정확해도 잘게 봐야 함).
#   크게 음수면 그 상자는 실제로 나쁜데 완화 상한만 높다 (완화가 느슨, 조이면 바로 가지치기). nonzero_sum/diag_oracle_stall.jl [3] 과 같은 진단의 loc_tkp 판.
#  환경 변수: FR (전선 파일), HUCAP 1, K 8 (상자 수, 상한 큰 순), BTL 60 (상자당 초), NT 2 (Gurobi 스레드), PICK top (top | spread: 상한 순으로 고르게)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization, Statistics
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("FR", ""))
K = parse(Int, envf("K", "8")); btl = parse(Float64, envf("BTL", "60")); nt = parse(Int, envf("NT", "2")); pick = envf("PICK", "top")
nd = make_loc_tkp(; g=fr.g, S=fr.S, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=fr.lam)
x̄ = fr.x; K_ = nd.meta[:K]; tiers = nd.meta[:tiers]; nA = nd.meta[:nA]
if envf("HUCAP", "1") == "1"                      # orcl_replay.jl 과 같은 정의역 (연 B 거점 약정 ≤ 용량)
    hU = copy(nd.hU); hidx = nd.meta[:hidx]; sx = nd.meta[:site_x]
    for i in nA+1:nd.meta[:nst]
        b = sum(tiers[(t - 1) % K_ + 1] * x̄[t] for t in sx[i]; init=0.0)
        k = hidx(i, 1); b > 0.5 && b < hU[k] && (hU[k] = b)
    end
    global nd = nz_with_hU(nd, hU)
end
hUp, _ = nz_presolve_hU(nd, x̄); nd = nz_with_hU(nd, hUp)       # α-B&B 안의 presolve 와 같은 정의역
nh = nz_nh(nd); LB = fr.LB; sc = max(1.0, abs(LB))
B = fr.boxes; ord = sortperm([b.ub for b in B]; rev=true)
sel = pick == "top" ? ord[1:min(K, end)] : ord[unique(round.(Int, range(1, length(ord); length=min(K, length(ord)))))]
@printf("--- 전선 %s: g=%d S=%d x=%s, 열린 상자 %d, incumbent %.2f, 상한 %.2f, 루트 상한 %.2f | 상자 %d 개 (%s), 상자당 %.0f s, 스레드 %d\n",
        envf("FR", ""), fr.g, fr.S, fr.xs, length(B), LB, fr.UB, fr.root_UB, length(sel), pick, btl, nt)
act = [i for i in 1:nh if nd.hU[i] > 1e-9]
@printf("    활성 좌표 %s (hᵁ %s), incumbent α %s\n", string(act), string(round.(nd.hU[act]; digits=1)), string(round.(fr.α[act]; digits=2)))
GRB_ENV = Gurobi.Env()
Ob = build_nz_omega(nd; optimizer=() -> Gurobi.Optimizer(GRB_ENV)); set_optimizer_attribute(Ob.model, "Threads", nt)
nz_set_objective!(Ob, nd, x̄)
println("    상자 | 상한 − inc | 상자 최선 − inc | Gurobi 상한 − inc | 상태 | 안 고정 쌍대 | 활성 좌표 [l, u]")
for (n, k) in enumerate(sel)
    b = B[k]
    for i in 1:nh; set_lower_bound(Ob.v[:α][i], b.l[i]); set_upper_bound(Ob.v[:α][i], b.u[i]); end
    set_time_limit_sec(Ob.model, btl); set_optimizer_attribute(Ob.model, "MIPGap", 1e-5)
    optimize!(Ob.model)
    st = termination_status(Ob.model)
    inc = has_values(Ob.model) ? objective_value(Ob.model) : NaN
    bd = st == MOI.INFEASIBLE ? -Inf : objective_bound(Ob.model)
    nun = count(b.hi .- b.lo .> 1e-4)
    @printf("    %3d | %8.2f | %s | %s | %s | %d | %s\n", n, b.ub - LB, isnan(inc) ? "해 없음" : @sprintf("%8.2f", inc - LB),
            isfinite(bd) ? @sprintf("%8.2f", bd - LB) : "실행 불가능", string(st), nun,
            join([@sprintf("%d:[%.1f,%.1f]", i, b.l[i], b.u[i]) for i in act], " "))
    flush(stdout)
end
