"""
diag_dual_range.jl — "상자별 쌍대 범위 조임" 이 노드 완화 상한을 얼마나 조이는지 (구현 전 잠재력 확인).

아이디어: Ω 의 recourse 쌍대해 ϖˢ/r_s (리더 블록) 와 π̃ˢ/d_s (follower 블록) 는 정확한 해에서 시나리오 s recourse LP 의 **α 에서의 최적 쌍대**다.
α 가 노드 상자 B 안에 있으므로, 예약 행 (H 행) k 의 성분은
    σ_ks ∈ [σlo_ks(B), σhi_ks(B)] := { σ_k : σ 는 어떤 α' ∈ B 에서 시나리오 s recourse LP 의 최적 쌍대, H 행 성분 ≤ ϖᵁ }
에 있다 (작은 KKT-MIP 2 개로 정확히 계산, 시나리오·행마다). 현재 완화는 전역 범위 [0, ϖᵁ] (= [0, p]) 만 쓴다.
  W : σlo r_s ≤ ϖ_ks ≤ σhi r_s  (+ 이 행과 α 상자 인수의 RLT)
  F : σlo d_s ≤ π̃_ks ≤ σhi d_s
  C : 상자가 함의하는 상보성 — l_i > 0 이면 follower 1단계 정상성 등식 (Fh_i 가 등식), Σ_i W_li u_i < w_l 이면 κ_l = 0
유효성: Wcap (ϖ ≤ ϖᵁ r) 과 같은 논증 — H 행 성분이 ϖᵁ 이하인 최적 쌍대가 존재하고, 그 쌍대는 α ∈ B 에서 최적이므로 범위 안.
        (follower 블록: 2단계 LP 의 최적 쌍대는 시나리오 블록에서 α 고정 recourse LP 의 최적 쌍대 × d_s.)

비교: 최적 α* 중심 폭 w 상자에서 cur / W / W+F / W+F+C 의 상한 − V*.
환경변수: DR_S = 3  DR_X = "1,2"  DR_EPS = 0.3  DR_DELTA = 0.1  DR_WIDTHS = "150,50,10,3,1"
          DR_ALPHA = kkt | abb  DR_ABB_TL = 60  DR_ABB_WORKERS = 4
실행: julia -t 6,1 nonzero_sum/diag_dual_range.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
const NTHR = parse(Int, get(ENV, "DR_THREADS", "2"))
GRB() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NTHR); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
const NONLOWER = envf("DR_NONLOWER", "1") == "1"
S = parse(Int, envf("DR_S", "3")); ε = parse(Float64, envf("DR_EPS", "0.3")); δ = pd(envf("DR_DELTA", "0.1"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                           eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[parse.(Int, split(envf("DR_X", "1,2"), ","))] .= 1
hUp, _ = nz_presolve_hU(ndf, x̄)
nd = nz_with_hU(ndf, hUp)
nh, m, ny = nz_nh(nd), nz_m(nd), nz_ny(nd)
Hr = findall(nd.hrow .> 0)

if envf("DR_ALPHA", "kkt") == "kkt"
    k = nz_kkt_value(ndf, x̄; optimizer=GRB, time_limit=600.0)
    V, αs = k[:value], min.(k[:α], nd.hU)
    @printf("%s x=%s  KKT %s V*=%.4f (bound %.4f)\n", nd.name, string(findall(x̄ .> 0.5)), string(k[:status]), V, k[:bound])
else
    r = nz_alpha_bnb(ndf, x̄; nworkers=parse(Int, envf("DR_ABB_WORKERS", "4")), time_limit=parse(Float64, envf("DR_ABB_TL", "60")),
                     rel_gap=1e-3, verbose=false)
    V, αs = r[:LB], min.(r[:α], nd.hU)
    @printf("%s x=%s  α-B&B incumbent LB=%.4f (UB %.4f) — '상한−V*' 는 이 LB 기준\n", nd.name, string(findall(x̄ .> 0.5)), V, r[:UB])
end
println("α* = ", round.(αs; digits=2)); flush(stdout)

# ---- 시나리오 s 의 recourse KKT-MIP (α ∈ 상자): 최적 쌍대의 H 행 성분 범위 ----
function build_range_mip(s)
    mdl = Model(GRB); set_silent(mdl); set_optimizer_attribute(mdl, "Threads", 1)
    @variable(mdl, 0 <= α[i=1:nh] <= nd.hU[i])
    @constraint(mdl, [l=1:size(nd.W, 1)], sum(nd.W[l, i] * α[i] for i in 1:nh) <= nd.wvec[l])
    @variable(mdl, y[1:ny] >= 0); @variable(mdl, σ[1:m] >= 0)
    for k in Hr; isfinite(nd.varpiU[k]) && set_upper_bound(σ[k], nd.varpiU[k]); end
    sP = [@expression(mdl, nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) +
                           (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0) -
                           sum(nd.A[k, j] * y[j] for j in 1:ny if nd.A[k, j] != 0)) for k in 1:m]
    sD = [@expression(mdl, sum(nd.A[k, j] * σ[k] for k in 1:m if nd.A[k, j] != 0) - nd.c[j]) for j in 1:ny]
    @constraint(mdl, [k=1:m], sP[k] >= 0); @constraint(mdl, [j=1:ny], sD[j] >= 0)
    for k in 1:m; _comp!(mdl, σ[k], sP[k]); end
    for j in 1:ny; _comp!(mdl, y[j], sD[j]); end
    # "더 낮출 수 없는" 쌍대만: H 행 k 의 σ_k > 0 이면 그 행에 걸린 열 (A_kj > 0) 중 하나의 쌍대 제약이 등호.
    #   우변이 양수 (α_i > 0) 인 행은 최적 쌍대라면 자동으로 성립 (낮출 수 있으면 쌍대목적이 줄어 모순).
    #   α_i = 0 인 행은 σ_k 를 낮춰도 쌍대 실행가능·최적이 유지되므로 일반성을 잃지 않는다 (follower 블록의 Fh 는 ≤ 방향).
    #   이 조건이 없으면 정의역 경계 (α_i = 0) 에서 쌍대 집합이 [우도함수, ϖᵁ] 로 부풀어 범위가 조여지지 않는다.
    if NONLOWER
        for k in Hr
            J = [j for j in 1:ny if nd.A[k, j] > 0]
            z = @variable(mdl, binary = true); @constraint(mdl, z => {σ[k] == 0})
            w = @variable(mdl, [1:length(J)], binary = true)
            for (n, j) in enumerate(J); @constraint(mdl, w[n] => {sD[j] == 0}); end
            @constraint(mdl, z + sum(w) >= 1)
        end
    end
    return mdl, α, σ, y
end
const RM = [build_range_mip(s) for s in 1:S]
"시나리오별 리더 평가의 상자 위 최댓값 φ̄_s(B) = max{ℓᵀy : α ∈ B, y 는 recourse 최적} (같은 KKT-MIP, 목적만 바꿈)"
function phi_bar(l, u)
    out = zeros(S)
    for s in 1:S
        mdl, α, σ, y = RM[s]
        for i in 1:nh; set_lower_bound(α[i], l[i]); set_upper_bound(α[i], u[i]); end
        @objective(mdl, Max, sum(nd.ell[j] * y[j] for j in 1:ny)); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("phi_bar MIP s=$s: $(termination_status(mdl))")
        out[s] = objective_bound(mdl) + 1e-6
    end
    return out
end
"상자 [l, u] 에서 σ_ks 범위 (H 행 중 상한이 양수인 좌표만). 반환: lo, hi (m × S, 나머지는 전역 범위), MIP 수, 시간"
function dual_ranges(l, u)
    lo = zeros(m, S); hi = [isfinite(nd.varpiU[k]) ? nd.varpiU[k] : Inf for k in 1:m, s in 1:S]
    nmip = 0
    t = @elapsed for s in 1:S
        mdl, α, σ, _ = RM[s]
        for i in 1:nh; set_lower_bound(α[i], l[i]); set_upper_bound(α[i], u[i]); end
        for k in Hr
            nd.hU[nd.hrow[k]] > 0 || continue
            for sense in (MIN_SENSE, MAX_SENSE)
                @objective(mdl, sense, σ[k]); optimize!(mdl); nmip += 1
                termination_status(mdl) == MOI.OPTIMAL || error("range MIP s=$s k=$k: $(termination_status(mdl))")
                # 허용오차만큼 바깥으로 (유효성)
                sense == MIN_SENSE ? (lo[k, s] = max(objective_bound(mdl) - 1e-7, 0.0)) : (hi[k, s] = objective_bound(mdl) + 1e-7)
            end
        end
    end
    return lo, hi, nmip, t
end

"""
시나리오별 리더 평가 φ_s 의 상자 위 affine 상계 (기울기 컷): φ_s(α) ≤ a_s + b_sᵀα  ∀α ∈ B  ⇒  ℓᵀŷˢ ≤ a_s r_s + b_sᵀ ζL_s.
b_s = 상자 중심에서의 국소 기울기 (φ = P − θQ 의 두 LP 쌍대), a_s = max_{α∈B} [φ_s(α) − b_sᵀα] (KKT-MIP, 어떤 b 든 유효).
꺾임이 없는 시나리오는 이 컷 하나로 상자 위에서 정확해진다.
"""
function phi_grad(l, u)
    θ = nd.thetaU
    αc = _nz_capH(0.5 .* (l .+ u), nd)
    avec = zeros(S); bmat = zeros(nh, S)
    for s in 1:S
        rr = nz_rhs(nd, x̄, αc, s)
        duals = Vector{Vector{Float64}}()
        for obj in (nd.ell .+ θ .* nd.c, nd.c)
            mdl = Model(GRB); set_silent(mdl)
            @variable(mdl, yv[1:ny] >= 0)
            con = @constraint(mdl, [k=1:m], sum(nd.A[k, j] * yv[j] for j in 1:ny if nd.A[k, j] != 0) <= rr[k])
            @objective(mdl, Max, dot(obj, yv)); optimize!(mdl)
            push!(duals, [shadow_price(con[k]) for k in 1:m])
        end
        for k in Hr; bmat[nd.hrow[k], s] += nd.hcoef[k] * (duals[1][k] - θ * duals[2][k]); end
        mdl, α, σ, y = RM[s]
        for i in 1:nh; set_lower_bound(α[i], l[i]); set_upper_bound(α[i], u[i]); end
        @objective(mdl, Max, sum(nd.ell[j] * y[j] for j in 1:ny) - sum(bmat[i, s] * α[i] for i in 1:nh)); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("phi_grad MIP s=$s: $(termination_status(mdl))")
        avec[s] = objective_bound(mdl) + 1e-6
    end
    return avec, bmat
end

function add_ranges!(P, lo, hi, l, u; W::Bool, F::Bool, C::Bool, V::Bool=false, G::Bool=false, X::Bool=false)
    v = P.O.v; mdl = P.O.model
    if X
        # 상보성의 RLT (follower 1단계): α_i · (Wᵀκ − c0 − Σ_s Bᵀπ̃ˢ)_i = 0,  κ_l · (w_l − W_l α) = 0 를 곱 변수로 선형화.
        #   Zκ_il = α_i κ_l (κ ≥ 0 × 상자 인수),  ζΠ_ks = α_i π̃_ks (범위 행 σlo d ≤ π̃ ≤ σhi d × 상자 인수: W 쪽과 같은 4 행)
        # 쌍대가 고정된 상자에서는 ζΠ = σ ζF 가 되어 follower 의 primal 값 (ζF 로 완화된 것) 이 정상성으로 묶인다.
        nW = size(nd.W, 1); κ = v[:κ]; α = v[:α]
        acti = [i for i in 1:nh if nd.hU[i] > 0]
        Zκ = @variable(mdl, [i in acti, ll in 1:nW], lower_bound = 0.0)
        for i in acti, ll in 1:nW
            @constraint(mdl, Zκ[i, ll] >= l[i] * κ[ll]); @constraint(mdl, Zκ[i, ll] <= u[i] * κ[ll])
        end
        @constraint(mdl, [ll in 1:nW], nd.wvec[ll] * κ[ll] == sum(nd.W[ll, i] * Zκ[i, ll] for i in acti))
        ζΠ = Dict{Tuple{Int,Int},VariableRef}()
        for k in Hr, s in 1:S
            i = nd.hrow[k]; nd.hU[i] > 0 || continue
            z = @variable(mdl, lower_bound = 0.0); ζΠ[(k, s)] = z
            π = v[:π̃][k, s]; d = v[:d][s]; zF = v[:ζF][i, s]; Lo, Hi = lo[k, s], hi[k, s]
            @constraint(mdl, z - l[i] * π - Lo * zF + Lo * l[i] * d >= 0)
            @constraint(mdl, -z + u[i] * π + Lo * zF - Lo * u[i] * d >= 0)
            @constraint(mdl, -z + l[i] * π + Hi * zF - Hi * l[i] * d >= 0)
            @constraint(mdl, z - u[i] * π - Hi * zF + Hi * u[i] * d >= 0)
        end
        for i in acti
            @constraint(mdl, sum(nd.W[ll, i] * Zκ[i, ll] for ll in 1:nW) - nd.c0[i] * α[i]
                             - sum(nd.hcoef[k] * ζΠ[(k, s)] for k in Hr, s in 1:S if nd.hrow[k] == i) == 0)
        end
    end
    if G                                          # 기울기 컷: ℓᵀŷˢ ≤ a_s r_s + b_sᵀ ζL_s
        av, bm = phi_grad(l, u)
        @constraint(mdl, [s=1:S], sum(nd.ell[j] * v[:ŷ][j, s] for j in 1:ny if nd.ell[j] != 0) <=
                                  av[s] * v[:r][s] + sum(bm[i, s] * v[:ζL][i, s] for i in 1:nh))
    end
    if V                                          # 시나리오별 값 상한: ℓᵀŷˢ ≤ φ̄_s(B) r_s  (ŷˢ = r_s × (α 에서의 pessimistic 최적해))
        φb = phi_bar(l, u)
        @constraint(mdl, [s=1:S], sum(nd.ell[j] * v[:ŷ][j, s] for j in 1:ny if nd.ell[j] != 0) <= φb[s] * v[:r][s])
    end
    if W
        for (idx, (jj, s, i)) in enumerate(P.wpos)
            k = Hr[jj]; nd.hU[i] > 0 || continue
            @constraint(mdl, v[:ϖ][k, s] <= hi[k, s] * v[:r][s]); @constraint(mdl, v[:ϖ][k, s] >= lo[k, s] * v[:r][s])
            push!(P.rows, _NZRow(0.0, [(:r, s, hi[k, s]), (:w, idx, -1.0)], i)); push!(P.rowscen, s)     # σhi r − ϖ ≥ 0
            push!(P.rows, _NZRow(0.0, [(:w, idx, 1.0), (:r, s, -lo[k, s])], i)); push!(P.rowscen, s)     # ϖ − σlo r ≥ 0
        end
    end
    if F
        for k in Hr, s in 1:S
            nd.hU[nd.hrow[k]] > 0 || continue
            @constraint(mdl, v[:π̃][k, s] <= hi[k, s] * v[:d][s]); @constraint(mdl, v[:π̃][k, s] >= lo[k, s] * v[:d][s])
        end
    end
    if C
        hrows_of = [[k for k in Hr if nd.hrow[k] == i] for i in 1:nh]
        for i in 1:nh
            l[i] > 1e-9 || continue                     # α_i > 0 ⇒ follower 1단계 정상성 등식
            @constraint(mdl, sum(nd.hcoef[k] * v[:π̃][k, s] for s in 1:S, k in hrows_of[i]) + nd.c0[i]
                             >= sum(nd.W[ll, i] * v[:κ][ll] for ll in 1:size(nd.W, 1)))
        end
        for ll in 1:size(nd.W, 1)
            dot(nd.W[ll, :], u) < nd.wvec[ll] - 1e-9 && fix(v[:κ][ll], 0.0; force=true)   # Wα < w ⇒ κ = 0
        end
    end
end

# ---- 2. 꺾임점 정렬 상자: α* 를 상자의 모서리에 두면 (분기점을 꺾임점에 맞추면) 상한이 어떻게 되는가 ----
# α*_i > 0 인 좌표마다 아래쪽 [α* − w, α*] 또는 위쪽 [α*, α* + w] (2^n 개 조합), α*_i = 0 인 좌표는 [0, w].
if envf("DR_ORTHANT", "0") == "1"
    pos = [i for i in 1:nh if nd.hU[i] > 0 && αs[i] > 1e-6 && αs[i] < nd.hU[i] - 1e-6]
    println("꺾임점 정렬 상자 (α* 가 모서리). 양수 좌표 ", pos, " 의 아래(L)/위(U) 조합별 상한 − V*")
    for w in pd.(split(envf("DR_OWIDTHS", "50,10,3"), ","))
        worst = Dict("cur" => -Inf, "W" => -Inf)
        for mask in 0:(2^length(pos) - 1)
            l = max.(0.0, αs .- w); u = min.(nd.hU, αs .+ w)
            for i in 1:nh
                nd.hU[i] > 0 || continue
                if αs[i] <= 1e-6; l[i] = 0.0                       # [0, w]
                elseif αs[i] >= nd.hU[i] - 1e-6; u[i] = nd.hU[i]   # [hU − w, hU]
                end
            end
            tag = ""
            for (b, i) in enumerate(pos)
                if (mask >> (b - 1)) & 1 == 1; l[i] = αs[i]; tag *= "U" else u[i] = αs[i]; tag *= "L" end
            end
            lo, hi, _, _ = dual_ranges(l, u)
            vals = Float64[]
            for W in (false, true)
                P = _nz_build_sep(nd, x̄; optimizer=GRB)
                _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                add_ranges!(P, lo, hi, l, u; W=W, F=W, C=W)
                push!(vals, _nz_sep_solve!(P) - V)
            end
            worst["cur"] = max(worst["cur"], vals[1]); worst["W"] = max(worst["W"], vals[2])
            act = [(k, s) for k in Hr, s in 1:S if nd.hU[nd.hrow[k]] > 0]
            @printf("  w=%5.1f %s | cur %9.3f  W+F+C %9.3f | 쌍대 고정 %d/%d\n", w, tag, vals[1], vals[2],
                    count(((k, s),) -> hi[k, s] - lo[k, s] < 1e-5, act), length(act))
            flush(stdout)
        end
        @printf("w=%5.1f 최악 조합: cur %.3f  W+F+C %.3f   (같은 폭의 중심 상자와 비교)\n", w, worst["cur"], worst["W"]); flush(stdout)
    end
end

# ---- 3. 꺾임점 ± η 로 3 분할: 아래 [α* − w, α* − η] / 얇은 판 [α* − η, α* + η] / 위 [α* + η, α* + w] ----
# 닫힌 상자에서 계산한 쌍대 범위만 쓰므로 한쪽 극한을 따질 필요가 없다. 얇은 판은 그 좌표의 폭이 2η 라 오차 ∝ η.
if envf("DR_SLAB", "0") == "1"
    η = parse(Float64, envf("DR_ETA", "0.05"))
    pos = [i for i in 1:nh if nd.hU[i] > 0 && αs[i] > 1e-6]
    println("꺾임점 ± η 3 분할 (η = $η). 양수 좌표 ", pos, " 마다 L / M (얇은 판) / U. 상한 − V* (음수면 가지치기됨)")
    for w in pd.(split(envf("DR_OWIDTHS", "50,10,3"), ","))
        worst = Dict("cur" => (-Inf, ""), "W" => (-Inf, "")); nbox = 0; npos = Dict("cur" => 0, "W" => 0); tolV = 5e-3 * abs(V)
        for code in Iterators.product(fill(0:2, length(pos))...)
            l = zeros(nh); u = min.(nd.hU, αs .+ w)
            for i in 1:nh; nd.hU[i] > 0 || (u[i] = 0.0); end
            tag = ""; ok = true
            for (b, i) in enumerate(pos)
                if code[b] == 0;     l[i] = max(0.0, αs[i] - w); u[i] = αs[i] - η; tag *= "L"
                elseif code[b] == 1; l[i] = max(0.0, αs[i] - η); u[i] = min(nd.hU[i], αs[i] + η); tag *= "M"
                else;                l[i] = αs[i] + η; u[i] = min(nd.hU[i], αs[i] + w); tag *= "U"
                end
                l[i] <= u[i] - 1e-9 || (ok = false)
            end
            (ok && all(nd.W * l .<= nd.wvec .+ 1e-9)) || continue
            nbox += 1
            lo, hi, _, _ = dual_ranges(l, u)
            for (name, W) in (("cur", false), ("W", true))
                P = _nz_build_sep(nd, x̄; optimizer=GRB)
                _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                add_ranges!(P, lo, hi, l, u; W=W, F=W, C=W)
                z = _nz_sep_solve!(P) - V
                z > worst[name][1] && (worst[name] = (z, tag))
                z > tolV && (npos[name] += 1)
            end
        end
        @printf("w=%5.1f 상자 %d 개 | cur: 최악 %.3f (%s), 허용오차 (%.2f) 초과 %d 개 | W+F+C: 최악 %.3f (%s), 초과 %d 개\n",
                w, nbox, worst["cur"]..., tolV, npos["cur"], worst["W"]..., npos["W"]); flush(stdout)
    end
end

for w in pd.(split(envf("DR_WIDTHS", "150,50,10,3,1"), ","))
    l = max.(0.0, αs .- w / 2); u = min.(nd.hU, αs .+ w / 2)
    lo, hi, nmip, tr = dual_ranges(l, u)
    act = [(k, s) for k in Hr, s in 1:S if nd.hU[nd.hrow[k]] > 0]
    nfixd = count(((k, s),) -> hi[k, s] - lo[k, s] < 1e-5, act)
    avgw = sum(hi[k, s] - lo[k, s] for (k, s) in act) / length(act)
    out = Float64[]
    for (W, F, C, Vc, Gc, Xc) in ((false, false, false, false, false, false), (true, true, true, false, false, false), (false, false, false, true, true, false),
                              (true, true, true, true, true, false), (true, true, true, false, false, true), (true, true, true, true, true, true))
        P = _nz_build_sep(nd, x̄; optimizer=GRB)
        _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
        add_ranges!(P, lo, hi, l, u; W=W, F=F, C=C, V=Vc, G=Gc, X=Xc)
        push!(out, _nz_sep_solve!(P) - V)
    end
    @printf("w=%6.1f | 상한−V*: cur %9.3f  W+F+C %9.3f  V+G %9.3f  W+F+C+V+G %9.3f  W+F+C+X %9.3f  전부 %9.3f | 쌍대 범위: 고정 %d/%d, 평균 폭 %.1f (MIP %d 개, %.1fs)
",
            w, out..., nfixd, length(act), avgw, nmip, tr)
    flush(stdout)
end
