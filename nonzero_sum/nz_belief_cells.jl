"""
nz_belief_cells.jl — 고정 x̄ 의 oracle 을 **belief 공간의 셀 열거** 로 정확히 푸는 시제품 (S = 3 전용). nz_alpha_bnb.jl 을 먼저 include.

배경 (문서 §10.17). follower 의 2 단계 LP 는 실행가능 집합이 belief d 와 무관하고 d 는 목적에만 들어간다. 그래서 최적 반응의 집합 Ψ(x̄, d) 는 d 의
조각별 상수 (LP 의 최적면) 이고, belief 볼 위에서 최적 반응 전체는 **차원이 S − 1 이하인 다면체 복합체** 다 — 반응 차원 n_h 와 무관하다.
α-B&B 는 n_h 차원 상자를 나누므로 n_h 가 크면 닫지 못하지만, S − 1 < n_h 이면 belief 쪽에서 세는 것이 싸다 (유효 차원은 min(n_h, S − 1)).

방법 (S = 3: d 공간은 2 차원 다각형)
  1. 층 (stratum) 분할. 볼록 다각형 P 의 무게중심 c 에서 follower LP 의 최적 반응 α_c 를 구한다. α_c 가 P 의 모든 꼭짓점에서도 최적이면 P 전체에서 최적이고
     (최적인 d 의 집합은 볼록), P 의 내부에서 최적면은 일정하다 (보조정리: 다른 반응 α′ 가 내부의 한 점에서 최적이면 g_d(α_c) − g_d(α′) ≥ 0 이 내부점에서 0 인
     선형함수라 P 전체에서 0 → α′ 도 c 에서 최적). 아니면 α_c 가 지는 꼭짓점의 반응 α_i 와의 동률선 g_d(α_c) = g_d(α_i) (d 에 선형) 로 P 를 자른다.
     같은 일을 각 다각형의 변 (1 차원) 에, 그리고 변의 끝점 (0 차원) 에 반복한다 — 셀 경계에서는 최적 반응이 면 전체라 따로 봐야 하기 때문이다.
  2. 층마다 값. 층의 대표점 d_rep 에서의 최적면 F = {(α, y) : c0ᵀα + Σ_s d_rep,s cᵀy_s ≥ v_F(d_rep)} 는 선형이고, 층 안에서 d 는 리더의 belief 집합 (결합) 에만 들어간다:
         max Σ_s r_s ℓᵀŷ_s   s.t.  d ∈ 층,  (a, r) ∈ 𝔅(d),  (α, y) ∈ F,  ŷ_s ∈ Y_s(α),  cᵀŷ_s ≥ cᵀy_s
     곱은 r_s × (ℓᵀŷ_s) 의 S 개뿐이다 (α 와의 곱, big-M, θ 벌점, 이진변수 없음). d_rep,s > 0 이면 F 위의 y_s 는 recourse 최적이므로 ŷ_s 는 recourse 최적해 중
     리더에게 가장 나쁜 것이 된다. (TV 볼에서 d_s ≥ q̂_s − ε̃ 이므로 q̂_s > ε̃ 이면 항상 양수. 아니면 이 시제품은 에러.)
  3. V*(x̄) = 모든 층의 값의 최댓값. 하한은 가장 좋은 층의 α 를 α 고정 LP 로 다시 평가한 값.

    nz_belief_cell_oracle(nd, x̄; nthreads=…, verbose=true) -> Dict(:LB, :UB, :α, :cells, :edges, :points, :time, …)
"""

using JuMP, Gurobi, LinearAlgebra, Printf
import MathOptInterface as MOI

# ---------------- 2 차원 볼록 다각형 ----------------
_bc_area(P) = length(P) < 3 ? 0.0 : 0.5 * abs(sum(P[i][1] * P[mod1(i + 1, end)][2] - P[mod1(i + 1, end)][1] * P[i][2] for i in eachindex(P)))
_bc_centroid(P) = [sum(p[1] for p in P) / length(P), sum(p[2] for p in P) / length(P)]
"볼록 다각형 P 를 반평면 a·p ≤ b 로 자른다 (Sutherland–Hodgman)"
function _bc_clip(P, a, b)
    out = Vector{Float64}[]
    n = length(P)
    for i in 1:n
        p, q = P[i], P[mod1(i + 1, n)]
        fp, fq = dot(a, p) - b, dot(a, q) - b
        fp <= 0 && push!(out, p)
        if (fp < 0 && fq > 0) || (fp > 0 && fq < 0)
            t = fp / (fp - fq); push!(out, p .+ t .* (q .- p))
        end
    end
    res = Vector{Float64}[]                       # 겹친 점 제거
    for p in out
        (isempty(res) || norm(p .- res[end]) > 1e-12) && push!(res, p)
    end
    length(res) > 1 && norm(res[1] .- res[end]) <= 1e-12 && pop!(res)
    return res
end

function nz_belief_cell_oracle(nd::NZData, x̄; nthreads::Int=max(1, Threads.nthreads() - 2), verbose::Bool=true, tol=1e-7, min_area=1e-10, min_len=1e-8,
                               maxdepth::Int=200)
    nd.S == 3 || error("nz_belief_cell_oracle: 시제품은 S = 3 전용 (d 공간이 2 차원)")
    t0 = time()
    hUp, _ = nz_presolve_hU(nd, x̄); nd = nz_with_hU(nd, hUp)
    S, m, ny, nh = nd.S, nz_m(nd), nz_ny(nd), nz_nh(nd); q = nd.q_hat; ε̃, ε̂, β = nd.eps_tilde, nd.eps_hat, nd.beta
    A, c, ℓ = nd.A, nd.c, nd.ell
    all(q[s] - ε̃ > 1e-6 for s in 1:S) || error("nz_belief_cell_oracle: d_s 가 0 이 될 수 있다 (q̂_s ≤ ε̃) — 그 시나리오의 recourse 최적성을 따로 걸어야 함 (미구현)")
    envs = _nz_envs(nthreads + 1)
    mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    dof(p) = [p[1], p[2], 1 - p[1] - p[2]]

    # follower LP: 반응 (α, [Q_s(α)]_s, 최적값)
    fm = Model(mk(envs[1])); @variable(fm, 0 <= αf[i = 1:nh] <= nd.hU[i]); @variable(fm, yf[1:ny, 1:S] >= 0); @constraint(fm, nd.W * αf .<= nd.wvec)
    for s in 1:S, k in 1:m
        @constraint(fm, sum(A[k, j] * yf[j, s] for j in 1:ny if A[k, j] != 0) <=
                        nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * αf[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0))
    end
    nlp = Ref(0)
    function resp(p)
        d = dof(p)
        @objective(fm, Max, dot(nd.c0, αf) + sum(d[s] * c[j] * yf[j, s] for j in 1:ny, s in 1:S if c[j] != 0)); optimize!(fm); nlp[] += 1
        termination_status(fm) == MOI.OPTIMAL || error("follower LP: $(termination_status(fm))")
        a = value.(αf); v = objective_value(fm)
        Q = [sum(c[j] * value(yf[j, s]) for j in 1:ny) for s in 1:S]            # d_s > 0 이므로 y_s 는 recourse 최적 → Q_s(α)
        return (α=a, Q=Q, c0=dot(nd.c0, a), v=v)
    end
    gval(R, p) = (d = dof(p); R.c0 + sum(d[s] * R.Q[s] for s in 1:S))
    vF(p) = resp(p).v
    isopt(R, p, vp) = gval(R, p) >= vp - tol * max(1.0, abs(vp))

    # belief 볼 (2 차원): 단체 ∩ {Σ_s σ_s (d_s − q̂_s) ≤ 2ε̃  ∀ 부호 σ}
    P0 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    for σ1 in (-1, 1), σ2 in (-1, 1), σ3 in (-1, 1)
        (σ1 == σ2 == σ3) && continue
        P0 = _bc_clip(P0, [σ1 - σ3, σ2 - σ3], 2ε̃ + σ1 * q[1] + σ2 * q[2] - σ3 * (1 - q[3]))
    end
    # 1. 2 차원 층
    cells = Any[]; nfuzzy = Ref(0)
    function split2(P, depth)
        cpt = _bc_centroid(P); Rc = resp(cpt)
        bad = 0; Rb = nothing
        for (i, p) in enumerate(P)
            Rp = resp(p)
            if !isopt(Rc, p, Rp.v)
                bad = i; Rb = Rp; break
            end
        end
        if bad == 0
            push!(cells, (P=P, rep=cpt, R=Rc, fuzzy=false)); return
        end
        # 동률선 h(d) = g_d(α_c) − g_d(α_b) = 0.  h(c) ≥ 0 > h(P[bad])
        ΔQ = Rc.Q .- Rb.Q; a = [ΔQ[1] - ΔQ[3], ΔQ[2] - ΔQ[3]]; b0 = (Rc.c0 - Rb.c0) + ΔQ[3]
        Pp = _bc_clip(P, -a, b0)            # h ≥ 0  ⟺  −a·p ≤ b0
        Pm = _bc_clip(P, a, -b0)            # h ≤ 0
        if depth >= maxdepth || _bc_area(Pp) < min_area || _bc_area(Pm) < min_area
            nfuzzy[] += 1; push!(cells, (P=P, rep=cpt, R=Rc, fuzzy=true)); return     # 더 못 자르는 조각: 최적면의 허용오차를 넓혀 평가 (상한 쪽으로 안전)
        end
        split2(Pp, depth + 1); split2(Pm, depth + 1)
    end
    split2(P0, 0)
    # 2. 1 차원 층 (각 셀의 변) 과 0 차원 층 (끝점)
    segs = Any[]; seen = Set{NTuple{4,Float64}}()
    function split1(p, qq, depth)
        norm(qq .- p) < min_len && return
        mid = 0.5 .* (p .+ qq); Rm = resp(mid)
        bad = nothing; Rb = nothing
        for e in (p, qq)
            Re = resp(e)
            if !isopt(Rm, e, Re.v)
                bad = e; Rb = Re; break
            end
        end
        if bad === nothing
            push!(segs, (P=[p, qq], rep=mid, R=Rm, fuzzy=false)); return
        end
        hm = gval(Rm, mid) - gval(Rb, mid); hb = gval(Rm, bad) - gval(Rb, bad)        # hm ≥ 0 > hb, 선분 위에서 선형
        t = hm / (hm - hb); xs = mid .+ t .* (bad .- mid)
        if depth >= maxdepth || norm(xs .- bad) < min_len
            nfuzzy[] += 1; push!(segs, (P=[p, qq], rep=mid, R=Rm, fuzzy=true)); return
        end
        split1(p, xs, depth + 1); split1(xs, qq, depth + 1)
    end
    for C in cells, i in eachindex(C.P)
        p, qq = C.P[i], C.P[mod1(i + 1, length(C.P))]
        key = let a = round.(p; digits=9), b = round.(qq; digits=9); a <= b ? (a[1], a[2], b[1], b[2]) : (b[1], b[2], a[1], a[2]) end
        key in seen && continue
        push!(seen, key); split1(p, qq, 0)
    end
    pts = Any[]; seenp = Set{NTuple{2,Float64}}()
    for sg in segs, p in sg.P
        key = (round(p[1]; digits=9), round(p[2]; digits=9))
        key in seenp && continue
        push!(seenp, key); push!(pts, (P=[p], rep=p, R=resp(p), fuzzy=false))
    end
    tsplit = time() - t0
    verbose && (@printf("  [belief-cells] 층: 셀 %d, 변 %d, 점 %d (더 못 자른 조각 %d) | follower LP %d 회, %.1fs\n", length(cells), length(segs), length(pts), nfuzzy[], nlp[], tsplit); flush(stdout))

    # 3. 층마다 값 (곱은 r_s φ_s 뿐)
    strata = vcat(cells, segs, pts)
    δc = nz_delta(nd)
    φr = [begin
              lm = Model(mk(envs[1])); @variable(lm, 0 <= al[i = 1:nh] <= nd.hU[i]); @variable(lm, yl[1:ny] >= 0); @constraint(lm, nd.W * al .<= nd.wvec)
              @constraint(lm, [k = 1:m], sum(A[k, j] * yl[j] for j in 1:ny if A[k, j] != 0) <=
                                         nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * al[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0))
              @objective(lm, Min, dot(ℓ, yl)); optimize!(lm); lo = objective_value(lm) - 1e-6
              @objective(lm, Max, dot(ℓ, yl)); optimize!(lm); (lo, objective_value(lm) + 1e-6)
          end for s in 1:S]
    function solve_stratum(st, env)
        M = Model(mk(env)); set_optimizer_attribute(M, "NonConvex", 2); set_optimizer_attribute(M, "MIPGap", 1e-7)
        nv = length(st.P)
        @variable(M, μ[1:nv] >= 0); @constraint(M, sum(μ) == 1)
        @variable(M, d[1:S] >= 0)
        for s in 1:S; @constraint(M, d[s] == sum(μ[v] * dof(st.P[v])[s] for v in 1:nv)); end
        @variable(M, 0 <= a[1:S] <= 1); @variable(M, b[1:S] >= 0); @variable(M, 0 <= r[1:S] <= 1)
        @constraint(M, sum(a) == 1); @constraint(M, sum(r) == 1)
        @constraint(M, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(M, [s = 1:S], a[s] + b[s] >= q[s])
        @constraint(M, sum(b) <= 2ε̂); @constraint(M, [s = 1:S], (1 - β) * r[s] <= a[s])
        if isfinite(δc)
            @variable(M, cpl[1:S] >= 0)
            @constraint(M, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(M, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(M, sum(cpl) <= 2δc)
        end
        @variable(M, 0 <= α[i = 1:nh] <= nd.hU[i]); @constraint(M, nd.W * α .<= nd.wvec)
        @variable(M, y[1:ny, 1:S] >= 0); @variable(M, ŷ[1:ny, 1:S] >= 0)
        for s in 1:S, k in 1:m
            rk = nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0)
            @constraint(M, sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0) <= rk)
            @constraint(M, sum(A[k, j] * ŷ[j, s] for j in 1:ny if A[k, j] != 0) <= rk)
        end
        drep = dof(st.rep); vrep = st.R.v
        ftol = (st.fuzzy ? 1e-4 : tol) * max(1.0, abs(vrep))
        @constraint(M, dot(nd.c0, α) + sum(drep[s] * c[j] * y[j, s] for j in 1:ny, s in 1:S if c[j] != 0) >= vrep - ftol)      # (α, y) ∈ 대표점의 최적면
        @constraint(M, [s = 1:S], sum(c[j] * ŷ[j, s] for j in 1:ny if c[j] != 0) >= sum(c[j] * y[j, s] for j in 1:ny if c[j] != 0))
        @variable(M, φr[s][1] <= φ[s = 1:S] <= φr[s][2]); @constraint(M, [s = 1:S], φ[s] == sum(ℓ[j] * ŷ[j, s] for j in 1:ny if ℓ[j] != 0))
        @objective(M, Max, sum(r[s] * φ[s] for s in 1:S))
        optimize!(M)
        st_ = termination_status(M)
        st_ == MOI.OPTIMAL || error("nz_belief_cell_oracle 층 문제: $st_")
        return (val=objective_value(M), bd=objective_bound(M), α=value.(α))
    end
    res = Vector{Any}(undef, length(strata)); nxt = Threads.Atomic{Int}(1)
    tasks = map(1:nthreads) do w
        Threads.@spawn while true
            n = Threads.atomic_add!(nxt, 1); n > length(strata) && break
            res[n] = solve_stratum(strata[n], envs[w + 1])
        end
    end
    foreach(wait, tasks)
    UB = maximum(r.bd for r in res); ib = argmax([r.val for r in res])
    F = nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[1]))
    LB = -Inf; αb = res[ib].α
    for n in sortperm([r.val for r in res]; rev=true)[1:min(5, end)]        # 좋은 층 몇 개의 α 를 α 고정 LP 로 다시 평가 (인증된 하한)
        z = nz_eval_alpha!(F, nd, res[n].α)
        isfinite(z) && z > LB && (LB = z; αb = res[n].α)
    end
    kind(n) = n <= length(cells) ? :cell : n <= length(cells) + length(segs) ? :edge : :point
    verbose && (@printf("  [belief-cells] UB=%.6f LB=%.6f (gap %.2e) | 최댓값의 층: %s | 셀만의 최댓값 %.6f, 변 %.6f, 점 %.6f | %.1fs\n", UB, LB,
                        (UB - LB) / max(1.0, abs(LB)), string(kind(ib)), maximum(res[n].bd for n in 1:length(cells)),
                        isempty(segs) ? NaN : maximum(res[length(cells) + n].bd for n in 1:length(segs)),
                        isempty(pts) ? NaN : maximum(res[length(cells) + length(segs) + n].bd for n in 1:length(pts)), time() - t0); flush(stdout))
    return Dict(:LB => LB, :UB => UB, :α => αb, :is_exact => (UB - LB <= 1e-3 * max(1.0, abs(LB))), :cells => length(cells), :edges => length(segs),
                :points => length(pts), :fuzzy => nfuzzy[], :time => time() - t0, :t_split => tsplit, :follower_lps => nlp[], :solver => :nz_belief_cell_oracle,
                :best_kind => kind(ib))
end
