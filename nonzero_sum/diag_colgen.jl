"""
diag_colgen.jl — 노드 완화의 시나리오별 Dantzig–Wolfe 열 생성 프로토타입 (진단, `diag_lagrange.jl` 의 분해를 그대로 씀).

RMP: 공통 변수 x_G (α, κ, ρ⁰) + 블록 열의 볼록조합 μ_sk.
  연결 행 (승수 π), 사본 행 α − Σ_k α_sk μ_sk = 0 (승수 λ_s), 볼록성 Σ_k μ_sk = 1, 공통 행. 연결·사본·볼록성 행에 벌점 M 의 인공 변수.
가격 결정: 블록 s LP max (c_s − A_Lsᵀπ)ᵀx_s + λ_sᵀα_s (블록 모델을 상자마다 한 번 만들고 목적만 바꿔 warm start).
상한: 매 반복 L(π̃, λ̃) (diag_lagrange 와 같은 식, 어떤 승수든 유효). 안정화: Wentges smoothing (π̃ = θ·중심 + (1−θ)·RMP 쌍대,
  중심 = 지금까지 L 이 가장 낮은 승수, 위반 열이 없으면 θ = 0 으로 다시).
시작: 부모 LP 의 쌍대 (중심·첫 가격 결정) + 부모 LP 해의 블록 부분 중 자식 상자에서 실행가능한 것 (열).

측정 (diag_lagrange 와 같은 하강): 깊이마다 두 자식에서 L_best 가 자식 LP 값 z_K 의 0.1% 안에 드는 반복 수·시간, 수렴 (L − RMP ≤ 1e-4) 까지.
환경변수: CG_S = "50"  CG_DEPTH = 2  CG_THETA = 0.5  CG_MAXIT = 300  CG_TL = 300  CG_M = 1e5  (LG_X, LG_EPS, LG_DELTA 는 diag_lagrange 와 같음)
"""
nothing
include(joinpath(@__DIR__, "diag_lagrange.jl"))
θ0 = parse(Float64, envf("CG_THETA", "0.5")); maxit = parse(Int, envf("CG_MAXIT", "300"))
cgtl = parse(Float64, envf("CG_TL", "300")); Mpen = parse(Float64, envf("CG_M", "1e5"))
Δrel0 = pd(envf("CG_DREL", "Inf")); Δabs0 = parse(Float64, envf("CG_DABS", "1.0"))
const ZMAP = Dict("Za" => "a", "Zb" => "b", "Ze" => "e", "Zc" => "cpl")

"부분 LP 모델 (목적은 나중에)"
function build_sub(A, lo, hi, xl, xu; method=0)
    m = Model(mk1); set_optimizer_attribute(m, "Method", method)
    n = size(A, 2)
    x = @variable(m, [1:n])
    for j in 1:n
        fin(xl[j]) && set_lower_bound(x[j], xl[j]); fin(xu[j]) && set_upper_bound(x[j], xu[j])
    end
    At = sparse(A')
    for r in 1:size(A, 1)
        e = AffExpr(0.0)
        for p in nzrange(At, r); add_to_expression!(e, nonzeros(At)[p], x[rowvals(At)[p]]); end
        l, h = lo[r], hi[r]
        if fin(l) && fin(h) && l == h
            @constraint(m, e == l)
        else
            fin(l) && @constraint(m, e >= l)
            fin(h) && @constraint(m, e <= h)
        end
    end
    return m, x
end

function solve_obj!(m, x, c)
    @objective(m, Max, dot(c, x))
    t = @elapsed optimize!(m)
    st = termination_status(m)
    st == MOI.OPTIMAL && return objective_value(m), value.(x), t
    st in (MOI.DUAL_INFEASIBLE, MOI.OBJECTIVE_LIMIT, MOI.INFEASIBLE_OR_UNBOUNDED) && return Inf, nothing, t
    error("부분 LP 상태 $st")
end

"자식 상자 dK 에서 열 생성. 반환: 반복 기록"
function colgen(dK, Pt, S, nh, πpar, λpar, seedcols, zK; θ0=0.5, maxit=300, tl=300.0, M=1e5, tol=1e-4, Δrel=Inf, Δabs=1.0)
    lo, hi, A = dK.b_lower, dK.b_upper, dK.A
    t0 = time()
    # ---- 블록·공통 모델 (상자마다 한 번) ----
    tbuild = @elapsed begin
        blk = map(1:S) do s
            allc = vcat(Pt.blkcols[s], Pt.αcols)
            build_sub(A[Pt.blkrows[s], allc], lo[Pt.blkrows[s]], hi[Pt.blkrows[s]], dK.x_lower[allc], dK.x_upper[allc])
        end
        gm, gx = build_sub(A[Pt.grows, Pt.gcols], lo[Pt.grows], hi[Pt.grows], dK.x_lower[Pt.gcols], dK.x_upper[Pt.gcols])
    end
    ALs = [A[Pt.link, Pt.blkcols[s]] for s in 1:S]       # 연결 행 × 블록 열
    ALG = A[Pt.link, Pt.gcols]
    αpos = [Pt.αidx[j] for j in Pt.αcols]                 # 블록 열 끝의 α 사본 순서 → α 번호
    gαk = [findfirst(==(j), Pt.gcols) for j in Pt.αcols]  # 공통 변수 안의 α 위치 (αcols 순서)
    nL = length(Pt.link)
    # 승수 → L 과 블록 해
    function Lval(πL, λ)
        cst = dK.c_offset
        for (k, r) in enumerate(Pt.link)
            p = πL[k]; abs(p) < 1e-12 && continue
            p > 0 ? (fin(hi[r]) ? (cst += p * hi[r]) : (return Inf, nothing, 0.0, 0.0)) :
                    (fin(lo[r]) ? (cst += p * lo[r]) : (return Inf, nothing, 0.0, 0.0))
        end
        tb = 0.0; tot = cst; sols = Vector{Any}(undef, S)
        for s in 1:S
            cs = dK.c[Pt.blkcols[s]] .- ALs[s]' * πL
            v, xs, t = solve_obj!(blk[s][1], blk[s][2], vcat(cs, [λ[s][i] for i in αpos]))
            tb += t; tot += v; sols[s] = xs
            isinf(v) && return Inf, nothing, tb, 0.0
        end
        cg = dK.c[Pt.gcols] .- ALG' * πL
        for (q, k) in enumerate(gαk); cg[k] -= sum(λ[s][αpos[q]] for s in 1:S); end
        vg, _, tg = solve_obj!(gm, gx, cg)
        return tot + vg, sols, tb, tg
    end
    # ---- RMP ----
    rmp = Model(mk1); set_optimizer_attribute(rmp, "Method", 0)
    xg = @variable(rmp, [1:length(Pt.gcols)])
    for (k, j) in enumerate(Pt.gcols)
        fin(dK.x_lower[j]) && set_lower_bound(xg[k], dK.x_lower[j]); fin(dK.x_upper[j]) && set_upper_bound(xg[k], dK.x_upper[j])
    end
    obj = AffExpr(0.0); add_to_expression!(obj, dot(dK.c[Pt.gcols], xg))
    Agt = sparse(A[Pt.grows, Pt.gcols]')
    for (q, r) in enumerate(Pt.grows)
        e = AffExpr(0.0)
        for p in nzrange(Agt, q); add_to_expression!(e, nonzeros(Agt)[p], xg[rowvals(Agt)[p]]); end
        fin(lo[r]) && fin(hi[r]) && lo[r] == hi[r] ? @constraint(rmp, e == lo[r]) :
            (fin(lo[r]) && @constraint(rmp, e >= lo[r]); fin(hi[r]) && @constraint(rmp, e <= hi[r]))
    end
    arts = Tuple{VariableRef,VariableRef}[]
    art(e) = (a1 = @variable(rmp, lower_bound = 0.0); a2 = @variable(rmp, lower_bound = 0.0);
              add_to_expression!(obj, -M, a1); add_to_expression!(obj, -M, a2); push!(arts, (a1, a2)); e + a1 - a2)
    ALGt = sparse(ALG')
    linkcon = Vector{Any}(nothing, nL); artL = Vector{Any}(nothing, nL)
    for (k, r) in enumerate(Pt.link)
        (fin(lo[r]) || fin(hi[r])) || continue
        e = AffExpr(0.0)
        for p in nzrange(ALGt, k); add_to_expression!(e, nonzeros(ALGt)[p], xg[rowvals(ALGt)[p]]); end
        e = art(e); artL[k] = arts[end]
        linkcon[k] = fin(lo[r]) && fin(hi[r]) ? (lo[r] == hi[r] ? @constraint(rmp, e == lo[r]) : @constraint(rmp, lo[r] <= e <= hi[r])) :
                     fin(hi[r]) ? @constraint(rmp, e <= hi[r]) : @constraint(rmp, e >= lo[r])
    end
    copycon = [[@constraint(rmp, art(1.0 * xg[gαk[q]]) == 0.0) for q in eachindex(αpos)] for s in 1:S]   # αcols 순서
    artC = reshape(arts[end-S*length(αpos)+1:end], length(αpos), S)
    convcon = [@constraint(rmp, art(AffExpr(0.0)) == 1.0) for s in 1:S]
    @objective(rmp, Max, obj)
    ncol = 0
    function addcol!(s, xs)
        nb = length(Pt.blkcols[s])
        xb = xs[1:nb]; xa = xs[nb+1:end]
        μ = @variable(rmp, lower_bound = 0.0)
        set_objective_coefficient(rmp, μ, dot(dK.c[Pt.blkcols[s]], xb))
        cl = ALs[s] * xb
        for k in 1:nL
            linkcon[k] === nothing && continue
            abs(cl[k]) > 1e-12 && set_normalized_coefficient(linkcon[k], μ, cl[k])
        end
        for q in eachindex(αpos)
            abs(xa[q]) > 1e-12 && set_normalized_coefficient(copycon[s][q], μ, -xa[q])
        end
        set_normalized_coefficient(convcon[s], μ, 1.0)
        ncol += 1
    end
    function rmpduals()
        πL = [c === nothing ? 0.0 : -dual(c) for c in linkcon]
        λ = [zeros(nh) for _ in 1:S]
        for s in 1:S, q in eachindex(αpos); λ[s][αpos[q]] = -dual(copycon[s][q]); end
        σ = [-dual(c) for c in convcon]
        return πL, λ, σ
    end
    # box-step 안정화: 연결·사본 행의 인공 변수 비용을 중심 ± Δ 로 → RMP 쌍대가 [중심 − Δ, 중심 + Δ] 안 (Δ = Δrel·max(|중심|, Δabs))
    function setbox!(πc, λc)
        isfinite(Δrel) || return
        sb(a, pc) = (Δ = Δrel * max(abs(pc), Δabs); set_objective_coefficient(rmp, a[1], -(Δ - pc)); set_objective_coefficient(rmp, a[2], -(Δ + pc)))
        for k in 1:nL; artL[k] === nothing || sb(artL[k], πc[k]); end
        for s in 1:S, q in eachindex(αpos); sb(artC[q, s], λc[s][αpos[q]]); end
    end
    artsum() = sum(value(a[1]) + value(a[2]) for a in arts)
    rc(s, xs, πL, λ, σ) = (nb = length(Pt.blkcols[s]);
        dot(dK.c[Pt.blkcols[s]], xs[1:nb]) - dot(πL, ALs[s] * xs[1:nb]) + sum(λ[s][αpos[q]] * xs[nb+q] for q in eachindex(αpos)) - σ[s])
    # ---- 시작: 부모 승수에서 가격 결정 + 부모 해 열 ----
    for (s, xs) in seedcols; addcol!(s, xs); end
    πc = πpar[Pt.link]; λc = λpar
    Lb, sols, tb, tg = Lval(πc, λc)
    hist = [(it=0, L=Lb, z=NaN, t=time() - t0, tb=tb, tr=0.0)]
    sols === nothing || for s in 1:S; addcol!(s, sols[s]); end
    setbox!(πc, λc)
    θ = isfinite(Δrel) ? 0.0 : θ0; it_hit = Lb <= zK + 1e-3 * max(1.0, abs(zK)) ? 0 : -1
    trtot = 0.0; tbtot = tb; zr = NaN
    for it in 1:maxit
        trr = @elapsed optimize!(rmp); trtot += trr
        termination_status(rmp) == MOI.OPTIMAL || error("RMP $(termination_status(rmp))")
        zr = objective_value(rmp); as = artsum()
        πr, λr, σr = rmpduals()
        added = 0; θit = θ; Lt = Inf
        while true
            πt = θit .* πc .+ (1 - θit) .* πr
            λt = [θit .* λc[s] .+ (1 - θit) .* λr[s] for s in 1:S]
            Lt, sols, tb, tg = Lval(πt, λt); tbtot += tb
            if Lt < Lb
                Lb = Lt; πc = πt; λc = λt; setbox!(πc, λc)
            end
            if sols !== nothing
                for s in 1:S
                    rc(s, sols[s], πr, λr, σr) > 1e-7 * max(1.0, abs(zr)) && (addcol!(s, sols[s]); added += 1)
                end
            end
            (added > 0 || θit == 0.0) && break
            θit = 0.0                                          # 안정화 점에서 위반 열 없음 → RMP 쌍대에서 다시
        end
        push!(hist, (it=it, L=Lb, z=zr, t=time() - t0, tb=tbtot, tr=trtot))
        it_hit < 0 && Lb <= zK + 1e-3 * max(1.0, abs(zK)) && (it_hit = it)
        time() - t0 > tl && break
        if added == 0
            (!isfinite(Δrel) || (as <= 1e-7 && Lt - zr <= tol * max(1.0, abs(zr)))) && break   # 안정화 문제의 최적 = 원래 최적
            πc = πr; λc = λr; Δrel *= 2; setbox!(πc, λc)       # 인공 변수가 남거나 쌍대가 상자 경계 → 중심 이동·상자 확대
        end
        as <= 1e-7 && Lb - zr <= tol * max(1.0, abs(zr)) && break
    end
    return (; hist, it_hit, tbuild, ncol, Lb, zr)
end

"블록 부분 (x_s, α) 이 자식 상자에서 블록 s 의 실행가능 열인지"
function feasible_col(dK, Pt, s, xs; tol=1e-7)
    allc = vcat(Pt.blkcols[s], Pt.αcols)
    all(dK.x_lower[allc] .- tol .<= xs .<= dK.x_upper[allc] .+ tol) || return false
    rows = Pt.blkrows[s]; ax = dK.A[rows, allc] * xs
    all((.!fin.(dK.b_lower[rows]) .| (ax .>= dK.b_lower[rows] .- tol .* max.(1, abs.(dK.b_lower[rows])))) .&
        (.!fin.(dK.b_upper[rows]) .| (ax .<= dK.b_upper[rows] .+ tol .* max.(1, abs.(dK.b_upper[rows])))))
end

for S in parse.(Int, split(envf("CG_S", "50"), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    x̄ = zeros(nd.nx); x̄[xi] .= 1
    hUp, _ = nz_presolve_hU(nd, x̄); nd = nz_with_hU(nd, hUp)
    nh = nz_nh(nd)
    P = _nz_build_sep(nd, x̄; optimizer=mk1)
    l = zeros(nh); u = copy(nd.hU)
    _nz_sep_set_box!(P, l, u)
    tr = @elapsed z0 = _nz_sep_solve!(P; maxrounds=30)
    d = lp_matrix_data(P.O.model)
    Pt = partition(d, S)
    π, λ = duals(d, Pt, S, nh)
    names = name.(d.variables)
    F = nz_build_fixed_alpha(nd, x̄; optimizer=mk1)
    @printf("\nS=%d: 루트 LP %.4f (%.1fs), 연결 행 %d, 사본 행 %d\n", S, z0, tr, length(Pt.link), nh * S); flush(stdout)
    zpar = z0
    for dep in 1:parse(Int, envf("CG_DEPTH", "2"))
        xsol = value.(d.variables)
        parcols = [(s, xsol[vcat(Pt.blkcols[s], Pt.αcols)]) for s in 1:S]
        i = argmax(u .- l); mid = 0.5 * (l[i] + u[i])
        res = []
        for side in (:lo, :hi)
            lc = copy(l); uc = copy(u)
            side == :lo ? (uc[i] = mid) : (lc[i] = mid)
            _nz_sep_set_box!(P, lc, uc)
            dK = lp_matrix_data(P.O.model)
            tz = @elapsed zK = _nz_solve_lp!(P.O.model)
            seeds = [(s, xs) for (s, xs) in parcols if feasible_col(dK, Pt, s, xs)]
            # 정확한 점 열: α' = 부모 α̂ 를 자식 상자로 자른 점에서 α 고정 LP (B&B 가 노드마다 이미 푸는 평가) 의 해 + 곱 = α'·변수
            αp = clamp.(value.(P.O.v[:α]) , lc, uc)
            te = @elapsed ze = nz_eval_alpha!(F, nd, αp)
            nex = 0
            if isfinite(ze)
                xe = map(names) do nm
                    fam = match(r"^([^\[]+)", nm)[1]
                    if haskey(ZMAP, fam)
                        ii, ss = parse.(Int, split(match(r"\[(.*)\]", nm)[1], ","))
                        αp[ii] * value(variable_by_name(F.model, "$(ZMAP[fam])[$ss]"))
                    else
                        value(variable_by_name(F.model, nm))
                    end
                end
                ex = [(s, xe[vcat(Pt.blkcols[s], Pt.αcols)]) for s in 1:S]
                nex = count(t -> feasible_col(dK, Pt, t...), ex)
                append!(seeds, [t for t in ex if feasible_col(dK, Pt, t...)])
            end
            @printf("    정확한 점 (α 고정 LP %.4f, %.2fs): 실행가능 블록 열 %d/%d\n", ze, te, nex, S)
            R = colgen(dK, Pt, S, nh, π, λ, seeds, zK; θ0=θ0, maxit=maxit, tl=cgtl, M=Mpen, Δrel=Δrel0, Δabs=Δabs0)
            h = R.hist; hl = h[end]
            th = R.it_hit >= 0 ? h[findfirst(x -> x.it == R.it_hit, h)].t : NaN
            @printf("  깊이 %d α%d %s: z_부모 %.2f z_자식 %.2f (full LP %.2fs) | 부모 열 %d/%d | 0.1%% 도달 반복 %d (%.1fs) | 종료 반복 %d L %.2f RMP %.2f (%.1fs: RMP %.1fs 블록 %.1fs, 모델 %.1fs) 열 %d\n",
                    dep, i, side == :lo ? "아래" : "위 ", zpar, zK, tz, length(seeds), S, R.it_hit, th,
                    hl.it, R.Lb, R.zr, hl.t, hl.tr, hl.tb, R.tbuild, R.ncol)
            for x in h
                (x.it <= 5 || x.it % 10 == 0 || x === hl) && @printf("      반복 %3d  L %.3f  RMP %.3f  (L−z_K)/|z_K| %.2e  t %.1fs\n",
                                                                  x.it, x.L, x.z, (x.L - zK) / max(1.0, abs(zK)), x.t)
            end
            flush(stdout)
            push!(res, (zK, lc, uc))
        end
        k = argmax([r[1] for r in res])
        zpar, l, u = res[k]
        _nz_sep_set_box!(P, l, u); _nz_solve_lp!(P.O.model)
        d = lp_matrix_data(P.O.model)
        π, λ = duals(d, Pt, S, nh)
    end
end
