"""
diag_lagrange.jl — α-B&B 노드 완화의 시나리오별 라그랑지안 분해 진단 (구현 전, `alpha_bnb_nonzero_analysis.md` §7 A 결과).

분해: 시나리오 s 의 모든 변수 (확률·곱·네트워크) 와 그 시나리오만 쓰는 행 (상자 의존 RLT 포함) = 블록 s.
α 가 들어간 블록 행은 블록 사본 α_s 를 쓰고 α_s = α 를 승수 λ_s 로 완화. 여러 시나리오 또는 κ·ρ⁰ 가 걸린 행 = 연결 행 (승수 π).
어떤 (π, λ) 든 L_K(π, λ) = 상수 + Σ_s 블록 s 최대값 + 공통 변수 LP 는 노드 상자 K 의 완화 LP 의 유효 상한.

질문: 부모 노드의 최적 쌍대로 자식 노드에서 블록을 한 번 풀면 (반복 없이) 상한이 얼마나 좋은가.
절차: 루트 완화 (분리 수렴) → 행 집합 고정 → 중점 분할로 하강하며 깊이마다 두 자식에서
  L_K(π_부모, λ_부모) 와 같은 행 집합의 자식 LP 값 z_K 비교. 포착률 = (z_부모 − L) / (z_부모 − z_K).
  더 큰 z_K 쪽 자식으로 내려가 그 자식 LP 의 쌍대를 다음 깊이의 부모 쌍대로.
환경변수: LG_S = "50,200"  LG_X = "1,2"  LG_EPS = 0.3  LG_DELTA = 0.1  LG_DEPTH = 8
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, SparseArrays
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
ε = parse(Float64, envf("LG_EPS", "0.3")); δ = pd(envf("LG_DELTA", "0.1")); depth = parse(Int, envf("LG_DEPTH", "8"))
xi = parse.(Int, split(envf("LG_X", "1,2"), ","))
mk1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)

const SCEN_VEC = Set(["a", "b", "r", "d", "e", "cpl"])                       # 이름[s]
const SCEN_MAT = Set(["ŷ", "ϖ", "ρ̂1", "ρ̂2", "ρ̂3", "ζL", "ζW", "ȳ", "π̃", "ρ̃1", "ρ̃2", "ρ̃3", "ζF",
                      "Za", "Zb", "Ze", "Zc"])                                 # 이름[·, s]
const GLOBAL = Set(["α", "κ", "ρ01", "ρ02", "ρ03"])

"변수 이름 → 시나리오 (0 = 공통)"
function scen_of(name)
    mt = match(r"^([^\[]+)\[(.*)\]$", name)
    mt === nothing && error("이름 없는 변수: $name")
    fam = mt[1]; idx = parse.(Int, split(mt[2], ","))
    fam in SCEN_VEC && return idx[1]
    fam in SCEN_MAT && return idx[2]
    fam in GLOBAL && return 0
    error("분류 안 된 변수 묶음: $fam")
end

const BIG = 1e29
fin(x) = abs(x) < BIG

"분해 구조 (행 집합이 고정이므로 한 번만)"
function partition(d, S)
    names = name.(d.variables)
    cs = scen_of.(names)
    isα = startswith.(names, "α[")
    αcols = findall(isα)
    αidx = Dict(j => parse(Int, match(r"\[(\d+)\]", names[j])[1]) for j in αcols)
    At = sparse(d.A')                                           # 열 r = 행 r
    nr = size(d.A, 1)
    kind = zeros(Int, nr)                                       # -1 연결, 0 공통, s 블록
    for r in 1:nr
        ss = Set{Int}(); gnon = false
        for p in nzrange(At, r)
            j = rowvals(At)[p]
            cs[j] == 0 ? (isα[j] || (gnon = true)) : push!(ss, cs[j])
        end
        kind[r] = isempty(ss) ? (gnon ? 0 : 0) : (length(ss) == 1 && !gnon ? first(ss) : -1)
    end
    blkrows = [findall(==(s), kind) for s in 1:S]
    blkcols = [findall(==(s), cs) for s in 1:S]
    return (; cs, isα, αcols, αidx, kind, link=findall(==(-1), kind), grows=findall(==(0), kind),
            blkrows, blkcols, gcols=findall(==(0), cs))
end

"부분 LP max cᵀx s.t. lo ≤ A x ≤ hi, xl ≤ x ≤ xu 를 Gurobi 로 풀고 (값, 풀이 시간)"
function solve_sub(A, lo, hi, c, xl, xu)
    m = Model(mk1); set_optimizer_attribute(m, "Method", 1)
    n = size(A, 2)
    x = @variable(m, [1:n])
    for j in 1:n
        fin(xl[j]) && set_lower_bound(x[j], xl[j]); fin(xu[j]) && set_upper_bound(x[j], xu[j])
    end
    At = sparse(A')
    for r in 1:size(A, 1)
        rg = nzrange(At, r)
        e = AffExpr(0.0)
        for p in rg; add_to_expression!(e, nonzeros(At)[p], x[rowvals(At)[p]]); end
        l, h = lo[r], hi[r]
        if fin(l) && fin(h) && l == h
            @constraint(m, e == l)
        else
            fin(l) && @constraint(m, e >= l)
            fin(h) && @constraint(m, e <= h)
        end
    end
    @objective(m, Max, dot(c, x))
    t = @elapsed optimize!(m)
    st = termination_status(m)
    st == MOI.OPTIMAL && return objective_value(m), t
    st == MOI.DUAL_INFEASIBLE && return Inf, t
    st == MOI.INFEASIBLE && return -Inf, t
    error("부분 LP 상태 $st")
end

"""L_K(π, λ): d 는 현재 상자의 행렬 자료. π 는 행 승수 (≤ 쪽 ≥ 0), λ[s] 는 α_s 사본 승수 (nh).
비활성 RLT 행 (rhs −1e30) 의 연결 승수는 0 으로."""
function lagrange(d, Pt, π, λ, S, nh)
    lo = d.b_lower; hi = d.b_upper
    πL = zeros(length(lo))
    cst = d.c_offset
    for r in Pt.link
        p = π[r]
        abs(p) < 1e-12 && continue
        if p > 0
            fin(hi[r]) || continue                              # 상한 없는 행에 양의 승수 → 0 으로 (유효성 유지)
            cst += p * hi[r]
        else
            fin(lo[r]) || continue
            cst += p * lo[r]
        end
        πL[r] = p
    end
    c̃ = d.c .- d.A' * πL
    tblk = Float64[]
    total = cst
    for s in 1:S
        rows = Pt.blkrows[s]; cols = Pt.blkcols[s]
        allc = vcat(cols, Pt.αcols)                             # α 열 → 블록 사본
        Asub = d.A[rows, allc]
        cobj = vcat(c̃[cols], [λ[s][Pt.αidx[j]] for j in Pt.αcols])
        xl = d.x_lower[allc]; xu = d.x_upper[allc]
        v, t = solve_sub(Asub, lo[rows], hi[rows], cobj, xl, xu)
        push!(tblk, t); total += v
    end
    # 공통 변수: 목적 c̃ − Σ_s λ_s (α 열), 공통 행만
    cg = copy(c̃[Pt.gcols])
    for (k, j) in enumerate(Pt.gcols)
        Pt.isα[j] && (cg[k] -= sum(λ[s][Pt.αidx[j]] for s in 1:S))
    end
    vg, tg = solve_sub(d.A[Pt.grows, Pt.gcols], lo[Pt.grows], hi[Pt.grows], cg, d.x_lower[Pt.gcols], d.x_upper[Pt.gcols])
    total += vg
    return total, tblk, tg
end

"현재 해의 행 승수 π (= −dual, max 문제에서 ≤ 행 ≥ 0) 와 사본 승수 λ_s = Σ_{r∈블록 s} π_r A[r, α]"
function duals(d, Pt, S, nh)
    π = [-dual(c) for c in d.affine_constraints]
    λ = [zeros(nh) for _ in 1:S]
    for s in 1:S
        rows = Pt.blkrows[s]
        for j in Pt.αcols
            λ[s][Pt.αidx[j]] = dot(π[rows], d.A[rows, j])
        end
    end
    return π, λ
end

LG_MAIN = abspath(PROGRAM_FILE) == @__FILE__
LG_MAIN && for S in parse.(Int, split(envf("LG_S", "50,200"), ","))
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
    bs = [length(Pt.blkcols[s]) for s in 1:S]; br = [length(Pt.blkrows[s]) for s in 1:S]
    @printf("\nS=%d: 루트 LP %.4f (%.1fs). 변수 %d, 행 %d → 연결 행 %d, 공통 행 %d, 공통 변수 %d, 블록 변수 %d~%d (+α 사본 %d), 블록 행 %d~%d\n",
            S, z0, tr, size(d.A, 2), size(d.A, 1), length(Pt.link), length(Pt.grows), length(Pt.gcols),
            minimum(bs), maximum(bs), nh, minimum(br), maximum(br))
    L0, tb, tg = lagrange(d, Pt, π, λ, S, nh)
    @printf("  루트 검증: L(π*, λ*) = %.4f (상대차 %.1e), 블록 풀이 합 %.2fs (최대 %.3fs), 공통 LP %.3fs\n",
            L0, (L0 - z0) / max(1.0, abs(z0)), sum(tb), maximum(tb), tg)
    flush(stdout)
    zpar = z0
    for dep in 1:depth
        α̂ = value.(P.O.v[:α])
        i = argmax(u .- l); mid = 0.5 * (l[i] + u[i])
        res = []
        for side in (:lo, :hi)
            lc = copy(l); uc = copy(u)
            side == :lo ? (uc[i] = mid) : (lc[i] = mid)
            _nz_sep_set_box!(P, lc, uc)
            dK = lp_matrix_data(P.O.model)
            LK, tb, tg = lagrange(dK, Pt, π, λ, S, nh)
            tz = @elapsed zK = _nz_solve_lp!(P.O.model)
            cap = (zpar - LK) / max(1e-9, zpar - zK)
            @printf("  깊이 %d α%d %s: z_부모 %.2f  L(부모 쌍대) %.2f  z_자식 %.2f  포착률 %5.1f%%  (L−z)/|z| %.2e | 블록 합 %.2fs 최대 %.3fs, 자식 LP %.2fs\n",
                    dep, i, side == :lo ? "아래" : "위 ", zpar, LK, zK, 100cap, (LK - zK) / max(1.0, abs(zK)), sum(tb), maximum(tb), tz)
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
