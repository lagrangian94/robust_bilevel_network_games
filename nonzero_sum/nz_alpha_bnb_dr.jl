"""
nz_alpha_bnb_dr.jl — α-B&B (nz_alpha_bnb.jl) + **상자별 쌍대 범위 조임** 시제품. nz_alpha_bnb.jl 을 먼저 include.

기존 α-B&B 의 노드 완화는 인증서 (리더 블록 ϖˢ, follower 블록 π̃ˢ) 의 예약 행 (H 행) 성분에 전역 범위 [0, ϖᵁ] 만 쓴다.
정확한 해에서 ϖˢ/r_s, π̃ˢ/d_s 는 시나리오 s recourse LP 의 **α 에서의 최적 쌍대** 이고 α 는 노드 상자 B 안에 있으므로
    σ_ks ∈ [σlo_ks(B), σhi_ks(B)] = { σ_k : σ 는 어떤 α' ∈ B 에서 시나리오 s recourse LP 의 최적 쌍대 (H 성분 ≤ ϖᵁ, 더 낮출 수 없는 것) }
로 조일 수 있다. 범위는 시나리오·행마다 작은 KKT-MIP 2 개로 정확히 계산 (상자가 줄면 범위도 줄어 대부분 한 점으로 고정됨,
부모에서 고정된 성분은 자식에서 다시 풀지 않음).

노드 LP 에 넣는 행 (모두 계수만 노드마다 갱신)
  W  : σlo r_s ≤ ϖ_ks ≤ σhi r_s                                       (리더 인증서)
  F  : σlo d_s ≤ π̃_ks ≤ σhi d_s                                       (follower 인증서)
  RLT: (ϖ − σlo r)(α_i − l_i) ≥ 0, (ϖ − σlo r)(u_i − α_i) ≥ 0, (σhi r − ϖ)(α_i − l_i) ≥ 0, (σhi r − ϖ)(u_i − α_i) ≥ 0
       (ζW = α ϖ, ζL = α r 로 선형. 범위가 한 점이면 ζW = σ ζL 로 인증서 곱이 정확해진다)
  C  : 상자가 함의하는 상보성 — l_i > 0 이면 follower 1단계 정상성이 등식, Σ_i W_li u_i < w_l 이면 κ_l = 0

유효성
  - 최적 쌍대: Ω 의 최적해에서 (벌점 항 0) ϖˢ = r_s σ, π̃ˢ = d_s σ' (σ, σ' 는 α 에서의 recourse 최적 쌍대). r_s = 0, d_s = 0 이면
    H 성분을 0 으로 둘 수 있다 (쌍대목적 불증가, Fh 는 ≤ 방향).
  - "더 낮출 수 없는" 쌍대: H 행 k 의 우변이 양수 (α_i > 0) 면 최적 쌍대는 자동으로 그렇고 (낮출 수 있으면 쌍대목적이 줄어 모순),
    α_i = 0 이면 σ_k 를 낮춰도 실행가능·최적이 유지된다 → 일반성을 잃지 않음. Wcap (ϖ ≤ ϖᵁ r) 과 같은 종류의 논증.
  - 범위 MIP 는 objective_bound (바깥쪽) 에 1e-6 여유를 더해 쓴다.
  **주의: 시제품. 유효성은 위 논증 + 작은 S 에서 KKT 값과의 대조 (diag_dual_range.jl, test) 로만 확인했다.**

사용: r = nz_alpha_bnb_dr(nd, x̄; nworkers=12, time_limit=600.0, rel_gap=5e-3)        (dual_range=false 면 기존 완화, 비교용)
"""

struct _NZRangeMIP
    model::Model
    α::Vector{VariableRef}
    σ::Vector{VariableRef}
    y::Vector{VariableRef}
end

"시나리오 s 의 recourse KKT-MIP (α 는 상자 안의 변수): 최적 쌍대의 H 행 성분 범위 계산용"
function _nz_dr_build_mip(nd::NZData, x̄, s; optimizer)
    m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd)
    Hr = findall(nd.hrow .> 0)
    mdl = Model(optimizer); set_silent(mdl)
    set_optimizer_attribute(mdl, "MIPGap", 0.0); set_optimizer_attribute(mdl, "MIPGapAbs", 0.0)
    @variable(mdl, 0 <= α[i=1:nh] <= nd.hU[i])
    @constraint(mdl, [l=1:size(nd.W, 1)], sum(nd.W[l, i] * α[i] for i in 1:nh) <= nd.wvec[l])
    @variable(mdl, y[1:ny] >= 0); @variable(mdl, σ[1:m] >= 0)
    for k in Hr; isfinite(nd.varpiU[k]) && set_upper_bound(σ[k], nd.varpiU[k]); end
    sP = [@expression(mdl, nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) +
                           (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0) -
                           sum(nd.A[k, j] * y[j] for j in 1:ny if nd.A[k, j] != 0)) for k in 1:m]
    sD = [@expression(mdl, sum(nd.A[k, j] * σ[k] for k in 1:m if nd.A[k, j] != 0) - nd.c[j]) for j in 1:ny]
    @constraint(mdl, [k=1:m], sP[k] >= 0); @constraint(mdl, [j=1:ny], sD[j] >= 0)
    comp(var, slack) = (z = @variable(mdl, binary = true); @constraint(mdl, z => {var == 0}); @constraint(mdl, !z => {slack == 0}))
    for k in 1:m; comp(σ[k], sP[k]); end
    for j in 1:ny; comp(y[j], sD[j]); end
    for k in Hr                                   # 더 낮출 수 없는 쌍대: σ_k = 0 이거나, 그 행에 걸린 열 중 하나의 쌍대 제약이 등호
        J = [j for j in 1:ny if nd.A[k, j] > 0]
        z = @variable(mdl, binary = true); @constraint(mdl, z => {σ[k] == 0})
        w = @variable(mdl, [1:length(J)], binary = true)
        for (n, j) in enumerate(J); @constraint(mdl, w[n] => {sD[j] == 0}); end
        @constraint(mdl, z + sum(w) >= 1)
    end
    return _NZRangeMIP(mdl, α, σ, y)
end

mutable struct _NZDR
    mips::Vector{_NZRangeMIP}
    Hk::Vector{Int}                              # 상한이 양수인 좌표의 H 행
    jj::Vector{Int}                              # Hk[n] 의 Hr 안 위치 (ζW 의 첫 인덱스)
    whi::Matrix{ConstraintRef}; wlo::Matrix{ConstraintRef}
    phi::Matrix{ConstraintRef}; plo::Matrix{ConstraintRef}
    rlt::Array{ConstraintRef,3}                  # 4 × |Hk| × S
    stat::Dict{Int,ConstraintRef}                # 좌표 i → 정상성 역방향 행
    vcut::Vector{ConstraintRef}                  # 시나리오별 값 상한 ℓᵀŷˢ ≤ φ̄_s(B) r_s (없으면 빈 벡터)
    zk::Vector{Tuple{Int,Int,ConstraintRef,ConstraintRef}}   # (i, l, Zκ ≥ l_i κ_l, Zκ ≤ u_i κ_l)  — 상보성 RLT (없으면 빈 벡터)
    prlt::Array{ConstraintRef,3}                 # 4 × |Hk| × S: ζΠ = α π̃ 의 범위 × 상자 RLT (없으면 크기 0)
    nmip::Int
    nfail::Int                                   # 범위·값 MIP 가 Gurobi 오류로 실패한 횟수 (그 성분은 부모 값 유지)
end

"노드 LP (P) 에 범위 행의 자리를 만든다 (계수는 _nz_dr_set! 가 노드마다 채움)"
function _nz_dr_attach!(P::_NZSepLP, nd::NZData, x̄; optimizer, vcut::Bool=true, cslift::Bool=true)
    v = P.O.v; mdl = P.O.model; S, nh = nd.S, nz_nh(nd)
    Hr = v[:Hr]
    Hk = [k for k in Hr if nd.hU[nd.hrow[k]] > 0 && isfinite(nd.varpiU[k])]
    jjof = Dict(k => jj for (jj, k) in enumerate(Hr))
    jj = [jjof[k] for k in Hk]
    n = length(Hk)
    whi = Matrix{ConstraintRef}(undef, n, S); wlo = similar(whi); phi = similar(whi); plo = similar(whi)
    rlt = Array{ConstraintRef,3}(undef, 4, n, S)
    for (a, k) in enumerate(Hk), s in 1:S
        i = nd.hrow[k]; ϖ = v[:ϖ][k, s]; π = v[:π̃][k, s]; r = v[:r][s]; d = v[:d][s]
        zW = v[:ζW][jj[a], s]; zL = v[:ζL][i, s]
        whi[a, s] = @constraint(mdl, ϖ - nd.varpiU[k] * r <= 0.0); wlo[a, s] = @constraint(mdl, ϖ - 0.0 * r >= 0.0)
        phi[a, s] = @constraint(mdl, π - nd.varpiU[k] * d <= 0.0); plo[a, s] = @constraint(mdl, π - 0.0 * d >= 0.0)
        rlt[1, a, s] = @constraint(mdl, zW + 0.0 * ϖ + 0.0 * zL + 0.0 * r >= 0.0)
        rlt[2, a, s] = @constraint(mdl, -zW + 0.0 * ϖ + 0.0 * zL + 0.0 * r >= 0.0)
        rlt[3, a, s] = @constraint(mdl, -zW + 0.0 * ϖ + 0.0 * zL + 0.0 * r >= 0.0)
        rlt[4, a, s] = @constraint(mdl, zW + 0.0 * ϖ + 0.0 * zL + 0.0 * r >= 0.0)
    end
    stat = Dict{Int,ConstraintRef}()
    for i in 1:nh
        nd.hU[i] > 0 || continue
        ks = [k for k in Hr if nd.hrow[k] == i]
        stat[i] = @constraint(mdl, sum(nd.hcoef[k] * v[:π̃][k, s] for s in 1:S, k in ks)
                                   - sum(nd.W[l, i] * v[:κ][l] for l in 1:size(nd.W, 1)) >= -1e30)
    end
    # V: 시나리오별 값 상한. ŷˢ = r_s × (α 에서의 pessimistic 최적해) 이므로 ℓᵀŷˢ ≤ φ̄_s(B) r_s, φ̄_s(B) = max_{α∈B} φ_s(α)
    vc = ConstraintRef[]
    if vcut
        ny = nz_ny(nd)
        vc = [@constraint(mdl, sum(nd.ell[j] * v[:ŷ][j, s] for j in 1:ny if nd.ell[j] != 0) - 0.0 * v[:r][s] <= 1e30) for s in 1:S]
    end
    # X: follower 1단계 상보성의 RLT.  α_i (Wᵀκ − c0 − Σ_s Bᵀπ̃ˢ)_i = 0,  κ_l (w_l − W_l α) = 0 를 곱 변수로 선형화
    #    Zκ_il = α_i κ_l (κ ≥ 0 × 상자 인수),  ζΠ_as = α_i π̃_ks (범위 행 × 상자 인수: W 쪽 RLT 와 같은 4 행)
    #    쌍대가 고정된 상자에서는 ζΠ = σ ζF 가 되어 follower 의 primal 값 (ζF 로 완화된 것) 이 정상성으로 묶인다.
    zk = Tuple{Int,Int,ConstraintRef,ConstraintRef}[]
    prlt = Array{ConstraintRef,3}(undef, 0, 0, 0)
    if cslift
        nW = size(nd.W, 1); κ = v[:κ]; α = v[:α]
        Zκ = Dict{Tuple{Int,Int},VariableRef}()
        for i in 1:nh, l in 1:nW
            (nd.hU[i] > 0 && nd.W[l, i] != 0) || continue
            z = @variable(mdl, lower_bound = 0.0); Zκ[(i, l)] = z
            push!(zk, (i, l, @constraint(mdl, z - 0.0 * κ[l] >= 0.0), @constraint(mdl, z - nd.hU[i] * κ[l] <= 0.0)))
        end
        @constraint(mdl, [l in 1:nW], nd.wvec[l] * κ[l] - sum(nd.W[l, i] * Zκ[(i, l)] for i in 1:nh if haskey(Zκ, (i, l))) == 0.0)
        ζΠ = @variable(mdl, [1:n, 1:S], lower_bound = 0.0)
        prlt = Array{ConstraintRef,3}(undef, 4, n, S)
        for (a, k) in enumerate(Hk), s in 1:S
            i = nd.hrow[k]; π = v[:π̃][k, s]; d = v[:d][s]; zF = v[:ζF][i, s]; z = ζΠ[a, s]
            prlt[1, a, s] = @constraint(mdl, z + 0.0 * π + 0.0 * zF + 0.0 * d >= 0.0)
            prlt[2, a, s] = @constraint(mdl, -z + nd.hU[i] * π + 0.0 * zF + 0.0 * d >= 0.0)
            prlt[3, a, s] = @constraint(mdl, -z + 0.0 * π + nd.varpiU[k] * zF + 0.0 * d >= 0.0)
            prlt[4, a, s] = @constraint(mdl, z - nd.hU[i] * π - nd.varpiU[k] * zF + nd.varpiU[k] * nd.hU[i] * d >= 0.0)
        end
        for i in 1:nh
            nd.hU[i] > 0 || continue
            @constraint(mdl, sum(nd.W[l, i] * Zκ[(i, l)] for l in 1:nW if haskey(Zκ, (i, l))) - nd.c0[i] * α[i]
                             - sum(nd.hcoef[Hk[a]] * ζΠ[a, s] for a in 1:n, s in 1:S if nd.hrow[Hk[a]] == i) == 0.0)
        end
    end
    mips = [_nz_dr_build_mip(nd, x̄, s; optimizer=optimizer) for s in 1:S]
    return _NZDR(mips, Hk, jj, whi, wlo, phi, plo, rlt, stat, vc, zk, prlt, 0, 0)
end

"전역 범위 (루트 노드의 부모 값)"
function _nz_dr_global(nd::NZData, DR::_NZDR)
    n, S = length(DR.Hk), nd.S
    return zeros(n, S), [nd.varpiU[k] for k in DR.Hk, s in 1:S]
end

const _NZ_DR_FAIL_SHOWN = Threads.Atomic{Int}(0)

"""
범위·값 MIP 풀이. Gurobi 가 optimize 도중 오류를 내면 (worker 12 개 실행에서 간헐적으로 "Gurobi Error 10005: Unable to retrieve
attribute 'ObjBound'" 관찰, 단일 스레드 무작위 상자 4.6 만 회에서는 재현 안 됨 — 원인 미확인) 한 번 다시 풀고, 또 실패하면 false.
호출자는 그 성분을 부모 값으로 둔다 (부모 범위는 자식 상자에서도 유효하므로 완화는 유효, 조금 느슨해질 뿐).
실패 횟수는 DR.nfail 에 세어 결과 Dict 의 :range_fail 로 보고하고, 처음 3 번은 상자를 stderr 에 남긴다.
"""
function _nz_dr_opt!(DR::_NZDR, M::_NZRangeMIP, l, u, what)
    for attempt in 1:2
        try
            optimize!(M.model); DR.nmip += 1
            return true
        catch e
            e isa ErrorException || rethrow()
            if attempt == 2
                DR.nfail += 1
                if Threads.atomic_add!(_NZ_DR_FAIL_SHOWN, 1) < 3
                    println(stderr, "nz_alpha_bnb_dr: 범위 MIP 실패 ($what): ", e.msg, "\n  l = ", l, "\n  u = ", u)
                end
            end
        end
    end
    return false
end

"""상자 [l, u] 의 쌍대 범위. lo0, hi0 (부모의 범위) 를 복사해 조인 것과 witness 를 반환. 이미 한 점으로 고정된 성분은 다시 풀지 않는다.
witness: 범위 MIP 의 해 α (그 값을 달성하는 점). 부모의 witness 가 자식 상자 안에 있으면 그 값은 자식에서도 그대로이므로
MIP 를 다시 풀지 않는다 (상자를 한 좌표에서 나누면 witness 는 자식 중 하나에는 반드시 남는다).
상자 안에 실행가능한 α 가 없으면 nothing."""
function _nz_dr_ranges(DR::_NZDR, nd::NZData, l, u, lo0, hi0, wit0, phib0; fixtol=1e-4, deadline=Inf)
    lo = copy(lo0); hi = copy(hi0); phib = copy(phib0)
    wit = Dict{Tuple{Int,Int,Bool},Vector{Float64}}()
    S, nh = nd.S, nz_nh(nd)
    inbox(α) = all(l[i] - 1e-12 <= α[i] <= u[i] + 1e-12 for i in 1:nh)
    for s in 1:S
        M = DR.mips[s]
        todo = [a for a in eachindex(DR.Hk) if hi[a, s] - lo[a, s] > fixtol]
        isempty(todo) && (isempty(DR.vcut) || isfinite(phib[s])) && continue
        boxset = false
        for a in todo
            k = DR.Hk[a]
            for ismax in (false, true)
                key = (a, s, ismax)
                w = get(wit0, key, nothing)
                if w !== nothing && inbox(w)
                    wit[key] = w; continue
                end
                time() > deadline && return lo, hi, wit, phib   # 시간이 다 되면 지금까지 조인 범위만 (유효)
                if !boxset
                    for i in 1:nh; set_lower_bound(M.α[i], l[i]); set_upper_bound(M.α[i], u[i]); end
                    boxset = true
                end
                @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[k])
                _nz_dr_opt!(DR, M, l, u, "s=$s k=$k $(ismax ? "max" : "min")") || continue    # 실패: 부모 값 유지
                st = termination_status(M.model)
                st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return nothing
                st == MOI.OPTIMAL || error("nz_alpha_bnb_dr 범위 MIP (s=$s, k=$k): $st")
                b = objective_bound(M.model)
                ismax ? (hi[a, s] = min(hi[a, s], b + 1e-6)) : (lo[a, s] = max(lo[a, s], b - 1e-6))
                wit[key] = clamp.(value.(M.α), l, u)
            end
            hi[a, s] < lo[a, s] && (hi[a, s] = lo[a, s] = 0.5 * (lo[a, s] + hi[a, s]))   # 여유로 생긴 역전 (폭 ≤ 2e-6)
        end
        # V: φ̄_s(B) = max{ℓᵀy : α ∈ B, y 는 recourse 최적}. 쌍대가 모두 고정된 시나리오는 부모 값을 그대로 (유효, 리더 블록이 이미 정확)
        if !isempty(DR.vcut) && (!isempty(todo) || !isfinite(phib[s]))
            key = (0, s, true)
            w = get(wit0, key, nothing)
            if w !== nothing && inbox(w) && isfinite(phib[s])
                wit[key] = w
            else
                time() > deadline && return lo, hi, wit, phib
                if !boxset
                    for i in 1:nh; set_lower_bound(M.α[i], l[i]); set_upper_bound(M.α[i], u[i]); end
                    boxset = true
                end
                @objective(M.model, Max, sum(nd.ell[j] * M.y[j] for j in eachindex(M.y) if nd.ell[j] != 0))
                if _nz_dr_opt!(DR, M, l, u, "s=$s 값")                                          # 실패: 부모 값 유지
                    st = termination_status(M.model)
                    st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return nothing
                    st == MOI.OPTIMAL || error("nz_alpha_bnb_dr 값 MIP (s=$s): $st")
                    b = objective_bound(M.model)
                    phib[s] = min(phib[s], b + 1e-6 * max(1.0, abs(b)))
                    wit[key] = clamp.(value.(M.α), l, u)
                end
            end
        end
    end
    return lo, hi, wit, phib
end

"""
좌표 i (H 행 Hk[a]) 에서 시나리오 s 의 꺾임점 위치: 상자 안에서 쌍대가 아직 최댓값 hi 인 가장 큰 α_i.
(recourse 값은 α_i 에 오목이라 쌍대 σ_k 는 α_i 에 비증가 → 이 값을 넘으면 쌍대가 떨어진다.) 못 구하면 nothing.
"""
function _nz_dr_kink(DR::_NZDR, nd::NZData, l, u, lo, hi, a, s)
    M = DR.mips[s]; k = DR.Hk[a]; i = nd.hrow[k]
    gap = hi[a, s] - lo[a, s]
    gap > 1e-4 || return nothing
    for j in 1:nz_nh(nd); set_lower_bound(M.α[j], l[j]); set_upper_bound(M.α[j], u[j]); end
    set_lower_bound(M.σ[k], hi[a, s] - min(1e-3, 0.01 * gap) - 2e-6)       # hi 에는 1e-6 여유가 들어 있음
    @objective(M.model, Max, M.α[i])
    ok = _nz_dr_opt!(DR, M, l, u, "s=$s 꺾임점")
    t = ok && termination_status(M.model) == MOI.OPTIMAL ? objective_value(M.model) : nothing
    set_lower_bound(M.σ[k], 0.0)
    return t
end

"노드 LP 의 범위 행 계수를 상자 [l, u] 와 범위 (lo, hi) 로 갱신"
function _nz_dr_set!(P::_NZSepLP, DR::_NZDR, nd::NZData, l, u, lo, hi, phib; comp::Bool=true)
    v = P.O.v; S = nd.S
    for (s, c) in enumerate(DR.vcut)                      # ℓᵀŷˢ − φ̄_s r_s ≤ 0 (φ̄ 를 아직 모르면 비활성)
        if isfinite(phib[s])
            set_normalized_coefficient(c, v[:r][s], -phib[s]); set_normalized_rhs(c, 0.0)
        else
            set_normalized_rhs(c, 1e30)
        end
    end
    for (i, ll, clo, chi) in DR.zk                        # l_i κ_l ≤ Zκ_il ≤ u_i κ_l
        set_normalized_coefficient(clo, v[:κ][ll], -l[i]); set_normalized_coefficient(chi, v[:κ][ll], -u[i])
    end
    if length(DR.prlt) > 0
        for (a, k) in enumerate(DR.Hk), s in 1:S
            i = nd.hrow[k]; π = v[:π̃][k, s]; d = v[:d][s]; zF = v[:ζF][i, s]
            L, H, li, ui = lo[a, s], hi[a, s], l[i], u[i]
            for (n, cw, cl, cr) in ((1, -li, -L, L * li), (2, ui, L, -L * ui), (3, li, H, -H * li), (4, -ui, -H, H * ui))
                c = DR.prlt[n, a, s]
                set_normalized_coefficient(c, π, cw); set_normalized_coefficient(c, zF, cl); set_normalized_coefficient(c, d, cr)
            end
        end
    end
    for (a, k) in enumerate(DR.Hk), s in 1:S
        i = nd.hrow[k]; ϖ = v[:ϖ][k, s]; r = v[:r][s]; d = v[:d][s]; zL = v[:ζL][i, s]
        L, H, li, ui = lo[a, s], hi[a, s], l[i], u[i]
        set_normalized_coefficient(DR.whi[a, s], r, -H); set_normalized_coefficient(DR.wlo[a, s], r, -L)
        set_normalized_coefficient(DR.phi[a, s], d, -H); set_normalized_coefficient(DR.plo[a, s], d, -L)
        # R1: ζW − l ϖ − lo ζL + lo l r ≥ 0      R2: −ζW + u ϖ + lo ζL − lo u r ≥ 0
        # R3: −ζW + l ϖ + hi ζL − hi l r ≥ 0     R4: ζW − u ϖ − hi ζL + hi u r ≥ 0
        for (n, cw, cl, cr) in ((1, -li, -L, L * li), (2, ui, L, -L * ui), (3, li, H, -H * li), (4, -ui, -H, H * ui))
            c = DR.rlt[n, a, s]
            set_normalized_coefficient(c, ϖ, cw); set_normalized_coefficient(c, zL, cl); set_normalized_coefficient(c, r, cr)
        end
    end
    for (i, c) in DR.stat                                 # α_i > 0 ⇒ Σ_s Bᵀπ̃ + c0 = Wᵀκ (≤ 는 Fh 로 이미 있음)
        set_normalized_rhs(c, comp && l[i] > 1e-9 ? -nd.c0[i] : -1e30)
    end
    κ = v[:κ]
    for ll in 1:size(nd.W, 1)                             # Wα < w ⇒ κ_l = 0
        if comp && dot(nd.W[ll, :], u) < nd.wvec[ll] - 1e-9
            set_upper_bound(κ[ll], 0.0)
        elseif has_upper_bound(κ[ll])
            delete_upper_bound(κ[ll])
        end
    end
end


"""
    nz_alpha_bnb_dr(nd, x̄; dual_range=true, comp=true, ...)

nz_alpha_bnb 의 탐색 (worker 병렬, best-first + diving, 위반 RLT 행 분리, Ipopt 휴리스틱) 을 그대로 쓰고 노드마다 쌍대 범위를 조인다.
노드 = (l, u, 부모 상한, lo, hi). L-shaped, ϖ 분기, barrier 계열 옵션은 뺐다 (dual simplex 전용).
"""
function nz_alpha_bnb_dr(nd::NZData, x̄; nworkers=Threads.nthreads() - 2, time_limit=300.0, rel_gap=1e-4,
                         dive_max=3, maxrounds=30, child_rounds::Int=1, min_width=1e-7, ipopt_time=60.0,
                         verbose=true, log_every=30.0, target=nothing, heuristic::Bool=true, presolve::Bool=true,
                         branch_point::Symbol=:mid, sep_maxadd::Int=5000, dual_range::Bool=true, comp::Bool=true,
                         kink_eta::Float64=0.01, kink_width::Float64=Inf, vcut::Bool=true, cslift::Bool=true)
    # vcut: 시나리오별 값 상한 (V), cslift: follower 1단계 상보성의 RLT (X). 둘 다 dual_range = true 에서만.
    # branch_point = :kink: 분기 좌표에서 쌍대 범위가 고정되지 않은 시나리오의 꺾임점 t 를 찾아 [l, t−η], [t−η, t+η], [t+η, u] 로
    #   3 분할 (η = kink_eta). 꺾임점이 없거나 못 찾거나 좌표 폭이 kink_width 보다 크면 중점 분할. dual_range = true 에서만.
    branch_point in (:alphahat, :mid, :kink) || error("branch_point = :alphahat | :mid | :kink")
    n_kink = Threads.Atomic{Int}(0)
    nfix = 0
    if presolve
        hUp, nfix = nz_presolve_hU(nd, x̄)
        nfix > 0 && (nd = nz_with_hU(nd, hUp))
    end
    Threads.nthreads(:interactive) >= 1 || error("nz_alpha_bnb_dr: julia -t $(nworkers + 2),1 로 실행하세요")
    t0 = time(); t_end = t0 + time_limit
    nh, S = nz_nh(nd), nd.S
    Hr = findall(nd.hrow .> 0)
    mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true);
                     MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    envs = _nz_envs(nworkers + 1)
    Ps = [_nz_build_sep(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    DRs = dual_range ? [_nz_dr_attach!(Ps[i], nd, x̄; optimizer=mk(envs[i]), vcut=vcut, cslift=cslift) for i in 1:nworkers] : nothing
    Fs = [nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    tol(v) = rel_gap * (isfinite(v) ? max(1.0, abs(v)) : 1.0)
    boxok(l) = all(nd.W * l .<= nd.wvec .+ 1e-9)

    lk = ReentrantLock()
    t_relax = Threads.Atomic{Float64}(0.0); t_eval = Threads.Atomic{Float64}(0.0); t_range = Threads.Atomic{Float64}(0.0)
    lo0, hi0 = dual_range ? _nz_dr_global(nd, DRs[1]) : (zeros(0, 0), zeros(0, 0))
    open = Any[(zeros(nh), copy(nd.hU), Inf, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S))]
    inflight = fill(-Inf, nworkers)
    LB = Ref(-Inf); best_α = Ref(zeros(nh))
    nodes = Ref(0); done = Ref(false); root_UB = Ref(Inf)
    root_rows = Ref{Any}(nothing)
    node_cands = Tuple{Float64,Vector{Float64}}[]
    ipopt_calls = Ref(0)
    pruned_ub = Ref(-Inf)
    nfixed_sum = Ref(0.0)                      # 노드별 "고정된 쌍대 성분 비율" 의 합 (평균 보고용)

    globalUB() = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open), maximum(inflight), pruned_ub[])
    offer!(z, a) = lock(lk) do
        z > LB[] && (LB[] = z; best_α[] = copy(a))
    end

    function worker(wid)
        P = Ps[wid]; F = Fs[wid]; DR = dual_range ? DRs[wid] : nothing
        localnode = nothing; dive_left = 0; synced = false
        while true
            yield()
            node = nothing
            lock(lk)
            try
                ub_now = globalUB()
                if time() >= t_end || ub_now - LB[] <= tol(LB[]) ||
                   (target !== nothing && (LB[] >= target || ub_now <= target))
                    done[] = true
                end
                if done[]
                    localnode !== nothing && push!(open, localnode)
                    return
                end
                if localnode !== nothing && dive_left > 0 && localnode[3] > LB[] + tol(LB[])
                    node = localnode; dive_left -= 1
                else
                    if localnode !== nothing
                        localnode[3] > LB[] + tol(LB[]) ? push!(open, localnode) :
                                                          (pruned_ub[] = max(pruned_ub[], localnode[3]))
                    end
                    if !isempty(open)
                        node = popat!(open, argmax([n[3] for n in open]))
                        dive_left = dive_max
                    end
                end
                localnode = nothing
                node !== nothing && (inflight[wid] = node[3])
            finally
                unlock(lk)
            end
            if node === nothing
                sleep(0.05); continue
            end
            l, u, ubp, lop, hip, witp, phibp = node
            if !synced && root_rows[] !== nothing && isempty(P.allrows) && t_end - time() > 30.0
                for (i, j, islo, b) in root_rows[]; _nz_sep_add!(P, i, j, islo, b); end
                synced = true
            end
            z = -Inf; α̂ = nothing; children = nothing; zE = -Inf; unfinished = false
            pruned_val = -Inf; lo = lop; hi = hip; wit = witp; phib = phibp; fixfrac = NaN
            boxok(l) && ubp <= LB[] + tol(LB[]) && (pruned_val = ubp)
            if ubp > LB[] + tol(LB[]) && boxok(l)
                _nz_sep_set_box!(P, l, u)
                _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                empty_box = false
                if dual_range
                    tg = @elapsed (rg = _nz_dr_ranges(DR, nd, l, u, lop, hip, witp, phibp; deadline=t_end))
                    Threads.atomic_add!(t_range, tg)
                    if rg === nothing
                        empty_box = true                 # 상자 안에 실행가능한 α 가 없음
                    else
                        lo, hi, wit, phib = rg
                        _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp)
                        fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                    end
                end
                if !empty_box
                    set_time_limit_sec(P.O.model, max(t_end - time(), 0.1))
                    tr = @elapsed (z = _nz_sep_solve!(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd))
                    Threads.atomic_add!(t_relax, tr)
                    unfinished = isnan(z)
                    numfail = z == Inf
                    if numfail
                        z = ubp
                        i = argmax(u .- l)
                        if u[i] - l[i] > min_width
                            t = 0.5 * (l[i] + u[i]); u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                            children = ((copy(l), u1, z, lo, hi, wit, phib), (l2, copy(u), z, lo, hi, wit, phib))
                        else
                            pruned_val = z
                        end
                    end
                    z = unfinished ? ubp : min(z, ubp)
                    (!unfinished && !numfail && z <= LB[] + tol(LB[])) && (pruned_val = z)
                    if !unfinished && !numfail && z > LB[] + tol(LB[])
                        v = P.O.v
                        α̂ = clamp.(value.(v[:α]), l, u)
                        rv = value.(v[:r]); dv = value.(v[:d])
                        ϖv = zeros(size(v[:ϖ]))
                        for k in Hr, s in 1:S; ϖv[k, s] = value(v[:ϖ][k, s]); end
                        ζLv = value.(v[:ζL]); ζFv = value.(v[:ζF]); ζWv = value.(v[:ζW])
                        set_time_limit_sec(F.model, max(t_end - time(), 0.1))
                        te = @elapsed (zE = nz_eval_alpha!(F, nd, α̂))
                        Threads.atomic_add!(t_eval, te)
                        isnan(zE) && (zE = -Inf)
                        viol = zeros(nh)
                        for i in 1:nh
                            u[i] - l[i] > min_width || continue
                            viol[i] = sum(abs(ζLv[i, s] - α̂[i] * rv[s]) + abs(ζFv[i, s] - α̂[i] * dv[s]) for s in 1:S)
                            for (jj, k) in enumerate(Hr)
                                nd.hrow[k] == i || continue
                                viol[i] += sum(abs(ζWv[jj, s] - α̂[i] * ϖv[k, s]) for s in 1:S)
                            end
                        end
                        i = argmax(viol)
                        if viol[i] > 1e-9
                            # 꺾임점 분기: 좌표 i 에서 범위가 고정되지 않은 시나리오 중 곱 위반이 가장 큰 것의 꺾임점 t 를 찾아
                            #   [l, t − η] / [t − η, t + η] (얇은 판) / [t + η, u] 로 나눈다. 양옆 자식에서는 그 시나리오의 쌍대가 한 점으로 고정됨.
                            if branch_point == :kink && dual_range && u[i] - l[i] <= kink_width
                                best = -1.0; ab = 0; sb = 0
                                for (a, k) in enumerate(DR.Hk)
                                    nd.hrow[k] == i || continue
                                    for s in 1:S
                                        hi[a, s] - lo[a, s] > 1e-4 || continue
                                        sc = abs(ζWv[DR.jj[a], s] - α̂[i] * ϖv[k, s]) + abs(ζFv[i, s] - α̂[i] * dv[s]) +
                                             abs(ζLv[i, s] - α̂[i] * rv[s]) + 1e-9 * (rv[s] + dv[s])
                                        sc > best && (best = sc; ab = a; sb = s)
                                    end
                                end
                                if ab > 0
                                    tk = @elapsed (t = _nz_dr_kink(DR, nd, l, u, lo, hi, ab, sb))
                                    Threads.atomic_add!(t_range, tk)
                                    if t !== nothing
                                        bp = [l[i]]
                                        for b in (t - kink_eta, t + kink_eta); b > bp[end] + 1e-9 && b < u[i] - 1e-9 && push!(bp, b); end
                                        push!(bp, u[i])
                                        if length(bp) >= 3
                                            kids = Any[]
                                            for n in 1:length(bp)-1
                                                l2 = copy(l); u2 = copy(u); l2[i] = bp[n]; u2[i] = bp[n+1]
                                                push!(kids, (l2, u2, z, lo, hi, wit, phib))
                                            end
                                            # 꺾임점을 품은 자식 (얇은 판) 을 먼저 (diving)
                                            mid = findfirst(c -> c[1][i] <= t <= c[2][i] && c[2][i] - c[1][i] <= 2 * kink_eta + 1e-9, kids)
                                            if mid !== nothing
                                                tmp = kids[1]; kids[1] = kids[mid]; kids[mid] = tmp
                                            end
                                            children = Tuple(kids); Threads.atomic_add!(n_kink, 1)
                                        end
                                    end
                                end
                            end
                            if children === nothing
                                width = u[i] - l[i]; t = α̂[i]
                                (branch_point != :alphahat || t - l[i] < 0.1 * width || u[i] - t < 0.1 * width) && (t = 0.5 * (l[i] + u[i]))
                                u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                                children = ((copy(l), u1, z, lo, hi, wit, phib), (l2, copy(u), z, lo, hi, wit, phib))
                            end
                        else
                            zE = max(zE, z)                # 곱 위반이 없으면 완화가 정확
                        end
                    end
                end
            end
            lock(lk)
            try
                if unfinished
                    push!(open, node); inflight[wid] = -Inf; dive_left = 0
                    continue
                end
                nodes[] += 1
                isnan(fixfrac) || (nfixed_sum[] += fixfrac)
                pruned_ub[] = max(pruned_ub[], pruned_val)
                if ubp == Inf
                    root_UB[] = z
                    root_rows[] = [(r[1], r[6], r[2], r[3]) for r in P.allrows]
                end
                zE > LB[] && (LB[] = zE; best_α[] = copy(α̂))
                if α̂ !== nothing
                    push!(node_cands, (z, copy(α̂)))
                    length(node_cands) > 200 && popfirst!(node_cands)
                end
                if children !== nothing
                    for c in children[2:end]; push!(open, c); end
                    localnode = children[1]
                else
                    dive_left = 0
                end
                inflight[wid] = -Inf
            finally
                unlock(lk)
            end
        end
    end

    function heuristic_task()
        Fh = nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[end]))
        tried = Set{Vector{Float64}}()
        a0 = _nz_capH(fill(minimum(nd.wvec) / nh, nh), nd)
        while !done[] && t_end - time() > 15.0
            push!(tried, round.(a0; digits=3))
            aloc = nz_ipopt_local_alpha(nd, x̄, Fh, a0; max_time=min(ipopt_time, t_end - time()), deadline=t_end)
            ipopt_calls[] += 1
            if aloc !== nothing
                z = nz_eval_alpha!(Fh, nd, aloc)
                isnan(z) || offer!(z, aloc)
            end
            a0 = nothing
            while a0 === nothing && !done[]
                lock(lk)
                try
                    best = -Inf
                    for (zc, a) in node_cands
                        (zc > best && !(round.(a; digits=3) in tried)) && (best = zc; a0 = a)
                    end
                finally
                    unlock(lk)
                end
                a0 === nothing && sleep(0.5)
            end
            a0 === nothing && break
            a0 = _nz_capH(a0, nd)
        end
    end

    tasks = [Threads.@spawn worker(i) for i in 1:nworkers]
    heuristic && push!(tasks, Threads.@spawn heuristic_task())
    t_log = time()
    while true
        sleep(0.2)
        fin = false; UBn = Inf
        lock(lk)
        try
            UBn = globalUB()
            idle = isempty(open) && all(inflight .== -Inf)
            fin = (time() >= t_end) || (UBn - LB[] <= tol(LB[])) || (idle && nodes[] > 0) ||
                  (target !== nothing && (LB[] >= target || UBn <= target))
            any(istaskfailed, tasks) && (fin = true)        # worker 가 예외로 죽으면 바로 멈추고 아래 wait 에서 예외를 올림
            fin && (done[] = true)
        finally
            unlock(lk)
        end
        if verbose && time() - t_log > log_every
            @printf("    [nz-α-B&B-dr] t=%6.1fs nodes=%5d open=%5d LB=%.6f UB=%.6f gap=%.3e\n",
                    time() - t0, nodes[], length(open), LB[], UBn, (UBn - LB[]) / max(1.0, abs(LB[])))
            flush(stdout); t_log = time()
        end
        fin && break
    end
    foreach(wait, tasks)
    UB = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open), pruned_ub[])
    return Dict(:LB => LB[], :UB => UB, :α => best_α[], :is_exact => (UB - LB[] <= tol(LB[])),
                :nodes => nodes[], :time => time() - t0, :root_UB => root_UB[], :ipopt_calls => ipopt_calls[],
                :presolve_fixed => nfix, :t_relax => t_relax[], :t_eval => t_eval[], :t_range => t_range[],
                :range_mips => dual_range ? sum(D.nmip for D in DRs) : 0, :range_fail => dual_range ? sum(D.nfail for D in DRs) : 0,
                :fixfrac => nodes[] > 0 ? nfixed_sum[] / nodes[] : NaN, :kink_branches => n_kink[],
                :numerr => _NZ_NUMERR[], :numerr_fail => _NZ_NUMERR_FAIL[])
end
