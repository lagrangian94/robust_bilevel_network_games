"""
nz_alpha_bnb_dr.jl — α-B&B (nz_alpha_bnb.jl) + **상자별 쌍대 범위 조임** 시제품. nz_alpha_bnb.jl 을 먼저 include.

기존 α-B&B 의 노드 완화는 recourse 쌍대해 (리더 블록 ϖˢ, follower 블록 π̃ˢ; 원고의 recourse dual solution) 의 예약 행 (H 행) 성분에 전역 범위 [0, ϖᵁ] 만 쓴다.
정확한 해에서 ϖˢ/r_s, π̃ˢ/d_s 는 시나리오 s recourse LP 의 **α 에서의 최적 쌍대** 이고 α 는 노드 상자 B 안에 있으므로
    σ_ks ∈ [σlo_ks(B), σhi_ks(B)] = { σ_k : σ 는 어떤 α' ∈ B 에서 시나리오 s recourse LP 의 최적 쌍대 (H 성분 ≤ ϖᵁ, 더 낮출 수 없는 것) }
로 조일 수 있다. 범위는 시나리오·행마다 작은 KKT-MIP 2 개로 정확히 계산 (상자가 줄면 범위도 줄어 대부분 한 점으로 고정됨,
부모에서 고정된 성분은 자식에서 다시 풀지 않음).

노드 LP 에 넣는 행 (모두 계수만 노드마다 갱신)
  W  : σlo r_s ≤ ϖ_ks ≤ σhi r_s                                       (리더 블록의 recourse 쌍대해)
  F  : σlo d_s ≤ π̃_ks ≤ σhi d_s                                       (follower 블록의 recourse 쌍대해)
  RLT: (ϖ − σlo r)(α_i − l_i) ≥ 0, (ϖ − σlo r)(u_i − α_i) ≥ 0, (σhi r − ϖ)(α_i − l_i) ≥ 0, (σhi r − ϖ)(u_i − α_i) ≥ 0
       (ζW = α ϖ, ζL = α r 로 선형. 범위가 한 점이면 ζW = σ ζL 로 subgradient 곱이 정확해진다)
  C  : 상자가 함의하는 상보성 — l_i > 0 이면 follower 1단계 정상성이 등식, Σ_i W_li u_i < w_l 이면 κ_l = 0
  V  : 시나리오별 값 상한 ℓᵀŷˢ ≤ φ̄_s(B) r_s          X : follower 1단계 상보성의 RLT (곱 변수 Zκ = α κ, ζΠ = α π̃)
2026-10-13 에 더한 것 (모두 기본 켬. 진단과 측정은 docs/dependent_ambiguity/alpha_bnb_nonzero_analysis.md §10.13)
  FWD: follower 블록의 시나리오별 약쌍대  cᵀȳˢ ≤ (uˢ + gˢx̄)ᵀπ̃ˢ + Σ_H hcoef ζΠ_ks   (리더 블록의 WD 와 같은 종류)
  KMC: κ_l ≤ κmax_l 과 Zκ = α κ 의 위쪽 McCormick
  DD : 쌍대 **차이** 의 범위  dlo d_s ≤ π̃_ks − π̃_k′s ≤ dhi d_s (리더: ϖ, r_s) — 성분별 범위가 놓치는, 같은 시나리오의 두 쌍대가 함께 움직이는 관계
  상보성 분기 (cbranch): l_i = 0 인 좌표에서 α̂_i > 0, 정상성 여유 ĝ_i > 0 이면 {α_i = 0} | {g_i = 0} 로 분기
  분리 행 제자리 갱신 (sticky), 노드 LP 허용오차 1e-7 (lp_tol), Gurobi 호출의 GC-safe 구간 (gurobi_gcsafe.jl)

유효성
  - 최적 쌍대: Ω 의 최적해에서 (벌점 항 0) ϖˢ = r_s σ, π̃ˢ = d_s σ' (σ, σ' 는 α 에서의 recourse 최적 쌍대). r_s = 0, d_s = 0 이면
    H 성분을 0 으로 둘 수 있다 (쌍대목적 불증가, Fh 는 ≤ 방향).
  - "더 낮출 수 없는" 쌍대: H 행 k 의 우변이 양수 (α_i > 0) 면 최적 쌍대는 자동으로 그렇고 (낮출 수 있으면 쌍대목적이 줄어 모순),
    α_i = 0 이면 σ_k 를 낮춰도 실행가능·최적이 유지된다 → 일반성을 잃지 않음. Wcap (ϖ ≤ ϖᵁ r) 과 같은 종류의 논증.
  - 범위 MIP 는 objective_bound (바깥쪽) 에 1e-6 여유를 더해 쓴다.
  **주의: 시제품. 유효성은 위 논증 + 작은 S 에서 KKT 값과의 대조 (diag_dual_range.jl, test) 로만 확인했다.**

사용: r = nz_alpha_bnb_dr(nd, x̄; nworkers=12, time_limit=600.0, rel_gap=1e-3)
      인자 전체와 기본값은 nz_alpha_bnb_dr 의 docstring. Benders 에서는 nz_benders.jl 의 `bnb = :dr` (기본).
"""

include(joinpath(@__DIR__, "nz_resp_heuristic.jl"))  # primal 휴리스틱: 완화 해의 belief 에서 follower 의 실제 최적 반응
include(joinpath(@__DIR__, "gurobi_gcsafe.jl"))     # Gurobi 풀이 호출을 GC-safe 구간에서 (worker 병렬 실행의 GC 대기 제거, 끄기: NZ_GCSAFE=0)

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
    zP::Matrix{VariableRef}                      # ζΠ (|Hk| × S, 없으면 크기 0)
    fwd::Vector{ConstraintRef}                   # follower 블록의 시나리오별 약쌍대 (없으면 빈 벡터)
    kmc::Vector{Tuple{Int,Int,ConstraintRef,ConstraintRef,Float64}}         # Zκ = α κ 의 κ 상한 쪽 McCormick (i, l, 위 행, 아래 행, κmax)
    pairs::Vector{Tuple{Int,Int}}                                           # 쌍대 차이 범위의 성분 쌍 (a, a′), a < a′
    ddF::Array{ConstraintRef,3}                                             # 2 × |pairs| × S: dlo d ≤ π̃_a − π̃_a′ ≤ dhi d
    ddL::Array{ConstraintRef,3}                                             # 2 × |pairs| × S: dlo r ≤ ϖ_a − ϖ_a′ ≤ dhi r
    nmip::Int
    nfail::Int                                   # 범위·값 MIP 가 Gurobi 오류로 실패한 횟수 (그 성분은 부모 값 유지)
end

"노드 LP (P) 에 범위 행의 자리를 만든다 (계수는 _nz_dr_set! 가 노드마다 채움)"
function _nz_dr_attach!(P::_NZSepLP, nd::NZData, x̄; optimizer, vcut::Bool=true, cslift::Bool=true, fwd::Bool=true,
                        kmc::Bool=true, dd::Bool=true)
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
    kmcv = Tuple{Int,Int,ConstraintRef,ConstraintRef,Float64}[]
    fwd = fwd && cslift; kmc = kmc && cslift             # 둘 다 X 의 곱 변수 (ζΠ, Zκ) 를 쓴다
    prlt = Array{ConstraintRef,3}(undef, 0, 0, 0)
    zPm = Matrix{VariableRef}(undef, 0, 0)
    if cslift
        nW = size(nd.W, 1); κ = v[:κ]; α = v[:α]
        Zκ = Dict{Tuple{Int,Int},VariableRef}()
        for i in 1:nh, l in 1:nW
            (nd.hU[i] > 0 && nd.W[l, i] != 0) || continue
            z = @variable(mdl, lower_bound = 0.0); Zκ[(i, l)] = z
            push!(zk, (i, l, @constraint(mdl, z - 0.0 * κ[l] >= 0.0), @constraint(mdl, z - nd.hU[i] * κ[l] <= 0.0)))
        end
        @constraint(mdl, [l in 1:nW], nd.wvec[l] * κ[l] - sum(nd.W[l, i] * Zκ[(i, l)] for i in 1:nh if haskey(Zκ, (i, l)) && nd.W[l, i] != 0) == 0.0)
        if kmc
            # KMC: κ_l 의 상한 κmax_l 과 그 McCormick. κ_l > 0 이면 묶음 l 이 꽉 차 있고 (상보성) 그 안에 α_i > 0 인 좌표가 있어 정상성이 등식:
            #   W_li κ_l = Σ_s Bᵀπ̃ + c0_i ≤ Σ_k hcoef_k ϖᵁ_k + c0_i  (π̃_ks ≤ ϖᵁ_k d_s, Σ d = 1).  κ ≥ 0 뿐이면 α̂_i = 0 인 넓은 좌표에서
            #   Zκ_il 이 u_i κ_l 까지 자유로워 κ_l (w_l − W_l α) = 0 의 선형화가 아무것도 묶지 못한다.
            all(nd.W .>= 0) || error("_nz_dr_attach!: kmc — W 에 음수")
            κmax = zeros(nW)
            for l in 1:nW
                cand = [(sum(nd.hcoef[k] * nd.varpiU[k] for k in Hk if nd.hrow[k] == ii; init=0.0) + nd.c0[ii]) / nd.W[l, ii]
                        for ii in 1:nh if nd.hU[ii] > 0 && nd.W[l, ii] > 0]
                κmax[l] = isempty(cand) ? 0.0 : max(0.0, maximum(cand))
                @constraint(mdl, κ[l] <= κmax[l])
            end
            for (i, l, _, _) in zk
                z = Zκ[(i, l)]
                c1 = @constraint(mdl, z - 0.0 * κ[l] - κmax[l] * α[i] <= 0.0)                          # (κmax − κ)(α − l_i) ≥ 0
                c2 = @constraint(mdl, z - nd.hU[i] * κ[l] - κmax[l] * α[i] >= -κmax[l] * nd.hU[i])     # (κmax − κ)(u_i − α) ≥ 0
                push!(kmcv, (i, l, c1, c2, κmax[l]))
            end
        end
        ζΠ = @variable(mdl, [1:n, 1:S], lower_bound = 0.0)
        zPm = Matrix{VariableRef}(ζΠ)
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
            @constraint(mdl, sum(nd.W[l, i] * Zκ[(i, l)] for l in 1:nW if haskey(Zκ, (i, l)) && nd.W[l, i] != 0) - nd.c0[i] * α[i]
                             - sum(nd.hcoef[Hk[a]] * ζΠ[a, s] for a in 1:n, s in 1:S if nd.hrow[Hk[a]] == i) == 0.0)
        end
    end
    # FWD: follower 블록의 시나리오별 약쌍대  cᵀȳˢ ≤ (uˢ + gˢ x̄)ᵀπ̃ˢ + Σ_{k∈H} hcoef_k ζΠ_ks   (ζΠ = α π̃, X 가 있어야 함)
    #   정확한 해에서 ȳˢ = d_s y, π̃ˢ = d_s σ (y: α 에서 실행가능, σ: 쌍대 실행가능) 이므로 약쌍대성 cᵀy ≤ (u + g x̄ + Bα)ᵀσ 에 d_s 를 곱한 것.
    #   리더 블록의 WD (nz_omega.jl) 와 같은 종류. Flam (합친 강쌍대) + X 의 항등식은 Σ_s [쌍대목적_s − primal_s] ≤ 0 만 주므로
    #   꺾임을 품은 상자에서는 시나리오끼리 부호가 다른 gap 을 상쇄할 수 있다 — 이 행이 시나리오마다 gap ≥ 0 을 강제한다.
    fw = ConstraintRef[]
    if fwd
        all(nd.hU[nd.hrow[k]] <= 0 || k in Hk for k in Hr) || error("_nz_dr_attach!: fwd — ζΠ 가 없는 H 행이 있음 (ϖᵁ 무한)")
        m_, ny_ = nz_m(nd), nz_ny(nd); ȳ = v[:ȳ]; πt = v[:π̃]
        coefπ(k, s) = nd.u[k, s] + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0)
        fw = [@constraint(mdl, sum(nd.c[j] * ȳ[j, s] for j in 1:ny_ if nd.c[j] != 0)
                               - sum(coefπ(k, s) * πt[k, s] for k in 1:m_ if coefπ(k, s) != 0)
                               - sum(nd.hcoef[Hk[a]] * zPm[a, s] for a in 1:n) <= 0.0) for s in 1:S]
    end
    # DD: 쌍대 **차이** 의 범위. 성분별 범위 [lo, hi] 는 상자 (곱집합) 라, 같은 시나리오의 두 성분이 함께 움직이는 것을 놓친다
    #   (예: 같은 고객의 두 점포 예약이 모두 묶이면 두 쌍대의 차이는 수송비 차이로 고정). 쌍 (a, a′) 마다 같은 KKT-MIP 로
    #   dlo ≤ σ_a − σ_a′ ≤ dhi 를 구해  dlo d_s ≤ π̃_a − π̃_a′ ≤ dhi d_s (리더: ϖ, r_s) 를 넣는다. 유효성은 성분별 범위와 같은 논증.
    prs = Tuple{Int,Int}[]; ddF = Array{ConstraintRef,3}(undef, 0, 0, 0); ddL = Array{ConstraintRef,3}(undef, 0, 0, 0)
    if dd
        prs = [(a, b) for a in 1:n for b in a+1:n]
        ddF = Array{ConstraintRef,3}(undef, 2, length(prs), S); ddL = Array{ConstraintRef,3}(undef, 2, length(prs), S)
        for (p, (a, b)) in enumerate(prs), s in 1:S
            ka, kb = Hk[a], Hk[b]
            ddF[1, p, s] = @constraint(mdl, v[:π̃][ka, s] - v[:π̃][kb, s] + nd.varpiU[kb] * v[:d][s] >= 0.0)
            ddF[2, p, s] = @constraint(mdl, v[:π̃][ka, s] - v[:π̃][kb, s] - nd.varpiU[ka] * v[:d][s] <= 0.0)
            ddL[1, p, s] = @constraint(mdl, v[:ϖ][ka, s] - v[:ϖ][kb, s] + nd.varpiU[kb] * v[:r][s] >= 0.0)
            ddL[2, p, s] = @constraint(mdl, v[:ϖ][ka, s] - v[:ϖ][kb, s] - nd.varpiU[ka] * v[:r][s] <= 0.0)
        end
    end
    mips = [_nz_dr_build_mip(nd, x̄, s; optimizer=optimizer) for s in 1:S]
    return _NZDR(mips, Hk, jj, whi, wlo, phi, plo, rlt, stat, vc, zk, prlt, zPm, fw, kmcv, prs, ddF, ddL, 0, 0)
end

"전역 범위 (루트 노드의 부모 값)"
function _nz_dr_global(nd::NZData, DR::_NZDR)
    n, S = length(DR.Hk), nd.S
    return zeros(n, S), [nd.varpiU[k] for k in DR.Hk, s in 1:S]
end

"쌍대 차이의 전역 범위 (dd 가 꺼져 있으면 0 × S 행렬)"
function _nz_dr_dglobal(nd::NZData, DR::_NZDR)
    S = nd.S
    return [-nd.varpiU[DR.Hk[b]] for (a, b) in DR.pairs, s in 1:S], [nd.varpiU[DR.Hk[a]] for (a, b) in DR.pairs, s in 1:S]
end

"""
상자 [l, u] 의 쌍대 차이 범위 (DD). lo, hi 는 이 상자의 성분별 범위 (_nz_dr_ranges 의 결과), dlo0, dhi0 는 부모의 차이 범위,
wit0 는 부모의 witness, wit 는 이 노드의 witness (여기에 차이 MIP 의 witness 를 더한다. 키는 (−쌍 번호, s, ismax)).
두 성분이 모두 고정되지 않은 쌍만 MIP 를 푼다 (하나라도 고정이면 성분별 범위가 함의하는 값이 정확).
"""
function _nz_dr_dranges!(DR::_NZDR, nd::NZData, l, u, lo, hi, dlo0, dhi0, wit0, wit; fixtol=1e-4, deadline=Inf, mip_deadline=Inf)
    dlo = copy(dlo0); dhi = copy(dhi0)
    S, nh = nd.S, nz_nh(nd)
    inbox(α) = all(l[i] - 1e-12 <= α[i] <= u[i] + 1e-12 for i in 1:nh)
    for s in 1:S
        M = DR.mips[s]; boxset = false
        for (p, (a, b)) in enumerate(DR.pairs)
            dlo[p, s] = max(dlo[p, s], lo[a, s] - hi[b, s]); dhi[p, s] = min(dhi[p, s], hi[a, s] - lo[b, s])
            (hi[a, s] - lo[a, s] > fixtol && hi[b, s] - lo[b, s] > fixtol && dhi[p, s] - dlo[p, s] > fixtol) || continue
            ka, kb = DR.Hk[a], DR.Hk[b]
            for ismax in (false, true)
                key = (-p, s, ismax)
                w = get(wit0, key, nothing)
                if w !== nothing && inbox(w)
                    wit[key] = w; continue
                end
                time() > deadline && return dlo, dhi
                if !boxset
                    for i in 1:nh; set_lower_bound(M.α[i], l[i]); set_upper_bound(M.α[i], u[i]); end
                    boxset = true
                end
                @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[ka] - M.σ[kb])
                isfinite(mip_deadline) && set_time_limit_sec(M.model, max(mip_deadline - time(), 0.01))
                _nz_dr_opt!(DR, M, l, u, "s=$s 쌍 ($ka, $kb) $(ismax ? "max" : "min")") || continue
                termination_status(M.model) == MOI.OPTIMAL || continue          # 상자가 비었으면 _nz_dr_ranges 가 이미 걸렀음. 그 밖에는 부모 값 유지
                bd = objective_bound(M.model)
                ismax ? (dhi[p, s] = min(dhi[p, s], bd + 1e-6)) : (dlo[p, s] = max(dlo[p, s], bd - 1e-6))
                wit[key] = clamp.(value.(M.α), l, u)
            end
            dhi[p, s] < dlo[p, s] && (dhi[p, s] = dlo[p, s] = 0.5 * (dlo[p, s] + dhi[p, s]))
        end
    end
    return dlo, dhi
end

const _NZ_DR_FAIL_SHOWN = Threads.Atomic{Int}(0)

"시간 제한으로 끝난 범위 MIP 의 쌍대 bound. 아주 짧은 제한에서는 Gurobi 가 bound 를 만들기 전에 멈춰 ObjBound 를 읽을 수 없다 → NaN (호출자는 부모 값 유지)"
function _nz_dr_bound_or_nan(model)
    try
        return objective_bound(model)
    catch e
        e isa MOI.GetAttributeNotAllowed || rethrow()
        return NaN
    end
end

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
function _nz_dr_ranges(DR::_NZDR, nd::NZData, l, u, lo0, hi0, wit0, phib0; fixtol=1e-4, deadline=Inf, mip_deadline=Inf)
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
                isfinite(mip_deadline) && set_time_limit_sec(M.model, max(mip_deadline - time(), 0.01))
                _nz_dr_opt!(DR, M, l, u, "s=$s k=$k $(ismax ? "max" : "min")") || continue    # 실패: 부모 값 유지
                st = termination_status(M.model)
                st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return nothing
                if st == MOI.TIME_LIMIT && isfinite(mip_deadline)     # range_time 마감: 쌍대 bound 는 유효한 범위 (못 구했으면 부모 값)
                    b = _nz_dr_bound_or_nan(M.model)
                    isfinite(b) && (ismax ? (hi[a, s] = min(hi[a, s], b + 1e-6)) : (lo[a, s] = max(lo[a, s], b - 1e-6)))
                    continue
                end
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
                isfinite(mip_deadline) && set_time_limit_sec(M.model, max(mip_deadline - time(), 0.01))
                if _nz_dr_opt!(DR, M, l, u, "s=$s 값")                                          # 실패: 부모 값 유지
                    st = termination_status(M.model)
                    st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return nothing
                    if st == MOI.TIME_LIMIT && isfinite(mip_deadline)                        # range_time 마감: bound 만 (witness 없음)
                        b = _nz_dr_bound_or_nan(M.model)
                        isfinite(b) && (phib[s] = min(phib[s], b + 1e-6 * max(1.0, abs(b))))
                    else
                        st == MOI.OPTIMAL || error("nz_alpha_bnb_dr 값 MIP (s=$s): $st")
                        b = objective_bound(M.model)
                        phib[s] = min(phib[s], b + 1e-6 * max(1.0, abs(b)))
                        wit[key] = clamp.(value.(M.α), l, u)
                    end
                end
            end
        end
    end
    return lo, hi, wit, phib
end

"""
꼭짓점 LP 범위 (range_lp = true, 2026-10-28 L6, 문서 §10.26): 시나리오 s 의 recourse 쌍대 LP
    min/max σ_k  s.t.  Aᵀσ ≥ c,  σ ≥ 0 (H 행 ≤ ϖᵁ),  rr(α)ᵀσ ≤ z_s(α)   (= α 에서의 최적 쌍대면)
전제: recourse 값함수 z_s(α) 가 submodular (H 행의 한계값 σ_k 가 모든 α 좌표에 단조 감소) →
    max_{α∈B} σ_k = α = l 에서 최적 쌍대면 위의 max,  min_{α∈B} σ_k = α = u 에서 최적 쌍대면 위의 min.
L6 (seed 7, x̄={4,5}) 의 상자 3 개에서 범위 MIP 와 63/63 성분 일치 (diag_root_l6.jl (6)). 전제의 증명은 아직 — 시제품.
"""
struct _NZCornerLP
    model::Model
    σ::Vector{VariableRef}
    opt::ConstraintRef
end

function _nz_dr_build_corner(nd::NZData, x̄, s; optimizer)
    m, ny = nz_m(nd), nz_ny(nd)
    M = Model(optimizer); set_silent(M)
    @variable(M, σ[1:m] >= 0)
    for k in 1:m; nd.hrow[k] > 0 && isfinite(nd.varpiU[k]) && set_upper_bound(σ[k], nd.varpiU[k]); end
    @constraint(M, [j=1:ny], sum(nd.A[k, j] * σ[k] for k in 1:m if nd.A[k, j] != 0) >= nd.c[j])
    rr = nz_rhs(nd, x̄, zeros(nz_nh(nd)), s)
    opt = @constraint(M, sum(rr[k] * σ[k] for k in 1:m) + sum(0.0 * σ[k] for k in 1:m if nd.hrow[k] > 0) <= 1e20)
    return _NZCornerLP(M, σ, opt)
end

"상자 [l, u] 의 쌍대 범위를 두 꼭짓점의 LP 로 (부모 범위 lo0, hi0 와 교집합). 범위 MIP 의 바깥쪽 여유 1e-6 과 같게."
function _nz_dr_corner_ranges(CL::Vector{_NZCornerLP}, DR::_NZDR, nd::NZData, x̄, l, u, lo0, hi0; fixtol=1e-4)
    lo = copy(lo0); hi = copy(hi0); S, m = nd.S, nz_m(nd)
    for s in 1:S
        todo = [a for a in eachindex(DR.Hk) if hi[a, s] - lo[a, s] > fixtol]
        isempty(todo) && continue
        C = CL[s]
        for (αc, ismax) in ((u, false), (l, true))
            rr = nz_rhs(nd, x̄, αc, s)
            for k in 1:m; nd.hrow[k] > 0 && set_normalized_coefficient(C.opt, C.σ[k], rr[k]); end
            set_normalized_rhs(C.opt, 1e20)
            @objective(C.model, Min, sum(rr[k] * C.σ[k] for k in 1:m if rr[k] != 0))
            optimize!(C.model)
            termination_status(C.model) == MOI.OPTIMAL || error("nz_alpha_bnb_dr 꼭짓점 LP (s=$s, 최적값): $(termination_status(C.model))")
            z = objective_value(C.model)
            set_normalized_rhs(C.opt, z + 1e-7 * max(1.0, abs(z)))
            for a in todo
                @objective(C.model, ismax ? MAX_SENSE : MIN_SENSE, C.σ[DR.Hk[a]])
                optimize!(C.model)
                termination_status(C.model) == MOI.OPTIMAL || error("nz_alpha_bnb_dr 꼭짓점 LP (s=$s, k=$(DR.Hk[a])): $(termination_status(C.model))")
                b = objective_value(C.model)
                ismax ? (hi[a, s] = min(hi[a, s], b + 1e-6)) : (lo[a, s] = max(lo[a, s], b - 1e-6))
            end
        end
        for a in todo
            hi[a, s] < lo[a, s] && (hi[a, s] = lo[a, s] = 0.5 * (lo[a, s] + hi[a, s]))   # 여유로 생긴 역전 (폭 ≤ 2e-6)
        end
    end
    return lo, hi
end

"max{Σ_s d_s v_s : d ≥ 0, Σ d = 1, ‖d − q‖₁ ≤ 2ε} (TV 볼, 질량 ε 를 값이 낮은 시나리오에서 가장 높은 시나리오로)"
function _nz_tv_max(q, v, ε)
    d = copy(q); smax = argmax(v); bud = ε
    for s in sortperm(v)
        (s == smax || bud <= 0) && break
        t = min(bud, d[s]); d[s] -= t; d[smax] += t; bud -= t
    end
    return dot(d, v)
end

"""
follower 최적성에 의한 상자 축소 (resp_fix, range_lp 전용, 2026-10-28 시제품).
예약 좌표 i 의 follower 한계 이득 g_i(α, d) = c0_i + Σ_s d_s Σ_{k: h(k)=i} hcoef_k σ_ks(α) (follower 는 c0ᵀα + Σ_s d_s z_s(α) 를 최대화).
[lo, hi] 가 상자 B 의 모든 α 에서 유효한 쌍대 범위이므로
  max_{d ∈ D̃} [c0_i + Σ_s d_s Σ_k hcoef_k hi_ks] < −τ                        → B 안의 follower 최적 α 는 α_i = l_i (줄이면 개선)
  min_{d ∈ D̃} [c0_i + Σ_s d_s Σ_k hcoef_k lo_ks] > τ, i 를 늘려도 H 안 (solo[i]) → α_i = u_i
follower 최적인 α 는 하나도 지우지 않으므로 상한은 그대로 V* 이상 (Corollary conservative bounds 와 같은 논증; λ 정확성 불필요).
D̃ 는 follower TV 볼 (belief 결합 δ 는 무시 = 더 큰 집합, 유효). 반환: (l, u, 고정한 좌표 수)
"""
function _nz_dr_resp_fix(nd::NZData, DR::_NZDR, l, u, lo, hi, solo; τ=1e-6)
    S = nd.S; q = nd.q_hat; ε = nd.eps_tilde
    l2 = copy(l); u2 = copy(u); n = 0
    for i in 1:nz_nh(nd)
        u[i] - l[i] <= 1e-9 && continue
        rows = [a for a in eachindex(DR.Hk) if nd.hrow[DR.Hk[a]] == i]
        isempty(rows) && continue
        vhi = [sum(nd.hcoef[DR.Hk[a]] * hi[a, s] for a in rows) for s in 1:S]
        vlo = [sum(nd.hcoef[DR.Hk[a]] * lo[a, s] for a in rows) for s in 1:S]
        if nd.c0[i] + _nz_tv_max(q, vhi, ε) < -τ
            u2[i] = l[i]; n += 1
        elseif solo[i] && nd.c0[i] - _nz_tv_max(q, -vlo, ε) > τ
            l2[i] = u[i]; n += 1
        end
    end
    return l2, u2, n
end

"""
실행 불가능 판정의 Farkas 검증 (farkas = true, 2026-10-29). 직전 solve 가 INFEASIBLE 이고 Gurobi 가 Farkas 쌍대 y (InfUnbdInfo = 1) 를 주면
행 쌍대로 유효 부등식 Σ_i y_i a_iᵀx ≥ Σ_i y_i b_i 를 만들고 (부호가 쌍대 원뿔에 안 맞는 성분은 0 으로 — 어떤 y 든 유효), 그 좌변의 변수 상자 위
최대가 우변보다 여유 이상 작으면 실행 불가능이 증명된다. 여유 = rtol·(1 + Σ|y_i b_i| + Σ_j (Σ_i |y_i a_ij|)·|쓴 경계|) (부동소수 상쇄 몫).
경계가 없는 쪽 (예: x ≥ 0 의 위) 으로 남은 계수는 상쇄 잔차가 상대 rtol 이하면 0 으로 본다 — LP 솔버의 실행 불가능 판정과 같은 허용오차 수준의 검증
(경계가 한쪽뿐인 변수에서 Gurobi 의 ray 는 g_j ≈ ±1e-12 를 남겨, 이것까지 엄밀히 요구하면 L6 S3 에서 897 개 중 0 개 증명). NZ_FARKAS_CHECK=1 로 재확인과 대조.
증명되면 true. 증명서가 없거나 (모델이 바뀜, 다른 상태, 비선형 행) 여유가 모자라면 false → 호출자가 기존 재확인 (_nz_dr_recheck_infeasible!) 으로.
"""
function _nz_dr_farkas_certified(model; rtol=1e-9)
    model.is_model_dirty && return false
    termination_status(model) == MOI.INFEASIBLE || return false
    dual_status(model) == MOI.INFEASIBILITY_CERTIFICATE || return false
    vars = all_variables(model)
    pos = Dict(index(v).value => j for (j, v) in enumerate(vars))
    g = zeros(length(vars)); ga = zeros(length(vars))
    β = 0.0; βa = 0.0
    for (F, Sset) in list_of_constraint_types(model)
        F == VariableRef && continue                       # 변수 경계는 아래에서 상자로
        F == AffExpr || return false
        for c in all_constraints(model, F, Sset)
            y = dual(c); y == 0 && continue
            set = constraint_object(c).set
            if set isa MOI.LessThan
                y = min(y, 0.0); b = set.upper
            elseif set isa MOI.GreaterThan
                y = max(y, 0.0); b = set.lower
            elseif set isa MOI.EqualTo
                b = set.value
            elseif set isa MOI.Interval
                b = y > 0 ? set.lower : set.upper
            else
                return false
            end
            y == 0 && continue
            β += y * b; βa += abs(y * b)
            for (v, a) in constraint_object(c).func.terms
                j = pos[index(v).value]; g[j] += y * a; ga[j] += abs(y * a)
            end
        end
    end
    mx = 0.0; slack = 0.0
    for (j, v) in enumerate(vars)
        g[j] == 0 && continue
        lb = is_fixed(v) ? fix_value(v) : has_lower_bound(v) ? lower_bound(v) : -Inf
        ub = is_fixed(v) ? fix_value(v) : has_upper_bound(v) ? upper_bound(v) : Inf
        bj = g[j] > 0 ? ub : lb
        if isfinite(bj)
            mx += g[j] * bj; slack += ga[j] * abs(bj)
        elseif abs(g[j]) > rtol * max(1.0, ga[j])            # 경계 없는 쪽으로 계수가 남음 → 증명 아님
            return false
        end                                                  # 경계 없는 쪽의 상쇄 잔차 (상대 rtol 이하) 는 0 으로 본다
    end
    return mx < β - rtol * (1.0 + βa + slack)
end

"""
실행 불가능 노드의 사전 판정 (infeas_pre = true, 2026-10-29, 문서 §10.29). 노드 LP 를 풀기 전에 Gurobi 모델을 복사해 recourse 블록
(이름 있는 행 Ffeas, N1, Lflow, Fflow 와 ŷ·ȳ 가 든 이름 없는 행 = WD/FWD/V) 을 지우고 목적을 0 으로 둔 부분 LP 를 푼다. 부분 LP 는 노드 LP 의
행 일부만 남긴 완화라 그 Farkas 증명서 λ 에 지운 행 0 을 붙이면 노드 LP 의 증명서다. Gurobi 규칙 (λᵀAx ≤ λᵀb 유효, ≤ 행 λ ≥ 0, ≥ 행 λ ≤ 0)
의 y = −λ 를 _nz_dr_farkas_certified 와 같은 규칙 (rtol, 경계 없는 쪽 상쇄 잔차) 으로 상자 위에서 검증해 통과하면 true.
부분 LP 가 실행 가능, 시간 초과, 다른 상태, 검증 실패면 false (호출자는 노드 LP 를 그대로 푼다).
L6 덤프 25 개 (실행 불가능 19 + 수치 오류 6) 에서 부분 LP 0.02~1.0 s, 검증 25/25 (lp_subfk.jl). 반환 (증명 여부, 부분 LP 상태 코드, 재시도로 증명했는가).
retry = true (infeas_pre_retry, 2026-10-30, 문서 §10.29): 증명 못 하면 (상태가 INFEASIBLE 이 아니거나 검증 실패) 처음부터 (GRBreset)
NumericFocus 3, 그래도 못 하면 NumericFocus 0·Seed 2 로 다시 푼다. 부분 LP 는 수치에 취약해서 같은 모델이 Seed 0 에서 NUMERIC 또는
검증 실패 INFEASIBLE 로 끝나는 일이 있다 (L6 S50 run23 에서 놓친 실행 불가능 노드 8/8, 이 두 재시도로 8/8 증명, lp_miss3.jl).
"""
function _nz_dr_pre_infeasible(model; rtol=1e-9, tlim=30.0, retry=false)
    o = JuMP.unsafe_backend(model); Gurobi.GRBupdatemodel(o)
    m = Gurobi.GRBcopymodel(o)
    m == C_NULL && return false, -1, false
    try
        ri = Ref{Cint}()
        Gurobi.GRBgetintattr(m, "NumConstrs", ri); nr = Int(ri[]); Gurobi.GRBgetintattr(m, "NumVars", ri); nc = Int(ri[])
        sp = Vector{Ptr{Cchar}}(undef, max(nr, nc))
        Gurobi.GRBgetstrattrarray(m, "VarName", 0, nc, sp)
        rec = BitVector(undef, nc)
        for j in 1:nc; s = unsafe_string(sp[j]); rec[j] = startswith(s, "ŷ[") || startswith(s, "ȳ["); end
        Gurobi.GRBgetstrattrarray(m, "ConstrName", 0, nr, sp)
        nz = Ref{Cint}(); Gurobi.GRBgetconstrs(m, nz, C_NULL, C_NULL, C_NULL, 0, nr)
        beg = zeros(Cint, nr); ind = zeros(Cint, nz[]); val = zeros(nz[]); Gurobi.GRBgetconstrs(m, nz, beg, ind, val, 0, nr)
        drop = Cint[]
        for i in 1:nr
            s = unsafe_string(sp[i]); k = findfirst('[', s)
            if k === nothing                                  # 이름 없는 행: ŷ·ȳ 가 들면 지움 (WD/FWD/V)
                e = i < nr ? beg[i+1] : nz[]
                any(rec[ind[q] + 1] for q in (beg[i] + 1):e) && push!(drop, i - 1)
            elseif s[1:prevind(s, k)] in ("Ffeas", "N1", "Lflow", "Fflow")
                push!(drop, i - 1)
            end
        end
        Gurobi.GRBdelconstrs(m, length(drop), drop)
        Gurobi.GRBsetdblattrarray(m, "Obj", 0, nc, zeros(nc))
        e = Gurobi.GRBgetenv(m)
        Gurobi.GRBsetintparam(e, "Method", 1); Gurobi.GRBsetintparam(e, "Presolve", 0); Gurobi.GRBsetintparam(e, "InfUnbdInfo", 1)
        Gurobi.GRBsetintparam(e, "NumericFocus", 0)
        Gurobi.GRBupdatemodel(m)
        lb = zeros(nc); ub = zeros(nc); Gurobi.GRBgetdblattrarray(m, "LB", 0, nc, lb); Gurobi.GRBgetdblattrarray(m, "UB", 0, nc, ub)
        Gurobi.GRBgetintattr(m, "NumConstrs", ri); n2 = Int(ri[])
        sense = zeros(Cchar, n2); Gurobi.GRBgetcharattrarray(m, "Sense", 0, n2, sense)
        rhs = zeros(n2); Gurobi.GRBgetdblattrarray(m, "RHS", 0, n2, rhs)
        b1 = zeros(Cint, 1)
        function certified()                                # 풀린 부분 LP 의 Farkas 쌍대를 상자 위에서 검증
            λ = zeros(n2); Gurobi.GRBgetdblattrarray(m, "FarkasDual", 0, n2, λ) == 0 || return false
            g = zeros(nc); ga = zeros(nc); β = 0.0; βa = 0.0
            for i in 1:n2
                y = -λ[i]; sg = Char(sense[i])
                y = sg == '<' ? min(y, 0.0) : sg == '>' ? max(y, 0.0) : y
                y == 0 && continue
                β += y * rhs[i]; βa += abs(y * rhs[i])
                Gurobi.GRBgetconstrs(m, nz, C_NULL, C_NULL, C_NULL, i - 1, 1); ki = zeros(Cint, nz[]); kv = zeros(nz[])
                Gurobi.GRBgetconstrs(m, nz, b1, ki, kv, i - 1, 1)
                for q in 1:nz[]; j = ki[q] + 1; g[j] += y * kv[q]; ga[j] += abs(y * kv[q]); end
            end
            mx = 0.0; slack = 0.0
            for j in 1:nc
                g[j] == 0 && continue
                bj = g[j] > 0 ? ub[j] : lb[j]
                if abs(bj) < 1e29
                    mx += g[j] * bj; slack += ga[j] * abs(bj)
                elseif abs(g[j]) > rtol * max(1.0, ga[j])
                    return false
                end
            end
            return mx < β - rtol * (1.0 + βa + slack)
        end
        dl = time() + max(tlim, 0.1); st0 = -1
        # 시도: 기본 (NumericFocus 0, Seed 0) → (retry) NumericFocus 3 → NumericFocus 0·Seed 2. 재시도는 GRBreset 뒤 처음부터
        for (a, (nf, sd)) in enumerate(retry ? ((0, 0), (3, 0), (0, 2)) : ((0, 0),))
            r = dl - time(); (a > 1 && r <= 0.1) && break
            if a > 1
                Gurobi.GRBreset(m, 0); Gurobi.GRBsetintparam(e, "NumericFocus", nf); Gurobi.GRBsetintparam(e, "Seed", sd)
            end
            Gurobi.GRBsetdblparam(e, "TimeLimit", max(r, 0.1))
            Gurobi.GRBoptimize(m)
            Gurobi.GRBgetintattr(m, "Status", ri); st = Int(ri[]); a == 1 && (st0 = st)
            st == 3 && certified() && return true, st, a > 1   # 3 = GRB_INFEASIBLE
            st == 9 && break                                    # 9 = TIME_LIMIT: 재시도하지 않음
        end
        return false, st0, false
    finally
        Gurobi.GRBfreemodel(m)
    end
end

"""
노드 LP 의 INFEASIBLE 판정 재확인. 반환: −Inf (재확인에서도 실행 불가능), 유한값 (풀림: 그 목적값), NaN·Inf (시간·수치 문제).

범위 행을 넣은 노드 LP 는 실행 가능한데도 INFEASIBLE 로 나오는 일이 있다 (test_dr_validity_boxes.jl: 최적 α* 를 품은 상자의
0.5~1%, 범위 행이 없는 기존 완화에서는 0%). 같은 상자를 새 모델에서 풀면 OPTIMAL 이고, 제약이 더 많은 변형이 실행 가능한데
더 적은 변형이 실행 불가능으로 나오기도 한다 → 수학적 무효가 아니라 **warm start 경로의 수치 문제**
(노드마다 행렬 계수를 바꾼 뒤의 dual simplex). FeasibilityTol 만 1e-6 으로 풀어 이어서 풀면 그대로 실행 불가능.
그래서 기저·해 정보를 버리고 (GRBreset) FeasibilityTol 1e-6 에서 처음부터 다시 푼다.

방법 (2026-10-13 변경, lab_recheck.jl): 처음부터 푸는 방법을 primal simplex (Presolve 끔) 에서 **dual simplex + Presolve** 로 바꿨다.
  S = 20 의 첫 판정이 INFEASIBLE 인 상자 40 개에서 primal 은 평균 8.4 s (최대 37 s, 노드 LP 자체는 0.23 s), dual + Presolve 는 0.06 s.
  primal 이 "되살린" 5 개는 목적값이 LB + 10⁶~10⁷ 인 쓰레기 해였고 (dual + Presolve, primal + Presolve, dual 모두 실행 불가능),
  그 노드들은 부모 상한을 그대로 물려받아 다시 나뉘기만 했다. 재확인 시간은 t_relax 에 잡히지 않아 worker 시간의 40~60% 가 보이지 않게 여기서 쓰였다.
  dual + Presolve 가 실행 불가능이라고 하면 Presolve 를 끈 dual simplex 로 한 번 더 확인하고 (서로 다른 경로 두 번), 둘 다 실행 불가능일 때만 −Inf.
  참으로 실행 가능한 상자를 실행 불가능으로 보지 않는지는 test_dr_validity_boxes.jl · test_dr_validity_incumbent.jl 의 강제 재확인 (TV_FORCE_RECHECK) 으로 본다.
`method = :primal` 은 이전 방식 (비교용).
"""
function _nz_dr_recheck_infeasible!(model; method::Symbol=:dual)
    restore!() = (set_optimizer_attribute(model, "FeasibilityTol", get(model.ext, :lp_tol, 1e-9));
                  set_optimizer_attribute(model, "Method", get(model.ext, :lp_method, 1));
                  set_optimizer_attribute(model, "Presolve", get(model.ext, :lp_presolve, 0)); set_optimizer_attribute(model, "DualReductions", 1))
    set_optimizer_attribute(model, "FeasibilityTol", 1e-6)
    if method == :primal
        set_optimizer_attribute(model, "Method", 0)
        Gurobi.GRBreset(JuMP.unsafe_backend(model), 0)
        z = _nz_solve_lp!(model); restore!()
        return z
    end
    method == :dual || error("_nz_dr_recheck_infeasible!: method = :dual | :primal")
    set_optimizer_attribute(model, "Method", 1); set_optimizer_attribute(model, "Presolve", -1); set_optimizer_attribute(model, "DualReductions", 0)
    Gurobi.GRBreset(JuMP.unsafe_backend(model), 0)
    z = _nz_solve_lp!(model)
    if z == -Inf                                           # 두 번째 경로: Presolve 없이 처음부터
        set_optimizer_attribute(model, "Presolve", 0); set_optimizer_attribute(model, "DualReductions", 1)
        Gurobi.GRBreset(JuMP.unsafe_backend(model), 0)
        z = _nz_solve_lp!(model)
    end
    restore!()
    return z
end

"노드 LP 의 범위 행 계수를 상자 [l, u] 와 범위 (lo, hi) 로 갱신"
function _nz_dr_set!(P::_NZSepLP, DR::_NZDR, nd::NZData, l, u, lo, hi, phib; comp::Bool=true, eq=nothing, dlo=nothing, dhi=nothing)
    # eq[i] = true: 좌표 i 의 정상성을 등식으로 (상보성 분기의 "α_i > 0 쪽" 자식). dlo, dhi: 쌍대 차이 범위 (DD)
    v = P.O.v; S = nd.S
    iseq(i) = comp && (l[i] > 1e-9 || (eq !== nothing && eq[i]))
    for (i, ll, c1, c2, κm) in DR.kmc
        set_normalized_coefficient(c1, v[:κ][ll], -l[i]); set_normalized_rhs(c1, -κm * l[i])
        set_normalized_coefficient(c2, v[:κ][ll], -u[i]); set_normalized_rhs(c2, -κm * u[i])
    end
    if dlo !== nothing && length(DR.ddF) > 0
        for p in eachindex(DR.pairs), s in 1:S
            set_normalized_coefficient(DR.ddF[1, p, s], v[:d][s], -dlo[p, s]); set_normalized_coefficient(DR.ddF[2, p, s], v[:d][s], -dhi[p, s])
            set_normalized_coefficient(DR.ddL[1, p, s], v[:r][s], -dlo[p, s]); set_normalized_coefficient(DR.ddL[2, p, s], v[:r][s], -dhi[p, s])
        end
    end
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
            for (n, cz, cw, cl, cr) in ((1, 1.0, -li, -L, L * li), (2, -1.0, ui, L, -L * ui), (3, -1.0, li, H, -H * li), (4, 1.0, -ui, -H, H * ui))
                c = DR.prlt[n, a, s]; sc = 1.0 / max(1.0, abs(cw), abs(cl), abs(cr))     # 행 스케일 (아래 rlt 주석 참고)
                set_normalized_coefficient(c, DR.zP[a, s], cz * sc)
                set_normalized_coefficient(c, π, cw * sc); set_normalized_coefficient(c, zF, cl * sc); set_normalized_coefficient(c, d, cr * sc)
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
        # 행 스케일: 계수가 1 (곱 변수) ~ hi·u (최대 3e4) 로 벌어지면 FeasibilityTol = 1e-9 (build_nz_omega) 에서 실행 가능한 LP 가
        #   INFEASIBLE 로 나오는 일이 있었다 (test_dr_validity_boxes.jl: α* 를 품은 상자, 1e-6 에서는 OPTIMAL). 가장 큰 계수로 나눈다.
        zW = v[:ζW][DR.jj[a], s]
        for (n, cz, cw, cl, cr) in ((1, 1.0, -li, -L, L * li), (2, -1.0, ui, L, -L * ui), (3, -1.0, li, H, -H * li), (4, 1.0, -ui, -H, H * ui))
            c = DR.rlt[n, a, s]; sc = 1.0 / max(1.0, abs(cw), abs(cl), abs(cr))
            set_normalized_coefficient(c, zW, cz * sc)
            set_normalized_coefficient(c, ϖ, cw * sc); set_normalized_coefficient(c, zL, cl * sc); set_normalized_coefficient(c, r, cr * sc)
        end
    end
    for (i, c) in DR.stat                                 # α_i > 0 ⇒ Σ_s Bᵀπ̃ + c0 = Wᵀκ (≤ 는 Fh 로 이미 있음)
        set_normalized_rhs(c, iseq(i) ? -nd.c0[i] : -1e30)
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


# =====================================================================
# 분리 행의 "제자리 갱신" (sticky): (좌표 i, 후보 행 j, 쪽) 마다 LP 행을 하나만 두고, 상자가 바뀌면 그 행의 상자 경계 b 를
# 지금 상자의 l_i (또는 u_i) 로 바꾼다. 기존 방식 (_nz_sep_add! / _nz_sep_set_box!) 은 경계가 다른 행을 새로 쌓고 안 맞는 행은
# rhs = −∞ 로 꺼 두므로 worker 의 LP 가 계속 커지고 (30 분에 분리 행 2 만 개), 자식 노드에서는 부모 경계의 (느슨한) 행만 남아
# 분리 라운드를 더 돌아야 조여진다. 제자리 갱신은 LP 크기가 "한 번이라도 위반된 (i, j, 쪽) 의 수" 로 묶이고, 노드에 들어설 때
# 이미 모든 행이 지금 상자 기준이다.  행 형태 (_nz_sep_add! 와 같음):  아래쪽  GZ − b·Gy ≥ b·c0,   위쪽  b·Gy − GZ ≥ −b·c0
# =====================================================================
function _nz_sticky_setb!(P::_NZSepLP, n::Int, b::Float64)
    i, islo, bold, cref, _, j = P.allrows[n]
    row = P.rows[j]
    if b != bold
        sgn = islo ? -1.0 : 1.0
        for (f, idx, c) in row.terms; set_normalized_coefficient(cref, P.Y[f][idx], sgn * b * c); end
    end
    rhs = islo ? b * row.c0 : -b * row.c0
    (b != bold || !P.active[n]) && set_normalized_rhs(cref, rhs)
    P.allrows[n] = (i, islo, b, cref, rhs, j); P.active[n] = true
    return
end

function _nz_sticky_add!(P::_NZSepLP, i, j, islo, b)
    idx = get!(() -> Dict{Tuple{Int,Int,Bool},Int}(), P.O.model.ext, :sticky)
    n = get(idx, (i, j, islo), 0)
    if n == 0
        _nz_sep_add!(P, i, j, islo, b); idx[(i, j, islo)] = length(P.allrows)
    else
        _nz_sticky_setb!(P, n, Float64(b))
    end
    return
end

"상자 설정 + 모든 분리 행의 경계를 지금 상자로"
function _nz_sticky_set_box!(P::_NZSepLP, l, u)
    α = P.O.v[:α]
    for i in 1:P.nh; set_lower_bound(α[i], l[i]); set_upper_bound(α[i], u[i]); end
    for n in eachindex(P.allrows)
        i, islo = P.allrows[n][1], P.allrows[n][2]
        _nz_sticky_setb!(P, n, islo ? l[i] : u[i])
    end
    P.l .= l; P.u .= u
    return
end

"""
부모 basis (parent_basis = true, 2026-10-29, 문서 §10.29): 노드 LP 를 푼 직후의 basis 를 자식 노드에 실어 두었다가, 다른 노드를 거친 worker 가
그 자식을 꺼낼 때 (pop) 노드 LP 의 시작 basis 로 넣는다. dive (같은 worker 가 바로 자식을 푸는 것) 는 모델에 이미 부모 basis 가 있다.
Gurobi 행 순서 = 노드와 무관한 기본 행 (nr0 개, B&B 중 추가·삭제 없음) + 분리 행 (P.allrows 순서, sticky 는 추가만). 분리 행은 worker 마다
순서·집합이 달라 (좌표 i, 후보 j, 쪽) 으로 저장한다. basis 는 시작점일 뿐이라 LP 값에는 영향이 없다 (맞지 않으면 Gurobi 가 고치거나 버림).
"""
function _nz_dr_get_basis(P::_NZSepLP)
    o = JuMP.unsafe_backend(P.O.model); ri = Ref{Cint}()
    Gurobi.GRBgetintattr(o, "NumConstrs", ri); nr = Int(ri[]); Gurobi.GRBgetintattr(o, "NumVars", ri); nc = Int(ri[])
    nr0 = nr - length(P.allrows)
    vb = zeros(Cint, nc); cb = zeros(Cint, nr)
    Gurobi.GRBgetintattrarray(o, "VBasis", 0, nc, vb) == 0 || return nothing
    Gurobi.GRBgetintattrarray(o, "CBasis", 0, nr, cb) == 0 || return nothing
    cs = Dict{Tuple{Int,Int,Bool},Int8}()
    for (n, r) in enumerate(P.allrows); cs[(r[1], r[6], r[2])] = Int8(cb[nr0+n]); end
    return (vb=Int8.(vb), cb=Int8.(cb[1:nr0]), cs=cs)
end

"부모 basis 를 지금 모델에 시작 basis 로 넣음. 넣었으면 true (열·기본 행 수가 다르면 false)"
function _nz_dr_set_basis!(P::_NZSepLP, bas)
    # 부모에서 비기저 (묶인) 분리 행인데 이 worker 에 없는 행만 지금 상자로 추가 (P.l, P.u = 이 노드 상자). 부모에서 기저였던 행은 빠져도
    #   행 수와 기저 수가 같이 줄어 basis 가 그대로 맞는다. 공유 (share_rows) 와 달리 LP 는 실제로 묶인 행만큼만 커짐 (문서 §10.29)
    have = get(P.O.model.ext, :sticky, nothing)
    for ((i, j, islo), s) in bas.cs
        s == 0 && continue
        (have !== nothing && haskey(have, (i, j, islo))) && continue
        _nz_sticky_add!(P, i, j, islo, islo ? P.l[i] : P.u[i]); have = P.O.model.ext[:sticky]
    end
    o = JuMP.unsafe_backend(P.O.model); Gurobi.GRBupdatemodel(o); ri = Ref{Cint}()
    Gurobi.GRBgetintattr(o, "NumConstrs", ri); nr = Int(ri[]); Gurobi.GRBgetintattr(o, "NumVars", ri); nc = Int(ri[])
    nr0 = nr - length(P.allrows)
    (nc == length(bas.vb) && nr0 == length(bas.cb)) || return false
    vb = Cint.(bas.vb); cb = zeros(Cint, nr); cb[1:nr0] .= bas.cb
    known = falses(nr)
    for (n, r) in enumerate(P.allrows)
        s = get(bas.cs, (r[1], r[6], r[2]), nothing)
        s === nothing ? (cb[nr0+n] = 0) : (cb[nr0+n] = s; known[nr0+n] = true)   # 부모에 없던 분리 행은 slack 을 기저로
    end
    # 기저 수를 행 수에 맞춤: 부모에 없던 분리 행부터 기저 ↔ 비기저를 바꾸고, 그래도 다르면 뒤쪽 분리 행에서
    nb = count(==(0), vb) + count(==(0), cb)
    for pass in (false, true), n in nr:-1:(nr0 + 1)
        nb == nr && break
        (pass || !known[n]) || continue
        if nb > nr && cb[n] == 0
            cb[n] = -1; nb -= 1
        elseif nb < nr && cb[n] != 0
            cb[n] = 0; nb += 1
        end
    end
    Gurobi.GRBsetintattrarray(o, "VBasis", 0, nc, vb) == 0 || return false
    Gurobi.GRBsetintattrarray(o, "CBasis", 0, nr, cb) == 0 || return false
    return true
end

"_nz_sep_solve! 의 제자리 갱신판 (같은 종료 규칙)"
function _nz_sticky_solve!(P::_NZSepLP; maxrounds=30, maxadd=5000, tol=1e-6, stall_tol=1e-6, stall_rounds=2)
    zprev = Inf; nstall = 0
    for rnd in 1:maxrounds
        z = _nz_solve_lp!(P.O.model)
        (isnan(z) || isinf(z)) && return z
        V = _nz_sep_violations(P; tol=tol)
        isempty(V) && return z
        nstall = (zprev - z <= stall_tol * max(1.0, abs(z))) ? nstall + 1 : 0
        zprev = z
        nstall >= stall_rounds && return z
        sort!(V; by=first)
        for (_, i, j, islo) in V[1:min(maxadd, length(V))]; _nz_sticky_add!(P, i, j, islo, islo ? P.l[i] : P.u[i]); end
    end
    return _nz_solve_lp!(P.O.model)
end


const _NZ_DR_ANNOUNCED = Set{Any}()
const _NZ_DR_ANNOUNCE_LOCK = ReentrantLock()

"기본 구성 (2026-10-13). 이 값과 다른 항목이 있으면 _nz_dr_announce 가 stderr 에 알린다"
const _NZ_DR_DEFAULT = (dual_range=true, comp=true, vcut=true, cslift=true, fwd=true, kmc=true, dd=true, cbranch=true, sticky=true,
                        lp_tol=1e-7, branch_point=:mid, child_rounds=1, resp=false, range_mip=true, range_nodes=typemax(Int), range_time=Inf,
                        range_lp=false, resp_fix=false, farkas=false, lp_method=1, lp_presolve=0, lp_crossover=-1, infeas_pre=false, infeas_pre_retry=false, parent_basis=false, share_rows=false, node_lp_cap=Inf)

"구성이 기본과 다르면 다른 항목을 stderr 에 한 번 알린다 (같은 구성은 프로세스당 한 번) — 의도하지 않은 구성으로 도는 것을 막기 위해"
function _nz_dr_announce(config)
    diff = [(n, getfield(config, n)) for n in keys(_NZ_DR_DEFAULT) if getfield(config, n) != getfield(_NZ_DR_DEFAULT, n)]
    isempty(diff) && return
    lock(_NZ_DR_ANNOUNCE_LOCK) do
        diff in _NZ_DR_ANNOUNCED && return
        push!(_NZ_DR_ANNOUNCED, diff)
        println(stderr, "nz_alpha_bnb_dr: 기본이 아닌 구성으로 실행 — ", join(["$n = $v" for (n, v) in diff], ", "))
    end
    return
end

"""
    nz_alpha_bnb_dr(nd, x̄; ...) -> Dict

비제로섬 Ω(x̄) 의 전역 해법. **비제로섬 α-B&B 의 기본 구현** (nz_benders.jl 의 `bnb = :dr`).
nz_alpha_bnb (기존, `bnb = :legacy`) 의 탐색 — worker 병렬, best-first + diving, 위반 RLT 행 분리, Ipopt 휴리스틱 — 에
노드마다 상자별 쌍대 범위 조임을 더한 것. 노드 LP 는 dual simplex 전용.

인자 (기본값 = 2026-10-13 측정에 쓴 구성 `_NZ_DR_DEFAULT`. 기본값 밖의 조합은 측정·검증하지 않았다)

| 묶음 | 인자 | 기본 | 뜻 |
|---|---|---|---|
| 호출마다 | `nworkers` | 스레드 − 2 | worker 수. Julia 는 `-t (nworkers+2),1` |
| | `time_limit` | 300.0 | 초 |
| | `rel_gap` | 1e-4 | 상대 gap (UB − LB) / max(1, abs(LB)) 가 이 값 이하면 종료 |
| | `target` | nothing | LB ≥ target 또는 UB ≤ target 이면 즉시 종료 (Benders 의 목표값 조기 종료) |
| | `verbose`, `log_every` | true, 30.0 | 진행 로그 |
| 자원 | `ipopt_time` | 60.0 | Ipopt 휴리스틱 1 회 시간 (벤치마크는 120) |
| | `heuristic` | true | Ipopt 휴리스틱 task 사용 |
| | `envs` | nothing | Gurobi Env 를 직접 넘김 (worker 당 하나 + 휴리스틱용). 세션 수가 제한된 라이선스에서만 |
| 탐색 | `branch_point` | :mid | 분할점. :mid (중점) 또는 :alphahat (완화 해, 끝 10% 면 중점) |
| | `child_rounds` | 1 | 루트가 아닌 노드의 RLT 분리 라운드 상한 |
| | `maxrounds` | 30 | 루트의 분리 라운드 상한 |
| | `dive_max` | 3 | 한 worker 가 자식으로 연속 내려가는 횟수 |
| | `sep_maxadd` | 5000 | 라운드당 추가 RLT 행 상한 |
| | `min_width` | 1e-7 | 이보다 좁은 좌표는 더 나누지 않음 |
| | `presolve` | true | x̄ 에서 닫힌 점포의 예약 좌표를 0 으로 고정 |
| 강화 (ablation 전용) | `dual_range` | true | recourse 쌍대해의 상자별 범위 (W, F) + RLT. false 면 아래 셋도 꺼지고 기존 완화와 같음 |
| | `comp` | true | 상자가 함의하는 상보성 (C) |
| | `vcut` | true | 시나리오별 값 상한 (V) |
| | `cslift` | true | follower 1 단계 상보성의 RLT (X) |
| | `fwd` | true | follower 블록의 시나리오별 약쌍대 (FWD). X 가 필요 |
| | `kmc` | true | κ 상한과 Zκ 의 위쪽 McCormick (KMC). X 가 필요 |
| | `dd` | true | 쌍대 차이 범위 (DD). 노드당 범위 MIP 가 늘지만 (S = 20 에서 37 → 69 개) 노드 수가 1/10 로 준다 |
| 탐색·LP (2026-10-13) | `cbranch` | true | 상보성 분기. 끄면 중점 분할만 |
| | `sticky` | true | 분리 행을 (좌표, 후보 행, 쪽) 마다 하나만 두고 상자 경계를 제자리 갱신. false 면 행을 쌓는 기존 방식 |
| | `lp_tol` | 1e-7 | 노드 LP 의 Feasibility/OptimalityTol (+ NumericFocus 0). 상한에 상대 1e-6 여유를 더함. 1e-9 면 build_nz_omega 의 설정 그대로 |
| | `resp` | false | primal 휴리스틱: 노드 완화의 belief 에서 follower 의 실제 최적 반응을 평가 (nz_resp_heuristic.jl). 이득을 확인하지 못해 기본 끔 |
| 이어 풀기 | `resume` | nothing | 같은 x̄ 의 이전 호출이 돌려준 `r[:state]`. 주면 그 호출이 멈춘 곳 (열린 노드, incumbent) 에서 이어 푼다 |

이어 풀기 (`resume`): Ω(x̄) 는 x̄ 만으로 정해지므로 같은 x̄ 에서는 이전 호출의 B&B 전선 (열린 노드와 그 상한, 가지친 노드의 상한,
incumbent) 이 그대로 유효하다. Benders 가 목표값 조기 종료로 끝난 호출을 같은 x̄ 에서 다시 부를 때 처음부터 다시 풀지 않게 한다
(S = 30 에서 같은 x̄ = {1} 을 260 s + 300 s 로 두 번 풀던 것). 상태는 강화 구성이 같은 호출 사이에서만 넘길 것 (구성이 다르면 오류).

기본 구성 (`_NZ_DR_DEFAULT`) 과 다른 항목이 있으면 stderr 에 한 줄 알린다 (의도하지 않은 구성으로 도는 것을 막기 위해).
시험했다가 뺀 옵션은 코드에서 지웠다 — 꺾임점 3 분할 분기·분리 행 줄이기·노드 LP barrier (문서 §10.5, 코드는 커밋 3480bae 이전),
교차 정상성 RLT·성분별 상보성·리더 강쌍대·분기 점수의 ζΠ 위반 (문서 §10.13, 코드는 커밋 32a64e4).

반환 Dict: `:LB` (incumbent 값), `:UB` (상한), `:α` (incumbent, 원래 정의역), `:is_exact`, `:nodes`, `:time`, `:root_UB`,
`:solver` (= :nz_alpha_bnb_dr), `:config` (강화·탐색 설정), `:state` (이어 풀기용), `:resumed`, 계측 (`:t_relax`, `:t_range`, `:range_mips`, `:fixfrac`),
수치 문제 횟수 (`:range_fail`, `:infeas`, `:infeas_recovered`, `:numerr`, `:numerr_fresh`, `:numerr_fail`), `:cbranch` (상보성 분기 횟수),
`:t_recheck` (실행 불가능 재확인 시간), `:slow_nodes` (20 s 넘게 걸린 노드의 (전체, 범위 MIP, 완화 LP, 재확인) 초),
`:gc_wait` (GC safepoint 대기, 초), `:gcsafe` (Gurobi 호출을 GC-safe 구간에서 했는가).
"""
function nz_alpha_bnb_dr(nd::NZData, x̄;
                         # 호출마다 정하는 것
                         nworkers=Threads.nthreads() - 2, time_limit=300.0, rel_gap=1e-4, target=nothing,
                         verbose=true, log_every=30.0,
                         # 자원
                         ipopt_time=60.0, heuristic::Bool=true, envs=nothing,
                         # 탐색 (기본값 = 측정에 쓴 값)
                         branch_point::Symbol=:mid, child_rounds::Int=1, maxrounds=30, dive_max=3, sep_maxadd::Int=5000,
                         min_width=1e-7, presolve::Bool=true,
                         # 강화 스위치 (ablation 전용, 기본 모두 켬)
                         dual_range::Bool=true, comp::Bool=true, vcut::Bool=true, cslift::Bool=true,
                         fwd::Bool=true, kmc::Bool=true, dd::Bool=true,
                         # 탐색·LP 구성 (ablation 전용, 기본 켬)
                         cbranch::Bool=true, sticky::Bool=true, lp_tol::Float64=1e-7,
                         # primal 휴리스틱: 노드 완화의 belief 에서 follower 의 최적 반응을 평가 (nz_resp_heuristic.jl)
                         resp::Bool=false,
                         # range_mip = false: 쌍대 범위 KKT-MIP 를 풀지 않고 범위를 전역 값 [0, ϖᵁ] 으로 둔다 (dual_range 의 나머지 강화 — FWD, KMC,
                         #   상보성·상보성 분기, RLT — 는 그대로). recourse 가 커서 범위 MIP 가 노드 시간을 다 쓰는 인스턴스용 (L6, 문서 §10.26)
                         range_mip::Bool=true,
                         # range_nodes: 범위 MIP 를 처음 range_nodes 개 노드에서만 푼다. 그 뒤 노드는 부모 범위를 물려받는다 (부모 상자의 범위는
                         #   자식 상자에서도 유효). range_mip = false 와 기본 (모든 노드) 의 중간 (L6, 문서 §10.26)
                         range_nodes::Int=typemax(Int),
                         # range_time: 호출 시작 뒤 range_time 초까지만 범위 MIP 를 푼다 (그 뒤 노드는 부모 범위). range_nodes 와 함께 쓰면 둘 다 만족할 때만
                         range_time::Float64=Inf,
                         # range_lp = true: 범위 MIP 대신 상자 두 꼭짓점의 LP (_nz_dr_corner_ranges, 모든 노드). 전제 = recourse 값함수의 submodularity (시제품, 문서 §10.26).
                         #   값 MIP (φ̄) 와 차이 범위 (dd) 는 풀지 않음 → dd = false 와 함께, range_time·range_nodes 는 범위 MIP 용이라 같이 쓰지 않음
                         range_lp::Bool=false,
                         # resp_fix = true (range_lp 전용): 꼭짓점 범위로 follower 한계 이득의 부호가 상자에서 정해지는 예약 좌표를 l 또는 u 로 고정 (_nz_dr_resp_fix)
                         resp_fix::Bool=false,
                         # farkas = true: 노드 LP 가 실행 불가능이면 Gurobi 의 Farkas 쌍대를 먼저 검증 (_nz_dr_farkas_certified). 증명되면 재풀이 없이 가지치기,
                         #   아니면 기존 재확인. 환경변수 NZ_FARKAS_CHECK=1 이면 증명된 노드도 재확인을 돌려 결과가 다르면 센다 (검증용)
                         farkas::Bool=false,
                         # 노드 LP 풀이 방법 (기본 = 기존 dual simplex, Presolve 끔): lp_method = 1 dual simplex | 2 barrier | -1 자동, lp_presolve = 0 끔 | -1 자동,
                         #   lp_crossover = -1 자동 | 0 끔 (내부점 해: _nz_solve_lp! 가 max(primal, dual 목적값) 을 상한으로). L6 S=50 은 B&B 안의 노드 LP 가
                         #   12 개 동시 cold 와 같은 시간이라 (warm start 가 거의 안 들음) barrier 를 시험하려고 둠 (문서 §10.29)
                         lp_method::Int=1, lp_presolve::Int=0, lp_crossover::Int=-1,
                         # infeas_pre = true: 루트가 아닌 노드에서 노드 LP 전에 recourse 블록을 뺀 목적 0 부분 LP 로 실행 불가능을 먼저 판정
                         #   (_nz_dr_pre_infeasible, Farkas 증명서 검증). 증명되면 노드 LP 없이 가지치기 (문서 §10.29)
                         infeas_pre::Bool=false,
                         # infeas_pre_retry = true (infeas_pre 전용): 부분 LP 가 증명 못 하면 NumericFocus 3, 그다음 Seed 2 로 다시 풂 (부분 LP 의 수치 취약성, 문서 §10.29)
                         infeas_pre_retry::Bool=false,
                         # parent_basis = true: 노드 LP 를 푼 직후의 basis 를 자식에 실어 두고, 꺼낸 (pop) 노드의 시작 basis 로 넣음 (_nz_dr_get_basis,
                         #   _nz_dr_set_basis!). dive 는 이미 부모 basis 에서 시작. LP 값에는 영향 없음 (문서 §10.29)
                         parent_basis::Bool=false,
                         # share_rows = true: 분리 행 (sticky) 집합을 worker 끼리 맞춤 (한 worker 가 만든 행을 다른 worker 도 추가). LP 가 커짐 (문서 §10.29)
                         share_rows::Bool=false,
                         # node_lp_cap: 루트가 아닌 노드의 LP 작업 (분리 라운드·수치 오류 재시도·실행 불가능 재확인) 시간 상한 (초). 넘으면 수치 오류 노드처럼
                         #   부모 상한을 물려 가장 넓은 좌표를 반으로 나눔 (노드 하나가 오래 전역 UB 를 붙잡지 않게, 문서 §10.29). 기본 Inf = 전체 마감까지
                         node_lp_cap::Float64=Inf,
                         # 이어 풀기: 같은 x̄ 의 이전 호출이 돌려준 r[:state]
                         resume=nothing)
    branch_point in (:alphahat, :mid) || error("branch_point = :alphahat | :mid")
    config = (dual_range=dual_range, comp=dual_range && comp, vcut=dual_range && vcut, cslift=dual_range && cslift,
              branch_point=branch_point, child_rounds=child_rounds, maxrounds=maxrounds, dive_max=dive_max,
              sep_maxadd=sep_maxadd, presolve=presolve, heuristic=heuristic,
              fwd=dual_range && cslift && fwd, kmc=dual_range && cslift && kmc, dd=dual_range && dd,
              cbranch=dual_range && comp && cbranch, sticky=sticky, lp_tol=lp_tol, resp=resp, range_mip=range_mip, range_nodes=range_nodes, range_time=range_time,
              range_lp=range_lp, resp_fix=resp_fix, farkas=farkas, lp_method=lp_method, lp_presolve=lp_presolve, lp_crossover=lp_crossover, infeas_pre=infeas_pre, infeas_pre_retry=infeas_pre && infeas_pre_retry, parent_basis=parent_basis, share_rows=share_rows, node_lp_cap=node_lp_cap)
    resp_fix && !range_lp && error("nz_alpha_bnb_dr: resp_fix = true 는 range_lp = true 와 함께")
    range_lp &&(dd || isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&
        error("nz_alpha_bnb_dr: range_lp = true 는 dual_range = true, dd = false 와 함께 쓰고 range_mip·range_time·range_nodes 는 기본값으로")
    _nz_dr_announce(config)
    n_rnode = Threads.Atomic{Int}(0)                                         # 범위 MIP 를 푼 노드 수 (range_nodes)
    n_cbranch = Threads.Atomic{Int}(0)                                       # 상보성 분기 횟수
    n_infeas = Threads.Atomic{Int}(0); n_infrec = Threads.Atomic{Int}(0)     # 실행 불가능 노드 (1e-6 재확인 통과) / 재확인에서 되살아난 노드
    n_fk = Threads.Atomic{Int}(0); n_fkdis = Threads.Atomic{Int}(0)          # Farkas 로 증명된 노드 / (NZ_FARKAS_CHECK) 재확인이 실행가능이라 한 노드
    n_fkto = Threads.Atomic{Int}(0)                                          # (NZ_FARKAS_CHECK) 재확인이 시간 제한으로 판정 못 한 노드
    n_pre = Threads.Atomic{Int}(0); n_pretry = Threads.Atomic{Int}(0)        # infeas_pre: 부분 LP 로 증명된 노드 / 부분 LP 를 푼 노드
    t_pre = Threads.Atomic{Float64}(0.0); n_preretry = Threads.Atomic{Int}(0)  # infeas_pre_retry: 재시도에서 증명된 노드
    n_bload = Threads.Atomic{Int}(0); n_bfail = Threads.Atomic{Int}(0)       # parent_basis: 부모 basis 를 넣은 노드 / 못 넣은 노드
    n_cap = Threads.Atomic{Int}(0)                                           # node_lp_cap 에 걸린 노드
    bdive = get(ENV, "NZ_BASIS_DIVE", "0") == "1"           # 진단 (parent_basis): dive 노드에도 실어 둔 부모 basis 를 명시적으로 넣음
    skeys = Tuple{Int,Int,Bool}[]; skset = Set{Tuple{Int,Int,Bool}}(); sk_lk = ReentrantLock()   # parent_basis: worker 공유 분리 행 키
    precheck = get(ENV, "NZ_PRE_CHECK", "0") == "1"; n_predis = Threads.Atomic{Int}(0); n_preto = Threads.Atomic{Int}(0)
    fkcheck = get(ENV, "NZ_FARKAS_CHECK", "0") == "1"
    prchk = get(ENV, "NZ_PRUNE_CHECK", "0") == "1"; n_prchk = Threads.Atomic{Int}(0); n_prdis = Threads.Atomic{Int}(0)
    nfix = 0
    if presolve
        hUp, nfix = nz_presolve_hU(nd, x̄)
        nfix > 0 && (nd = nz_with_hU(nd, hUp))
    end
    Threads.nthreads(:interactive) >= 1 || error("nz_alpha_bnb_dr: julia -t $(nworkers + 2),1 로 실행하세요")
    t0 = time(); t_end = t0 + time_limit
    gcw0 = Base.gc_num().total_time_to_safepoint       # GC safepoint 대기 (전체 스레드가 멈춘 wall 시간) — :gc_wait 로 보고
    nh, S = nz_nh(nd), nd.S
    Hr = findall(nd.hrow .> 0)
    mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true);
                     MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    envs = envs === nothing ? _nz_envs(nworkers + 1) : envs
    length(envs) >= nworkers + (heuristic ? 1 : 0) || error("nz_alpha_bnb_dr: envs 가 부족함 (worker $nworkers + heuristic $(heuristic))")
    Ps = [_nz_build_sep(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    numerr0 = _NZ_NUMERR[]; numfail0 = _NZ_NUMERR_FAIL[]
    DRs = dual_range ? [_nz_dr_attach!(Ps[i], nd, x̄; optimizer=mk(envs[i]), vcut=vcut, cslift=cslift,
                                       fwd=config.fwd, kmc=config.kmc, dd=config.dd) for i in 1:nworkers] : nothing
    CLs = range_lp ? [[_nz_dr_build_corner(nd, x̄, s; optimizer=mk(envs[i])) for s in 1:nd.S] for i in 1:nworkers] : nothing
    # solo[i]: α_i 를 늘릴 때 걸리는 W 행이 i 하나만 포함 (거점별 한도) → 상자 안에서 늘려도 H 안 (resp_fix 의 위쪽 고정 조건)
    solo = [all(all(nd.W[ll, i2] == 0 || i2 == i for i2 in 1:nh) && nd.W[ll, i] * nd.hU[i] <= nd.wvec[ll] + 1e-9
                for ll in 1:size(nd.W, 1) if nd.W[ll, i] > 0) for i in 1:nh]
    n_rfix = Threads.Atomic{Int}(0)                                          # resp_fix 로 고정한 좌표 수 (노드 합)
    # 노드 LP 허용오차. build_nz_omega 의 기본은 1e-9 + NumericFocus 2 (θ 항의 상쇄 때문에 조였던 것). lp_tol > 1e-9 이면 그 값 + NumericFocus 0 으로 풀고
    #   노드 상한에 상대 1e-6 의 여유를 더한다 (허용오차만큼 덜 수렴한 목적값을 상한으로 쓰는 것에 대한 여유).
    lp_tol > 1e-9 && for P in Ps
        set_optimizer_attribute(P.O.model, "FeasibilityTol", lp_tol); set_optimizer_attribute(P.O.model, "OptimalityTol", lp_tol)
        set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = lp_tol
    end
    zpad(z) = (lp_tol > 1e-9 && isfinite(z)) ? z + 1e-6 * max(1.0, abs(z)) : z
    farkas && for P in Ps; set_optimizer_attribute(P.O.model, "InfUnbdInfo", 1); end   # Farkas 쌍대를 받기 위해
    for P in Ps; P.O.model.ext[:retry_fresh] = true; end          # 노드 LP 수치 오류: NumericFocus 3 전에 처음부터 한 번 (_nz_solve_lp!)
    for P in Ps                                                   # 노드 LP 풀이 방법. ext 값은 재시도·재확인 뒤 되돌릴 값 (기본이면 기존 설정 그대로)
        P.O.model.ext[:lp_method] = lp_method; P.O.model.ext[:lp_presolve] = lp_presolve; P.O.model.ext[:lp_crossover] = lp_crossover
        (lp_method, lp_presolve, lp_crossover) == (1, 0, -1) && continue
        set_optimizer_attribute(P.O.model, "Method", lp_method); set_optimizer_attribute(P.O.model, "Presolve", lp_presolve)
        set_optimizer_attribute(P.O.model, "Crossover", lp_crossover)
    end
    # 진단 (NZ_NODELP_LOG=<파일>): 노드마다 한 줄 — 시각, worker, 종류 (root / dive = 이 worker 가 직전에 푼 노드의 자식 / pop = 열린 노드에서 꺼냄),
    #   LP 횟수, simplex 반복, barrier 반복, 완화 LP 시간, 범위 LP 시간, 결과. 부모에서 이어 푼 LP (dive) 와 그렇지 않은 LP (pop) 의 비용 비교용
    nlog = get(ENV, "NZ_NODELP_LOG", ""); nlog_lk = ReentrantLock()
    if !isempty(nlog)
        for P in Ps; P.O.model.ext[:itlog] = zeros(Int, 3); end
        Base.open(io -> @printf(io, "# t wid kind nlp simplex_it barrier_it t_lp t_range result t_pre | S=%d nworkers=%d lp_method=%d lp_presolve=%d lp_crossover=%d\n",
                           nd.S, nworkers, lp_method, lp_presolve, lp_crossover), nlog, "a")
    end
    # parent_basis: Gurobi 모델을 지금 붙인다 (JuMP 는 첫 optimize 때 붙이며 그때까지 쌓인 행을 제약 종류별로 묶어 복사한다). worker 2~12 는 첫 풀이 전에
    #   루트의 분리 행을 받으므로, 늦게 붙이면 분리 행이 기본 행 사이에 끼어 "기본 행 nr0 개 + 분리 행" 순서가 깨진다 (2026-10-29: 다른 worker 의 부모 basis 무효의 원인)
    parent_basis && for P in Ps; MOI.Utilities.attach_optimizer(P.O.model); end
    if get(ENV, "NZ_BASIS_DEBUG", "0") == "1" && nworkers >= 2       # 진단: worker 1·2 모델의 행·열 순서가 같은가 (행 = 감각·rhs·열 번호·계수)
        sigs = map(Ps[1:2]) do Pk
            MOI.Utilities.attach_optimizer(Pk.O.model); o = JuMP.unsafe_backend(Pk.O.model); Gurobi.GRBupdatemodel(o); ri = Ref{Cint}()
            Gurobi.GRBgetintattr(o, "NumConstrs", ri); nr = Int(ri[]); Gurobi.GRBgetintattr(o, "NumVars", ri); nc = Int(ri[])
            nz = Ref{Cint}(); Gurobi.GRBgetconstrs(o, nz, C_NULL, C_NULL, C_NULL, 0, nr)
            beg = zeros(Cint, nr); ind = zeros(Cint, nz[]); val = zeros(nz[]); Gurobi.GRBgetconstrs(o, nz, beg, ind, val, 0, nr)
            sn = zeros(Cchar, nr); Gurobi.GRBgetcharattrarray(o, "Sense", 0, nr, sn)
            lb = zeros(nc); Gurobi.GRBgetdblattrarray(o, "LB", 0, nc, lb); ub = zeros(nc); Gurobi.GRBgetdblattrarray(o, "UB", 0, nc, ub)
            rows = [(sn[i], sort(collect(zip(ind[(beg[i]+1):(i < nr ? beg[i+1] : nz[])], round.(val[(beg[i]+1):(i < nr ? beg[i+1] : nz[])]; sigdigits=8))))) for i in 1:nr]
            (nr, nc, rows, lb, ub)
        end
        (a, b) = sigs
        nrd = count(i -> a[3][i] != b[3][i], 1:min(a[1], b[1])); ncd = count(j -> a[4][j] != b[4][j] || a[5][j] != b[5][j], 1:min(a[2], b[2]))
        @printf(stderr, "  [basis debug] worker 1·2: 행 %d·%d, 열 %d·%d, 순서가 다른 행 %d, 경계가 다른 열 %d (같은 행 집합: %s)\n",
                a[1], b[1], a[2], b[2], nrd, ncd, sort(a[3]) == sort(b[3]))
    end
    fresh0 = _NZ_FRESH_OK[]
    slow = Tuple{Float64,Float64,Float64,Float64}[]               # 20 s 넘게 걸린 노드: (전체, 범위 MIP, 완화 LP, 실행 불가능 재확인) 초
    Fs = [nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    Rs = resp ? [_nz_build_resp(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers] : nothing
    n_resp = Threads.Atomic{Int}(0)                    # follower 반응 휴리스틱이 incumbent 를 갱신한 횟수
    tol(v) = rel_gap * (isfinite(v) ? max(1.0, abs(v)) : 1.0)
    boxok(l) = all(nd.W * l .<= nd.wvec .+ 1e-9)

    lk = ReentrantLock()
    t_relax = Threads.Atomic{Float64}(0.0); t_eval = Threads.Atomic{Float64}(0.0); t_range = Threads.Atomic{Float64}(0.0)
    t_recheck = Threads.Atomic{Float64}(0.0)
    lo0, hi0 = dual_range ? _nz_dr_global(nd, DRs[1]) : (zeros(0, 0), zeros(0, 0))
    dlo0, dhi0 = dual_range ? _nz_dr_dglobal(nd, DRs[1]) : (zeros(0, 0), zeros(0, 0))
    # 노드 = (l, u, 상한, lo, hi, witness, φ̄, ext).  ext = (eq: 정상성을 등식으로 둔 좌표 (상보성 분기), dlo·dhi: 쌍대 차이 범위)
    open = Any[(zeros(nh), copy(nd.hU), Inf, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S), (eq=falses(nh), dlo=dlo0, dhi=dhi0))]
    if resume !== nothing
        resume.x == x̄ || error("nz_alpha_bnb_dr: resume 상태의 x̄ 가 다름 (Ω 는 x̄ 마다 다른 문제)")
        resume.config == config || error("nz_alpha_bnb_dr: resume 상태의 구성이 다름 ($(resume.config) vs $config)")
        open = Any[resume.open...]                 # 노드 튜플 안의 배열은 worker 가 제자리에서 고치지 않으므로 얕은 복사로 충분
    end
    inflight = fill(-Inf, nworkers)
    LB = Ref(-Inf); best_α = Ref(zeros(nh))
    nodes = Ref(0); done = Ref(false); root_UB = Ref(Inf)
    root_rows = Ref{Any}(nothing)
    if resume !== nothing
        LB[] = resume.LB; best_α[] = copy(resume.α); root_UB[] = resume.root_UB; root_rows[] = resume.root_rows
    end
    node_cands = Tuple{Float64,Vector{Float64}}[]
    ipopt_calls = Ref(0)
    pruned_ub = Ref(resume === nothing ? -Inf : resume.pruned_ub)
    nfixed_sum = Ref(0.0)                      # 노드별 "고정된 쌍대 성분 비율" 의 합 (평균 보고용)

    globalUB() = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open), maximum(inflight), pruned_ub[])
    offer!(z, a) = lock(lk) do
        z > LB[] && (LB[] = z; best_α[] = copy(a))
    end

    function worker(wid)
        P = Ps[wid]; F = Fs[wid]; DR = dual_range ? DRs[wid] : nothing
        localnode = nothing; dive_left = 0; synced = false
        nown = 0; nsk = 0                                   # parent_basis: 공유 목록에 올린 자기 분리 행 수 / 읽은 공유 키 수
        while true
            yield()
            node = nothing; fromdive = false
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
                    node = localnode; dive_left -= 1; fromdive = true
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
            # 진단 (NZ_BASIS_ROUNDTRIP=1, parent_basis): dive 노드에서 모델의 부모 basis 를 읽어 두었다가 노드 설정 뒤 그대로 다시 넣음 (넣는 경로의 손실 확인)
            rtbas = (parent_basis && fromdive && get(ENV, "NZ_BASIS_ROUNDTRIP", "0") == "1") ? _nz_dr_get_basis(P) : nothing
            tn0 = time(); tn_rg = 0.0; tn_lp = 0.0; tn_rc = 0.0
            ext = node[8]; eqp = ext.eq; dlo = ext.dlo; dhi = ext.dhi
            if !synced && root_rows[] !== nothing && isempty(P.allrows) && t_end - time() > 30.0
                for (i, j, islo, b) in root_rows[]; sticky ? _nz_sticky_add!(P, i, j, islo, b) : _nz_sep_add!(P, i, j, islo, b); end
                synced = true
            end
            # share_rows: 분리 행 (sticky) 집합을 worker 끼리 맞춤 — 자기가 새로 만든 행의 키를 공유 목록에 올리고, 남이 올린 키 중 없는 행을 추가.
            #   행은 어떤 상자에서도 유효한 RLT 행이라 더해도 완화는 조여지기만 한다. parent_basis 와 함께 시험했으나 다른 worker 의 basis 무효의 원인은
            #   모델을 늦게 붙인 것이었고 (위 attach), 공유는 LP 를 키워 S50 노드 LP 를 느리게 함 (문서 §10.29)
            if share_rows && sticky && root_rows[] !== nothing
                lock(sk_lk) do
                    for n in (nown + 1):length(P.allrows)
                        r = P.allrows[n]; key = (r[1], r[6], r[2]); key in skset || (push!(skset, key); push!(skeys, key))
                    end
                    newk = skeys[(nsk + 1):end]; nsk = length(skeys)
                    have = get(P.O.model.ext, :sticky, Dict{Tuple{Int,Int,Bool},Int}())
                    for (i, j, islo) in newk
                        haskey(have, (i, j, islo)) || _nz_sticky_add!(P, i, j, islo, islo ? l[i] : u[i])
                    end
                end
                nown = length(P.allrows)
            end
            z = -Inf; α̂ = nothing; children = nothing; zE = -Inf; unfinished = false; αE = nothing
            pruned_val = -Inf; lo = lop; hi = hip; wit = witp; phib = phibp; fixfrac = NaN
            boxok(l) && ubp <= LB[] + tol(LB[]) && (pruned_val = ubp)
            if ubp > LB[] + tol(LB[]) && boxok(l)
                sticky ? _nz_sticky_set_box!(P, l, u) : _nz_sep_set_box!(P, l, u)
                _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                empty_box = false
                if dual_range && range_lp                     # 꼭짓점 LP 범위 (모든 노드). 값 상한 φ̄ 는 부모 값
                    tg = @elapsed ((lo, hi) = _nz_dr_corner_ranges(CLs[wid], DR, nd, x̄, l, u, lop, hip))
                    Threads.atomic_add!(t_range, tg); tn_rg += tg
                    if resp_fix                               # 상자를 줄이면 범위는 그대로 유효 (부분 상자). 줄인 상자로 범위를 한 번 더
                        l, u, nrf = _nz_dr_resp_fix(nd, DR, l, u, lo, hi, solo)
                        if nrf > 0
                            Threads.atomic_add!(n_rfix, nrf)
                            sticky ? _nz_sticky_set_box!(P, l, u) : _nz_sep_set_box!(P, l, u)
                            _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                            tg = @elapsed ((lo, hi) = _nz_dr_corner_ranges(CLs[wid], DR, nd, x̄, l, u, lo, hi))
                            Threads.atomic_add!(t_range, tg); tn_rg += tg
                        end
                    end
                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                    fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                elseif dual_range && !(range_mip && time() - t0 < range_time && Threads.atomic_add!(n_rnode, 1) < range_nodes)   # 범위는 부모 값 그대로, 나머지 강화만 상자에 맞춤
                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                    fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                elseif dual_range
                    rdl = min(t_end, t0 + range_time)            # range_time 마감 (기본 Inf 면 t_end)
                    tg = @elapsed (rg = _nz_dr_ranges(DR, nd, l, u, lop, hip, witp, phibp; deadline=rdl,
                                                      mip_deadline=(isfinite(range_time) ? rdl : Inf)))
                    Threads.atomic_add!(t_range, tg); tn_rg += tg
                    if rg === nothing
                        empty_box = true                 # 상자 안에 실행가능한 α 가 없음
                    else
                        lo, hi, wit, phib = rg
                        if config.dd
                            tg2 = @elapsed ((dlo, dhi) = _nz_dr_dranges!(DR, nd, l, u, lo, hi, ext.dlo, ext.dhi, witp, wit; deadline=rdl,
                                                                          mip_deadline=(isfinite(range_time) ? rdl : Inf)))
                            Threads.atomic_add!(t_range, tg2); tn_rg += tg2
                        end
                        _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                        fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                    end
                end
                if !empty_box
                    set_time_limit_sec(P.O.model, max(t_end - time(), 0.1))
                    isempty(nlog) || (P.O.model.ext[:itlog] .= 0)
                    # 이 노드의 LP 작업 마감 (_nz_solve_lp! 가 재시도마다 남은 시간으로 TimeLimit 을 다시 둠). 루트는 전체 마감까지
                    ndl = ubp == Inf ? t_end : min(t_end, time() + node_lp_cap); P.O.model.ext[:deadline] = ndl
                    pre_inf = false; tp = 0.0                 # infeas_pre: 부분 LP 로 실행 불가능이 증명되면 노드 LP 를 풀지 않음
                    if infeas_pre && ubp != Inf
                        Threads.atomic_add!(n_pretry, 1)
                        tp = @elapsed ((pre_inf, _, pre_rt) = _nz_dr_pre_infeasible(P.O.model; tlim=min(30.0, ndl - time()), retry=config.infeas_pre_retry))
                        Threads.atomic_add!(t_pre, tp); pre_inf && Threads.atomic_add!(n_pre, 1); pre_rt && Threads.atomic_add!(n_preretry, 1)
                    end
                    bload = false                             # parent_basis: 꺼낸 노드에 부모 basis 가 있으면 시작 basis 로
                    if parent_basis && !pre_inf && (!fromdive || bdive) && get(ext, :bas, nothing) !== nothing
                        bload = _nz_dr_set_basis!(P, ext.bas); Threads.atomic_add!(bload ? n_bload : n_bfail, 1)
                    end
                    rtbas !== nothing && !pre_inf && (bload = _nz_dr_set_basis!(P, rtbas))
                    tr = pre_inf ? 0.0 : @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd)))
                    pre_inf && (z = -Inf)
                    if isnan(z) && ndl < t_end && time() < t_end - 1.0  # node_lp_cap 에 걸림 (전체 마감 전): 수치 오류 노드처럼 나눔
                        Threads.atomic_add!(n_cap, 1); z = Inf
                    end
                    if !isempty(nlog)
                        il = P.O.model.ext[:itlog]; kind = ubp == Inf ? "root" : fromdive ? (bload ? "diveb" : "dive") : bload ? "popb$(get(ext, :side, 1))$(ext.bas.wid == wid ? "s" : "o")" : "pop"
                        res = pre_inf ? "preinf" : isnan(z) ? "time" : z == -Inf ? "infeas" : z == Inf ? "numerr" : z <= LB[] + tol(LB[]) ? "prune" : "branch"
                        lock(nlog_lk) do
                            Base.open(io -> @printf(io, "%.2f %d %s %d %d %d %.3f %.3f %s %.3f\n", time() - t0, wid, kind, il[1], il[2], il[3], tr, tn_rg, res, tp), nlog, "a")
                        end
                    end
                    # 진단 (NZ_PRUNE_CHECK=1): LP 값으로 가지칠 노드 (OPTIMAL) 를 기저 없이 처음부터 다시 풀어 기준을 넘으면 출력
                    #   (이어 푼 dual simplex 의 최적값 검증. 2026-10-29 L6 S3 1,031 개·S20 68 개 불일치 0)
                    if prchk && isfinite(z) && z <= LB[] + tol(LB[]) && termination_status(P.O.model) == MOI.OPTIMAL
                        Threads.atomic_add!(n_prchk, 1); cthr = LB[] + tol(LB[])
                        Gurobi.GRBreset(JuMP.unsafe_backend(P.O.model), 0); zc0 = zpad(_nz_solve_lp!(P.O.model))
                        if isfinite(zc0) && zc0 > cthr
                            Threads.atomic_add!(n_prdis, 1)
                            @printf(stderr, "  [가지치기 불일치] 기준 %.10g, 이어 푼 값 %.10g, 처음부터 %.10g (초과 상대 %.2e)\n", cthr, z, zc0, (zc0 - cthr) / max(1.0, abs(cthr)))
                        end
                    end
                    Threads.atomic_add!(t_relax, tr); tn_lp += tr
                    # 실행 불가능 판정 재확인 (_nz_dr_recheck_infeasible!): 그래도 실행 불가능일 때만 가지치기한다. 재확인에서 풀리면
                    #   그 값을 상한으로 쓰고 해는 읽지 않은 채 가장 넓은 좌표를 반으로 나눈다 (수치 오류 노드와 같은 처리).
                    zrec = NaN
                    if z == -Inf && dual_range
                        fk = pre_inf                          # 부분 LP 의 Farkas 증명서로 이미 검증됨
                        if pre_inf && precheck                # 검증용 (NZ_PRE_CHECK=1): 노드 LP 를 처음부터 다시 풀어 대조
                            tn_rc += @elapsed (zchk = _nz_dr_recheck_infeasible!(P.O.model))
                            isfinite(zchk) && Threads.atomic_add!(n_predis, 1)         # 재확인이 해를 찾음 = 불일치
                            isnan(zchk) && Threads.atomic_add!(n_preto, 1)
                        end
                        if farkas && !pre_inf
                            tn_rc += @elapsed (fk = _nz_dr_farkas_certified(P.O.model))
                            fk && Threads.atomic_add!(n_fk, 1)
                            if fk && fkcheck
                                tn_rc += @elapsed (zchk = _nz_dr_recheck_infeasible!(P.O.model))
                                isfinite(zchk) && Threads.atomic_add!(n_fkdis, 1)      # 재확인이 해를 찾음 = 불일치
                                isnan(zchk) && Threads.atomic_add!(n_fkto, 1)          # 재확인이 시간 제한에 걸림 (판정 없음)
                            end
                        end
                        tn_rc += @elapsed (zc = fk ? -Inf : _nz_dr_recheck_infeasible!(P.O.model))
                        Threads.atomic_add!(t_recheck, tn_rc)
                        if zc == -Inf
                            Threads.atomic_add!(n_infeas, 1)
                        else
                            Threads.atomic_add!(n_infrec, 1)
                            zrec = isfinite(zc) ? zpad(zc) : ubp
                            z = Inf
                        end
                    end
                    unfinished = isnan(z)
                    numfail = z == Inf
                    if numfail
                        z = isnan(zrec) ? ubp : min(zrec, ubp)
                        i = argmax(u .- l)
                        if u[i] - l[i] > min_width
                            t = 0.5 * (l[i] + u[i]); u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi)))
                        else
                            pruned_val = z
                        end
                    end
                    z = unfinished ? ubp : min(z, ubp)
                    (!unfinished && !numfail && z <= LB[] + tol(LB[])) && (pruned_val = z)
                    if !unfinished && !numfail && z > LB[] + tol(LB[])
                        v = P.O.v
                        basP = parent_basis ? _nz_dr_get_basis(P) : nothing
                        basP === nothing || (basP = merge(basP, (wid=wid,)))   # 자식에 실을 부모 basis (값을 읽기만 한 뒤라 모델은 그대로)
                        α̂ = clamp.(value.(v[:α]), l, u)
                        rv = value.(v[:r]); dv = value.(v[:d])
                        ϖv = zeros(size(v[:ϖ]))
                        for k in Hr, s in 1:S; ϖv[k, s] = value(v[:ϖ][k, s]); end
                        ζLv = value.(v[:ζL]); ζFv = value.(v[:ζF]); ζWv = value.(v[:ζW])
                        set_time_limit_sec(F.model, max(t_end - time(), 0.1))
                        te = @elapsed (zE = nz_eval_alpha!(F, nd, α̂))
                        isnan(zE) && (zE = -Inf)
                        αE = α̂                                  # zE 를 준 점 (incumbent 후보)
                        if resp
                            te += @elapsed begin
                                αr = _nz_resp!(Rs[wid], nd, max.(dv, 0.0), α̂)
                                if αr !== nothing
                                    set_time_limit_sec(F.model, max(t_end - time(), 0.1))
                                    zR = nz_eval_alpha!(F, nd, αr)
                                    (!isnan(zR) && zR > zE) && (zE = zR; αE = αr)
                                end
                            end
                        end
                        Threads.atomic_add!(t_eval, te)
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
                        # 상보성 분기 (cbranch): 아래 끝이 0 인 좌표에서 α̂_i > 0 이면서 정상성 여유 ĝ_i > 0 이면 (상보성 α_i g_i = 0 위반)
                        #   상자를 반으로 나누지 않고 {α_i = 0} | {g_i = 0 (정상성 등식), 상자 그대로} 로 나눈다. 중점 분할은 왼쪽 자식 [0, u/2] 에
                        #   완화된 상보성이 그대로 남아 폭에 비례하는 오차가 반씩만 줄어든다 (2026-10-13 진단: 정체한 상자의 주된 오차).
                        ic = 0
                        if config.cbranch && viol[i] > 1e-9
                            κv = value.(v[:κ]); bestc = 1e-4
                            for i2 in 1:nh
                                (nd.hU[i2] > 0 && u[i2] - l[i2] > min_width && l[i2] <= 1e-9 && !eqp[i2]) || continue
                                g = sum(nd.W[ll, i2] * κv[ll] for ll in 1:size(nd.W, 1)) - nd.c0[i2]
                                for k in DR.Hk
                                    nd.hrow[k] == i2 && (g -= nd.hcoef[k] * sum(value(v[:π̃][k, s]) for s in 1:S))
                                end
                                sc = (α̂[i2] / nd.hU[i2]) * (g / max(1.0, abs(nd.c0[i2])))
                                sc > bestc && (bestc = sc; ic = i2)
                            end
                        end
                        if ic > 0
                            u1 = copy(u); u1[ic] = l[ic]; eqB = copy(eqp); eqB[ic] = true
                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP)), (copy(l), copy(u), z, lo, hi, wit, phib, (eq=eqB, dlo=dlo, dhi=dhi, bas=basP, side=2)))
                            Threads.atomic_add!(n_cbranch, 1)
                        elseif viol[i] > 1e-9
                            width = u[i] - l[i]; t = α̂[i]
                            (branch_point != :alphahat || t - l[i] < 0.1 * width || u[i] - t < 0.1 * width) && (t = 0.5 * (l[i] + u[i]))
                            u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, side=2)))
                        else
                            z > zE && (zE = z; αE = α̂)     # 곱 위반이 없으면 완화가 정확
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
                time() - tn0 > 20.0 && push!(slow, (time() - tn0, tn_rg, tn_lp, tn_rc))
                isnan(fixfrac) || (nfixed_sum[] += fixfrac)
                pruned_ub[] = max(pruned_ub[], pruned_val)
                if ubp == Inf
                    root_UB[] = z
                    root_rows[] = [(r[1], r[6], r[2], r[3]) for r in P.allrows]
                end
                if zE > LB[]
                    αE === α̂ || Threads.atomic_add!(n_resp, 1)
                    LB[] = zE; best_α[] = copy(αE)
                end
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
            fin = (time() >= t_end) || (UBn - LB[] <= tol(LB[])) || (idle && (nodes[] > 0 || resume !== nothing)) ||
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
                :solver => :nz_alpha_bnb_dr, :config => config, :presolve_fixed => nfix, :resumed => resume !== nothing,
                :state => (x=copy(x̄), config=config, open=open, LB=LB[], α=copy(best_α[]), pruned_ub=pruned_ub[],
                           root_UB=root_UB[], root_rows=root_rows[]), :t_relax => t_relax[], :t_eval => t_eval[], :t_range => t_range[],
                :t_recheck => t_recheck[], :slow_nodes => sort(slow; rev=true), :numerr_fresh => _NZ_FRESH_OK[] - fresh0,
                :range_mips => dual_range ? sum(D.nmip for D in DRs) : 0, :range_fail => dual_range ? sum(D.nfail for D in DRs) : 0,
                :fixfrac => nodes[] > 0 ? nfixed_sum[] / nodes[] : NaN, :infeas => n_infeas[], :infeas_recovered => n_infrec[],
                :numerr => _NZ_NUMERR[] - numerr0, :numerr_fail => _NZ_NUMERR_FAIL[] - numfail0,
                :cbranch => n_cbranch[], :resp_fix => n_rfix[], :farkas => n_fk[], :farkas_disagree => n_fkdis[], :farkas_check_timeout => n_fkto[], :infeas_pre => n_pre[], :infeas_pre_tried => n_pretry[], :infeas_pre_retry => n_preretry[], :t_pre => t_pre[], :infeas_pre_disagree => n_predis[], :infeas_pre_check_timeout => n_preto[], :basis_loaded => n_bload[], :node_lp_cap_hits => n_cap[], :basis_failed => n_bfail[], :prune_checked => n_prchk[], :prune_disagree => n_prdis[], :resp_hits => n_resp[], :gcsafe => _NZ_GCSAFE_STATE[], :gc_wait => (Base.gc_num().total_time_to_safepoint - gcw0) / 1e9)
end
