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

사용: r = nz_alpha_bnb_dr(nd, x̄; nworkers=12, time_limit=600.0, rel_gap=1e-3)
      인자 전체와 기본값은 nz_alpha_bnb_dr 의 docstring. Benders 에서는 nz_benders.jl 의 `bnb = :dr` (기본).
"""

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
    cc::Vector{Tuple{Int,Int,ConstraintRef,ConstraintRef,Vector{Tuple{VariableRef,Float64}}}}   # 성분별 상보성 (a, s, ≥ lo 행, ≤ hi 행, 행 활동의 (변수, 계수))
    xx::Vector{Tuple{Int,Int,ConstraintRef,ConstraintRef,ConstraintRef}}   # 교차 정상성 RLT (i, i′, A 행, B 행, C 행)
    kmc::Vector{Tuple{Int,Int,ConstraintRef,ConstraintRef,Float64}}         # Zκ = α κ 의 κ 상한 쪽 McCormick (i, l, 위 행, 아래 행, κmax)
    pairs::Vector{Tuple{Int,Int}}                                           # 쌍대 차이 범위의 성분 쌍 (a, a′), a < a′
    ddF::Array{ConstraintRef,3}                                             # 2 × |pairs| × S: dlo d ≤ π̃_a − π̃_a′ ≤ dhi d
    ddL::Array{ConstraintRef,3}                                             # 2 × |pairs| × S: dlo r ≤ ϖ_a − ϖ_a′ ≤ dhi r
    nmip::Int
    nfail::Int                                   # 범위·값 MIP 가 Gurobi 오류로 실패한 횟수 (그 성분은 부모 값 유지)
end

"노드 LP (P) 에 범위 행의 자리를 만든다 (계수는 _nz_dr_set! 가 노드마다 채움)"
function _nz_dr_attach!(P::_NZSepLP, nd::NZData, x̄; optimizer, vcut::Bool=true, cslift::Bool=true, fwd::Bool=false,
                        lsd::Bool=false, cc::Bool=false, xx::Bool=false, kmc::Bool=false, dd::Bool=false)
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
    xxv = Tuple{Int,Int,ConstraintRef,ConstraintRef,ConstraintRef}[]
    kmcv = Tuple{Int,Int,ConstraintRef,ConstraintRef,Float64}[]
    xx && !cslift && error("_nz_dr_attach!: xx 는 cslift 가 필요")
    kmc && !cslift && error("_nz_dr_attach!: kmc 는 cslift 가 필요")
    prlt = Array{ConstraintRef,3}(undef, 0, 0, 0)
    zPm = Matrix{VariableRef}(undef, 0, 0)
    if cslift
        nW = size(nd.W, 1); κ = v[:κ]; α = v[:α]
        Zκ = Dict{Tuple{Int,Int},VariableRef}()
        for i in 1:nh, l in 1:nW
            (nd.hU[i] > 0 && (nd.W[l, i] != 0 || xx)) || continue      # xx: 다른 묶음의 κ 와의 곱도 필요
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
        if xx                                             # 교차 정상성 RLT (설명은 아래 XX 주석)
            all(nd.hcoef[k] > 0 for k in Hk) || error("_nz_dr_attach!: xx — hcoef ≤ 0 인 H 행")
            ζF = v[:ζF]; act = [i for i in 1:nh if nd.hU[i] > 0]
            for i in act, ip in act
                ip == i && continue
                e0 = sum(0.0 * κ[l] for l in 1:nW) + sum(0.0 * v[:π̃][k, s] for k in Hk, s in 1:S if nd.hrow[k] == i) + sum(0.0 * ζF[ip, s] for s in 1:S)
                WZ = sum(nd.W[l, i] * Zκ[(ip, l)] for l in 1:nW if nd.W[l, i] != 0; init=AffExpr(0.0))
                cA = @constraint(mdl, WZ - nd.c0[i] * α[ip] + e0 >= 0.0)
                cB = @constraint(mdl, -WZ + nd.c0[i] * α[ip] + e0 >= 0.0)
                cC = @constraint(mdl, WZ - nd.c0[i] * α[ip] + sum(0.0 * ζF[ip, s] for s in 1:S) <= 1e30)
                push!(xxv, (i, ip, cA, cB, cC))
            end
        end
    end
    # FWD: follower 블록의 시나리오별 약쌍대  cᵀȳˢ ≤ (uˢ + gˢ x̄)ᵀπ̃ˢ + Σ_{k∈H} hcoef_k ζΠ_ks   (ζΠ = α π̃, X 가 있어야 함)
    #   정확한 해에서 ȳˢ = d_s y, π̃ˢ = d_s σ (y: α 에서 실행가능, σ: 쌍대 실행가능) 이므로 약쌍대성 cᵀy ≤ (u + g x̄ + Bα)ᵀσ 에 d_s 를 곱한 것.
    #   리더 블록의 WD (nz_omega.jl) 와 같은 종류. Flam (합친 강쌍대) + X 의 항등식은 Σ_s [쌍대목적_s − primal_s] ≤ 0 만 주므로
    #   꺾임을 품은 상자에서는 시나리오끼리 부호가 다른 gap 을 상쇄할 수 있다 — 이 행이 시나리오마다 gap ≥ 0 을 강제한다.
    fw = ConstraintRef[]
    if fwd
        cslift || error("_nz_dr_attach!: fwd 는 cslift (ζΠ) 가 필요")
        all(nd.hU[nd.hrow[k]] <= 0 || k in Hk for k in Hr) || error("_nz_dr_attach!: fwd — ζΠ 가 없는 H 행이 있음 (ϖᵁ 무한)")
        m_, ny_ = nz_m(nd), nz_ny(nd); ȳ = v[:ȳ]; πt = v[:π̃]
        coefπ(k, s) = nd.u[k, s] + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0)
        fw = [@constraint(mdl, sum(nd.c[j] * ȳ[j, s] for j in 1:ny_ if nd.c[j] != 0)
                               - sum(coefπ(k, s) * πt[k, s] for k in 1:m_ if coefπ(k, s) != 0)
                               - sum(nd.hcoef[Hk[a]] * zPm[a, s] for a in 1:n) <= 0.0) for s in 1:S]
    end
    # LSD: 리더 블록의 시나리오별 강쌍대 (WD 의 반대 방향)  cᵀŷˢ ≥ (uˢ + gˢ x̄)ᵀϖˢ + Σ_{k∈H} hcoef_k ζW_ks.
    #   정확한 해에서 ŷˢ = r_s y (y: pessimistic recourse 최적해), ϖˢ = r_s σ (σ: 최적 쌍대) 이므로 등식 (벌점 항 0). WD 와 합쳐 등식이 된다.
    if lsd
        m_, ny_ = nz_m(nd), nz_ny(nd); ŷ = v[:ŷ]; ϖ = v[:ϖ]
        coefϖ(k, s) = nd.u[k, s] + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0)
        jjof2 = Dict(k => j for (j, k) in enumerate(Hr))
        @constraint(mdl, [s = 1:S], sum(nd.c[j] * ŷ[j, s] for j in 1:ny_ if nd.c[j] != 0)
                                    - sum(coefϖ(k, s) * ϖ[k, s] for k in 1:m_ if coefϖ(k, s) != 0)
                                    - sum(nd.hcoef[k] * v[:ζW][jjof2[k], s] for k in Hr) >= 0.0)
    end
    # CC: 성분별 상보성의 곱 형태. H 행 k (좌표 i) 의 상보성 σ_k (u_k + hcoef α_i − (Ay)_k) = 0 에 d_s (r_s) 를 곱하면
    #   hcoef ζΠ_ks = σ_k · act^F_ks,  act^F_ks = (Aȳˢ)_k − u_ks d_s        (리더: hcoef ζW_ks = σ_k · act^L_ks, act^L = (Aŷˢ)_k − u_ks r_s)
    #   이고 σ_k ∈ [lo, hi], act ≥ 0 이므로  lo·act ≤ hcoef ζ ≤ hi·act.  act ≥ 0 이 보장되는 행 (A_k ≥ 0, u_k ≤ 0, x 없음) 에만 넣는다.
    ccv = Tuple{Int,Int,ConstraintRef,ConstraintRef,Vector{Tuple{VariableRef,Float64}}}[]
    if cc
        ny_ = nz_ny(nd)
        for (a, k) in enumerate(Hk), s in 1:S
            (all(nd.A[k, :] .>= 0) && nd.u[k, s] <= 0 && nd.xrow[k] == 0) || continue
            cols = [j for j in 1:ny_ if nd.A[k, j] != 0]
            if cslift
                act = Tuple{VariableRef,Float64}[(v[:ȳ][j, s], nd.A[k, j]) for j in cols]
                nd.u[k, s] != 0 && push!(act, (v[:d][s], -nd.u[k, s]))
                e0 = sum(0.0 * x for (x, _) in act)
                push!(ccv, (a, s, @constraint(mdl, nd.hcoef[k] * zPm[a, s] - e0 >= 0.0), @constraint(mdl, nd.hcoef[k] * zPm[a, s] - e0 <= 0.0), act))
            end
            act = Tuple{VariableRef,Float64}[(v[:ŷ][j, s], nd.A[k, j]) for j in cols]
            nd.u[k, s] != 0 && push!(act, (v[:r][s], -nd.u[k, s]))
            e0 = sum(0.0 * x for (x, _) in act)
            zW = v[:ζW][jj[a], s]
            push!(ccv, (a, s, @constraint(mdl, nd.hcoef[k] * zW - e0 >= 0.0), @constraint(mdl, nd.hcoef[k] * zW - e0 <= 0.0), act))
        end
    end
    # XX: follower 1 단계 정상성 g_i = (Wᵀκ − c0 − Σ_s Bᵀπ̃ˢ)_i ≥ 0 과 **다른 좌표** i′ 의 상자 인수의 RLT (X 는 자기 좌표 i′ = i 만).
    #   (α_i′ − l_i′) g_i ≥ 0,  (u_i′ − α_i′) g_i ≥ 0,  l_i > 0 이면 g_i = 0 이므로 α_i′ g_i = 0.
    #   곱 α_i′ κ_l = Zκ_i′l (새 변수, l κ ≤ Z ≤ u κ),  α_i′ π̃_ks 는 변수를 만들지 않고 범위로:  lo_ks ζF_i′s ≤ α_i′ π̃_ks ≤ hi_ks ζF_i′s
    #   (π̃_ks = d_s σ_ks, σ_ks ∈ [lo, hi], ζF_i′s = α_i′ d_s ≥ 0).  쌍대가 고정된 상자에서는 Σ_s σ_ks ζF_i′s = Σ_l W_li Zκ_i′l − c0_i α_i′ 로
    #   "σ_k 로 가중한 α_i′ 의 시나리오 평균이 α_i′ 와 같다" 를 모든 (k, i′) 쌍에 대해 강제한다 (X 는 k 가 i′ 자신의 행일 때만).
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
    return _NZDR(mips, Hk, jj, whi, wlo, phi, plo, rlt, stat, vc, zk, prlt, zPm, fw, ccv, xxv, kmcv, prs, ddF, ddL, 0, 0)
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
function _nz_dr_dranges!(DR::_NZDR, nd::NZData, l, u, lo, hi, dlo0, dhi0, wit0, wit; fixtol=1e-4, deadline=Inf)
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
노드 LP 의 INFEASIBLE 판정 재확인. 반환: −Inf (재확인에서도 실행 불가능), 유한값 (풀림: 그 목적값), NaN·Inf (시간·수치 문제).

범위 행을 넣은 노드 LP 는 실행 가능한데도 INFEASIBLE 로 나오는 일이 있다 (test_dr_validity_boxes.jl: 최적 α* 를 품은 상자의
0.5~1%, 범위 행이 없는 기존 완화에서는 0%). 같은 상자를 새 모델에서 풀면 OPTIMAL 이고, 제약이 더 많은 변형이 실행 가능한데
더 적은 변형이 실행 불가능으로 나오기도 한다 → 수학적 무효가 아니라 **warm start 경로의 수치 문제**
(노드마다 행렬 계수를 바꾼 뒤의 dual simplex + FeasibilityTol 1e-9). FeasibilityTol 만 1e-6 으로 풀어 이어서 풀면 그대로 실행 불가능.
여기서는 기저·해 정보를 버리고 (GRBreset) primal simplex 로 FeasibilityTol 1e-6 에서 처음부터 다시 푼다.
"""
function _nz_dr_recheck_infeasible!(model)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-6); set_optimizer_attribute(model, "Method", 0)
    Gurobi.GRBreset(JuMP.unsafe_backend(model), 0)
    z = _nz_solve_lp!(model)
    set_optimizer_attribute(model, "FeasibilityTol", get(model.ext, :lp_tol, 1e-9)); set_optimizer_attribute(model, "Method", get(model.ext, :lp_method, 1))
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
    if !isempty(DR.xx)
        nW = size(nd.W, 1); κv = v[:κ]; ζF = v[:ζF]
        rowsof = Dict{Int,Vector{Int}}()                  # 좌표 i → Hk 안 위치들
        for (a, k) in enumerate(DR.Hk); push!(get!(rowsof, nd.hrow[k], Int[]), a); end
        for (i, ip, cA, cB, cC) in DR.xx
            li, ui = l[ip], u[ip]; as = get(rowsof, i, Int[])
            for ll in 1:nW
                nd.W[ll, i] != 0 || continue
                set_normalized_coefficient(cA, κv[ll], -li * nd.W[ll, i]); set_normalized_coefficient(cB, κv[ll], ui * nd.W[ll, i])
            end
            for s in 1:S
                slo = 0.0; shi = 0.0
                for a in as
                    k = DR.Hk[a]; slo += nd.hcoef[k] * lo[a, s]; shi += nd.hcoef[k] * hi[a, s]
                    set_normalized_coefficient(cA, v[:π̃][k, s], li * nd.hcoef[k]); set_normalized_coefficient(cB, v[:π̃][k, s], -ui * nd.hcoef[k])
                end
                set_normalized_coefficient(cA, ζF[ip, s], -slo); set_normalized_coefficient(cB, ζF[ip, s], shi)
                set_normalized_coefficient(cC, ζF[ip, s], -shi)
            end
            set_normalized_rhs(cA, -li * nd.c0[i]); set_normalized_rhs(cB, ui * nd.c0[i])
            set_normalized_rhs(cC, iseq(i) ? 0.0 : 1e30)
        end
    end
    for (a, s, clo, chi, act) in DR.cc                    # lo·act ≤ hcoef ζ ≤ hi·act
        for (x, cf) in act
            set_normalized_coefficient(clo, x, -lo[a, s] * cf); set_normalized_coefficient(chi, x, -hi[a, s] * cf)
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

"강화 스위치가 기본 (모두 켬) 이 아니면 stderr 에 한 번 알린다 (같은 구성은 프로세스당 한 번)"
function _nz_dr_announce(config)
    off = [n for n in (:dual_range, :comp, :vcut, :cslift) if !getfield(config, n)]
    extra = (fwd=config.fwd, lsd=config.lsd, cc=config.cc, xx=config.xx, lp_tol=config.lp_tol, child_rounds=config.child_rounds, branch_pi=config.branch_pi, sticky=config.sticky, kmc=config.kmc, dd=config.dd, cbranch=config.cbranch)
    isempty(off) && config.branch_point == :mid && extra == (fwd=false, lsd=false, cc=false, xx=false, lp_tol=1e-9, child_rounds=1, branch_pi=false, sticky=false,
                                                                 kmc=false, dd=false, cbranch=false) && return
    lock(_NZ_DR_ANNOUNCE_LOCK) do
        key = (Tuple(off), config.branch_point, extra)
        key in _NZ_DR_ANNOUNCED && return
        push!(_NZ_DR_ANNOUNCED, key)
        println(stderr, "nz_alpha_bnb_dr: 기본이 아닌 구성으로 실행 — 끈 강화: ", isempty(off) ? "없음" : join(off, ", "),
                ", branch_point = ", config.branch_point, ", ", extra)
    end
    return
end

"""
    nz_alpha_bnb_dr(nd, x̄; ...) -> Dict

비제로섬 Ω(x̄) 의 전역 해법. **비제로섬 α-B&B 의 기본 구현** (nz_benders.jl 의 `bnb = :dr`).
nz_alpha_bnb (기존, `bnb = :legacy`) 의 탐색 — worker 병렬, best-first + diving, 위반 RLT 행 분리, Ipopt 휴리스틱 — 에
노드마다 상자별 쌍대 범위 조임을 더한 것. 노드 LP 는 dual simplex 전용.

인자 (기본값 = 2026-10-11 측정에 쓴 구성. 기본값 밖의 조합은 측정·검증하지 않았다)

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
| 강화 (ablation 전용) | `dual_range` | true | 인증서의 상자별 쌍대 범위 (W, F) + RLT. false 면 아래 셋도 꺼지고 기존 완화와 같음 |
| | `comp` | true | 상자가 함의하는 상보성 (C) |
| | `vcut` | true | 시나리오별 값 상한 (V) |
| | `cslift` | true | follower 1 단계 상보성의 RLT (X) |
| 이어 풀기 | `resume` | nothing | 같은 x̄ 의 이전 호출이 돌려준 `r[:state]`. 주면 그 호출이 멈춘 곳 (열린 노드, incumbent) 에서 이어 푼다 |

이어 풀기 (`resume`): Ω(x̄) 는 x̄ 만으로 정해지므로 같은 x̄ 에서는 이전 호출의 B&B 전선 (열린 노드와 그 상한, 가지친 노드의 상한,
incumbent) 이 그대로 유효하다. Benders 가 목표값 조기 종료로 끝난 호출을 같은 x̄ 에서 다시 부를 때 처음부터 다시 풀지 않게 한다
(S = 30 에서 같은 x̄ = {1} 을 260 s + 300 s 로 두 번 풀던 것). 상태는 강화 구성이 같은 호출 사이에서만 넘길 것 (구성이 다르면 오류).

강화 스위치를 하나라도 끄면 stderr 에 한 줄 알린다 (의도하지 않은 구성으로 도는 것을 막기 위해).
시험했다가 뺀 옵션 (꺾임점 3 분할 분기, 분리 행 줄이기, 노드 LP barrier) 은 코드에서 지웠다 — 결과는
docs/dependent_ambiguity/alpha_bnb_nonzero_analysis.md §10.5, 코드는 커밋 3480bae 이전.

반환 Dict: `:LB` (incumbent 값), `:UB` (상한), `:α` (incumbent, 원래 정의역), `:is_exact`, `:nodes`, `:time`, `:root_UB`,
`:solver` (= :nz_alpha_bnb_dr), `:config` (강화·탐색 설정), `:state` (이어 풀기용), `:resumed`, 계측 (`:t_relax`, `:t_range`, `:range_mips`, `:fixfrac`),
수치 문제 횟수 (`:range_fail`, `:infeas`, `:infeas_recovered`, `:numerr`, `:numerr_fail`).
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
                         # 시험 중 (기본 끔): follower 시나리오별 약쌍대, 리더 시나리오별 강쌍대, 성분별 상보성, 노드 LP 허용오차
                         fwd::Bool=false, lsd::Bool=false, cc::Bool=false, xx::Bool=false, lp_tol::Float64=1e-9, branch_pi::Bool=false, sticky::Bool=false,
                         kmc::Bool=false, dd::Bool=false, cbranch::Bool=false,
                         # 이어 풀기: 같은 x̄ 의 이전 호출이 돌려준 r[:state]
                         resume=nothing)
    branch_point in (:alphahat, :mid) || error("branch_point = :alphahat | :mid")
    config = (dual_range=dual_range, comp=dual_range && comp, vcut=dual_range && vcut, cslift=dual_range && cslift,
              branch_point=branch_point, child_rounds=child_rounds, maxrounds=maxrounds, dive_max=dive_max,
              sep_maxadd=sep_maxadd, presolve=presolve, heuristic=heuristic,
              fwd=dual_range && cslift && fwd, lsd=dual_range && lsd, cc=dual_range && cc, xx=dual_range && cslift && xx, lp_tol=lp_tol, branch_pi=dual_range && cslift && branch_pi, sticky=sticky,
              kmc=dual_range && cslift && kmc, dd=dual_range && dd, cbranch=dual_range && comp && cbranch)
    _nz_dr_announce(config)
    n_cbranch = Threads.Atomic{Int}(0)                                       # 상보성 분기 횟수
    n_infeas = Threads.Atomic{Int}(0); n_infrec = Threads.Atomic{Int}(0)     # 실행 불가능 노드 (1e-6 재확인 통과) / 재확인에서 되살아난 노드
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
                                       fwd=config.fwd, lsd=config.lsd, cc=config.cc, xx=config.xx, kmc=config.kmc, dd=config.dd) for i in 1:nworkers] : nothing
    # 노드 LP 허용오차. build_nz_omega 의 기본은 1e-9 + NumericFocus 2 (θ 항의 상쇄 때문에 조였던 것). lp_tol > 1e-9 이면 그 값 + NumericFocus 0 으로 풀고
    #   노드 상한에 상대 1e-6 의 여유를 더한다 (허용오차만큼 덜 수렴한 목적값을 상한으로 쓰는 것에 대한 여유).
    lp_tol > 1e-9 && for P in Ps
        set_optimizer_attribute(P.O.model, "FeasibilityTol", lp_tol); set_optimizer_attribute(P.O.model, "OptimalityTol", lp_tol)
        set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = lp_tol
    end
    zpad(z) = (lp_tol > 1e-9 && isfinite(z)) ? z + 1e-6 * max(1.0, abs(z)) : z
    Fs = [nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    tol(v) = rel_gap * (isfinite(v) ? max(1.0, abs(v)) : 1.0)
    boxok(l) = all(nd.W * l .<= nd.wvec .+ 1e-9)

    lk = ReentrantLock()
    t_relax = Threads.Atomic{Float64}(0.0); t_eval = Threads.Atomic{Float64}(0.0); t_range = Threads.Atomic{Float64}(0.0)
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
            ext = node[8]; eqp = ext.eq; dlo = ext.dlo; dhi = ext.dhi
            if !synced && root_rows[] !== nothing && isempty(P.allrows) && t_end - time() > 30.0
                for (i, j, islo, b) in root_rows[]; sticky ? _nz_sticky_add!(P, i, j, islo, b) : _nz_sep_add!(P, i, j, islo, b); end
                synced = true
            end
            z = -Inf; α̂ = nothing; children = nothing; zE = -Inf; unfinished = false
            pruned_val = -Inf; lo = lop; hi = hip; wit = witp; phib = phibp; fixfrac = NaN
            boxok(l) && ubp <= LB[] + tol(LB[]) && (pruned_val = ubp)
            if ubp > LB[] + tol(LB[]) && boxok(l)
                sticky ? _nz_sticky_set_box!(P, l, u) : _nz_sep_set_box!(P, l, u)
                _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                empty_box = false
                if dual_range
                    tg = @elapsed (rg = _nz_dr_ranges(DR, nd, l, u, lop, hip, witp, phibp; deadline=t_end))
                    Threads.atomic_add!(t_range, tg)
                    if rg === nothing
                        empty_box = true                 # 상자 안에 실행가능한 α 가 없음
                    else
                        lo, hi, wit, phib = rg
                        if config.dd
                            tg2 = @elapsed ((dlo, dhi) = _nz_dr_dranges!(DR, nd, l, u, lo, hi, ext.dlo, ext.dhi, witp, wit; deadline=t_end))
                            Threads.atomic_add!(t_range, tg2)
                        end
                        _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                        fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                    end
                end
                if !empty_box
                    set_time_limit_sec(P.O.model, max(t_end - time(), 0.1))
                    tr = @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd)))
                    Threads.atomic_add!(t_relax, tr)
                    # 실행 불가능 판정 재확인 (_nz_dr_recheck_infeasible!): 그래도 실행 불가능일 때만 가지치기한다. 재확인에서 풀리면
                    #   그 값을 상한으로 쓰고 해는 읽지 않은 채 가장 넓은 좌표를 반으로 나눈다 (수치 오류 노드와 같은 처리).
                    zrec = NaN
                    if z == -Inf && dual_range
                        zc = _nz_dr_recheck_infeasible!(P.O.model)
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
                        if branch_pi && dual_range && length(DR.zP) > 0     # follower 인증서 곱 ζΠ = α π̃ 의 위반도 분기 점수에
                            for (a, k) in enumerate(DR.Hk)
                                i = nd.hrow[k]
                                u[i] - l[i] > min_width || continue
                                viol[i] += sum(abs(value(DR.zP[a, s]) - α̂[i] * value(v[:π̃][k, s])) for s in 1:S)
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
                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi)), (copy(l), copy(u), z, lo, hi, wit, phib, (eq=eqB, dlo=dlo, dhi=dhi)))
                            Threads.atomic_add!(n_cbranch, 1)
                        elseif viol[i] > 1e-9
                            width = u[i] - l[i]; t = α̂[i]
                            (branch_point != :alphahat || t - l[i] < 0.1 * width || u[i] - t < 0.1 * width) && (t = 0.5 * (l[i] + u[i]))
                            u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi)))
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
                :range_mips => dual_range ? sum(D.nmip for D in DRs) : 0, :range_fail => dual_range ? sum(D.nfail for D in DRs) : 0,
                :fixfrac => nodes[] > 0 ? nfixed_sum[] / nodes[] : NaN, :infeas => n_infeas[], :infeas_recovered => n_infrec[],
                :numerr => _NZ_NUMERR[] - numerr0, :numerr_fail => _NZ_NUMERR_FAIL[] - numfail0,
                :cbranch => n_cbranch[], :gcsafe => _NZ_GCSAFE_STATE[], :gc_wait => (Base.gc_num().total_time_to_safepoint - gcw0) / 1e9)
end
