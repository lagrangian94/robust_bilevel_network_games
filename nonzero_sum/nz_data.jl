"""
nz_data.jl — 비제로섬 확장 (docs/nonzero_sum_extension.html) 용 일반 LP recourse 데이터.

모델
  리더:     min_{x∈X} q_xᵀx + sup_{(P,P̃)} sup_{h∈Ψ(x,P̃)} CVaR_β^P[ φ(x,h,ξ) ]
  follower: Ψ(x,P̃) = argmax_{h∈H} c0ᵀh + E^P̃[Q(x,h,ξ)],   H = {h ≥ 0 : W h ≤ w}
  recourse: Q(x,h,ξˢ) = max{ cᵀy : A y ≤ r_s(x,h), y ≥ 0 }
            r_s(x,h)_k = u_k^s + hcoef_k · h_{hrow(k)} + g_k^s · x_{xrow(k)}     (행마다 h, x 는 많아야 하나)
  리더 평가: φ_s = max{ ℓᵀy : y ∈ argmax Q }   (pessimistic, ℓ = c 이면 zero-sum)

상계 (Ω 의 McCormick / exact penalty)
  thetaU  : follower 최적성 벌점 θ^U (html Step 2, Lemma 2)
  piLU[k] : 리더 쪽 π̂_k 상계 (A^Tπ ≥ ℓ + θ^U c 의 최적 꼭짓점), x 행에서만 사용
  piFU[k] : follower 자신의 recourse dual 상계 (A^Tπ ≥ c), x 행에서만 사용. π̃ 상계 = λ^U·piFU
  lambdaU : follower 블록 exact penalty λ^U (ψ⁰ = λx McCormick)
  varpiU[k]: ϖ_k/r_s 의 상계 (h 행에서만 사용: bilinear α·ϖ 의 spatial B&B 용)
"""

using LinearAlgebra, Random, Printf

struct NZData
    name::String
    # recourse
    A::Matrix{Float64}          # m × ny
    c::Vector{Float64}          # ny, follower recourse 목적 (max)
    ell::Vector{Float64}        # ny, 리더 평가 (max 형태 손실)
    u::Matrix{Float64}          # m × S
    hrow::Vector{Int}           # m, 0 = h 없음
    hcoef::Vector{Float64}      # m
    xrow::Vector{Int}           # m, 0 = x 없음
    g::Matrix{Float64}          # m × S
    # follower here-and-now
    W::Matrix{Float64}          # nW × nh
    wvec::Vector{Float64}       # nW
    c0::Vector{Float64}         # nh, follower 1단계 목적 (max)
    hU::Vector{Float64}         # nh, h 상한 (H 에서 유도)
    # 리더
    nx::Int
    x_allowed::Vector{Bool}
    gamma::Int                  # 1ᵀx ≤ γ
    qx::Vector{Float64}         # 출점비 등 (OMP 목적에만)
    # ambiguity
    S::Int
    q_hat::Vector{Float64}
    eps_hat::Float64
    eps_tilde::Float64
    beta::Float64
    # 상계
    thetaU::Float64
    piLU::Vector{Float64}
    piFU::Vector{Float64}
    lambdaU::Float64
    varpiU::Vector{Float64}
    # 진단용 (location)
    meta::Dict{Symbol,Any}
end

nz_m(nd::NZData) = size(nd.A, 1)
nz_ny(nd::NZData) = size(nd.A, 2)
nz_nh(nd::NZData) = length(nd.c0)

"r_s(x,h) 계산 (x, h 는 실수 벡터)"
function nz_rhs(nd::NZData, x, h, s)
    r = copy(nd.u[:, s])
    for k in eachindex(r)
        nd.hrow[k] > 0 && (r[k] += nd.hcoef[k] * h[nd.hrow[k]])
        nd.xrow[k] > 0 && (r[k] += nd.g[k, s] * x[nd.xrow[k]])
    end
    return r
end

"같은 데이터에서 상계만 바꾼 사본"
function nz_with_bounds(nd::NZData; thetaU=nd.thetaU, lambdaU=nd.lambdaU, piLU=nothing)
    if piLU === nothing
        # piLU 가 θ^U 에 비례하도록 만든 인스턴스 (location) 는 다시 계산
        piLU = haskey(nd.meta, :piLU_fn) ? nd.meta[:piLU_fn](thetaU) : nd.piLU
    end
    return NZData(nd.name, nd.A, nd.c, nd.ell, nd.u, nd.hrow, nd.hcoef, nd.xrow, nd.g,
                  nd.W, nd.wvec, nd.c0, nd.hU, nd.nx, nd.x_allowed, nd.gamma, nd.qx,
                  nd.S, nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta,
                  thetaU, piLU, nd.piFU, lambdaU, nd.varpiU, nd.meta)
end


# =====================================================================
# SGB128 (John Burkardt, https://people.sc.fsu.edu/~jburkardt/datasets/cities/cities.html), data/sgb128/
# =====================================================================
const _SGB128_DIR = joinpath(@__DIR__, "data", "sgb128")
_sgb_lines(file) = [l for l in eachline(joinpath(_SGB128_DIR, file)) if !isempty(strip(l)) && !startswith(l, "#")]
"128 × 2 XY 좌표"
sgb128_xy() = permutedims(hcat([parse.(Float64, split(l)) for l in _sgb_lines("sgb128_xy.txt")]...))
"128 개 도시 이름"
sgb128_names() = strip.(_sgb_lines("sgb128_name.txt"))


# =====================================================================
# 사전예약 location (html "적용 예", Goyal et al. 2023 JOC §7.1 기본 사례 규모)
# =====================================================================
"""
    make_location_instance(; n_loc=8, nA=1, nB=4, NB=4, S=5, seed=1, ...)

Goyal et al. (2023) §7.1 기본 사례를 따른다 (SGB128 좌표 대신 [0,1]² 균등 좌표).
  위치 n_loc 개 중 A 점포 nA, B 후보 nB, 나머지가 고객.
  용량: A 점포 bA (기본 720), B 후보 bB (기본 360). 출점비 qB (기본 305), 출점 수 ≤ NB.
  수요: 고객마다 U[dlo, dhi] (기본 [30, 240]), 시나리오 S 개, q̂ 균등.
  수송비 c_ij = Euclidean 거리를 dist_res (기본 0.1) 단위로 반올림.
  리더 손실 ℓ = −v (B 점포에서 나가는 모든 flow, v=5).
확장 (html): 고객이 사전예약 h 를 정함 (단위 예약비 f), 예약분은 c_ij, 현장구매는 c_ij + p.
  reservation = :pooled (기존) — 점포별 예약 풀 h_i, Σ_j y^R_ij ≤ h_i, 한도 1ᵀh ≤ wres
              | :pair         — 고객×점포별 선주문 h_ij, y^R_ij ≤ h_ij, 점포별 선판매 할당량 Σ_j h_ij ≤ quota
                                (aggregate LP 가 개별 고객 선주문의 가격 균형과 같아지는 형태, test_nz_equilibrium.jl)
RCR: A 총용량 > 최대 총수요 (assert).
"""
function make_location_instance(; coords::Symbol=:sgb128, n_loc=8, nA=1, nB=4, NB=4, S=5, seed=1,
        bA=720.0, bB=360.0, qB=305.0, dlo=30.0, dhi=240.0, v=5.0,
        p=nothing, f=nothing, wres=300.0, dist_res=nothing,
        reservation::Symbol=:pooled, quota=wres,
        eps_hat=0.2, eps_tilde=0.2, beta=0.4,
        thetaU=:circuit, lambdaU=nothing)
    reservation in (:pooled, :pair) || error("reservation = :pooled | :pair")
    coords in (:sgb128, :random) || error("coords = :sgb128 | :random")
    # λᵁ (follower exact penalty) 는 비용 단위에 묶인 상수. follower McCormick 상한이 λᵁ·(c^max + p) 라
    # 척도를 따라가야 한다. random 에서 경계가 v/해상도 = 50 → λᵁ = 1000. sgb128 (해상도 1) 은 경계 ~5 로 보고 100.
    lambdaU = lambdaU === nothing ? (coords == :sgb128 ? 100.0 : 1000.0) : Float64(lambdaU)
    # 거리 척도가 좌표에 따라 다르므로 p, f, 해상도의 기본값도 다름
    #   :sgb128 — XY 좌표가 마일 척도 (점포-고객 거리 217~2,971, 평균 1,162) → 정수 반올림, p = 200, f = 50
    #   :random — [0,1]² → 0.1 단위 반올림, p = 0.3, f = 0.1 (2026-10-06~07 의 초기 실험)
    sgb = coords == :sgb128
    p = p === nothing ? (sgb ? 200.0 : 0.3) : Float64(p)
    f = f === nothing ? (sgb ? 50.0 : 0.1) : Float64(f)
    dist_res = dist_res === nothing ? (sgb ? 1.0 : 0.1) : Float64(dist_res)
    rng = MersenneTwister(seed)
    nst = nA + nB
    ncu = n_loc - nst
    @assert ncu >= 1
    st = 1:nst                      # 1..nA = A, nA+1..nst = B 후보
    cu = nst+1:n_loc
    if sgb
        # Goyal et al. (2023) §7.1 기본 사례: SGB128 (Burkardt) 의 처음 8 개 도시, 적격 위치 1,2,3,4,6 중 A 는 6,
        # 나머지 5,7,8 이 고객. c_ij = 좌표 Euclidean 거리. 여기서는 점포 (A 먼저) → 고객 순으로 재배열
        @assert (n_loc, nA, nB) == (8, 1, 4) "coords=:sgb128 은 Goyal 기본 사례 (n_loc=8, nA=1, nB=4) 전용"
        xy = sgb128_xy()
        order = [6, 1, 2, 3, 4, 5, 7, 8]
        coord = xy[order, :]
        cities = sgb128_names()[order]
    else
        coord = rand(rng, n_loc, 2)
        cities = String[]
    end
    dist = [round(norm(coord[i, :] .- coord[j, :]) / dist_res) * dist_res for i in st, j in cu]
    ξ = round.(dlo .+ (dhi - dlo) .* rand(rng, ncu, S); digits=1)
    @assert nA * bA > maximum(sum(ξ; dims=1)) "RCR/상계 논증: A 총용량 > 최대 총수요 필요"

    isB = [i > nA for i in 1:nst]
    # y 인덱스: yR[i,j], yS[i,j]
    iyR(i, j) = (i - 1) * ncu + j
    iyS(i, j) = nst * ncu + (i - 1) * ncu + j
    ny = 2 * nst * ncu
    # 행: 수요 ncu, 예약 (pooled: nst, pair: nst·ncu), 용량 nst
    pair = reservation == :pair
    nres = pair ? nst * ncu : nst
    hidx(i, j) = pair ? (i - 1) * ncu + j : i          # 예약 행 / here-and-now 인덱스
    rdem(j) = j
    rres(i, j) = ncu + hidx(i, j)
    rcap(i) = ncu + nres + i
    m = ncu + nres + nst
    A = zeros(m, ny)
    for i in 1:nst, j in 1:ncu
        A[rdem(j), iyR(i, j)] = -1; A[rdem(j), iyS(i, j)] = -1
        A[rres(i, j), iyR(i, j)] = 1
        A[rcap(i), iyR(i, j)] = 1;  A[rcap(i), iyS(i, j)] = 1
    end
    c = zeros(ny); ell = zeros(ny)
    for i in 1:nst, j in 1:ncu
        c[iyR(i, j)] = -dist[i, j]
        c[iyS(i, j)] = -(dist[i, j] + p)
        if isB[i]
            ell[iyR(i, j)] = -v; ell[iyS(i, j)] = -v
        end
    end
    u = zeros(m, S); g = zeros(m, S)
    hrow = zeros(Int, m); hcoef = zeros(m); xrow = zeros(Int, m)
    for s in 1:S, j in 1:ncu
        u[rdem(j), s] = -ξ[j, s]
    end
    for i in 1:nst, j in (pair ? (1:ncu) : (1:1))
        hrow[rres(i, j)] = hidx(i, j); hcoef[rres(i, j)] = 1.0
    end
    for i in 1:nst
        if isB[i]
            xrow[rcap(i)] = i - nA
            g[rcap(i), :] .= bB
        else
            u[rcap(i), :] .= bA
        end
    end
    nx = nB
    if pair
        nh = nst * ncu
        W = zeros(nst, nh)
        for i in 1:nst, j in 1:ncu; W[i, hidx(i, j)] = 1.0; end
        qv = quota isa Real ? fill(Float64(quota), nst) : Float64.(quota)
        wvec = qv; c0 = fill(-f, nh); hU = [qv[i] for i in 1:nst for j in 1:ncu]
    else
        W = ones(1, nst); wvec = [wres]; c0 = fill(-f, nst); hU = fill(wres, nst)
    end

    # ---- 상계 (html Step 2 + location 구조) ----
    cmax = maximum(dist)
    # θ^U 두 가지 (html Step 2). 거리 해상도 dist_res, 프리미엄 p 의 공통분모 D.
    #   :lemma2  — Cramer: θ^U = D‖ℓ‖₁ (일반 TU 논증, 매우 느슨: 기본 사례 1200)
    #   :circuit — Y 의 edge 방향은 TU 행렬의 circuit (성분 ±1). circuit 하나가 바꾸는 B 총판매량은 ≤ 1 단위라
    #              Δ(ℓᵀy) ≤ v, follower 비용 변화는 0 이 아니면 dist_res 의 배수라 ≥ dist_res.
    #              θ* = max(Δℓ / −Δc) ≤ v / dist_res (기본 사례 50; nz_theta_diag 로 50 이 실제로 달성됨을 확인)
    D = 1 / dist_res                 # 비용이 1/D 단위 정수배 (sgb128: D = 1)
    @assert isapprox(p * D, round(p * D); atol=1e-9) "p 는 dist_res 의 배수여야 함"
    θU = thetaU isa Real ? Float64(thetaU) :
         thetaU == :lemma2 ? D * norm(ell, 1) :
         thetaU == :circuit ? v * D : error("thetaU = 숫자 | :circuit | :lemma2")
    # A 점포에 항상 여유 → 수요 dual μ_j ≤ θ(cmax+p), 용량 dual ≤ max μ (리더 벌점 LP: 비용 θc + v[B])
    piLU_fn(θ) = [xrow[k] > 0 ? θ * (cmax + p) + v : 0.0 for k in 1:m]
    piFU = [xrow[k] > 0 ? cmax + p : 0.0 for k in 1:m]          # follower 자신: 같은 논증, θ=1, v=0
    varpiU = [hrow[k] > 0 ? p : Inf for k in 1:m]               # 예약 dual ≤ p (현장구매로 대체 가능)

    meta = Dict{Symbol,Any}(:kind => :location, :coords => coords, :cities => cities, :coord => coord, :dist => dist, :xi => ξ,
        :nA => nA, :nB => nB, :ncu => ncu, :p => p, :f => f, :v => v, :wres => wres,
        :bA => bA, :bB => bB, :cmax => cmax, :D => D, :piLU_fn => piLU_fn,
        :iyR => iyR, :iyS => iyS, :reservation => reservation, :nst => nst,
        :rdem => rdem, :rres => rres, :rcap => rcap, :hidx => hidx, :isB => isB)
    return NZData((sgb ? "location_sgb128" : "location_n$(n_loc)_A$(nA)_B$(nB)") * "_S$(S)_seed$(seed)" * (pair ? "_pair" : ""),
        A, c, ell, u, hrow, hcoef, xrow, g, W, wvec, c0, hU,
        nx, fill(true, nx), NB, fill(qB, nx),
        S, fill(1.0 / S, S), eps_hat, eps_tilde, beta,
        θU, piLU_fn(θU), piFU, lambdaU, varpiU, meta)
end


# =====================================================================
# 원고 max-flow interdiction (TrueDROData) → NZData  (zero-sum 교차검증용)
# =====================================================================
"""
    nz_from_true_dro(td; ell_equals_c=true, thetaU=0.0)

y = [arc flow (K); dummy t→s flow], 보존식은 부등식 두 개, 용량 행 y_k ≤ ξ_k + α_k − v ξ_k x_k.
ℓ = c = e_ts 이고 θ^U = 0 이면 원고 Ω (build_true_dro_subproblem) 와 같은 값을 줘야 한다.
"""
function nz_from_true_dro(td; thetaU=0.0)
    K, S, nv1 = td.num_arcs, td.S, td.nv1
    ny = K + 1
    N = hcat(td.Ny, td.Nts)
    A = vcat(N, -N, hcat(Matrix{Float64}(I, K, K), zeros(K)))
    m = size(A, 1)
    c = zeros(ny); c[end] = 1.0
    u = zeros(m, S); g = zeros(m, S)
    hrow = zeros(Int, m); hcoef = zeros(m); xrow = zeros(Int, m)
    for k in 1:K
        row = 2nv1 + k
        u[row, :] .= td.xi_bar[k, :]
        hrow[row] = k; hcoef[row] = 1.0
        xrow[row] = k
        g[row, :] .= -td.v[k, :] .* td.xi_bar[k, :]
    end
    piLU = [xrow[k] > 0 ? td.phi_hat_U : 0.0 for k in 1:m]
    piFU = [xrow[k] > 0 ? td.phi_tilde_U / td.lambda_U : 0.0 for k in 1:m]
    varpiU = [hrow[k] > 0 ? 1.0 : Inf for k in 1:m]
    return NZData("maxflow_K$(K)_S$(S)", A, c, copy(c), u, hrow, hcoef, xrow, g,
        ones(1, K), [td.w], zeros(K), fill(td.w, K),
        K, Vector{Bool}(td.interdictable_arcs[1:K]), td.gamma, zeros(K),
        S, td.q_hat, td.eps_hat, td.eps_tilde, td.beta === nothing ? 0.0 : td.beta,
        thetaU, piLU, piFU, td.lambda_U, varpiU, Dict{Symbol,Any}(:kind => :maxflow))
end
