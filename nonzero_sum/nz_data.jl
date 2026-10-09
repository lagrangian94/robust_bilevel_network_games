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

"""
종속 ambiguity set (belief 결합) 반경 δ: 𝒟_δ = {(p, p̃) ∈ D̂ × D̃ : d_TV(p, p̃) ≤ δ}.
meta[:delta] 에 저장 (없으면 Inf = 기존 rectangular). δ ≥ ε̂ + ε̃ 이면 결합은 비활성 (삼각부등식).
"""
nz_delta(nd::NZData) = Float64(get(nd.meta, :delta, Inf))
"결합 제약을 Ω 등에 넣어야 하는가 (유한 δ)"
nz_coupled(nd::NZData) = isfinite(nz_delta(nd))

"δ 만 바꾼 사본 (meta 는 얕은 복사라 원본과 공유하지 않음). δ = Inf 는 rectangular"
function nz_with_delta(nd::NZData, δ::Real)
    meta = copy(nd.meta); meta[:delta] = Float64(δ)
    return NZData(nd.name, nd.A, nd.c, nd.ell, nd.u, nd.hrow, nd.hcoef, nd.xrow, nd.g,
                  nd.W, nd.wvec, nd.c0, nd.hU, nd.nx, nd.x_allowed, nd.gamma, nd.qx,
                  nd.S, nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta,
                  nd.thetaU, nd.piLU, nd.piFU, nd.lambdaU, nd.varpiU, meta)
end

"""
    nz_theta_circuit_exact(nd; maxpaths=10^8) -> (θ̄, 정보)

location (reservation = :pair) 의 정확한 circuit 상계 θ̄ = max { ℓᵀg / (−cᵀg) : g 는 [A; −I] 의 circuit, cᵀg < 0, ℓᵀg > 0 }.

Lemma phi (ii) 에 쓰이는 근거: 비관적 최적 꼭짓점 y* 에서 Y(b) = {y ≥ 0 : Ay ≤ b} 의 접뿔은 모서리 방향 (= circuit) 들로
생성되고, 최적면 안의 방향은 ℓᵀg ≤ 0 이므로 ℓᵀ(y − y*) ≤ θ̄ (z* − cᵀy) 가 모든 y ∈ Y(b), 모든 b 에서 성립 → θᵁ = θ̄ 로 충분.

pair 행렬의 행은 수요 (고객 j), 선주문 (y^R_ij 하나만), 용량 (점포 i) 이고 선주문 행은 열 하나에만 걸려 circuit 조건에
제약을 더하지 않는다. 따라서 circuit 은 점포–고객 이분 다중그래프 (쌍마다 R, S 두 호) 의 단순 교대 경로 (부호 +, −, + …,
끝점 행은 여유) 또는 짝수 사이클이다. 사이클과 고객–고객 경로는 점포마다 들어오고 나가는 흐름이 같아 ℓᵀg = 0 이므로,
모든 단순 경로를 열거해 비율의 최댓값을 구한다 (g 와 −g 모두). 기존 :circuit (v / 거리 해상도) 은 비용 변화를 해상도로
하한한 것이라 이 값 이상이다.
"""
function nz_theta_circuit_exact(nd::NZData; maxpaths::Int=10^8)
    meta = nd.meta
    get(meta, :reservation, :none) == :pair || error("nz_theta_circuit_exact: location reservation = :pair 전용")
    nst, ncu = meta[:nst], meta[:ncu]
    iyR, iyS = meta[:iyR], meta[:iyS]
    cost(col) = -nd.c[col]                                  # follower 비용 (nd.c 는 follower 가 최대화하는 −비용)
    # 호: (열, 점포, 고객). 노드 번호: 점포 1..nst, 고객 nst+1..nst+ncu
    arcs = [(col, i, j) for i in 1:nst, j in 1:ncu for col in (iyR(i, j), iyS(i, j))]
    inc = [Int[] for _ in 1:(nst + ncu)]
    for (a, (_, i, j)) in enumerate(arcs); push!(inc[i], a); push!(inc[nst + j], a); end
    best = 0.0; arg = nothing; npaths = 0
    visited = falses(nst + ncu)
    path = Int[]
    function visit(node, sign, gc, gl)
        for a in inc[node]
            col, i, j = arcs[a]
            nxt = node == i ? nst + j : i
            visited[nxt] && continue
            gc2 = gc + sign * cost(col); gl2 = gl + sign * nd.ell[col]
            npaths += 1
            npaths > maxpaths && error("nz_theta_circuit_exact: 경로 수가 maxpaths 를 넘음")
            push!(path, a)
            for σ in (1.0, -1.0)                            # g 와 −g
                dc, dl = σ * gc2, σ * gl2                     # dc = 비용 증가 = −cᵀg, dl = ℓᵀg
                if dc > 1e-9 && dl > 1e-12 && dl / dc > best
                    best = dl / dc; arg = (copy(path), σ, dc, dl)
                end
            end
            visited[nxt] = true
            visit(nxt, -sign, gc2, gl2)
            visited[nxt] = false
            pop!(path)
        end
    end
    for start in 1:(nst + ncu)
        visited[start] = true
        visit(start, 1.0, 0.0, 0.0)
        visited[start] = false
    end
    desc = arg === nothing ? "ℓᵀg > 0 인 비용 증가 circuit 없음" :
        join([(s = (k % 2 == 1 ? "+" : "−"); (col, i, j) = arcs[a];
               "$(s)$(col == iyR(i, j) ? "R" : "S")($(i),$(j))") for (k, a) in enumerate(arg[1])], " ") *
        "  (부호 $(arg[2] > 0 ? "+" : "−"), 비용 증가 $(arg[3]), ℓ 변화 $(arg[4]))"
    return best, Dict(:npaths => npaths, :circuit => desc)
end

"h 상한 (hU) 만 바꾼 사본 (α-B&B 프리솔브용)"
nz_with_hU(nd::NZData, hU) = NZData(nd.name, nd.A, nd.c, nd.ell, nd.u, nd.hrow, nd.hcoef, nd.xrow, nd.g,
    nd.W, nd.wvec, nd.c0, Float64.(hU), nd.nx, nd.x_allowed, nd.gamma, nd.qx,
    nd.S, nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta,
    nd.thetaU, nd.piLU, nd.piFU, nd.lambdaU, nd.varpiU, nd.meta)

"""
    nz_presolve_hU(nd, x̄) -> (hU', 고정된 좌표 수)

x̄ 에서 follower 의 최적 반응이 항상 0 인 예약 좌표의 상한을 0 으로 (location 전용, 다른 인스턴스는 그대로).
x̄ 에서 닫힌 B 점포는 용량이 0 이라 그 점포의 예약분을 받을 수 없는데 예약비 f > 0 (c0 < 0) 는 내므로, 그 예약은
엄격히 지배되어 Ψ(x̄, p̃) 의 모든 원소에서 0 이다. 정의역을 {그 좌표 = 0} 으로 줄여도 Ψ 를 포함하므로 V*(x̄) 는 그대로이고,
줄인 정의역의 점은 원래 Ω 의 실행가능해라 거기서 만든 cut 도 유효하다.
"""
function nz_presolve_hU(nd::NZData, x̄)
    hU = copy(nd.hU)
    get(nd.meta, :kind, nothing) == :location || return hU, 0
    nA, nst, ncu = nd.meta[:nA], nd.meta[:nst], nd.meta[:ncu]
    hidx = nd.meta[:hidx]
    nfix = 0
    for i in nA+1:nst
        x̄[i - nA] > 0.5 && continue
        for j in (nd.meta[:reservation] == :pair ? (1:ncu) : (1:1))
            k = hidx(i, j)
            nd.c0[k] < 0 || error("nz_presolve_hU: 예약비가 양수가 아님 (c0[$k] = $(nd.c0[k])) — 지배 논증이 성립 안 함")
            hU[k] > 0 && (hU[k] = 0.0; nfix += 1)
        end
    end
    return hU, nfix
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
        thetaU=:circuit, lambdaU=nothing, bigm::Symbol=:tight)
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
    # x 행 (B 후보의 용량 행) dual 상한. A 점포 총용량 > 총수요라 모든 최적해에서 A 에 여유가 있고 A 의 용량 dual 은 0.
    #   → 수요 dual μ_j ≤ (A 에서 현장구매 비용) = θ(c_Aj + p)  (follower 자신은 θ = 1)
    #   → B 점포 i 의 용량 dual 을 최소로 고른 최적 dual 이 존재하고 (용량 우변 ≥ 0 이라 목적 불증가),
    #     그 값은 max_j (μ_j − 점포 i 의 구매비용)⁺ ≤ θ · max_j (c_Aj + p − c_ij)⁺
    #   bigm = :tight (기본) 은 이 점포별 상한, :loose 는 이전의 c^max + p 공통 상한 (리더는 +v).
    cA = [maximum(dist[a, j] for a in 1:nA) for j in 1:ncu]
    gapB = [maximum(max(cA[j] + p - dist[i, j], 0.0) for j in 1:ncu) for i in 1:nst]
    store_of_xrow = Dict(rcap(i) => i for i in 1:nst)
    # (분기 안에서 같은 이름의 지역 함수를 정의하면 Julia 가 뒤의 정의로 덮어쓰므로 익명 함수로 대입)
    bigm in (:tight, :loose) || error("bigm = :tight | :loose")
    piLU_fn = bigm == :tight ?
        (θ -> [xrow[k] > 0 ? θ * gapB[store_of_xrow[k]] : 0.0 for k in 1:m]) :
        (θ -> [xrow[k] > 0 ? θ * (cmax + p) + v : 0.0 for k in 1:m])
    piFU = bigm == :tight ? [xrow[k] > 0 ? gapB[store_of_xrow[k]] : 0.0 for k in 1:m] :
                            [xrow[k] > 0 ? cmax + p : 0.0 for k in 1:m]
    varpiU = [hrow[k] > 0 ? p : Inf for k in 1:m]               # 예약 dual ≤ p (현장구매로 대체 가능)

    meta = Dict{Symbol,Any}(:kind => :location, :coords => coords, :cities => cities, :coord => coord, :dist => dist, :xi => ξ,
        :nA => nA, :nB => nB, :ncu => ncu, :p => p, :f => f, :v => v, :wres => wres,
        :bA => bA, :bB => bB, :cmax => cmax, :D => D, :piLU_fn => piLU_fn,
        :iyR => iyR, :iyS => iyS, :reservation => reservation, :nst => nst,
        :rdem => rdem, :rres => rres, :rcap => rcap, :hidx => hidx, :isB => isB, :gapB => gapB, :bigm => bigm)
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
