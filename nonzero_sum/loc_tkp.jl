"""
loc_tkp.jl — 원고 6.1 (Non-zero-sum Case: Facility Location Problem) 의 인스턴스 (2026-11-20).

가치 실험은 지난 세션 스크래치의 사본 (geo/nz_data_geo2.jl, geo/fac_builder_geo2.jl, dwLeanB.jl 의 with_takeorpay·setvs!,
설정 dwps3) 에서만 돌았다. 그 구성을 저장소로 옮긴 것이고, 한 가지만 다르다: 크기 단계를 거점별 연속값 x̄_i = b_i/400 (고정 x 평가용)
대신 단계마다 이진변수 x_i^k 로 두어 (용량 행 하나에 x 항목 셋, nz_xentries) Benders 로 풀 수 있게 했다.

  도시: data/sgb128/loc_geos.txt 의 지리 g (SGB128 의 15 도시, A 2 곳), 도시마다 점포 노드와 고객 노드.
  수요: μ_j = 50~100 (인구 affine), ξ_j = μ_j (1/2 + Φ(0.9 Z_{r(j)} + √0.19 E_j)), 지역 r = 경도 3 구간, 난수 MersenneTwister(hash((4242, seed))).
  A: 용량 0.75 Σμ. B: 후보 13 곳, 크기 tiers (기본 200·300·400) 중 하나, 출점 ≤ NB (기본 3), 점포비 q0 + q1 b.
  소매상: 거점마다 약정 h_i ≤ w_i (300), 약정가 f 를 쓰든 안 쓰든 냄, 추가분은 p. B 거점은 Σ_j y^R_ij + s_i = h_i (s = 남는 약정분),
        s 는 용량 행 안 (스크래치와 같음). A 거점은 Σ_j y^R_ij ≤ h_i.
  리더 (B) 손실: −Σ_{i∈B} [v (Σ_j y^R_ij + s_i) + (p/f) v Σ_j y^S_ij]  (가격 기준 수입 f h + p 추가분을 f = v 로 나눈 단위).
  단위: 코드의 리더 단위는 원고 값 / 120 (v = 5 ↔ f = 600, q0 = 150 ↔ 18,000, q1 = 2 ↔ 240). 소매상 비용 (거리, p = 800, f = 600) 은 원고 단위.

θᵁ (원고 Section recourse, Lemma phi (ii) 의 circuit 상계):
  recourse 를 (거점 출하 F_ij, 약정 출하 res_i, 추가분 spot_i) 로 사영하면 s_i = h_i − res_i, B 의 용량 행은 spot_i ≤ b_i − h_i 가 되어
  순환 네트워크 (s → 거점 i 의 두 호 res·spot, 거점 → 고객, 고객 → t, t → s) 의 흐름이다. 리더 손실은 사영에서
  −v Σ_B h_i − (p/f) v Σ_B spot_i 이고 앞 항은 recourse 안에서 상수다. circuit (단순 사이클) 은 s 를 많아야 한 번 지나므로
  B 추가분 합을 많아야 1 단위 바꿔 |ℓᵀg| ≤ (p/f) v, 소매상 비용은 모두 dist_res (10) 의 배수라 비용이 늘면 ≥ dist_res.
  → θᵁ = (p/f) v / dist_res 는 θ̄ 이상 (유효한 상계). 정확한 θ̄ 는 열거하지 않았다.
등식 쌍: 약정 등식은 우변 +h 행과 −h 행 둘이다. α-B&B 는 이 쌍을 nz_eq_pairs 로 찾아 정규형 쌍대로 다룬다 (nz_alpha_bnb_dr.jl 머리말).
π 상계 (piLU, piFU): make_location_instance 의 점포별 상계 gapB_i = max_j (c_Aj + p − c_ij)⁺ (× θ) 를 그대로 쓴다. 약정 물량에서는
  용량 한 단위를 더하면 추가분 하나만 늘릴 수 있어 (res + s = h 고정) 최소 쌍대가 max_j (μ_j − c_ij − p)⁺ ≤ max_j (c_Aj − c_ij)⁺ 이고
  (B 추가분의 ℓ ≤ 0), gapB 는 그보다 크다.

사용:  nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
       (box=true 가 기본: 볼마다 |q_s − q̂_s| ≤ ε/√S)
"""

isdefined(Main, :NZData) || include(joinpath(@__DIR__, "nz_data.jl"))
using Distributions: Normal, cdf

const _LOC_GEOS_FILE = joinpath(@__DIR__, "data", "sgb128", "loc_geos.txt")

"지리 g 의 (도시 15 개, A 2 곳). data/sgb128/loc_geos.txt"
function loc_geo(g::Int)
    for l in eachline(_LOC_GEOS_FILE)
        (isempty(strip(l)) || startswith(l, "#")) && continue
        f = split(l, "|")
        parse(Int, f[1]) == g || continue
        return String.(strip.(split(f[2], ";"))), String.(strip.(split(f[3], ";")))
    end
    error("loc_geo: 지리 $g 가 $_LOC_GEOS_FILE 에 없음")
end

"지리 g 의 도시 집합 (make_location_instance 의 sgb15_set): 이름·SGB128 번호·인구 (sgb128_weight.txt)"
function loc_geo_set(g::Int)
    C, A = loc_geo(g)
    full = String.(sgb128_names())
    idx = [findfirst(==(c), full) for c in C]
    any(isnothing, idx) && error("loc_geo_set: SGB128 에 없는 도시 $(C[isnothing.(idx)])")
    w = [parse(Float64, l) for l in eachline(joinpath(_SGB128_DIR, "sgb128_weight.txt")) if !isempty(strip(l)) && !startswith(l, "#")]
    length(w) == 128 || error("loc_geo_set: sgb128_weight.txt 가 128 줄이 아님")
    return (; names=C, A=A, pop=w[idx], idx=Int.(idx))
end

"도시의 서경 (도). 지역: ≥ 105 서부 1, 89~105 중부 2, < 89 동부 3 (스크래치 fac_builder_geo2.jl 의 lon 규칙)"
function _loc_lon(c)
    full = String.(sgb128_names()); k = findfirst(==(c), full)
    L = [split(l) for l in eachline(joinpath(_SGB128_DIR, "sgb128_dms.txt")) if !isempty(strip(l)) && !startswith(l, "#")]
    v = L[k]
    return parse(Float64, v[5]) + parse(Float64, v[6]) / 60 + parse(Float64, v[7]) / 3600
end
_loc_region(c) = (lon = _loc_lon(c); lon >= 105 ? 1 : lon >= 89 ? 2 : 3)

"""
    make_loc_tkp(; g, S=10, seed=1, eps_hat, eps_tilde, delta=Inf, beta=0.4, box=true,
                  tiers=[200,300,400], NB=3, q0=150.0, q1=2.0, p=800.0, f=600.0, quota=300.0, ρ=0.9, lambdaU=100.0)

원고 6.1 입지 인스턴스 (파일 머리말). x 번호: B 거점 i (= nA+1..nst) 의 단계 k 가 (i − nA − 1)·K + k.
"""
function make_loc_tkp(; g::Int, S::Int=10, seed::Int=1, eps_hat::Real, eps_tilde::Real, delta::Real=Inf, beta::Real=0.4,
                      box::Bool=true, tiers=[200.0, 300.0, 400.0], NB::Int=3, q0::Real=150.0, q1::Real=2.0,
                      p::Real=800.0, f::Real=600.0, quota::Real=300.0, ρ::Real=0.9, lambdaU::Real=100.0)
    tiers = Float64.(tiers); K = length(tiers)
    issorted(tiers) && tiers[1] > 0 || error("make_loc_tkp: tiers 는 양수 오름차순")
    set = loc_geo_set(g)
    v = 5.0; vsm = p / f; dist_res = 10.0
    θU = vsm * v / dist_res
    nd0 = make_sgb15_instance(; S=S, seed=seed, dist=:euclid, dist_res=dist_res, sgb15_set=set,
        p=p, f=f, eps_hat=eps_hat, eps_tilde=eps_tilde, beta=beta, quota=fill(Float64(quota), 15), bB=maximum(tiers),
        thetaU=θU, lambdaU=lambdaU)
    M = nd0.meta
    nA, nst, ncu = M[:nA], M[:nst], M[:ncu]; nB = nst - nA
    iyR, iyS, rres, rcap, rdem, hidx = M[:iyR], M[:iyS], M[:rres], M[:rcap], M[:rdem], M[:hidx]

    # ---- 수요: 지역 요인 (Gaussian copula), 스크래치 make_sgb15_fac 와 같은 난수 ----
    custs = M[:cities][nst+1:nst+ncu]
    reg = [_loc_region(c) for c in custs]
    μ = sgb15_mu(set.pop)
    rng = MersenneTwister(hash((4242, seed)))
    Z = randn(rng, 3, S); E = randn(rng, ncu, S)
    ξ = [round(μ[j] * (0.5 + cdf(Normal(), ρ * Z[reg[j], s] + sqrt(1 - ρ^2) * E[j, s])); digits=1) for j in 1:ncu, s in 1:S]
    u = copy(nd0.u)
    for s in 1:S, j in 1:ncu; u[rdem(j), s] = -ξ[j, s]; end
    nA * M[:bA] > maximum(sum(ξ; dims=1)) || error("make_loc_tkp: A 총용량 > 최대 총수요 필요")

    # ---- 약정 물량: B 거점마다 s_i 열과 −Σ_j y^R_ij − s_i ≤ −h_i 행 ----
    m0, ny0 = nz_m(nd0), nz_ny(nd0)
    A = zeros(m0 + nB, ny0 + nB); A[1:m0, 1:ny0] = nd0.A
    hrow = vcat(nd0.hrow, zeros(Int, nB)); hcoef = vcat(nd0.hcoef, zeros(nB))
    ys(i) = ny0 + (i - nA); rtop(i) = m0 + (i - nA)
    for i in nA+1:nst
        A[rres(i, 1), ys(i)] = 1.0
        A[rcap(i), ys(i)] = 1.0
        for j in 1:ncu; A[rtop(i), iyR(i, j)] = -1.0; end
        A[rtop(i), ys(i)] = -1.0; hrow[rtop(i)] = hidx(i, 1); hcoef[rtop(i)] = -1.0
    end
    c = vcat(nd0.c, zeros(nB))
    ell = vcat(nd0.ell, fill(-v, nB))
    for i in nA+1:nst, j in 1:ncu; ell[iyS(i, j)] = -vsm * v; end          # 가격 기준: 추가분은 p/f 배
    u = vcat(u, zeros(nB, S))
    m, ny = m0 + nB, ny0 + nB

    # ---- 크기 단계: 용량 행 rcap(i) 에 x 항목 K 개 ----
    nx = nB * K
    xt(i, k) = (i - nA - 1) * K + k
    xrow = zeros(Int, m); gmat = zeros(m, S)
    xextra = Tuple{Int,Int,Vector{Float64}}[]
    for i in nA+1:nst, k in 1:K
        push!(xextra, (rcap(i), xt(i, k), fill(tiers[k], S)))
    end
    qx = [q0 + q1 * tiers[k] for i in nA+1:nst for k in 1:K]
    site_x = [i > nA ? [xt(i, k) for k in 1:K] : Int[] for i in 1:nst]
    omp_card = [(site_x[i], 1) for i in nA+1:nst]

    gapB = M[:gapB]
    isBcap = Dict(rcap(i) => i for i in nA+1:nst)
    piLU_fn = θ -> [haskey(isBcap, k) ? θ * gapB[isBcap[k]] : 0.0 for k in 1:m]
    piFU = [haskey(isBcap, k) ? gapB[isBcap[k]] : 0.0 for k in 1:m]
    # 약정 행 (우변 −h) 의 쌍대 상계 ϖᵁ = max_j (c_Aj − c_ij)⁺. 정규형 (nz_eq_pairs) 에서 σ⁻ = ν⁻ 이고, s_i 열의 쌍대 제약
    # σ⁺ + σ_cap − σ⁻ ≥ 0 에서 ν ≥ −σ_cap. 용량 쌍대는 약정 승수와 맞바꿔 (σ_cap ↓ t, ν ↑ t, 목적 변화 (h_i − b_i) t ≤ 0) 추가분 열에서
    # 막힐 때까지 낮출 수 있어 max_j (μ_j − c_ij − p)⁺ ≤ max_j (c_Aj − c_ij)⁺ (여유 있는 A 의 추가분 열에서 μ_j ≤ c_Aj + p).
    # 그 뒤 ν 를 막힐 때까지 낮추면 (우변 h ≥ 0) ν ≤ p 도 성립해 예약 행의 기존 상계 p 와 함께 쓸 수 있다.
    dist = M[:dist]
    cA = [maximum(dist[a, j] for a in 1:nA) for j in 1:ncu]
    Utop = [max(maximum(cA[j] - dist[i, j] for j in 1:ncu), 0.0) for i in nA+1:nst]
    varpiU = vcat(nd0.varpiU, Utop)

    meta = copy(M)
    meta[:xi] = ξ; meta[:region] = reg; meta[:geo] = g; meta[:tiers] = tiers; meta[:K] = K
    meta[:xextra] = xextra; meta[:site_x] = site_x; meta[:omp_card] = omp_card; meta[:piLU_fn] = piLU_fn
    meta[:ys] = ys; meta[:rtop] = rtop; meta[:vsm] = vsm; meta[:takeorpay] = true
    meta[:q0] = q0; meta[:q1] = q1; meta[:quota] = quota
    meta[:bB] = maximum(tiers)
    isfinite(delta) && (meta[:delta] = Float64(delta))
    nd = NZData("loc_tkp_g$(g)_S$(S)_seed$(seed)", A, c, ell, u, hrow, hcoef, xrow, gmat,
        nd0.W, nd0.wvec, nd0.c0, nd0.hU, nx, fill(true, nx), NB, qx,
        S, nd0.q_hat, Float64(eps_hat), Float64(eps_tilde), Float64(beta),
        θU, piLU_fn(θU), piFU, Float64(lambdaU), varpiU, meta)
    box && nz_set_box!(nd)
    return nd
end

"x 벡터 ↔ 결정 문자열 {거점:크기;...} (가치 실험 csv 의 dstr 과 같은 형식, 거점 번호는 점포 노드 번호)"
function loc_tkp_xstr(nd::NZData, x̄)
    K, tiers, nA, nst = nd.meta[:K], nd.meta[:tiers], nd.meta[:nA], nd.meta[:nst]
    parts = String[]
    for i in nA+1:nst, k in 1:K
        x̄[(i - nA - 1) * K + k] > 0.5 && push!(parts, "$(i):$(round(Int, tiers[k]))")
    end
    return "{" * join(parts, ";") * "}"
end
