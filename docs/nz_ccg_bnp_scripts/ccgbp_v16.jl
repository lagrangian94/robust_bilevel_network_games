# ccgbp_v16.jl — gen_v16.py 가 v15 에서 만든 파일 (손으로 고치지 말 것). v15 + 1 단계 총량 몫 PH1_SHARE (2026-11-26 시험판).
#  docs/nz_ccg_bnp_2026-11-23.md 11.17. 1 단계는 경과 PH1_SHARE·TL (기본 0.3) 이 넘으면 다음 반복을 시작하지 않음.
# (아래는 v15 머리말) # ccgbp_v15.jl — gen_v15.py 가 v14 에서 만든 파일 (손으로 고치지 말 것). v14 + 적응 선별 DD oracle + 1 단계 결론 없는 oracle 이어 풀기 (2026-11-26 시험판).
#  docs/nz_ccg_bnp_2026-11-23.md 11.16. (1) ABB_PATCH (기본 abb_dd_patch.jl), ORCL_KW (기본 "(dd=true, dd_maxw=25.0)").
#  (2) 1 단계 oracle 이 위반도 인증도 없이 끝나면 같은 점을 누적 PH1_TOT (900 s) 까지 이어 풂. 인증 (상한 ≤ 목표) 이거나 누적이 차면 1 단계 끝.
# (아래는 v14 머리말) # ccgbp_v14.jl — v13 + 1 단계 oracle 상한 (새 점도) + 하한이 생기면 마감 루트 CG 몫을 oracle 에 (2026-11-25 시험판). docs/nz_ccg_bnp_2026-11-23.md 11.12.
#  (1) 1 단계 oracle 은 새 점이든 같은 점이든 상한 PH1_CAP (300 s). 1 단계의 일은 belief 모으기이고 인증은 2 단계의 마스터 incumbent 몫.
#      v13 S = 20 지리 1: 1 단계가 3,182 s (새 점 둘이 위반 1.5 를 들고 900 s 씩, 둘 다 마스터 incumbent 아님). S = 20 의 위반 발견은 79 ~ 286 s.
#  (2) 나무가 유효 하한 (LB 유한) 을 가지면 마감을 TL 로 늘려 oracle·나무를 계속 돌리고 마감 루트 정확 CG 는 건너뜀. 루트 CG 는 나무 하한이 없을 때의 대비책.
#      v13 S = 20 지리 1: 루트 CG 255 s 의 하한 −1,555.53 이 나무 하한 −1,554.88 보다 약해 안 쓰였고, 남긴 1,200 s 중 약 940 s 를 쓰지 않고 끝남.
# (아래는 v13 머리말) # ccgbp_v13.jl — v12 + oracle α 재사용 + 1 단계 반복 금지 (2026-11-25 시험판). docs/nz_ccg_bnp_2026-11-23.md 11.12.
#  (1) 앞선 oracle 호출이 찾은 incumbent α 를 다른 점에서 정확 평가 (exact_value, LP 하나). t₀ + 0.05 m 을 넘으면 그 belief 를 블록으로 넣고
#      α-B&B 를 부르지 않음. 대등한 점들은 같은 α 에서 위반이 나는 경우가 많음 (v12 S = 20: 세 점의 oracle incumbent 가 −3,403.40 · −3,403.48 · −3,403.40,
#      마스터 incumbent 의 위반 1.58 을 찾는 데 900 s). (점, α) 쌍은 한 번만 평가 (블록이 늘면 t₀ 는 오르기만 하므로 실패한 쌍은 계속 실패).
#      oracle_at! 의 처음과 trysearch (belief 탐색이 실패한 점) 에서 부름. 유효성: 정확 평가한 α 의 belief 블록은 다른 블록과 같은 하한.
#      다른 점의 α 는 그 점에서 실행 불가능할 수 있음 (닫힌 거점의 약정 등, 지리 13 점검에서 거리 0.73) → 먼저 그 점의 follower 실행 가능 영역으로
#      ℓ₁ 최근접 이동 (transplant_alpha, nz_feasible_alpha 와 같은 LP 에서 거리 제한만 없앰) 한 뒤 정확 평가. 옮긴 α 도 정확값이라 하한 유효.
#  (2) 1 단계에서 같은 점을 다시 부르면 (이어 풀기) 상한 PH1_RECAP (300 s). v12 S = 20: 같은 점 이어 풀기 900 s 가 결론 없음, 그 점은 마스터 incumbent 가 아니었음.
#      처음 판은 아예 다시 안 불렀는데, 지리 13 점검에서 1 단계 점이 최적점이라 나무를 세 번 다시 풀어 515 → 897 s (v12 는 1 단계 세 호출 55·43·40 s 로 197 s 에 인증).
#      이어 풀기는 α-B&B 상태가 남으므로 끊어도 잃는 작업이 없음.
# (아래는 v12 머리말) # ccgbp_v12.jl — v11 + 허용오차 예산 분할 (원고 Algorithm 2 의 τ⁺ = t₀ + m 을 한 나무로, 2026-11-25 시험판). docs/nz_ccg_bnp_2026-11-23.md 11.10·11.11.
#  (1) oracle 목표 τ⁺ = t₀ + ORCL_THETA·m (m = tol·max(1, |mval(x̄)|)), 지금 LB 와 무관. v11 은 LB + 0.95 tol − q 라 LB 가 생기면 목표가 조여짐.
#  (2) 나무는 노드 하한이 PT = mval(cinc) − TREE_SHARE·m 이상이면 그 노드를 주차 (증명된 하한 유지, 분기 안 함). 마스터를 TREE_SHARE·m 안으로 푸는 것.
#      PT 가 오르면 (cinc 에 belief) 하한 < PT 인 주차 노드를 다시 엶. UB ≤ mval(cinc) + ORCL_THETA·m 이면 주차를 끄고 모두 다시 엶 (가지치기로 끝남).
#  (3) oracle 은 노드마다 부르지 않고, 열린 노드가 없을 때 (마스터가 풀렸을 때) 마스터 incumbent 한 점에서 이어 풀기로 ORCL_TL 씩. 대등한 점은 주차되어 따로 부르지 않음.
#  (4) 주차·닫는 노드의 하한은 증명된 Farley 하한만 (v11 의 closed_lb 는 mval(c) 를 썼음). 정수 노드가 하한이 모자라면 엄격한 정확 CG 를 한 번 더.
#  (5) t_low (no-good 상수) 는 follower 최적성을 뺀 낙관값 LP (x 완화) 로. 가격 MILP 를 끝까지 풀면 S = 20 에서 355 ~ 376 s.
#  (6) 루트 정확 CG 는 기본이 마감 직전 한 번 (ROOT_EXACT = end, 마지막 min(1200 s, 0.2 TL) 를 남김). v11 처럼 2 단계 전이면 S = 10 에서 전체의 38 ~ 72 %.
#  환경 변수 추가: ORCL_THETA 0.75, TREE_SHARE 0.2, ROOT_EXACT end (start·off). 평가기 여유 κ 는 저장소 _bccg_value! (2026-11-25 수정) 를 따름.
# (아래는 v11 머리말) ccgbp_v11.jl — v10 + oracle 정의역 HUCAP (연 B 거점 약정 상한 = 용량, 기본 1; 2026-11-25 시험판).
# (아래는 v10 머리말) # ccgbp_v10.jl — v9 + 2 단계 전 루트 정확 CG 하한 (ROOT_EXACT, 기본 1; 2026-11-25 시험판).
# (아래는 v9 머리말) ccgbp_v9.jl — v8 (ccgbp.jl) + oracle 위반 판정을 incumbent α 의 정확값 (exact_value) 으로 (2026-11-24 시험판).
# (아래는 v8 머리말) ccgbp.jl — belief 블록 CCG 를 Dantzig–Wolfe branch-and-price 한 나무로. **기본 판 (v8, 2026-11-24 확정)**. 절차는 docs/nz_ccg_bnp_2026-11-23.md 10·11 절.
#  v8 = v6 (ccgbp_v6.jl) + 같은 점의 oracle 재호출은 이전 α-B&B 상태에서 이어 풂 (resume, v7) + oracle 의 follower 반응 휴리스틱 resp (ORCL_RESP, 기본 1).
#  저장소 부품 (블록 생성기 _bccg_gen, belief 탐색 평가기, α-B&B oracle nz_alpha_bnb_dr, 확보 규칙) 을 그대로 부름. X 를 미리 나열하지 않음 (만난 점에만 번호).
#  기본값 = 인증용 설정: η = 1/λᵁ, μ = μ 상한 = λᵁ = 1e4 (블록 = 원고 Prop certificate (ii) 의 벌점값을 belief 로 제한한 것, μ 올림 없음 → 모자라면 no-good).
#  환경 변수 (기본값): GEO 1, R 0.2, NB 3 (거점 수 상한), TL 3600, TOL 5e-3, ETA 1e-4, MU 1e4, MU_MAX 1e4, LAMU 1e4 (oracle λᵁ),
#    S 10 (시나리오 수), NS 10 (지지 열당 이웃 표본, 0 = 전부), SEED_ALL 0 (1 이면 새 블록에 평가한 점 전부를 LP 로), NTOP 5 (탐색할 상위 정수점), CGTOL 1e-3,
#    NRAND 40 (무작위 belief 후보), SB_K 3 (strong branching 후보), PH1 1 (1 단계), ORCL_TL 900 (oracle 호출당 상한), PSTOP 1 (가지치기 기준 가격 멈춤), ORCL_RESP 1 (oracle resp)
#  실행: GRB_LICENSE_FILE=... julia -t 12,1 --project=. docs/nz_ccg_bnp_scripts/ccgbp.jl   (oracle 때문에 interactive 스레드 1 개 필요)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl")); include(joinpath(root, "nonzero_sum", "nz_belief_ccg_tree.jl"))
const ABB_PATCH = get(ENV, "ABB_PATCH", joinpath(@__DIR__, "abb_dd_patch.jl")); isempty(ABB_PATCH) || include(ABB_PATCH)   # v15: 실험용 α-B&B 재정의 ("" 이면 저장소 판)
envf(k, d) = get(ENV, k, d)
const GEO = parse(Int, envf("GEO", "1")); const RR = parse(Float64, envf("R", "0.2"))
const TL = parse(Float64, envf("TL", "3600")); const TOL = parse(Float64, envf("TOL", "5e-3"))
const MU = parse(Float64, envf("MU", "1e4")); const ETA = parse(Float64, envf("ETA", "1e-4")); const MU_MAX = parse(Float64, envf("MU_MAX", "1e4"))
const NS = parse(Int, envf("NS", "10")); const SEED_ALL = envf("SEED_ALL", "0") == "1"; const NTOP = parse(Int, envf("NTOP", "5"))
const CGTOL = parse(Float64, envf("CGTOL", "1e-3")); const NRAND = parse(Int, envf("NRAND", "40"))
const PH1_CAP = parse(Float64, envf("PH1_CAP", "300")); const SB_K = parse(Int, envf("SB_K", "3")); const PH1 = envf("PH1", "1") == "1"; const ORCL_TL = parse(Float64, envf("ORCL_TL", "900")); const PSTOP = envf("PSTOP", "1") == "1"
const LAMU = parse(Float64, envf("LAMU", "1e4")); const ORCL_RESP = envf("ORCL_RESP", "1") == "1"   # oracle 의 λᵁ (loc_tkp 자체 기본은 100), resp 켬/끔
const HUCAP = envf("HUCAP", "1") == "1"   # v11: oracle 정의역 h_i ≤ b_i(x̄)
const ORCL_KW = eval(Meta.parse(envf("ORCL_KW", isempty(ABB_PATCH) ? "NamedTuple()" : "(dd=true, dd_maxw=25.0)")))   # v15: α-B&B 추가 인자
const PH1_TOT = parse(Float64, envf("PH1_TOT", "900"))   # v15: 1 단계에서 결론 없는 점의 누적 oracle 시간 상한
const PH1_SHARE = parse(Float64, envf("PH1_SHARE", "0.3"))   # v16: 1 단계 총량 몫 (경과가 PH1_SHARE·TL 을 넘으면 다음 반복 안 함)
const ORCL_THETA = parse(Float64, envf("ORCL_THETA", "0.75")); const TREE_SHARE = parse(Float64, envf("TREE_SHARE", "0.2"))
const ROOT_EXACT = envf("ROOT_EXACT", "end")   # end (기본: B&P 가 끝나지 않으면 마감 직전에 루트 정확 CG 한 번) | start (v10·v11 처럼 2 단계 전) | off
ROOT_EXACT in ("end", "start", "off") || error("ccgbp v16: ROOT_EXACT 는 end·start·off")
const RESERVE = ROOT_EXACT == "end" ? min(1200.0, 0.2 * TL) : 0.0; const TLB = TL - RESERVE   # B&P·oracle 마감 (남은 RESERVE 는 마감 루트 CG 몫)
dl() = isfinite(LB) ? TL : TLB   # v14: 나무가 유효 하한을 가지면 루트 CG 대비책이 필요 없으므로 TL 까지
0 < PH1_SHARE <= 1 || error("ccgbp v16: PH1_SHARE 는 (0, 1]")
ORCL_THETA + TREE_SHARE < 1 || error("ccgbp v16: ORCL_THETA + TREE_SHARE 는 1 보다 작아야 함 (여유 = |UB| 와 |mval| 차이)")
nd = make_loc_tkp(; g=GEO, S=parse(Int, envf("S", "10")), seed=1, eps_hat=RR, eps_tilde=RR, delta=RR / 3, beta=0.4, lambdaU=LAMU, NB=parse(Int, envf("NB", "3")))
@printf("  [설정] η %.1e, μ %.1e (상한 %.1e), oracle λᵁ %.1e, μη = %.3g\n", ETA, MU, MU_MAX, LAMU, MU * ETA)
K = nd.meta[:K]; nx = nd.nx; m, ny = nz_m(nd), nz_ny(nd); A = nd.A; XE = nz_xentries(nd); S = nd.S
NT = Threads.nthreads()
@printf("--- ccgbp v16 (v15 + 1 단계 총량 몫 %.2f·TL = %.0fs) | ccgbp v15 (v14 + oracle 패치 %s 인자 %s, 1 단계 결론 없으면 누적 %.0fs 까지 이어 풂) | ccgbp v14 (v13 + 1 단계 oracle 상한 %.0fs (새 점도), LB 가 생기면 마감까지 oracle·나무; v13 = v12 + oracle α 재사용; 예산 분할: oracle θ %.2f, 나무 몫 %.2f, 루트 정확 CG %s; HUCAP=%s, resp=%s; 평가기 κ 단계 %s) | SB_K %d PH1 %s ORCL_TL %.0f PSTOP %s\n",
        PH1_SHARE, PH1_SHARE * TL, isempty(ABB_PATCH) ? "없음" : basename(ABB_PATCH), string(ORCL_KW), PH1_TOT, PH1_CAP, ORCL_THETA, TREE_SHARE, ROOT_EXACT, HUCAP, ORCL_RESP, string(_BCCG_KAPPAS), SB_K, PH1, ORCL_TL, PSTOP); @printf(" g=%d coupled r=%.2f S=%d, 스레드 %d, tol %.1e, μ %.3g, η %.3g, NS %d, SEED_ALL %s, NTOP %d, CGTOL %.1e\n", GEO, RR, S, NT, TOL, MU, ETA, NS, SEED_ALL, NTOP, CGTOL); flush(stdout)
wall = time()
site(i) = (i - 1) ÷ K + 1
al = findall(nd.x_allowed); sites = unique(site.(al))
# 점 번호는 만난 점에만 붙인다 (v6, 2026-11-23): X 를 미리 나열하지 않음. |X| 는 보고용으로 공식으로만 계산
pts = Vector{Vector{Float64}}(); qv = Float64[]; idx = Dict{Tuple,Int}()
key(z) = Tuple(findall(>(0.5), z))
"점 z 의 번호 (처음 만나면 등록). z 는 X 의 점이어야 함 (호출하는 곳이 실행가능하게 만듦)"
function pid(z)
    k = key(z); c = get(idx, k, 0); c > 0 && return c
    push!(pts, Float64.(z .> 0.5)); push!(qv, dot(nd.qx, pts[end])); idx[k] = length(pts)
    return length(pts)
end
pid(zeros(nx))                                                   # 빈 x = 번호 1
NP = sum(binomial(length(sites), n) * K^n for n in 0:nd.gamma)   # |X| (보고용)
nbrs_cache = Dict{Int,Vector{Int}}()
function nbrs(t)
    get!(nbrs_cache, t) do
        z = pts[t]; on = [(site(i), (i - 1) % K + 1) for i in findall(>(0.5), z)]; out = Int[]
        put(conf) = (w = zeros(nx); for (s, k) in conf; w[(s - 1) * K + k] = 1; end; pid(w))
        for a in eachindex(on)
            rest = [on[b] for b in eachindex(on) if b != a]; push!(out, put(rest))
            for s in sites, k in 1:K; any(o -> o[1] == s, rest) && continue; push!(out, put(vcat(rest, [(s, k)]))); end
        end
        if length(on) < nd.gamma
            for s in sites, k in 1:K; any(o -> o[1] == s, on) && continue; push!(out, put(vcat(on, [(s, k)]))); end
        end
        filter(>(0), unique(out))
    end
end
envs = _nz_envs(NT + 3)                    # 1..NT 생성기·평가기, NT+1 가격 MILP, NT+2 RMP, NT+3 검사 LP
lp_opt = () -> (o = Gurobi.Optimizer(envs[NT + 3]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
C = (nd = nd, eta = ETA, vub = false, wd = false, XE = XE, lp_opt = lp_opt,
     rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m], colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny],
     ents_of_row = [[e for e in XE if e.k == k] for k in 1:m])
pool_ev = Channel{BCCGEval}(NT); for k in 1:NT; put!(pool_ev, BCCGEval(nd, envs[k])); end

# ---------------- 블록 ----------------
mutable struct Blk
    kind::Symbol                    # :lp (belief 블록) | :aff (안전장치 아핀식 c0 + cᵀz)
    d::Vector{Float64}; r::Vector{Float64}; μ::Float64
    gens::Vector{Any}               # 스레드마다 생성기
    c0::Float64; c::Vector{Float64}
end
blks = Blk[]; cache = Dict{Tuple{Int,Int},Float64}(); visited = Int[]; visset = Set{Int}(); lastsupp = Int[]
t_build = Ref(0.0); t_lp = Ref(0.0); n_lp = Ref(0); t_mip = Ref(0.0); n_mip = Ref(0); nodes_mip = Ref(0); t_rmp = Ref(0.0); n_rmp = Ref(0)
function mkgens(d, r, μ)
    out = Vector{Any}(undef, NT)
    t_build[] += @elapsed (@sync for th in 1:NT; Threads.@spawn (out[th] = _bccg_gen(C, d, r, μ, envs[th], th)); end)
    out
end
add_lpblk!(d, r, μ) = (push!(blks, Blk(:lp, copy(d), copy(r), μ, mkgens(d, r, μ), 0.0, Float64[])); length(blks))
add_affblk!(c0, c) = (push!(blks, Blk(:aff, Float64[], Float64[], 0.0, Any[], c0, copy(c))); length(blks))
getL(c, b) = blks[b].kind == :aff ? blks[b].c0 + dot(blks[b].c, pts[c]) : cache[(c, b)]
"(점, 블록) 들을 블록 LP 로 (캐시 밖만, 스레드 병렬)"
function evalpairs!(prs)
    todo = unique([p for p in prs if blks[p[2]].kind == :lp && !haskey(cache, p)]); isempty(todo) && return
    out = Vector{Float64}(undef, length(todo))
    t_lp[] += @elapsed begin
        ch = Channel{Int}(length(todo)); for i in eachindex(todo); put!(ch, i); end; close(ch)
        @sync for th in 1:NT
            Threads.@spawn for i in ch
                c, b = todo[i]; out[i] = first(_bccg_gen_eval!(blks[b].gens[th], pts[c]))
            end
        end
    end
    for i in eachindex(todo); cache[todo[i]] = out[i]; c = todo[i][1]; c in visset || (push!(visset, c); push!(visited, c)); end; n_lp[] += length(todo)
end
evalall!(cs) = evalpairs!([(c, b) for c in cs for b in eachindex(blks)])
mval(c) = qv[c] + maximum(getL(c, b) for b in eachindex(blks))      # 마스터 목적 (모든 블록이 평가되어 있어야)


# ---------------- 노드 (분기 고정) ----------------
struct Node
    f0::Set{Int}; f1::Set{Int}; opn::Set{Int}     # x_jk = 0, x_jk = 1, 거점 j 개설 강제
    lb::Float64; est::Float64; depth::Int; tag::String
end
sitevars(j) = [(j - 1) * K + k for k in 1:K if nd.x_allowed[(j - 1) * K + k]]
consistent(c, N) = (z = pts[c]; all(z[i] < 0.5 for i in N.f0) && all(z[i] > 0.5 for i in N.f1) && all(any(z[i] > 0.5 for i in sitevars(j)) for j in N.opn))
"노드와 맞는 정수점 하나 (없으면 0)"
function seedpt(N)
    z = zeros(nx); for i in N.f1; z[i] = 1.0; end
    for j in N.opn
        any(z[i] > 0.5 for i in sitevars(j)) && continue
        ks = [i for i in sitevars(j) if !(i in N.f0)]; isempty(ks) && return 0
        z[ks[1]] = 1.0
    end
    sum(z) <= nd.gamma + 0.5 || return 0
    for j in sites; count(i -> z[i] > 0.5, sitevars(j)) <= 1 || return 0; end
    any(z[i] > 0.5 for i in N.f0) && return 0
    pid(z)
end
"노드 위 X 에서 선형 최소 min cᵀz (탐욕, 정확)"
function linmin(c, N)
    forced = 0.0; nf = 0; opt = Tuple{Float64,Int}[]; z = zeros(nx)
    for j in sites
        sv = sitevars(j); f1j = [i for i in sv if i in N.f1]
        if !isempty(f1j); z[f1j[1]] = 1.0; forced += c[f1j[1]]; nf += 1; continue; end
        av = [i for i in sv if !(i in N.f0)]
        if isempty(av); j in N.opn && return Inf, 0; continue; end
        ib = av[argmin([c[i] for i in av])]
        if j in N.opn; z[ib] = 1.0; forced += c[ib]; nf += 1; else; push!(opt, (c[ib], ib)); end
    end
    nf > nd.gamma && return Inf, 0
    sort!(opt; by=first); v = forced
    for (val, ib) in opt[1:min(nd.gamma - nf, length(opt))]; val < 0 || break; z[ib] = 1.0; v += val; end
    return v, pid(z)
end

# ---------------- 가격 MILP (노드 고정 반영) ----------------
pmods = Dict{Int,Any}()
function pricemip(b, wb, πb, σb, stopv, N; tlim=600.0)
    if !haskey(pmods, b)
        Mp = Model(() -> (o = Gurobi.Optimizer(envs[NT + 1]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o))
        @variable(Mp, xb[1:nx], Bin); @variable(Mp, tb >= -1e7); @constraint(Mp, sum(xb) <= nd.gamma)
        for (ix, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(Mp, sum(xb[i] for i in ix) <= rhs); end
        for i in 1:nx; nd.x_allowed[i] || fix(xb[i], 0.0; force=true); end
        _bccg_add_block!(merge(C, (M = Mp, x = xb, t0 = tb)), blks[b].d, blks[b].r, blks[b].μ)
        set_optimizer_attribute(Mp, "MIPGap", 1e-6); set_optimizer_attribute(Mp, "IntFeasTol", 1e-9); set_optimizer_attribute(Mp, "IntegralityFocus", 1)
        set_optimizer_attribute(Mp, "PoolSolutions", 10)
        pmods[b] = (Mp, xb, tb)
    end
    Mp, xb, tb = pmods[b]
    fx = Int[]
    for i in N.f0; is_fixed(xb[i]) || (fix(xb[i], 0.0; force=true); push!(fx, i)); end
    for i in N.f1; is_fixed(xb[i]) || (fix(xb[i], 1.0; force=true); push!(fx, i)); end
    oc = [@constraint(Mp, sum(xb[i] for i in sitevars(j)) >= 1) for j in N.opn]
    @objective(Mp, Min, wb * tb - sum(πb[i] * xb[i] for i in al))
    set_optimizer_attribute(Mp, "BestBdStop", stopv); set_optimizer_attribute(Mp, "TimeLimit", tlim)
    t_mip[] += @elapsed optimize!(Mp); n_mip[] += 1; nodes_mip[] += Int(MOI.get(Mp, MOI.NodeCount()))
    st = termination_status(Mp)
    bnd = st == MOI.INFEASIBLE ? Inf : objective_bound(Mp)
    zs = [pid(round.(value.(xb; result=k))) for k in 1:result_count(Mp)]
    inc = has_values(Mp) ? objective_value(Mp) : Inf
    for i in fx; unfix(xb[i]); end
    for c in oc; delete(Mp, c); end
    return bnd, zs, inc
end

# ---------------- DW 열 생성 (노드) ----------------
cols = Vector{Set{Int}}(); n_rmpfix = Ref(0)
function rmp(N)
    M = Model(() -> Gurobi.Optimizer(envs[NT + 2])); set_silent(M)
    set_optimizer_attribute(M, "Method", 2); set_optimizer_attribute(M, "Crossover", 0)
    nb = length(blks); cN = [[c for c in cols[b] if consistent(c, N)] for b in 1:nb]
    @variable(M, x[al]); @variable(M, t)
    lam = Dict((b, c) => @variable(M, lower_bound = 0) for b in 1:nb for c in cN[b])
    ct = [@constraint(M, t - sum(getL(c, b) * lam[(b, c)] for c in cN[b]) >= 0) for b in 1:nb]
    cl = [Dict(i => @constraint(M, sum(pts[c][i] * lam[(b, c)] for c in cN[b]) - x[i] == 0) for i in al) for b in 1:nb]
    cc = [@constraint(M, sum(lam[(b, c)] for c in cN[b]) == 1) for b in 1:nb]
    @objective(M, Min, sum(nd.qx[i] * x[i] for i in al) + t)
    t_rmp[] += @elapsed optimize!(M); n_rmp[] += 1
    if termination_status(M) != MOI.OPTIMAL       # barrier (crossover 끔) 가 수치 문제로 멈춤 → crossover 켜고 다시 (쌍대 실행가능성이 Farley 하한에 필요)
        n_rmpfix[] += 1
        set_optimizer_attribute(M, "Crossover", -1); t_rmp[] += @elapsed optimize!(M)
        if termination_status(M) != MOI.OPTIMAL
            set_optimizer_attribute(M, "Method", 1); t_rmp[] += @elapsed optimize!(M)
        end
    end
    termination_status(M) == MOI.OPTIMAL || error("ccgbp: RMP $(termination_status(M))")
    w = dual.(ct); π = [[i in al ? dual(cl[b][i]) : 0.0 for i in 1:nx] for b in 1:nb]; σ = dual.(cc)
    supp = [Set(c for c in cN[b] if value(lam[(b, c)]) > 1e-9) for b in 1:nb]
    xv = zeros(nx); for i in al; xv[i] = value(x[i]); end
    objective_value(M), w, π, σ, supp, xv
end
rcv(b, c, w, π, σ) = w[b] * getL(c, b) - dot(π[b], pts[c]) - σ[b]
rng_ns = MersenneTwister(7)
"""노드 N 에서 열 생성. 반환 (lb (exact 일 때만, 아니면 −Inf; 실행불가면 Inf), RMP 값, x̃, 마스터 UB, 그 점, 상위 정수점, 정보).
v12: stopat (주차 기준 PT) 과 가지치기 기준 UB − tol 중 작은 값 cut 에 lb 가 닿으면 멈춤. RMP 값 v ≥ cut 인 동안은 CGTOL 로 멈추지 않음
(그래야 주차할 노드의 하한이 cut 에 닿음). cgtol·trel 은 엄격한 정확 CG 용."""
function dwsolve!(N; exact=true, ubglob=Inf, stopat=Inf, cgtol=CGTOL, trel=1e-4, deadline=Inf)
    lb = -Inf; ubm = Inf; cbest = 0; it = 0; nex = 0; topl = Int[]; v = NaN; xv = zeros(nx)
    s0 = seedpt(N); s0 == 0 && return Inf, Inf, xv, Inf, 0, Int[], (it = 0, nex = 0)
    evalall!([s0]); for b in eachindex(blks); push!(cols[b], s0); end
    while true
        it += 1
        v, w, π, σ, supp, xv = rmp(N)
        tolabs = trel * max(1, abs(v))
        need = Tuple{Int,Int}[]; candb = Dict{Int,Vector{Int}}()
        for b in eachindex(blks)
            (w[b] > 1e-9 && blks[b].kind == :lp) || continue
            cs = Set{Int}()
            for c in supp[b]
                push!(cs, c); nb_ = filter(c2 -> consistent(c2, N), nbrs(c))
                union!(cs, (NS == 0 || length(nb_) <= NS) ? nb_ : shuffle(rng_ns, nb_)[1:NS])
            end
            for c in visited; haskey(cache, (c, b)) && consistent(c, N) && push!(cs, c); end
            candb[b] = collect(cs); append!(need, [(c, b) for c in cs])
        end
        evalpairs!(need)
        added = 0
        for b in eachindex(blks)
            if haskey(candb, b)
                c = argmin(c -> rcv(b, c, w, π, σ), candb[b]); r = rcv(b, c, w, π, σ)
            else
                cl = blks[b].kind == :aff ? w[b] .* blks[b].c .- π[b] : -π[b]
                r, c = linmin(cl, N); r += (blks[b].kind == :aff ? w[b] * blks[b].c0 : 0.0) - σ[b]; evalpairs!([(c, b)])
            end
            if r < -tolabs && !(c in cols[b]); push!(cols[b], c); added += 1; end
        end
        added > 0 && continue
        sp = unique([c for b in eachindex(blks) for c in supp[b]]); evalall!(sp); global lastsupp = sp
        for c in sp; mv = mval(c); mv < ubm && (ubm = mv; cbest = c); end
        topl = sort(sp; by=mval)[1:min(NTOP, length(sp))]
        exact || return -Inf, v, xv, ubm, cbest, topl, (it = it, nex = 0)
        lbit = v; added2 = 0
        # 가격 MILP 멈춤 기준: 가지치기·주차에 필요한 여유 (RMP − cut) 를 블록에 나눔. 여유가 없으면 엄격 (tolabs)
        nw = count(b -> blks[b].kind == :lp && w[b] > 1e-9, eachindex(blks))
        cut = min(isfinite(ubglob) ? ubglob - TOL * max(1.0, abs(ubglob)) : Inf, stopat)
        need = isfinite(cut) ? v - cut : 0.0
        allow = (PSTOP && need > 0 && nw > 0) ? max(tolabs, 0.9 * need / nw) : tolabs
        for b in eachindex(blks)
            if blks[b].kind == :lp && w[b] > 1e-9
                bnd, zs, inc = pricemip(b, w[b], π[b], σ[b], σ[b] - allow, N; tlim=min(600.0, max(1.0, deadline - time())))
                lbit += min(0.0, bnd - σ[b])
                if inc - σ[b] < -tolabs
                    evalpairs!([(zc, b) for zc in zs])
                    for zc in zs
                        if consistent(zc, N) && rcv(b, zc, w, π, σ) < -tolabs && !(zc in cols[b]); push!(cols[b], zc); added2 += 1; end
                    end
                end
            else
                cl = blks[b].kind == :aff ? w[b] .* blks[b].c .- π[b] : -π[b]
                r, c = linmin(cl, N); r += (blks[b].kind == :aff ? w[b] * blks[b].c0 : 0.0) - σ[b]; lbit += min(0.0, r)
                if r < -tolabs && !(c in cols[b]); evalpairs!([(c, b)]); push!(cols[b], c); added2 += 1; end
            end
        end
        nex += 1; lb = max(lb, lbit)
        time() > deadline && return lb, v, xv, ubm, cbest, topl, (it = it, nex = nex)   # 마감 (Farley 하한은 어느 반복에서나 유효, 시간 상한에 멈춘 MILP 도 bound 를 씀)
        # 노드를 자르거나 주차할 수 있으면 더 조일 필요 없음
        (isfinite(cut) && lb >= cut) && return lb, v, xv, ubm, cbest, topl, (it = it, nex = nex)
        tgt = min(ubm, ubglob)
        (added2 == 0 || (tgt - lb <= cgtol * max(1, abs(tgt)) && !(isfinite(cut) && v >= cut))) && return lb, v, xv, ubm, cbest, topl, (it = it, nex = nex)
    end
end
cinc = 0          # 마스터 incumbent (모든 블록으로 평가된 점 중 최소 mval)
"새 블록의 열 씨앗"
function seed!(b, c=0)
    base = unique(vcat(lastsupp, c > 0 ? [c] : Int[], [1], cinc > 0 ? [cinc] : Int[]))
    if blks[b].kind == :lp
        evalpairs!([(cc, b) for cc in base])
        SEED_ALL && evalpairs!([(cc, b) for cc in copy(visited)])
    end
    push!(cols, Set(base))
end

# ---------------- belief 탐색·확보 (nz_belief_ccg_tree 와 같은 규칙) ----------------
qh = nd.q_hat; bx = nd.meta[:box]
inset(d) = all(d .>= bx.dl .- 1e-12) && all(d .<= bx.du .+ 1e-12) && sum(abs.(d .- qh)) <= 2nd.eps_tilde + 1e-12
pairbels = Vector{Vector{Float64}}()
for i in 1:S, j in 1:S
    i == j && continue
    mv = min(qh[i] - bx.dl[i], bx.du[j] - qh[j], nd.eps_tilde); mv <= 1e-9 && continue
    dd = copy(qh); dd[i] -= mv; dd[j] += mv; inset(dd) && push!(pairbels, dd)
end
rng = MersenneTwister(2026)
function randbel()
    d = copy(qh)
    for t in 1:rand(rng, 1:4)
        i, j = randperm(rng, S)[1:2]; mv = rand(rng) * min(d[i] - bx.dl[i], bx.du[j] - d[j]); dd = copy(d); dd[i] -= mv; dd[j] += mv; inset(dd) && (d = dd)
    end
    return d
end
t_srch = Ref(0.0); n_srch = Ref(0)
function search(xb)
    n_srch[] += 1
    t_srch[] += @elapsed begin
        cand = vcat([B.d for B in blks if B.kind == :lp], pairbels, [randbel() for _ in 1:NRAND]); vals = fill(-Inf, length(cand))
        Threads.@threads for t in eachindex(cand)
            local ev = take!(pool_ev)
            try; _bccg_setx!(ev, nd, xb); local v1, _, _, _ = _bccg_g!(ev, nd, cand[t]); vals[t] = v1; finally; put!(pool_ev, ev); end
        end
        tb = argmax(vals); best = (v = -Inf, d = nothing, r = nothing, α = nothing)
        ev2 = take!(pool_ev)
        try
            _bccg_setx!(ev2, nd, xb); dcur = cand[tb]
            for it in 1:4
                v2, dd, r2, α2 = _bccg_g!(ev2, nd, dcur)
                dd === nothing && break
                v2 > best.v + 1e-9 || break
                best = (v = v2, d = dd, r = r2, α = α2)
                inset(dd) || break
                dcur = dd
            end
        finally; put!(pool_ev, ev2); end
    end
    return best
end
function exact_value(xb, α)
    ev3 = take!(pool_ev)
    try
        _bccg_setx!(ev3, nd, xb)
        a1 = [(v = clamp(α[i], 0.0, nd.hU[i]); v < 1e-6 * max(1.0, nd.hU[i]) ? 0.0 : v) for i in eachindex(α)]
        v3, dd = _bccg_value!(ev3, nd, a1)
        if dd === nothing
            a2 = nz_feasible_alpha(nd, xb, a1; lp_optimizer=lp_opt, tol=1e-3); a1 = a2; v3, dd = _bccg_value!(ev3, nd, a2)
        end
        return v3, dd, (dd === nothing ? nothing : copy(ev3.lastr)), a1
    finally; put!(pool_ev, ev3); end
end
tlow = -1e7; n_ng = Ref(0); n_mu = Ref(0); n_rc = Ref(0)
function nogood!(c, v)
    xb = pts[c]; cv = zeros(nx); c0 = v
    for i in al; if xb[i] > 0.5; c0 -= (v - tlow); cv[i] = (v - tlow); else; cv[i] = -(v - tlow); end; end
    b = add_affblk!(c0, cv); seed!(b, c); n_ng[] += 1
end
function restricted!(c, α)
    F = nz_build_fixed_alpha(nd, pts[c]; optimizer=lp_opt); z = nz_eval_alpha!(F, nd, α); isfinite(z) || return false
    cut = nz_cut_from_solution(F, nd, pts[c]); b = add_affblk!(cut[:intercept], cut[:slope]); seed!(b, c); n_rc[] += 1; true
end
"belief 를 블록으로 넣고 c 에서 확보 (μ 올리기 → no-good)"
function add_and_secure!(c, v, d, r)
    xb = pts[c]; thr = 1e-4 * max(1.0, abs(v))
    b = add_lpblk!(d, r, MU); evalpairs!([(c, b)])
    getL(c, b) >= v - thr && (seed!(b, c); return :block)
    _, Linf, μstar = _bccg_block_at(C, xb, d, r, MU)
    if MU < MU_MAX && Linf >= v - thr && μstar <= MU_MAX      # μ 가 이미 상한이면 올리지 않고 바로 no-good
        μn = min(MU_MAX, max(2MU, 1.1μstar)); blks[b] = Blk(:lp, copy(d), copy(r), μn, mkgens(d, r, μn), 0.0, Float64[]); n_mu[] += 1
        for k in collect(keys(cache)); k[2] == b && delete!(cache, k); end
        evalpairs!([(c, b)])
        getL(c, b) >= v - thr && (seed!(b, c); return :mu)
    end
    seed!(b, c); nogood!(c, v); return :safeguard
end


# ---------------- CCG + branch-and-price ----------------
ROOT = Node(Set{Int}(), Set{Int}(), Set{Int}(), -Inf, -Inf, 0, "root")
"상위 정수점에서 싼 탐색, belief 하나 넣으면 true"
function trysearch(topl)
    for c in topl
        bb = search(pts[c]); tc = mval(c) - qv[c]
        if bb.d !== nothing && bb.v > tc + 1e-3 * max(1.0, abs(bb.v))
            how = add_and_secure!(c, bb.v, bb.d, bb.r)
            @printf("    [탐색] x=%-24s t %.2f → %.2f (%s), 블록 %d\n", loc_tkp_xstr(nd, pts[c]), tc, bb.v, how, length(blks)); flush(stdout)
            return true
        end
        alseed!(c) && return true                 # v13
    end
    false
end
UB = Inf; LB = -Inf; xbest = 0; status = :TimeLimit; t_or = 0.0; n_or = 0
intpts = Set{Int}()      # v12: 만난 정수 후보점 (belief 가 늘면 이 점들로 마스터 incumbent 를 다시 고름)
function upd_inc!(cs)
    for c in cs
        push!(intpts, c)
        all(b -> blks[b].kind == :aff || haskey(cache, (c, b)), eachindex(blks)) || continue
        (cinc == 0 || mval(c) < mval(cinc) - 1e-9) && (global cinc = c)
    end
end
"v12: belief 가 늘었을 때 마스터 incumbent 를 후보점 전체에서 다시 고름 (새 블록으로 평가). cinc 의 값이 오르면 다른 점이 incumbent 가 될 수 있음"
function refresh_cinc!()
    cs = collect(intpts); evalall!(cs)
    isempty(cs) || (global cinc = cs[argmin([mval(c) for c in cs])])
end
mref(c) = TOL * max(1.0, abs(mval(c)))                                              # 허용 m (원고 Algorithm 2 의 m, 마스터 값 기준)
ub_ok() = cinc > 0 && isfinite(UB) && UB <= mval(cinc) + ORCL_THETA * mref(cinc) + 1e-9 * max(1.0, abs(UB))   # UB 가 cinc 기준으로 충분 → 주차 끔
PT() = (cinc > 0 && !ub_ok()) ? mval(cinc) - TREE_SHARE * mref(cinc) : Inf        # 주차 기준 (Inf = 주차 안 함)
"oracle (α-B&B) 을 c 에서. v12: 목표 τ⁺ = t₀ + ORCL_THETA·m (원고 Algorithm 2 의 τ⁺ = t₀ + m 을 나무 몫만큼 줄인 것, 지금 LB 와 무관)"
orLB = Dict{Int,Float64}(); orcap = Dict{Int,Float64}()
orcert = Dict{Int,Bool}(); orph1 = Dict{Int,Float64}()   # v15: 점 → 마지막 호출이 인증 (상한 ≤ 목표) 했는지, 1 단계 누적 oracle 시간
orcl_log = NamedTuple[]
orstate = Dict{Int,Any}()     # v7: 점 → 마지막 α-B&B 상태 (같은 점을 다시 부르면 이어 풂)
alpool = Vector{Vector{Float64}}()       # v13: oracle incumbent α
alv = Dict{Tuple{Int,Int},Float64}()     # v13: (점, α 번호) → 정확값 (한 번만 평가)
n_seed = Ref(0); t_seed = Ref(0.0); n_seedev = Ref(0)
"v13: α̂ 를 xb 의 follower 실행 가능 영역 (α ∈ H, 모든 시나리오 recourse 실행 가능) 으로 ℓ₁ 최근접 이동. 없으면 nothing"
function transplant_alpha(xb, α̂)
    nh = nz_nh(nd); M = Model(lp_opt)
    @variable(M, 0 <= a[i=1:nh] <= nd.hU[i]); @variable(M, dev[1:nh] >= 0); @variable(M, y[1:ny, 1:S] >= 0)
    @constraint(M, nd.W * a .<= nd.wvec)
    @constraint(M, [i=1:nh], dev[i] >= a[i] - α̂[i]); @constraint(M, [i=1:nh], dev[i] >= α̂[i] - a[i])
    for k in 1:m, s in 1:S
        @constraint(M, sum(av * y[j, s] for (j, av) in C.rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * a[nd.hrow[k]] : 0.0) <= nd.u[k, s] + nz_gx(nd, k, s, xb))
    end
    @objective(M, Min, sum(dev)); optimize!(M)
    termination_status(M) == MOI.OPTIMAL || return nothing
    return [clamp(value(a[i]), 0.0, nd.hU[i]) for i in 1:nh]
end
"v13: 앞선 oracle 의 incumbent α 를 c 에서 정확 평가. t₀ + 0.05 m 을 넘으면 그 belief 를 블록으로 넣고 true"
function alseed!(c)
    isempty(alpool) && return false
    evalall!([c]); t0v = mval(c) - qv[c]; thr = 0.05 * mref(c)
    for k in eachindex(alpool)
        haskey(alv, (c, k)) && continue
        vx = nothing
        t_seed[] += @elapsed begin
            at = transplant_alpha(pts[c], alpool[k])
            vx = at === nothing ? (-Inf, nothing, nothing, nothing) : exact_value(pts[c], at)
        end
        n_seedev[] += 1
        alv[(c, k)] = vx[2] === nothing ? -Inf : vx[1]
        if vx[2] !== nothing && vx[1] > t0v + thr
            add_and_secure!(c, vx[1], vx[2], vx[3]); n_seed[] += 1
            @printf("    [α 재사용] x=%-24s t %.2f → %.2f (oracle α %d), 블록 %d\n", loc_tkp_xstr(nd, pts[c]), t0v, vx[1], k, length(blks)); flush(stdout)
            return true
        end
    end
    false
end
function alpush!(α)
    α === nothing && return
    any(a -> maximum(abs.(a .- α)) <= 1e-6, alpool) || push!(alpool, copy(α))
end
# v11: oracle 정의역에서 연 B 거점 약정 상한 = 그 거점 용량 (h_i ≤ b_i(x̄)). 약정 등식과 용량 행에서 h_i > b_i 면 retailer recourse 가 실행 불가능 →
#   그런 α 는 retailer 의 실행 가능한 1 단계가 아니라 어떤 belief 의 최적 반응도 아님. 정의역을 줄인 Ω_λ 도 최적 반응 α* (벌점 0) 를 품으므로 ≥ V*(x̄): 상한 인증은 그대로 유효.
#   줄인 정의역의 α 는 원래 정의역에도 있으므로 하한 (incumbent) 쪽도 그대로. orcl_replay.jl HUCAP=1 로 지리 1 S=10 인증 386 → 211 s, 지리 5 같은 시간 상한 간격 절반 (2026-11-25)
nd_hc = Dict{Int,Any}()
function nd_hucap(c)
    haskey(nd_hc, c) && return nd_hc[c]
    xb = pts[c]; K = nd.meta[:K]; tiers = nd.meta[:tiers]; hU = copy(nd.hU); hidx = nd.meta[:hidx]; sx = nd.meta[:site_x]
    for i in nd.meta[:nA]+1:nd.meta[:nst]
        b = sum(tiers[(t - 1) % K + 1] * xb[t] for t in sx[i]; init=0.0)
        k = hidx(i, 1); (b > 0.5 && b < hU[k]) && (hU[k] = b)
    end
    nd_hc[c] = nz_with_hU(nd, hU)
end
function oracle_at!(c; cap=ORCL_TL)
    alseed!(c) && return :seed                    # v13
    evalall!([c]); t0v = mval(c) - qv[c]; xb = pts[c]
    tau = t0v + ORCL_THETA * mref(c)
    gap_o = min(1e-3, 0.4 * mref(c) / max(1.0, abs(t0v)))      # 원고: 허용오차 < m/2 (멈추면 상한 ≤ 하한 + 0.4 m ≤ τ⁺ 또는 위반)
    lim = max(10.0, min(cap, dl() - (time() - wall))); orcap[c] = cap
    to = @elapsed (res = nz_alpha_bnb_dr(HUCAP ? nd_hucap(c) : nd, xb; nworkers=max(1, NT - 1), time_limit=lim, rel_gap=gap_o, target=tau, verbose=false, resume=get(orstate, c, nothing), resp=ORCL_RESP, ORCL_KW...))
    orcert[c] = res[:UB] <= tau
    orstate[c] = res[:state]; alpush!(res[:α])
    global t_or += to; global n_or += 1
    # v9: 위반 판정은 incumbent α 의 정확값으로. 벌점 Ω 의 incumbent 값 res[:LB] 는 α 가 꺾임점에서 1e-5 만 벗어나도 λᵁ 벌점으로 몇 단위 낮다 (2026-11-24 진단)
    vx = res[:α] !== nothing ? exact_value(xb, res[:α]) : nothing
    lbx = max(res[:LB], (vx !== nothing && vx[2] !== nothing) ? vx[1] : -Inf)
    orLB[c] = max(get(orLB, c, -Inf), lbx)
    if qv[c] + res[:UB] < UB; global UB = qv[c] + res[:UB]; global xbest = c; end
    how = :none
    if vx !== nothing && lbx > t0v + 1e-6 * max(1.0, abs(t0v))
        vv, dd, r2, a1 = vx
        if dd !== nothing
            how = add_and_secure!(c, vv, dd, r2)
            vv <= t0v + 1e-6 * max(1.0, abs(t0v)) && restricted!(c, a1)
        end
    end
    if how == :none && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v)); nogood!(c, res[:LB]); how = :nogood_lb; end
    @printf("  [orcl] x=%-24s V in [%.2f, %.2f] (정확값 %.2f), t0 %.2f, 목표 %.2f (%.0fs, 상한 %.0fs%s) %s, 블록 %d, LB %.2f UB %.2f (경과 %.0fs)\n", loc_tkp_xstr(nd, xb), res[:LB], res[:UB], lbx, t0v, tau, to, lim, res[:resumed] ? ", 이어 풂" : "", how, length(blks), LB, UB, time() - wall)
    @printf("         α-B&B: 노드 %d, 이완 %.0fs, 평가 %.0fs, 범위 %.0fs (MIP %d, 실패 %d), 재확인 %.0fs, ipopt %d, 고정 비율 %.2f, 수치 오류 %d\n",
            res[:nodes], res[:t_relax], res[:t_eval], res[:t_range], res[:range_mips], res[:range_fail], res[:t_recheck], res[:ipopt_calls], res[:fixfrac], res[:numerr]); flush(stdout)
    # 재현용 기록 (α-B&B 조정): 같은 인자로 nz_alpha_bnb_dr 를 따로 부를 수 있게
    push!(orcl_log, (g=GEO, S=S, x=copy(xb), tau=tau, gap=gap_o, lim=lim, LB=res[:LB], UB=res[:UB], time=to, nodes=res[:nodes]))
    serialize(@sprintf("orcl_calls_g%d_S%d.jls", GEO, S), orcl_log)
    return how
end

# ---- 시작 ----
b1 = add_lpblk!(qh, qh, MU); evalpairs!([(1, b1)]); push!(cols, Set([1]))
# v12: t_low (no-good 상수) 는 min_{x ∈ X} V*(x) 의 유효한 하한이면 된다. 블록 L_b(x) = min_plan [C_L + μ(C_F − ηC_L⁺ + 쌍대 목적)] 의 벌점항은
#   정수 x 에서 약쌍대로 ≥ 0 이므로 L_b(x) ≥ min_plan C_L. 그래서 follower 최적성을 뺀 낙관값 LP (x 는 conv X 로 완화, 명목 r = q̂) 의 값이 하한이다.
#   v11 까지는 명목 블록 가격 MILP 를 끝까지 (S = 20 에서 355 ~ 376 s). 시간 상한을 둔 MILP bound 는 블록 LP 완화가 약해 −10⁷ (변수 하한) 에 머묾 (지리 13 점검).
function tlow_lp()
    M = Model(lp_opt); nh = nz_nh(nd)
    @variable(M, 0 <= xr[i=1:nx] <= (nd.x_allowed[i] ? 1.0 : 0.0)); @constraint(M, sum(xr) <= nd.gamma)
    for (ix, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(xr[i] for i in ix) <= rhs); end
    @variable(M, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(M, nd.W * h .<= nd.wvec); @variable(M, y[1:ny, 1:S] >= 0)
    for k in 1:m, s in 1:S
        @constraint(M, sum(a * y[j, s] for (j, a) in C.rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) - sum(e.g[s] * xr[e.i] for e in C.ents_of_row[k]; init=0.0) <= nd.u[k, s])
    end
    @objective(M, Min, sum(qh[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0; init=AffExpr(0.0)))
    optimize!(M)
    return termination_status(M) == MOI.OPTIMAL ? objective_value(M) : -Inf
end
bnd = 0.0; tl0 = @elapsed (global bnd = tlow_lp()); how_tl = "낙관값 LP"
if !isfinite(bnd)                                   # LP 가 안 풀리면 예전처럼 명목 블록 가격 MILP 를 끝까지
    tl0 += @elapsed (global bnd = first(pricemip(b1, 1.0, zeros(nx), 0.0, Inf, ROOT))); how_tl = "가격 MILP"
end
isfinite(bnd) || error("ccgbp v16: t_low 를 못 구함")
global tlow = bnd - 1e-6 * max(1.0, abs(bnd))
@printf("  [시작] t 하한 %.2f (%s %.1fs)\n", tlow, how_tl, tl0); flush(stdout)

# ---- 1 단계 (루트, 휴리스틱 열 생성 + 탐색 + oracle): belief 와 UB 모으기 ----
ph1_why = "총량 몫"   # v16: 1 단계가 끝난 까닭 (break 가 고쳐 씀)
if PH1
    while time() - wall < min(TLB, PH1_SHARE * TL)   # v16: 1 단계 총량 몫
        r = dwsolve!(ROOT; exact=false); upd_inc!(r[6])
        @printf("  [ph1] 루트 RMP %.2f, 마스터 UB %.2f (x=%s), 블록 %d, LP 누적 %d (경과 %.0fs)\n", r[2], r[4], loc_tkp_xstr(nd, pts[r[5]]), length(blks), n_lp[], time() - wall); flush(stdout)
        trysearch(r[6]) && continue
        tph = time(); hw = oracle_at!(r[5]; cap=PH1_CAP); orph1[r[5]] = get(orph1, r[5], 0.0) + (time() - tph)   # v14: 1 단계 oracle 은 모두 PH1_CAP 까지
        if hw == :none                                # v15: 결론 없음이면 같은 점을 누적 PH1_TOT 까지 이어 풂 (다음 반복의 dwsolve 가 같은 점을 줌)
            (get(orcert, r[5], false) || orph1[r[5]] >= PH1_TOT) && (global ph1_why = get(orcert, r[5], false) ? "인증" : "같은 점 누적 PH1_TOT"; break)
            @printf("  [ph1] 결론 없음 → 같은 점 이어 풂 (누적 %.0fs / %.0fs)\n", orph1[r[5]], PH1_TOT); flush(stdout)
        end
    end
end
t_ph1 = time() - wall
@printf("  [1 단계 끝] %.0fs, belief %d, UB %.2f%s\n", t_ph1, count(B -> B.kind == :lp, blks), UB,
        PH1 ? @sprintf(" (v16: 끝난 까닭 %s, 총량 몫 %.0fs)", ph1_why, PH1_SHARE * TL) : ""); flush(stdout)

# ---- 2 단계: branch-and-price ----
"분기: 거점 개설 o_j (분수) 를 strong branching (휴리스틱 열 생성 값의 곱 점수) 으로, 없으면 크기 x_jk"
function children(N, xv, vpar)
    o = Dict(j => sum(xv[i] for i in sitevars(j)) for j in sites)
    frac = [j for j in sites if !(j in N.opn) && 1e-5 < o[j] < 1 - 1e-5]
    mk_open(j, e) = Node(copy(N.f0), copy(N.f1), union(N.opn, [j]), N.lb, e, N.depth + 1, N.tag * @sprintf(" o%d=1", j))
    mk_close(j, e) = Node(union(N.f0, sitevars(j)), copy(N.f1), copy(N.opn), N.lb, e, N.depth + 1, N.tag * @sprintf(" o%d=0", j))
    if !isempty(frac)
        sort!(frac; by=j -> -min(o[j], 1 - o[j])); cands = frac[1:min(max(SB_K, 1), length(frac))]
        SB_K <= 1 && return (mk_open(cands[1], vpar), mk_close(cands[1], vpar))
        best = nothing; bsc = -Inf
        for j in cands
            r1 = dwsolve!(mk_open(j, vpar); exact=false); r0 = dwsolve!(mk_close(j, vpar); exact=false)
            e1 = r1[2]; e0 = r0[2]; upd_inc!(vcat(r1[6], r0[6]))
            sc = max(e1 - vpar, 1e-6) * max(e0 - vpar, 1e-6)
            @printf("      [sb] o%d (%.3f): 열림 %.2f, 닫힘 %.2f, 점수 %.3g\n", j, o[j], e1, e0, sc); flush(stdout)
            sc > bsc && (bsc = sc; best = (mk_open(j, e1), mk_close(j, e0)))
        end
        return best
    end
    fx = [i for i in al if 1e-5 < xv[i] < 1 - 1e-5]
    isempty(fx) && return nothing
    i = fx[argmax([min(xv[i], 1 - xv[i]) for i in fx])]
    return (Node(copy(N.f0), union(N.f1, [i]), copy(N.opn), N.lb, vpar, N.depth + 1, N.tag * @sprintf(" x%d=1", i)),
            Node(union(N.f0, [i]), copy(N.f1), copy(N.opn), N.lb, vpar, N.depth + 1, N.tag * @sprintf(" x%d=0", i)))
end
# v12: 주차 노드 = 하한 ≥ PT 라 분기하지 않고 둔 노드 (노드, 점; 점은 정수 노드면 그 점, 아니면 0). 하한은 증명된 Farley 하한만 쓴다
#   (v11 의 closed_lb 는 정수 노드에서 mval(c) 와 q + oracle 하한을 썼는데, 둘 다 노드 전체의 증명된 하한이 아님).
#   막힌 정수 노드 (엄격한 정확 CG 뒤에도 하한 < 기준) 는 stuckq 에 남겨 LB 에만 넣는다.
parkq = Tuple{Node,Int}[]; stuckq = Tuple{Node,Int}[]
gapcut() = isfinite(UB) ? UB - TOL * max(1.0, abs(UB)) : Inf
gapok(lb) = isfinite(UB) && lb >= gapcut()
global_lb(extra=Inf) = min(extra, minimum((o.lb for o in openq); init=Inf), minimum((e[1].lb for e in parkq); init=Inf), minimum((e[1].lb for e in stuckq); init=Inf))
nb_seen = Ref(0); n_park = Ref(0); n_reopen = Ref(0); n_cinc_or = Ref(0); n_strict = Ref(0)
"belief·UB 가 바뀌면 cinc 를 다시 고르고, 하한 < PT 인 주차 노드를 다시 엶 (PT = Inf, 곧 UB 가 충분하면 모두 다시 열어 가지치기로 정리)"
function sync_park!()
    if length(blks) != nb_seen[]
        refresh_cinc!(); nb_seen[] = length(blks)
    end
    pt = PT(); keep = Tuple{Node,Int}[]
    for e in parkq
        if !isfinite(pt) || e[1].lb < pt - 1e-9 * max(1.0, abs(pt))
            push!(openq, e[1]); n_reopen[] += 1
        else
            push!(keep, e)
        end
    end
    global parkq = keep
end
park!(N, lb, v, c) = (push!(parkq, (Node(N.f0, N.f1, N.opn, lb, v, N.depth, N.tag), c)); n_park[] += 1)
# v10: 루트 정확 가격 CG (Lagrangian 하한, X 전체에 유효). v12: ROOT_EXACT = end 가 기본 — B&P 가 마감 (TL − RESERVE) 까지 못 끝내면 그때 한 번 돌려 LB 로 보고.
#   start (v10·v11) 는 2 단계 전에 돌리는데, S = 10 에서 전체 시간의 38 ~ 72 % 였다 (지리 1 1,268 s, 지리 13 점검 1,200 s, 대부분 가지치기에 안 쓰이는 정밀도).
function root_exact!(tag; deadline=Inf)
    tre = @elapsed (rr = dwsolve!(ROOT; exact=true, ubglob=UB, stopat=PT(), deadline=deadline))
    rr[1] < Inf && (upd_inc!(rr[6]); global LB = max(LB, rr[1]))
    @printf("  [루트 정확 CG, %s] 하한 %.2f, RMP %.2f, UB %.2f, 간격 %.3f%%, 주차 기준 PT %.2f (%.0fs, 정확 반복 %d)\n", tag, rr[1], rr[2], UB, 100 * (UB - rr[1]) / abs(UB), PT(), tre, rr[7].nex); flush(stdout)
    rr
end
if ROOT_EXACT == "start" && time() - wall < TLB
    rr = root_exact!("시작")
    rr[1] < Inf && (global ROOT = Node(ROOT.f0, ROOT.f1, ROOT.opn, rr[1], rr[2], 0, "root"))
end
openq = [ROOT]; nproc = 0; nprune = 0; nexact = 0; nint = 0; nb_seen[] = length(blks)
while time() - wall < dl()
    sync_park!()
    global LB = max(LB, min(gapcut(), global_lb()))
    if isfinite(UB) && UB - LB <= TOL * max(1.0, abs(UB)) * (1 + 1e-9); global status = :Optimal; break; end
    if isempty(openq)
        # 마스터가 나무 몫 안으로 풀림 (열린 노드 없음, 남은 것은 주차) → 마스터 incumbent 에서 oracle (같은 점이면 이어 풂)
        ub_ok() && error("ccgbp v16: UB 가 충분한데 열린·주차 노드 없이 간격이 남음 (LB $LB, UB $UB)")
        cinc > 0 || error("ccgbp v16: 마스터 incumbent 없음")
        n_cinc_or[] += 1
        @printf("  [마스터 풀림] 주차 %d, 막힘 %d, LB %.2f, cinc %s (mval %.2f, PT %.2f, 목표 UB ≤ %.2f) → oracle\n", length(parkq), length(stuckq), LB,
                loc_tkp_xstr(nd, pts[cinc]), mval(cinc), PT(), mval(cinc) + ORCL_THETA * mref(cinc)); flush(stdout)
        oracle_at!(cinc)
        continue
    end
    sort!(openq; by=N -> (N.lb, N.est)); N = popfirst!(openq)
    gapok(N.lb) && (global nprune += 1; continue)
    (isfinite(PT()) && N.lb >= PT()) && (park!(N, N.lb, N.est, 0); continue)
    global nproc += 1
    local r
    while true
        r = dwsolve!(N; exact=false); r[1] == Inf && break
        upd_inc!(r[6]); trysearch(filter(c -> !gapok(mval(c)), r[6])) || break    # 이미 UB − tol 위인 점의 belief 는 가지치기를 못 바꿈
    end
    if r[1] == Inf; global nprune += 1; continue; end            # 실행불가 노드
    length(blks) != nb_seen[] && (refresh_cinc!(); nb_seen[] = length(blks))
    v, xv = r[2], r[3]; lbN = N.lb
    integral = all(i -> abs(xv[i] - round(xv[i])) < 1e-6, al)
    if v >= min(gapcut(), PT()) || integral                     # 자르거나 주차할 수 있을 때만 정확 가격 (v12: 주차 기준 PT 도)
        r = dwsolve!(N; exact=true, ubglob=UB, stopat=PT()); global nexact += 1; upd_inc!(r[6])
        lbN = max(lbN, r[1]); v, xv = r[2], r[3]; integral = all(i -> abs(xv[i] - round(xv[i])) < 1e-6, al)
    end
    c = integral ? pid(round.(xv)) : 0
    if integral
        global nint += 1; upd_inc!([c])
        if lbN < min(gapcut(), PT())                              # 정수 RMP 인데 증명된 하한이 모자람 → 엄격한 정확 CG 한 번
            r = dwsolve!(N; exact=true, ubglob=UB, stopat=PT(), cgtol=0.0, trel=1e-7); n_strict[] += 1; lbN = max(lbN, r[1])
        end
    end
    @printf("  [node %3d] %-34s 깊이 %d: RMP %.2f, LB %.2f, %s, 열린 %d, 주차 %d, LB* %.2f UB %.2f, 마스터 inc %.2f (PT %.2f), 블록 %d, MILP %d (경과 %.0fs)\n",
            nproc, N.tag, N.depth, v, lbN, integral ? "정수" : "분수", length(openq), length(parkq), LB, UB, cinc > 0 ? mval(cinc) : NaN, PT(), length(blks), n_mip[], time() - wall); flush(stdout)
    gapok(lbN) && (global nprune += 1; continue)
    (isfinite(PT()) && lbN >= PT()) && (park!(N, lbN, v, c); continue)
    if integral
        push!(stuckq, (Node(N.f0, N.f1, N.opn, lbN, v, N.depth, N.tag), c))
        @printf("  [!] 정수 노드 %s 의 하한 %.2f 가 기준 %.2f 아래 (엄격 CG 뒤) → 막힌 노드로 둠\n", N.tag, lbN, min(gapcut(), PT())); flush(stdout)
        continue
    end
    ch = children(N, xv, v)
    ch === nothing && error("ccgbp v16: 분수 노드인데 분기 변수 없음 ($(N.tag))")
    push!(openq, ch[1]); push!(openq, ch[2])
end
# v16 (2026-11-26 밤 수정): 마지막 노드를 마감 TLB 뒤에 가지치기하면 while 머리의 LB 갱신을 못 거치고 루프가 끝나 나무가 닫혔는데도
#   TimeLimit 으로 보고되고 마감 루트 CG 가 쓸모없는 하한을 냄 (N_B 4 지리 12 v17: 3,568 s 에 루트 가지치기, 보고 LB −9,999,422). 루프 머리와 같은 갱신을 한 번 더.
global LB = max(LB, min(gapcut(), global_lb()))
isfinite(UB) && UB - LB <= TOL * max(1.0, abs(UB)) * (1 + 1e-9) && (global status = :Optimal)
ran_final_cg = false
if status != :Optimal && ROOT_EXACT == "end" && !isfinite(LB) && time() - wall < TL   # v14: 나무 하한이 있으면 dl() = TL 이라 여기 오지 않음
    root_exact!("마감"; deadline=wall + TL); global ran_final_cg = true
    isfinite(UB) && UB - LB <= TOL * max(1.0, abs(UB)) * (1 + 1e-9) && (global status = :Optimal)
end
tot = time() - wall
nbel = count(B -> B.kind == :lp, blks)
@printf("RESULT ccgbp g=%d status=%s x=%s LB=%.4f UB=%.4f gap=%.3f%% beliefs=%d 노드 %d (가지침 %d, 정확 %d, 정수 %d, 남은 %d, 주차 %d, 막힘 %d) 안전장치 %d (no-good %d, restricted %d, μ 올림 %d) time=%.1f (1 단계 %.0fs)\n",
        GEO, status, xbest == 0 ? "-" : loc_tkp_xstr(nd, pts[xbest]), LB, UB, 100 * (UB - LB) / abs(UB), nbel, nproc, nprune, nexact, nint, length(openq), length(parkq), length(stuckq), n_ng[] + n_rc[], n_ng[], n_rc[], n_mu[], tot, t_ph1)
@printf("  시간: oracle %d 번 %.0fs, 탐색 %d 번 %.0fs, 블록 LP %d 번 %.0fs, 가격 MILP %d 번 %.0fs (노드 %d), RMP %d 번 %.0fs (crossover 재시도 %d), 생성기 구성 %.0fs; 서로 다른 점 %d / %d\n",
        n_or, t_or, n_srch[], t_srch[], n_lp[], t_lp[], n_mip[], t_mip[], nodes_mip[], n_rmp[], t_rmp[], n_rmpfix[], t_build[], length(unique([c for (c, _) in keys(cache)])), NP)
@printf("  v12: 주차 %d 번, 다시 엶 %d 번, 마스터 incumbent oracle %d 번, 엄격 정확 CG %d 번, 후보점 %d\n", n_park[], n_reopen[], n_cinc_or[], n_strict[], length(intpts))
@printf("  v13: α 재사용 위반 %d 번 (정확 평가 %d 번 %.0fs), α 풀 %d\n", n_seed[], n_seedev[], t_seed[], length(alpool))
@printf("  v14: 1 단계 상한 %.0fs, 마감 루트 CG %s, 2 단계 마감 %s\n", PH1_CAP, ran_final_cg ? "돌림" : "건너뜀", isfinite(LB) ? "TL (하한 있음)" : "TL − RESERVE")
@printf("  열: RMP 에 들어간 열 (블록 합) %d, 서로 다른 열 %d, 번호를 붙인 점 %d / |X| %d\n", sum(length.(cols)), length(union(cols...)), length(pts), NP)
