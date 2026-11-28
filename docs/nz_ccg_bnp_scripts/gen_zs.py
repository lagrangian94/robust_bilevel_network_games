# gen_zs.py — B&P 기본 판 ccgbp.jl (v17) 에서 zero-sum 네트워크 차단 판을 만든다 (2026-11-28 새벽, 사용자 "제로섬도 ccg bnp로 풀어보고").
#   ccgbp.jl 은 그대로 두고, 바꾼 사본을 출력 경로 (기본 scratchpad 의 ccgbp_zs.jl) 에 쓴다. 저장소 코드는 건드리지 않음.
#   인스턴스: true_dro/factor_5/run_nig_coupled_scaling.jl 과 같은 규칙 (factor_additive k = 5, 시드 ZSEED, w = 0.5 γ median, v ~ Bernoulli(0.75),
#     γ = 2, grid5x5 구조는 seed 42 고정, 실망은 모든 호 차단 가능 + source/sink cut 카디널리티) → nz_from_true_dro → nz_with_delta.
#   블록 LP 의 g < 0 McCormick 은 2026-11-28 부터 저장소 판 (nz_belief_ccg.jl) 을 씀.
#   입지 전용 부분 바꿈: 거점 = 호 (K = 1), 이웃·씨앗점·선형 최소 (linmin) 가 카디널리티 (meta[:omp_card]) 를 지킴, HUCAP 끔 (입지 전용),
#     상자 없음 (TV 만, 탐색의 상자는 [0, 1]), x 출력은 호 번호, RESULT 에 net·seed·ε·β·δ·λᵁ.
#   환경 변수 (기본): NET abilene, ZSEED 42, S 10, ZEPS 0.3, ZBETA 0.4, ZDELTA 0.1, LAMU 1e4 (ccgbp 와 같은 이름, oracle·모형 λᵁ), 나머지는 ccgbp.jl 과 같음.
#   사용: python gen_zs.py [출력 경로]
import re, sys
here = __file__.rsplit('\\', 1)[0] if '\\' in __file__ else __file__.rsplit('/', 1)[0]
src = here + '/ccgbp.jl'
out = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\user\AppData\Local\Temp\claude\a31bffbb1639accd\310f4530-ff69-4912-9a26-69df1bdd76ba\scratchpad\ccgbp_zs.jl'
t = open(src, encoding='utf-8').read()

def rep(t, a, b, cnt=1):
    n = t.count(a)
    assert n == cnt, (n, a[:100])
    return t.replace(a, b)

# (1) 머리말
t = rep(t, "# ccgbp.jl — belief 블록 CCG", "# ccgbp_zs.jl — gen_zs.py 가 ccgbp.jl 에서 만든 zero-sum 네트워크 차단 판 (직접 고치지 말 것).\n# ccgbp.jl — belief 블록 CCG")

# (2) zero-sum 인스턴스 (ABB_PATCH 줄 뒤, nd 생성 전)
inst = r'''
# ---- zero-sum 네트워크 차단 인스턴스 (gen_zs.py) ----
include(joinpath(root, "network_generator.jl")); using .NetworkGenerator; const NG = NetworkGenerator
include(joinpath(root, "true_dro", "true_dro_data.jl"))
using Statistics
const NET = get(ENV, "NET", "abilene"); const ZSEED = parse(Int, get(ENV, "ZSEED", "42"))
const ZEPS = parse(Float64, get(ENV, "ZEPS", "0.3")); const ZBETA = parse(Float64, get(ENV, "ZBETA", "0.4")); const ZDELTA = parse(Float64, get(ENV, "ZDELTA", "0.1"))
"run_nig_coupled_scaling.jl 의 make_network·인스턴스 규칙과 같음 (grid5x5 구조는 seed 42 고정, 용량·v 는 ZSEED)"
function zs_instance(S, lamU)
    if NET == "grid5x5"
        net = NG.generate_grid_network(5, 5; seed=42); intd = net.interdictable_arcs
    else
        gen = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network,
                   "nobel_us" => NG.generate_nobel_us_network, "sioux_falls" => NG.generate_sioux_falls_network)
        haskey(gen, NET) || error("zs_instance: NET = $(NET) 모름")
        net0 = gen[NET](); intd = fill(true, length(net0.arcs))
        net = NG.RealWorldNetworkData(net0.name, net0.original_node_names, net0.nodes, net0.arcs, net0.N, intd, net0.arc_adjacency, net0.node_arc_incidence)
    end
    γ = 2; Kz = length(net.arcs) - 1
    caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S; interdictable_arcs=intd, seed=ZSEED, num_factors=5)
    w = round(0.5 * γ * median(caps[findall(intd[1:Kz]), :]); digits=4)
    Random.seed!(ZSEED); v = zeros(Kz, S)
    for k in 1:Kz, s in 1:S; v[k, s] = intd[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0; end
    td = make_true_dro_data(net, caps, fill(1.0 / S, S), ZEPS, ZEPS; w=w, lambda_U=lamU, gamma=γ, beta=ZBETA, v_scenarios=v)
    nd0 = nz_from_true_dro(td)
    card = Tuple{Vector{Int},Int}[]
    if NET != "grid5x5"
        src = [i for i in 1:Kz if net.arcs[i][1] == "s"]; snk = [i for i in 1:Kz if net.arcs[i][2] == "t"]
        length(src) >= 2 && push!(card, (src, length(src) - 1)); length(snk) >= 2 && push!(card, (snk, length(snk) - 1))
    end
    # linmin 의 탐욕이 정확하려면 카디널리티 집합이 서로 겹치지 않아야 함 (전체 상한 γ 와 함께 laminar matroid)
    length(card) == 2 && !isempty(intersect(card[1][1], card[2][1])) && error("zs_instance: source·sink 집합이 겹침 (linmin 탐욕이 정확하지 않음)")
    nd0.meta[:omp_card] = card; nd0.meta[:network] = NET; nd0.meta[:w] = w; nd0.meta[:K] = 1; nd0.meta[:nB] = nd0.nx
    return nz_with_delta(nd0, ZDELTA), td
end
zs_xstr(nd, z) = "{" * join(string.(findall(>(0.5), z)), ";") * "}"
"x 가 X 의 점인가 (카디널리티 γ, 차단 가능 호, source/sink cut)"
zs_ok(z) = sum(z) <= nd.gamma + 0.5 && all(nd.x_allowed[i] || z[i] < 0.5 for i in eachindex(z)) &&
           all(sum(z[i] for i in ix) <= rhs + 0.5 for (ix, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]))
'''
t = rep(t, 'envf(k, d) = get(ENV, k, d)\n', 'envf(k, d) = get(ENV, k, d)\n' + inst)

# (2b) 블록 LP 의 x 곱 McCormick: 2026-11-28 사용자 "다 고쳐" 로 저장소 nz_belief_ccg.jl 의 _bccg_add_block! 에 반영됨
#      (g ≥ 0 은 아래쪽 식, g < 0 (차단 g = −vξ) 은 위쪽 식 ps ≤ pd, ps ≤ U x). 예전에는 이 사본에서 재정의했다. 저장소 판이 그 식인지만 확인한다.
bsrc = open(here + '/../../nonzero_sum/nz_belief_ccg.jl', encoding='utf-8').read()
assert 'push!(cons, @constraint(M, ps <= pd[e.k, s])); push!(cons, @constraint(M, ps <= U * x[e.i]))' in bsrc, '저장소 _bccg_add_block! 에 g < 0 McCormick 없음'

# (3) 인스턴스 생성
lines = t.split('\n')
ix = [i for i, l in enumerate(lines) if l.startswith('nd = make_loc_tkp(')]
assert len(ix) == 1, ix
lines[ix[0]] = ('nd, td_zs = zs_instance(parse(Int, envf("S", "10")), LAMU)\n'
                '@printf("  [zero-sum] net %s, seed %d, S %d, 호 %d (차단 가능 %d), γ %d, w %.4f, ε̂ = ε̃ = %.2f, β %.2f, δ %.3f, λᵁ %.1e, 카디널리티 %s\\n", '
                'NET, ZSEED, nd.S, nd.nx, count(nd.x_allowed), nd.gamma, nd.meta[:w], ZEPS, ZBETA, ZDELTA, LAMU, string([(length(a), b) for (a, b) in nd.meta[:omp_card]]))')
t = '\n'.join(lines)

# (4) HUCAP 은 입지 전용
t = rep(t, 'const HUCAP = envf("HUCAP", "1") == "1"', 'const HUCAP = envf("HUCAP", "0") == "1"; HUCAP && error("ccgbp_zs: HUCAP 은 입지 전용")')

# (5) 이웃·씨앗점이 카디널리티를 지킴
t = rep(t, "put(conf) = (w = zeros(nx); for (s, k) in conf; w[(s - 1) * K + k] = 1; end; pid(w))",
           "put(conf) = (w = zeros(nx); for (s, k) in conf; w[(s - 1) * K + k] = 1; end; zs_ok(w) ? pid(w) : 0)")
t = rep(t, "    any(z[i] > 0.5 for i in N.f0) && return 0\n    pid(z)\nend",
           "    any(z[i] > 0.5 for i in N.f0) && return 0\n    zs_ok(z) || return 0\n    pid(z)\nend")

# (6) linmin: 카디널리티를 지키는 탐욕 (전체 상한 + 서로 겹치지 않는 집합 상한 = laminar matroid 라 정확)
i0 = t.index('"노드 위 X 에서 선형 최소 min cᵀz (탐욕, 정확)"')
i1 = t.index('# ---------------- 가격 MILP')
new_linmin = '''"노드 위 X 에서 선형 최소 min cᵀz (zero-sum: K = 1, 전체 상한 γ + 서로 겹치지 않는 source/sink 상한 → laminar matroid 탐욕, 정확)"
function linmin(c, N)
    card = get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[])
    memb = [Int[] for _ in 1:nx]; for (q, (ixs, _)) in enumerate(card), i in ixs; push!(memb[i], q); end
    cnt = zeros(Int, length(card)); z = zeros(nx); v = 0.0; nf = 0
    for j in sites
        sv = sitevars(j); isempty(sv) && continue; i = sv[1]
        if i in N.f1 || j in N.opn
            i in N.f0 && return Inf, 0
            z[i] = 1.0; v += c[i]; nf += 1; for q in memb[i]; cnt[q] += 1; end
        end
    end
    (nf > nd.gamma || any(cnt[q] > card[q][2] for q in eachindex(card))) && return Inf, 0
    for (val, i) in sort([(c[i], i) for i in al if z[i] < 0.5 && !(i in N.f0) && c[i] < 0]; by=first)
        nf >= nd.gamma && break
        all(cnt[q] < card[q][2] for q in memb[i]) || continue
        z[i] = 1.0; v += val; nf += 1; for q in memb[i]; cnt[q] += 1; end
    end
    return v, pid(z)
end

'''
t = t[:i0] + new_linmin + t[i1:]

# (7) 상자 없음 (TV 만)
t = rep(t, "qh = nd.q_hat; bx = nd.meta[:box]", "qh = nd.q_hat; bx = haskey(nd.meta, :box) ? nd.meta[:box] : (dl = zeros(S), du = ones(S))   # zero-sum: 상자 없음")

# (7b) oracle: nz_alpha_bnb_dr 의 기본 (range_lp = true, 꼭짓점 LP 쌍대 범위) 은 recourse 값함수의 submodularity 를 전제 (입지에서만 측정, 증명 없음).
#      최대흐름은 직렬 호에서 보완재라 전제가 깨짐 → 2026-11-28 Polska s42 {6,33}: dr 기본 [17.2996, 17.2996] (24.6 s, 틀림),
#      range_lp = false [17.9315, 18.2326] (300 s), nz_alpha_bnb [17.9315, 17.9349] (300 s), 독립 하한 zsi.jl 17.9315 (zs_orcl_check.jl).
#      ZORCL = plain (기본, nz_alpha_bnb: zero-sum 교차검증된 판, 이어 풀기 없음) | dr (nz_alpha_bnb_dr, ORCL_KW 에 range_lp = false 를 줘야 유효)
orcl_old = "res = nz_alpha_bnb_dr(HUCAP ? nd_hucap(c) : nd, xb;"
assert t.count(orcl_old) == 1
t = t.replace(orcl_old, "res = zs_oracle(HUCAP ? nd_hucap(c) : nd, xb;")
zs_orc = r'''
const ZORCL = get(ENV, "ZORCL", "plain"); ZORCL in ("plain", "dr") || error("ccgbp_zs: ZORCL = plain | dr")
ZORCL == "dr" && !(get(ORCL_KW, :range_lp, true) === false) && error("ccgbp_zs: ZORCL = dr 은 ORCL_KW 에 range_lp = false 필요 (꼭짓점 LP 범위는 최대흐름에서 틀림, gen_zs.py (7b))")
"zero-sum oracle (gen_zs.py (7b)): plain = nz_alpha_bnb (이어 풀기 없음, 보고용 칸은 빈 값), dr = nz_alpha_bnb_dr 그대로"
function zs_oracle(ndo, xb; nworkers, time_limit, rel_gap, target, verbose, resume, resp, kw...)
    ZORCL == "dr" && return nz_alpha_bnb_dr(ndo, xb; nworkers, time_limit, rel_gap, target, verbose, resume, resp, kw...)
    r = nz_alpha_bnb(ndo, xb; nworkers=max(1, nworkers - 1), time_limit, rel_gap, target, verbose=false)
    return merge(r, Dict{Symbol,Any}(:state => nothing, :resumed => false, :t_range => 0.0, :range_mips => 0, :range_fail => 0,
                                     :t_recheck => 0.0, :fixfrac => NaN))
end
'''
# zs_oracle 정의는 인스턴스 코드 뒤 (ORCL_KW 정의 뒤여야 함) 에 넣는다
okw = 'const ORCL_KW = eval(Meta.parse('
i_okw = t.index(okw); i_eol = t.index('\n', i_okw)
t = t[:i_eol + 1] + zs_orc + t[i_eol + 1:]

# (8) 출력
t = t.replace("loc_tkp_xstr(nd, ", "zs_xstr(nd, ")
t = rep(t, '@sprintf("orcl_calls_g%d_S%d.jls", GEO, S)', '@sprintf("orcl_calls_zs_%s_s%d_S%d.jls", NET, ZSEED, S)')
t = rep(t, '@printf("RESULT ccgbp g=%d status=', '@printf("RESULT ccgbp_zs net=%s seed=%d eps=%.2f beta=%.2f delta=%.3f lamU=%.0e status=')
t = rep(t, '        GEO, status, xbest == 0 ?', '        NET, ZSEED, ZEPS, ZBETA, ZDELTA, LAMU, status, xbest == 0 ?')
assert 'loc_tkp_xstr' not in t
open(out, 'w', encoding='utf-8', newline='\n').write(t)
print('ok', out)
