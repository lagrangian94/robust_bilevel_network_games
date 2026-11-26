# -*- coding: utf-8 -*-
# gen_v18.py — ccgbp_v17.jl 에서 ccgbp_v18.jl 을 만든다 (2026-11-26 밤 시험판, md 11.19, 사용자 "PADM 도 궁금한데").
#  v17 + PADM 가격 생성기 (Kleinert & Schmidt 2021, IJOC 33(1), 원·쌍대 교대 벌점법).
#   블록 b = 강쌍대 간격 벌점 문제 (논문 식 9) 와 같은 꼴. 우리 MILP 는 쌍대 p 와 x 의 곱을 big-M 으로 풀지만, PADM 은 교대로 푼다:
#   (a) p̂ 고정 → (x, h, y) MILP: min w_b [C_L + ρ (C_F − η C_L⁺) + ρ Σ_e Σ_s g_eˢ p̂_{k_e,s} x_{i_e}] − π_bᵀx,  x ∈ X(N) (노드 고정 포함), big-M 없음
#   (b) x 고정 → 블록 LP (x 고정) 를 풀어 그 쌍대 pd 를 새 p̂ 로 (블록 값 L_b(x) 와 같은 LP)
#   벌점 ρ 는 PD_RHO 일정 (기본 0.01, 0.1, 1, 10, 100, 1e3, 1e4 = μ) 으로 키움 (μ 를 바로 쓰면 지금 쌍대가 계속 최적인 x 만 골라 움직이지 않음).
#   시작점 = 블록마다 λ 가 큰 지지점 PD_START 개. 나온 x 는 x 고정 블록 LP 로 정확히 재서 감축 비용 < −tol 이면 열 (블록당 TB_CMAX 개까지).
#   순서: 한 걸음 생성기 → tabu (v17) → PADM → 정확 MILP. 기본은 정확 가격을 할 dwsolve! 에서만 (PD_WHEN = exact; all 이면 휴리스틱 열 생성에도).
#   정확 가격 MILP 와 Farley 하한은 그대로라 하한 유효성은 바뀌지 않음.
import os
here = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(here, 'ccgbp_v17.jl'), encoding='utf-8').read()

def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:90]!r}')
    s = s.replace(a, b)

head = ('# ccgbp_v18.jl — gen_v18.py 가 v17 에서 만든 파일 (손으로 고치지 말 것). v17 + PADM 가격 생성기 (2026-11-26 밤 시험판).\n'
        '#  docs/nz_ccg_bnp_2026-11-23.md 11.19. tabu 가 실패하면 PADM (Kleinert & Schmidt 2021) 으로 전역 x 후보, 그다음 정확 MILP.\n'
        '# (아래는 v17 머리말) ')
s = head + s
rep('(TB_START >= 1 && TB_IT >= 1 && TB_NS >= 1 && TB_LEN >= 1 && TB_CMAX >= 1) || error("ccgbp v17: TB_* 는 1 이상")',
    '(TB_START >= 1 && TB_IT >= 1 && TB_NS >= 1 && TB_LEN >= 1 && TB_CMAX >= 1) || error("ccgbp v17: TB_* 는 1 이상")\n'
    'const PADM = envf("PADM", "1") == "1"                                       # v18: PADM 가격 생성기 켬/끔\n'
    'const PD_RHO = parse.(Float64, split(envf("PD_RHO", "0.01,0.1,1,10,100,1000,10000"), ","))   # v18: 벌점 일정\n'
    'const PD_INNER = parse(Int, envf("PD_INNER", "3"))      # v18: 벌점마다 교대 상한\n'
    'const PD_START = parse(Int, envf("PD_START", "1"))      # v18: 블록당 시작점 수\n'
    'const PD_MAXMIP = parse(Int, envf("PD_MAXMIP", "60"))   # v18: 라운드당 x 단계 MILP 상한\n'
    'const PD_WHEN = envf("PD_WHEN", "exact")                # v18: exact (정확 가격을 할 열 생성에서만) | all\n'
    '(PD_INNER >= 1 && PD_START >= 1 && PD_MAXMIP >= 1 && !isempty(PD_RHO) && all(>(0), PD_RHO)) || error("ccgbp v17: PD_* 값이 잘못됨")\n'
    'PD_WHEN in ("exact", "all") || error("ccgbp v17: PD_WHEN 은 exact·all")')
rep('@printf("--- ccgbp v17 (v16 + ',
    '@printf("--- ccgbp v18 (v17 + PADM 가격 %s: 벌점 %s·교대 %d·시작 %d·MILP 상한 %d·%s) | ccgbp v17 (v16 + ')
rep('        TABU ? "켬" : "끔", TB_START, TB_IT, TB_NS, TB_LEN, TB_CMAX, PH1_SHARE',
    '        PADM ? "켬" : "끔", join(string.(PD_RHO), ","), PD_INNER, PD_START, PD_MAXMIP, PD_WHEN, TABU ? "켬" : "끔", TB_START, TB_IT, TB_NS, TB_LEN, TB_CMAX, PH1_SHARE')
# PADM 부품 (tabu_round! 바로 뒤)
rep('    n_tblp[] += n_lp[] - lp0; n_tbcol[] += added; added > 0 && (n_tbok[] += 1)\n    return added\nend\n',
    '    n_tblp[] += n_lp[] - lp0; n_tbcol[] += added; added > 0 && (n_tbok[] += 1)\n    return added\nend\n'
    '# ---------------- v18: PADM 가격 생성기 (Kleinert & Schmidt 2021) ----------------\n'
    'n_pdr = Ref(0); n_pdok = Ref(0); n_pdmip = Ref(0); t_pdmip = Ref(0.0); n_pdlp = Ref(0); n_pdprop = Ref(0); n_pdcol = Ref(0); t_pd = Ref(0.0)\n'
    'n_pdstay = Ref(0); n_pdnone = Ref(0); n_pdneg = Ref(0)   # x 단계가 제자리 / 해 없음, 제안 중 감축 비용 음수\n'
    'pddist = Dict{Int,Int}()   # 진단: PADM 이 넣은 열과 그 블록 지지점의 최소 해밍 거리 → 개수\n'
    'pdx = Dict{Int,Any}(); pdb = Dict{Int,Any}()\n'
    '"블록 b 의 x 단계 모형 (x 이진, follower 원변수 h·y, big-M·쌍대 변수 없음)"\n'
    'function padm_xmodel(b)\n'
    '    get!(pdx, b) do\n'
    '        B = blks[b]; d, r = B.d, B.r; nh = nz_nh(nd)\n'
    '        M = Model(() -> (o = Gurobi.Optimizer(envs[NT + 1]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o))\n'
    '        @variable(M, xb[1:nx], Bin); @constraint(M, sum(xb) <= nd.gamma)\n'
    '        for (ix, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(xb[i] for i in ix) <= rhs); end\n'
    '        for i in 1:nx; nd.x_allowed[i] || fix(xb[i], 0.0; force=true); end\n'
    '        h = @variable(M, [i=1:nh], lower_bound = 0.0, upper_bound = nd.hU[i]); @constraint(M, nd.W * h .<= nd.wvec)\n'
    '        y = @variable(M, [1:ny, 1:S], lower_bound = 0.0)\n'
    '        for k in 1:m, s in 1:S\n'
    '            @constraint(M, sum(a * y[j, s] for (j, a) in C.rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0)\n'
    '                           - sum(e.g[s] * xb[e.i] for e in C.ents_of_row[k]; init=0.0) <= nd.u[k, s])\n'
    '        end\n'
    '        CF = -(dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0 && d[s] != 0; init=AffExpr(0.0)))\n'
    '        CL = sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0; init=AffExpr(0.0))\n'
    '        CLp = sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0 && d[s] > 0; init=AffExpr(0.0))\n'
    '        set_optimizer_attribute(M, "MIPGap", 1e-4); set_optimizer_attribute(M, "TimeLimit", 30.0)\n'
    '        (M, xb, CL, CF, CLp)\n'
    '    end\n'
    'end\n'
    '"""x 고정 블록 LP 의 쌍대 pd (블록 값과 같은 LP). 닫힌 거점 행 (u = 0, 그 행의 x 가 모두 0) 은 쌍대 비용이 0 이라 첫 LP 가 상한 U 에\n'
    '붙은 값을 고른다 (2026-11-26 진단: pd/U = 1.00, 그러면 x 단계가 새 거점을 못 엶). 그래서 블록 값을 유지한 채 닫힌 행 쌍대 합을 최소로 (사전식 두 번째 LP)."""\n'
    'function padm_dual(b, z)\n'
    '    P, fixc, pd, tP = get!(pdb, b) do\n'
    '        B = blks[b]; P = Model(lp_opt); xv = @variable(P, [1:nx]); tP = @variable(P)\n'
    '        fixc = @constraint(P, [i=1:nx], xv[i] == 0.0)\n'
    '        bk = _bccg_add_block!(merge(C, (M = P, x = xv, t0 = tP)), B.d, B.r, B.μ); @objective(P, Min, tP)\n'
    '        (P, fixc, bk.pd, tP)\n'
    '    end\n'
    '    for i in 1:nx; set_normalized_rhs(fixc[i], z[i]); end\n'
    '    optimize!(P); n_pdlp[] += 1\n'
    '    termination_status(P) == MOI.OPTIMAL || error("ccgbp v17: PADM 쌍대 LP $(termination_status(P))")\n'
    '    closed = [k for k in 1:m if !isempty(C.ents_of_row[k]) && all(z[e.i] < 0.5 for e in C.ents_of_row[k])]\n'
    '    isempty(closed) && return value.(pd)\n'
    '    tstar = objective_value(P); cc = @constraint(P, tP <= tstar + 1e-7 * max(1.0, abs(tstar)))\n'
    '    @objective(P, Min, sum(pd[k, s] for k in closed, s in 1:S)); optimize!(P); n_pdlp[] += 1\n'
    '    termination_status(P) == MOI.OPTIMAL || error("ccgbp v17: PADM 쌍대 최소화 LP $(termination_status(P))")\n'
    '    out = value.(pd); delete(P, cc); @objective(P, Min, tP)\n'
    '    return out\n'
    'end\n'
    '"p̂ 고정 x 단계. 반환 = 점 번호 (실행불가·해 없음이면 0)"\n'
    'function padm_xstep(b, w, π, ρ, p̂, N)\n'
    '    M, xb, CL, CF, CLp = padm_xmodel(b)\n'
    '    fx = Int[]\n'
    '    for i in N.f0; is_fixed(xb[i]) || (fix(xb[i], 0.0; force=true); push!(fx, i)); end\n'
    '    for i in N.f1; is_fixed(xb[i]) || (fix(xb[i], 1.0; force=true); push!(fx, i)); end\n'
    '    oc = [@constraint(M, sum(xb[i] for i in sitevars(j)) >= 1) for j in N.opn]\n'
    '    coef = zeros(nx); for e in C.XE, s in 1:S; coef[e.i] += e.g[s] * p̂[e.k, s]; end\n'
    '    @objective(M, Min, w[b] * (CL + ρ * (CF - ETA * CLp)) + sum((w[b] * ρ * coef[i] - π[b][i]) * xb[i] for i in al))\n'
    '    t_pdmip[] += @elapsed optimize!(M); n_pdmip[] += 1\n'
    '    st = termination_status(M)\n'
    '    st in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.INFEASIBLE) || error("ccgbp v17: PADM x 단계 MILP $(st)")\n'
    '    z = has_values(M) ? round.(value.(xb)) : nothing\n'
    '    for i in fx; unfix(xb[i]); end\n'
    '    for c in oc; delete(M, c); end\n'
    '    (st == MOI.INFEASIBLE || z === nothing) && return 0\n'
    '    return pid(z)\n'
    'end\n'
    '"tabu 도 실패했을 때의 PADM. 반환 = 넣은 열 수"\n'
    'function padm_round!(N, w, π, σ, supp, tolabs)\n'
    '    n_pdr[] += 1; added = 0; nmip = 0\n'
    '    t_pd[] += @elapsed begin\n'
    '        props = Dict{Int,Set{Int}}()\n'
    '        for b in eachindex(blks)\n'
    '            (w[b] > 1e-9 && blks[b].kind == :lp) || continue\n'
    '            st = sort(collect(supp[b]); by=c -> -get(lamval, (b, c), 0.0))\n'
    '            for c0 in st[1:min(PD_START, length(st))]\n'
    '                c = c0; p̂ = padm_dual(b, pts[c])\n'
    '                for ρ in PD_RHO\n'
    '                    nmip >= PD_MAXMIP && break\n'
    '                    for it in 1:PD_INNER\n'
    '                        nmip >= PD_MAXMIP && break\n'
    '                        c2 = padm_xstep(b, w, π, ρ, p̂, N); nmip += 1\n'
    '                        c2 == 0 && (n_pdnone[] += 1); c2 == c && (n_pdstay[] += 1)\n'
    '                        (c2 == 0 || c2 == c) && break             # 이 벌점에서 부분 최소점 → 다음 벌점\n'
    '                        push!(get!(props, b, Set{Int}()), c2); c = c2; p̂ = padm_dual(b, pts[c])\n'
    '                    end\n'
    '                end\n'
    '            end\n'
    '        end\n'
    '        n_pdprop[] += sum(length, values(props); init=0)\n'
    '        evalpairs!([(c, b) for (b, cs) in props for c in cs])\n'
    '        for (b, cs) in props\n'
    '            cand = [(c, rcv(b, c, w, π, σ)) for c in cs if consistent(c, N) && !(c in cols[b])]\n'
    '            cand = sort(filter(t -> t[2] < -tolabs, cand); by=last); n_pdneg[] += length(cand)\n'
    '            for (c, _) in cand[1:min(TB_CMAX, length(cand))]\n'
    '                dd = isempty(supp[b]) ? -1 : minimum(hamd(c, sc) for sc in supp[b]); pddist[dd] = get(pddist, dd, 0) + 1\n'
    '                push!(cols[b], c); added += 1\n'
    '            end\n'
    '        end\n'
    '    end\n'
    '    n_pdcol[] += added; added > 0 && (n_pdok[] += 1)\n'
    '    return added\n'
    'end\n')
rep('        TABU && tabu_round!(N, w, π, σ, supp, tolabs) > 0 && continue   # v17: 한 걸음 생성기 실패 → tabu 걸음 (DLH 순서, 찾으면 RMP 로)\n',
    '        TABU && tabu_round!(N, w, π, σ, supp, tolabs) > 0 && continue   # v17: 한 걸음 생성기 실패 → tabu 걸음 (DLH 순서, 찾으면 RMP 로)\n'
    '        PADM && (PD_WHEN == "all" || exact) && padm_round!(N, w, π, σ, supp, tolabs) > 0 && continue   # v18: tabu 실패 → PADM (찾으면 RMP 로)\n')
rep('        TABU ? "켬" : "끔", n_tbr[], n_tbok[], n_tbcol[], n_tblp[], t_tb[], sum(values(mipdist); init=0), join(["$k:$(mipdist[k])" for k in sort(collect(keys(mipdist)))], " "))',
    '        TABU ? "켬" : "끔", n_tbr[], n_tbok[], n_tbcol[], n_tblp[], t_tb[], sum(values(mipdist); init=0), join(["$k:$(mipdist[k])" for k in sort(collect(keys(mipdist)))], " "))\n'
    '@printf("  v18: PADM %s (%s), 라운드 %d (열 찾음 %d), x 단계 MILP %d 번 %.0fs (제자리 %d, 해 없음 %d), 쌍대 LP %d 번, 제안 %d (감축 비용 음수 %d), 넣은 열 %d, 전체 %.0fs; 넣은 열의 지지점 최소 해밍 거리 %s\\n",\n'
    '        PADM ? "켬" : "끔", PD_WHEN, n_pdr[], n_pdok[], n_pdmip[], t_pdmip[], n_pdstay[], n_pdnone[], n_pdlp[], n_pdprop[], n_pdneg[], n_pdcol[], t_pd[], join(["$k:$(pddist[k])" for k in sort(collect(keys(pddist)))], " "))')
s = s.replace('ccgbp v17:', 'ccgbp v18:')
assert 'ccgbp v17:' not in s
open(os.path.join(here, 'ccgbp_v18.jl'), 'w', encoding='utf-8').write(s)
print('ok')
