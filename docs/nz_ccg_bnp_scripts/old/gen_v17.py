# -*- coding: utf-8 -*-
# gen_v17.py — ccgbp_v16.jl 에서 ccgbp_v17.jl 을 만든다 (2026-11-26 시험판, md 11.18, 사용자 "tabu 서치로 좀 해볼래?").
#  v16 + tabu 가격 생성기 (Desaulniers, Lessard, Hadjar 2008, Transp. Sci. 42(3) §4.3–4.4 식):
#   지금의 한 걸음 생성기 (지지점 이웃 표본 + 캐시) 가 음의 감축 비용 열을 못 찾을 때만 tabu 걸음을 돌린다 (빠른 생성기부터, 열을 찾으면 RMP 로).
#   시작점 = 블록마다 λ 가 큰 지지점 TB_START 개 (DLH 의 기저 열, 감축 비용 ≈ 0). 걸음마다 노드와 맞고 tabu 가 아닌 한 걸음 이웃 중 TB_NS 개를
#   x 고정 블록 LP 로 재고 (모든 걸음을 묶어 병렬), 감축 비용이 가장 낮은 이웃으로 옮김 (나빠져도). tabu 목록 = 최근 지난 점 TB_LEN 개 (DLH 는 역이동).
#   시작점당 TB_IT 걸음. 지나며 본 음의 감축 비용 점을 블록당 TB_CMAX 개까지 열로 (DLH 의 C_max). 정확 가격 MILP·하한은 v16 그대로.
#  진단: 정확 가격 MILP 가 넣은 열과 그 블록 지지점의 최소 해밍 거리 분포 (한 걸음 = 1 또는 2).
import os
here = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(here, 'ccgbp_v16.jl'), encoding='utf-8').read()

def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:90]!r}')
    s = s.replace(a, b)

head = ('# ccgbp_v17.jl — gen_v17.py 가 v16 에서 만든 파일 (손으로 고치지 말 것). v16 + tabu 가격 생성기 (2026-11-26 시험판).\n'
        '#  docs/nz_ccg_bnp_2026-11-23.md 11.18. 한 걸음 생성기가 열을 못 찾을 때만 지지점에서 tabu 걸음 (DLH 2008 §4.3–4.4).\n'
        '# (아래는 v16 머리말) ')
s = head + s
# 상수 (머리 출력보다 앞)
rep('const PH1_SHARE = parse(Float64, envf("PH1_SHARE", "0.3"))',
    'const PH1_SHARE = parse(Float64, envf("PH1_SHARE", "0.3"))',)
rep('0 < PH1_SHARE <= 1 || error("ccgbp v16: PH1_SHARE 는 (0, 1]")',
    '0 < PH1_SHARE <= 1 || error("ccgbp v16: PH1_SHARE 는 (0, 1]")\n'
    'const TABU = envf("TABU", "1") == "1"                    # v17: tabu 가격 생성기 켬/끔\n'
    'const TB_START = parse(Int, envf("TB_START", "3"))       # v17: 블록당 시작점 수 (λ 큰 지지점부터)\n'
    'const TB_IT = parse(Int, envf("TB_IT", "5"))             # v17: 시작점당 걸음 수 (DLH 의 I_max)\n'
    'const TB_NS = parse(Int, envf("TB_NS", "10"))            # v17: 걸음마다 재는 이웃 수\n'
    'const TB_LEN = parse(Int, envf("TB_LEN", "5"))           # v17: tabu 목록 길이 (최근 지난 점 재방문 금지)\n'
    'const TB_CMAX = parse(Int, envf("TB_CMAX", "10"))        # v17: tabu 한 번에 블록당 넣는 열 상한 (DLH 의 C_max)\n'
    '(TB_START >= 1 && TB_IT >= 1 && TB_NS >= 1 && TB_LEN >= 1 && TB_CMAX >= 1) || error("ccgbp v16: TB_* 는 1 이상")')
rep('@printf("--- ccgbp v16 (v15 + ',
    '@printf("--- ccgbp v17 (v16 + tabu 가격 %s: 시작 %d·걸음 %d·이웃 %d·tabu 길이 %d·열 상한 %d) | ccgbp v16 (v15 + ')
rep('        PH1_SHARE, PH1_SHARE * TL, isempty(ABB_PATCH)',
    '        TABU ? "켬" : "끔", TB_START, TB_IT, TB_NS, TB_LEN, TB_CMAX, PH1_SHARE, PH1_SHARE * TL, isempty(ABB_PATCH)')
# RMP 의 λ 값 (시작점 고르기)
rep('cols = Vector{Set{Int}}(); n_rmpfix = Ref(0)',
    'cols = Vector{Set{Int}}(); n_rmpfix = Ref(0); lamval = Dict{Tuple{Int,Int},Float64}()   # v17: 마지막 RMP 의 λ')
rep('    supp = [Set(c for c in cN[b] if value(lam[(b, c)]) > 1e-9) for b in 1:nb]',
    '    supp = [Set(c for c in cN[b] if value(lam[(b, c)]) > 1e-9) for b in 1:nb]\n'
    '    global lamval = Dict(k => value(v) for (k, v) in lam)   # v17')
# tabu 생성기
rep('rng_ns = MersenneTwister(7)\n',
    'rng_ns = MersenneTwister(7)\n'
    '# ---------------- v17: tabu 가격 생성기 (DLH 2008 §4.3–4.4) ----------------\n'
    'rng_tb = MersenneTwister(11)\n'
    'n_tbr = Ref(0); n_tbok = Ref(0); n_tbcol = Ref(0); n_tblp = Ref(0); t_tb = Ref(0.0)\n'
    'hamd(a, c) = count(i -> (pts[a][i] > 0.5) != (pts[c][i] > 0.5), eachindex(pts[a]))\n'
    'mipdist = Dict{Int,Int}()   # 진단: 정확 가격 MILP 가 넣은 열과 그 블록 지지점의 최소 해밍 거리 → 개수\n'
    '"한 걸음 생성기가 열을 못 찾았을 때의 tabu 걸음. 반환 = 넣은 열 수"\n'
    'function tabu_round!(N, w, π, σ, supp, tolabs)\n'
    '    n_tbr[] += 1; lp0 = n_lp[]; added = 0\n'
    '    t_tb[] += @elapsed begin\n'
    '        walks = Tuple{Int,Int,Vector{Int}}[]           # (블록, 현재 점, tabu 목록)\n'
    '        for b in eachindex(blks)\n'
    '            (w[b] > 1e-9 && blks[b].kind == :lp) || continue\n'
    '            st = sort(collect(supp[b]); by=c -> -get(lamval, (b, c), 0.0))\n'
    '            for c in st[1:min(TB_START, length(st))]; push!(walks, (b, c, [c])); end\n'
    '        end\n'
    '        found = Dict{Int,Dict{Int,Float64}}()            # 블록 → (점 → 감축 비용 < −tolabs)\n'
    '        for it in 1:TB_IT\n'
    '            isempty(walks) && break\n'
    '            samp = Vector{Vector{Int}}(undef, length(walks))\n'
    '            for k in eachindex(walks)\n'
    '                b, c, tl = walks[k]\n'
    '                nb_ = filter(c2 -> consistent(c2, N) && !(c2 in tl), nbrs(c))\n'
    '                samp[k] = length(nb_) <= TB_NS ? nb_ : shuffle(rng_tb, nb_)[1:TB_NS]\n'
    '            end\n'
    '            evalpairs!([(c2, walks[k][1]) for k in eachindex(walks) for c2 in samp[k]])\n'
    '            nw = Tuple{Int,Int,Vector{Int}}[]\n'
    '            for k in eachindex(walks)\n'
    '                b, c, tl = walks[k]; isempty(samp[k]) && continue\n'
    '                rs = [rcv(b, c2, w, π, σ) for c2 in samp[k]]\n'
    '                for (c2, r) in zip(samp[k], rs)\n'
    '                    (r < -tolabs && !(c2 in cols[b])) && (get!(found, b, Dict{Int,Float64}())[c2] = r)\n'
    '                end\n'
    '                cn = samp[k][argmin(rs)]                  # 나빠져도 가장 좋은 이웃으로\n'
    '                push!(tl, cn); length(tl) > TB_LEN && popfirst!(tl)\n'
    '                push!(nw, (b, cn, tl))\n'
    '            end\n'
    '            walks = nw\n'
    '            all(k -> length(get(found, walks[k][1], Dict{Int,Float64}())) >= TB_CMAX, eachindex(walks)) && break   # 모든 블록이 열 상한을 채움\n'
    '        end\n'
    '        for (b, fd) in found\n'
    '            for (c2, _) in sort(collect(fd); by=last)[1:min(TB_CMAX, length(fd))]; push!(cols[b], c2); added += 1; end\n'
    '        end\n'
    '    end\n'
    '    n_tblp[] += n_lp[] - lp0; n_tbcol[] += added; added > 0 && (n_tbok[] += 1)\n'
    '    return added\n'
    'end\n')
rep('        added > 0 && continue\n',
    '        added > 0 && continue\n'
    '        TABU && tabu_round!(N, w, π, σ, supp, tolabs) > 0 && continue   # v17: 한 걸음 생성기 실패 → tabu 걸음 (DLH 순서, 찾으면 RMP 로)\n')
rep('                        if consistent(zc, N) && rcv(b, zc, w, π, σ) < -tolabs && !(zc in cols[b]); push!(cols[b], zc); added2 += 1; end',
    '                        if consistent(zc, N) && rcv(b, zc, w, π, σ) < -tolabs && !(zc in cols[b])\n'
    '                            dd = isempty(supp[b]) ? -1 : minimum(hamd(zc, sc) for sc in supp[b]); mipdist[dd] = get(mipdist, dd, 0) + 1   # v17 진단\n'
    '                            push!(cols[b], zc); added2 += 1\n'
    '                        end')
rep('@printf("  열: RMP 에 들어간 열 (블록 합) %d, 서로 다른 열 %d, 번호를 붙인 점 %d / |X| %d\\n", sum(length.(cols)), length(union(cols...)), length(pts), NP)',
    '@printf("  열: RMP 에 들어간 열 (블록 합) %d, 서로 다른 열 %d, 번호를 붙인 점 %d / |X| %d\\n", sum(length.(cols)), length(union(cols...)), length(pts), NP)\n'
    '@printf("  v17: tabu %s, 라운드 %d (열 찾음 %d), 넣은 열 %d, LP %d 번, %.0fs; 정확 MILP 가 넣은 열 %d 개의 지지점 최소 해밍 거리 %s\\n",\n'
    '        TABU ? "켬" : "끔", n_tbr[], n_tbok[], n_tbcol[], n_tblp[], t_tb[], sum(values(mipdist); init=0), join(["$k:$(mipdist[k])" for k in sort(collect(keys(mipdist)))], " "))')
s = s.replace('ccgbp v16:', 'ccgbp v17:')
assert 'ccgbp v16:' not in s
open(os.path.join(here, 'ccgbp_v17.jl'), 'w', encoding='utf-8').write(s)
print('ok')
