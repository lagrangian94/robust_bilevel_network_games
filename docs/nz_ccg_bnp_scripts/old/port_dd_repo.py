# (old/ 로 옮긴 기록, 2026-11-28) 저장소 반영에 쓴 일회용 스크립트. 반영 뒤에는 저장소 파일이 바뀌어 다시 돌면 assert 에서 멈춤 (의도).
# port_dd.py (2026-11-28, 사용자 승인 "ㅇㅇㅇㅇ"): abb_dd_patch.jl 의 선별 DD (+ 강쌍대 cut) 를 저장소 nonzero_sum/nz_alpha_bnb_dr.jl 로 옮긴다.
#   생성기 gen_abb_dd_patch.py 의 치환을 그대로 쓰고 (exec), 쓰인 경로 (dd_mode = :screen, dd_inc = :viol1, dd_gap = NaN) 만 남긴다.
#   뺀 경로: dd_mode = :full, dd_inc = :off (_abb_dd_screen!)·:ratio·:viol1s, dd_gap. 이름 _abb_ → _nz_, _ABB_ → _NZ_, 환경 변수 ABB_SD_* → NZ_SD_*.
#   dd_sdcut 은 dd = true 이고 range_lp 일 때만 붙음 (기본 켬). dd 기본은 끔 그대로.
import os, re, sys
R = 'C:/Users/user/robust_bilevel_network_games'
G = R + '/docs/nz_ccg_bnp_scripts/gen_abb_dd_patch.py'
SRC = R + '/nonzero_sum/nz_alpha_bnb_dr.jl'
gtxt = open(G, encoding='utf-8').read()
cut = gtxt.index("hdr = ('# abb_dd_patch.jl")
g = {'__file__': G, '__name__': 'gen'}
exec(compile(gtxt[:cut], G, 'exec'), g)
L, start, end, body, helper = g['L'], g['start'], g['end'], g['body'], g['helper']
assert '\n'.join(L) == open(SRC, encoding='utf-8').read()

def sub(s, a, b, cnt=1):
    n = s.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:100]!r}')
    return s.replace(a, b)

# ---------- 본문 ----------
# (a) 인자
i0 = body.index('                         resume=nothing,\n                         # 패치: dd = true')
i1 = body.index('dd_sdcut::Bool=false)') + len('dd_sdcut::Bool=false)')
body = body[:i0] + '''                         resume=nothing,
                         # 선별 DD (dd = true 이고 range_lp = true 일 때, 2026-11-25 ~ 11-28, 문서 docs/nz_ccg_bnp_2026-11-23.md 11.13 ~ 11.33).
                         #   노드 LP 로 가지치기 못 한 노드에서 두 꼭짓점의 최적 쌍대 면을 보고 follower 쌍대 차이가 모인 (쌍, 시나리오) 를 고른 뒤,
                         #   지금 노드 LP 해가 그 꼭짓점 구간에서 벗어난 양이 큰 후보부터 dd_batch 개씩 KKT-MIP 로 차이 범위를 인증하고 다시 푼다
                         #   (가지치기되거나 자를 후보가 없으면 멈춤, 남은 후보는 덜 조일 뿐). 유효성은 범위 MIP 판 DD 와 같은 논증.
                         #   dd_maxw: 활성 좌표의 최대 상자 폭이 이 값 이하인 노드만. dd_mip_tl: MIP 하나의 시간 상한 (넘으면 objective_bound)
                         #   dd_sigbd: KKT-MIP 에 상자의 꼭짓점 범위를 H 행 쌍대 경계로 넣음 (실행 불가능이면 경계를 빼고 재확인한 뒤에만 상자를 비움)
                         #   dd_adapt: DD 를 받은 노드가 dd_warm 개를 넘었는데 상한이 내려간 비율이 dd_minrate 미만이면 dd_probe 번째 노드마다만
                         #   dd_objstop: 노드 LP 해를 자를 수 없다고 판명되면 MIP 를 멈춤 (BestObjStop). dd_fcache: 꼭짓점 면 LP 결과를 자식에 물려줌
                         #   dd_sdcut: KKT-MIP 에 강쌍대 b(α)ᵀσ ≤ cᵀy 의 McCormick 선형화 (KKT 점을 자르지 않음, 계수는 풀이마다 지금 경계로)
                         dd_maxw::Float64=25.0, dd_mip_tl::Float64=2.0, dd_sigbd::Bool=true, dd_adapt::Bool=true, dd_warm::Int=200,
                         dd_minrate::Float64=0.25, dd_probe::Int=10, dd_batch::Int=2, dd_objstop::Bool=true, dd_fcache::Bool=true,
                         dd_sdcut::Bool=true)''' + body[i1:]
# (b) 강쌍대 행은 dd·range_lp 일 때만
body = sub(body, '    dd_sdcut && dual_range && _abb_sd_attach_all!(DRs, nd, x̄)            # 패치 (dd_sdcut)\n',
           '    config.dd && range_lp && dd_sdcut && dual_range && _abb_sd_attach_all!(DRs, nd, x̄)   # 선별 DD 의 KKT-MIP 에 강쌍대 행 (dd_sdcut)\n')
# (c) range_lp + dd 허용
body = sub(body, '    range_lp &&(isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&   # 패치: dd 허용\n'
                 '        error("nz_alpha_bnb_dr: range_lp = true 는 dual_range = true, dd = false 와 함께 쓰고 range_mip·range_time·range_nodes 는 기본값으로")',
                 '    range_lp &&(isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&   # dd = true 는 선별 DD (2026-11-28)\n'
                 '        error("nz_alpha_bnb_dr: range_lp = true 는 dual_range = true 와 함께 쓰고 range_mip·range_time·range_nodes 는 기본값으로")')
body = sub(body, '#   값 MIP (φ̄) 와 차이 범위 (dd) 는 풀지 않음 → dd = false 와 함께, range_time·range_nodes 는 범위 MIP 용이라 같이 쓰지 않음',
                 '#   값 MIP (φ̄) 는 풀지 않음. dd = true 면 노드 LP 뒤 선별 DD (아래 dd_* 인자). range_time·range_nodes 는 범위 MIP 용이라 같이 쓰지 않음')
# (d) :full 블록 제거
i0 = body.index('                    if config.dd && dd_mode == :full')
i1 = body.index('                    end\n', i0) + len('                    end\n')
body = body[:i0] + body[i1:]
# (e) 선별 블록을 :viol1 경로만으로
i0 = body.index('                    dd_empty = false')
mk = '                            z = z2\n                        end\n                    end\n'
i1 = body.index(mk, i0) + len(mk)
body = body[:i0] + '''                    dd_empty = false                          # 선별 DD: 가지치기 못 한 노드에서 → 상자가 비었거나 조여지면 다시 풂
                    if config.dd && range_lp && !pre_inf && ubp != Inf && isfinite(z) && z > LB[] + tol(LB[]) &&
                       maximum((u[i] - l[i] for i in 1:nh if nd.hU[i] > 1e-9); init=0.0) <= dd_maxw &&
                       _abb_dd_go(dd_adapt, n_ddelig, n_ddnode, n_ddcut, n_ddskip, dd_warm, dd_minrate, dd_probe)
                        # 후보를 dd_batch 개씩 인증 → 조여지면 다시 풂 → 가지치기 (또는 비움) 되면 멈춤. 남은 후보는 풀지 않음 (덜 조일 뿐)
                        Threads.atomic_add!(n_ddnode, 1)
                        if dd_fcache                                           # 부모와 같은 꼭짓점의 면 LP 결과를 복사해 씀 (자식끼리 공유하지 않게 복사)
                            ddfl = (get(ext, :sl, false) && get(ext, :fl, nothing) !== nothing) ? deepcopy(ext.fl) : Dict{Int,Dict{Int,Tuple{Float64,Float64}}}()
                            ddfu = (get(ext, :su, false) && get(ext, :fu, nothing) !== nothing) ? deepcopy(ext.fu) : Dict{Int,Dict{Int,Tuple{Float64,Float64}}}()
                        end
                        tdd = @elapsed (cands = _abb_dd_cands(CLs[wid], DR, nd, x̄, l, u, lo, hi, dlo, dhi; deadline=t_end, fl=ddfl, fu=ddfu))
                        dlo = copy(dlo); dhi = copy(dhi); z0dd = z; nmdd = 0; kdd = 0
                        while kdd < length(cands) && !dd_empty && isfinite(z) && z > LB[] + tol(LB[]) && time() < t_end
                            rest = cands[(kdd+1):end]; sv = [_abb_dd_lpviol(P, DR, c[2], c[3], c[4], c[5]) for c in rest]   # 지금 LP 해 기준으로 다시 줄 세움
                            od = sortperm([x[1] for x in sv]; rev=true); cands = vcat(cands[1:kdd], rest[od]); sv = sv[od]
                            sv[1][1] <= 1e-6 && break                          # 지금 LP 해를 자를 후보 없음
                            bt = cands[(kdd+1):min(kdd + dd_batch, end)]; kdd += length(bt); tgt = false
                            tdd += @elapsed for c in bt
                                _, p_, s_, c1_, c2_ = c
                                _, sd_, δ1_ = _abb_dd_lpviol(P, DR, p_, s_, c1_, c2_); sd_ == 0 && continue   # 벗어난 쪽만 인증
                                rr_, nm_ = _abb_dd_mip!(DR, nd, l, u, lo, hi, dlo, dhi, p_, s_, sd_ > 0; mip_tl=dd_mip_tl, deadline=t_end, sigbd=dd_sigbd,
                                                        objstop=(dd_objstop ? δ1_ : NaN))
                                nmdd += nm_; rr_ === :empty && (dd_empty = true; break); rr_ === :tight && (tgt = true)
                            end
                            if dd_empty
                                z = -Inf; Threads.atomic_add!(n_ddempty, 1)
                            elseif tgt
                                _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                                tr += @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=child_rounds, maxadd=sep_maxadd)))
                            end
                        end
                        Threads.atomic_add!(t_range, tdd); tn_rg += tdd; Threads.atomic_add!(n_ddmip, nmdd)
                        (!isnan(z) && z < z0dd) && Threads.atomic_add!(n_ddcut, 1)
                        (!isnan(z) && z <= LB[] + tol(LB[])) && Threads.atomic_add!(_ABB_DD_T.nprune, 1)
                    end
''' + body[i1:]
body = sub(body, '                    if z == -Inf && dual_range && !dd_empty      # 패치: DD MIP 가 상자를 비웠으면 (KKT-MIP 실행 불가능) 재확인할 LP 가 없음',
                 '                    if z == -Inf && dual_range && !dd_empty      # 선별 DD 가 상자를 비웠으면 (KKT-MIP 실행 불가능) 재확인할 LP 가 없음')
for a in ('dd_mode', 'dd_inc', 'dd_gap', ':viol1s', ':ratio', '_abb_dd_screen!', 'only_s'):
    if a in body:
        raise SystemExit(f'본문에 남음: {a}')

# (f) 생성기의 :gc_wait 치환이 주석 안까지 바꾼 줄은 원래대로
orig_gc = [l for l in L if 'gcw0 = Base.gc_num().total_time_to_safepoint' in l]; assert len(orig_gc) == 1
bl = body.split(chr(10)); ix = [i for i, l in enumerate(bl) if 'gcw0 = Base.gc_num().total_time_to_safepoint' in l]; assert len(ix) == 1
bl[ix[0]] = orig_gc[0]; body = chr(10).join(bl)
assert body.count(':dd_screen =>') == body.count(':gc_wait =>'), (body.count(':dd_screen =>'), body.count(':gc_wait =>'))
# (g) n_cap 줄의 주석이 생성기가 붙인 줄 끝으로 밀린 것을 되돌림
orig_cap = [l for l in L if l.startswith('    n_cap = Threads.Atomic{Int}(0)')]; assert len(orig_cap) == 1
cap_cmt = orig_cap[0][len('    n_cap = Threads.Atomic{Int}(0)'):]
body = sub(body, '    n_cap = Threads.Atomic{Int}(0)\n', orig_cap[0] + '\n')
body = sub(body, '_ABB_SD_C.maxv[] = -Inf   # 패치 (dd_sdcut 대조)' + cap_cmt, '_ABB_SD_C.maxv[] = -Inf   # 패치 (dd_sdcut 대조)')

# ---------- 보조 함수 ----------
i0 = helper.index('"패치 (screen): 두 꼭짓점의 최적 쌍대 면에서')
i1 = helper.index('"패치 (dd_inc): 선별만')
helper = helper[:i0] + helper[i1:]
helper = sub(helper, '"패치 (dd_inc): 선별만 — _abb_dd_screen! 의 앞부분. 반환', '"선별 DD: 두 꼭짓점의 최적 쌍대 면에서 차이가 함의 폭의 절반 안에 모인 (쌍 p, 시나리오 s) 후보. 반환')
helper = sub(helper, '; deadline=Inf, fixtol=1e-4, only_s::Int=0, fl=nothing, fu=nothing)', '; deadline=Inf, fixtol=1e-4, fl=nothing, fu=nothing)')
helper = sub(helper, '    for s in (only_s > 0 ? (only_s:only_s) : (1:S))\n', '    for s in 1:S\n')
helper = sub(helper, 'objstop::Float64=NaN, gap::Float64=NaN)', 'objstop::Float64=NaN)')
helper = sub(helper, '    #   멈춰도 objective_bound 는 유효한 범위. gap (dd_gap): MIPGap. 둘 다 풀이 뒤 원래 값으로\n', '    #   멈춰도 objective_bound 는 유효한 범위. 풀이 뒤 원래 값으로\n')
helper = sub(helper, '    ob0 = isfinite(objstop) ? get_optimizer_attribute(M.model, "BestObjStop") : NaN; gp0 = isfinite(gap) ? get_optimizer_attribute(M.model, "MIPGap") : NaN\n',
                     '    ob0 = isfinite(objstop) ? get_optimizer_attribute(M.model, "BestObjStop") : NaN\n')
helper = sub(helper, '    isfinite(objstop) && set_optimizer_attribute(M.model, "BestObjStop", objstop); isfinite(gap) && set_optimizer_attribute(M.model, "MIPGap", gap)\n',
                     '    isfinite(objstop) && set_optimizer_attribute(M.model, "BestObjStop", objstop)\n')
helper = sub(helper, '    isfinite(objstop) && set_optimizer_attribute(M.model, "BestObjStop", ob0); isfinite(gap) && set_optimizer_attribute(M.model, "MIPGap", gp0)\n',
                     '    isfinite(objstop) && set_optimizer_attribute(M.model, "BestObjStop", ob0)\n')
# _nz_dr_opt! 재정의는 저장소 함수를 바꾸는 것으로
i0 = helper.index('"패치: 저장소 _nz_dr_opt! (nz_alpha_bnb_dr.jl) 와 같은 동작')
optfun = helper[helper.index('function _nz_dr_opt!', i0):]
helper = helper[:i0].rstrip('\n') + '\n'
assert optfun.rstrip().endswith('end')
helper = sub(helper, '"패치 (dd_inc): (쌍 p', '"선별 DD: (쌍 p')
helper = sub(helper, '"패치 (dd_inc = :viol1): 노드 LP', '"선별 DD: 노드 LP')
helper = helper.replace('패치 (', 'DD (').replace('# ---- DD (dd_sdcut', '# ---- 선별 DD (dd_sdcut')
helper = sub(helper, 'println(stderr, "[abb_dd_patch] dd_sdcut: KKT-MIP 에', 'println(stderr, "nz_alpha_bnb_dr: dd_sdcut — KKT-MIP 에')
helper = sub(helper, 'println(stderr, "[abb_dd_patch] dd_sdcut 대조: "', 'println(stderr, "nz_alpha_bnb_dr: dd_sdcut 대조: "')
for a in ('gap', 'only_s', '_abb_dd_screen!', '패치', 'abb_dd_patch'):
    if a in helper:
        raise SystemExit(f'보조 함수에 남음: {a}: ' + helper[max(0, helper.index(a) - 80):helper.index(a) + 40])

# ---------- 조립 ----------
pre = '\n'.join(L[:start]); post = '\n'.join(L[end + 1:])
old_opt = pre[pre.index('function _nz_dr_opt!(DR::_NZDR, M::_NZRangeMIP, l, u, what)\n'):]
old_opt = old_opt[:old_opt.index('\n    return false\nend\n') + len('\n    return false\nend\n')]
pre = sub(pre, '실패 횟수는 DR.nfail 에 세어 결과 Dict 의 :range_fail 로 보고하고, 처음 3 번은 상자를 stderr 에 남긴다.\n"""\n',
          '실패 횟수는 DR.nfail 에 세어 결과 Dict 의 :range_fail 로 보고하고, 처음 3 번은 상자를 stderr 에 남긴다.\n'
          '강쌍대 행 (dd_sdcut, M.model.ext[:nz_sd]) 이 있으면 풀기 전에 McCormick 계수를 그 MIP 의 지금 경계로 다시 쓰고,\n'
          '환경 변수 NZ_SD_CHECK = N > 0 이면 N 번째마다 행 없는 판과 대조한다 (_nz_sd_nocut!·_nz_sd_compare!).\n"""\n')
pre = sub(pre, old_opt, optfun.rstrip('\n') + '\n\n' + helper.lstrip('\n'))
# 설명 표
pre = sub(pre, '| | `dd` | false | 쌍대 차이 범위 (DD, 범위 MIP 전용). 2026-10-13 기본은 true (노드당 범위 MIP 37 → 69 개, 노드 1/10). range_lp = true 와 함께 쓸 수 없음 |',
          '| | `dd` | false | 쌍대 차이 범위 (DD). range_lp = false 면 범위 MIP 안의 차이 범위 (2026-10-13 기본은 true, 노드당 범위 MIP 37 → 69 개, 노드 1/10), range_lp = true 면 노드 LP 뒤 선별 DD (아래 `dd_*`, 2026-11-28 저장소 반영) |')
pre = sub(pre, 'dd = false 필요. `dual_range = false` ablation 은', '`dual_range = false` ablation 은')
anchor = '| 이어 풀기 | `resume` | nothing |'
pre = sub(pre, anchor,
          '| 선별 DD (2026-11-28, `dd = true`·`range_lp` 전용) | `dd_maxw` | 25.0 | 활성 좌표의 최대 상자 폭이 이 값 이하인 노드에서만 선별 (넓은 상자에서는 차이가 모이지 않아 비쌈) |\n'
          '| | `dd_mip_tl`, `dd_batch` | 2.0, 2 | KKT-MIP 하나의 시간 상한 (넘으면 objective_bound), 한 번에 인증하는 후보 수 (그 뒤 노드 LP 를 다시 풂) |\n'
          '| | `dd_sigbd` | true | KKT-MIP 에 상자의 꼭짓점 범위를 H 행 쌍대 경계로 (MIP 평균 0.87 → 0.19 s). 실행 불가능이면 경계를 빼고 재확인 |\n'
          '| | `dd_adapt`, `dd_warm`, `dd_minrate`, `dd_probe` | true, 200, 0.25, 10 | DD 받은 노드 중 상한이 내려간 비율이 낮으면 선별 빈도를 1/dd_probe 로 |\n'
          '| | `dd_objstop` | true | 노드 LP 해를 자를 수 없다고 판명되면 KKT-MIP 를 멈춤 (BestObjStop) |\n'
          '| | `dd_fcache` | true | 꼭짓점 면 LP 결과를 자식에 물려줌 (자식은 부모와 꼭짓점 하나를 공유, 같은 LP) |\n'
          '| | `dd_sdcut` | true | KKT-MIP 에 강쌍대 McCormick 행 (MIP 하나 2.5 ~ 3.8 배, 문서 11.30·11.31). 환경 변수 NZ_SD_CHECK = N 이면 N 번째마다 행 없는 판과 대조, NZ_SD_TIGHT = 1 이면 허용오차 1e-9 |\n'
          + anchor)
new = pre + '\n' + body + '\n' + post
# 이름
new = new.replace('_abb_', '_nz_').replace('_ABB_', '_NZ_').replace(':abb_sd', ':nz_sd').replace('"ABB_SD_CHECK"', '"NZ_SD_CHECK"').replace('"ABB_SD_TIGHT"', '"NZ_SD_TIGHT"')
new = new.replace('ABB_SD_CHECK', 'NZ_SD_CHECK').replace('ABB_SD_TIGHT', 'NZ_SD_TIGHT')
new = new.replace('# 패치 (', '# DD (')
body2 = new  # 남은 '패치' 단어 확인
for m_ in re.finditer('패치', body2):
    ln = body2.count('\n', 0, m_.start()) + 1
    print('패치 남음 줄', ln, body2.split('\n')[ln - 1][:160])
for a in ('_abb_', '_ABB_', 'abb_sd', 'ABB_SD'):
    assert a not in new, a
if '--write' in sys.argv:
    open(SRC, 'w', encoding='utf-8', newline='\n').write(new)
    print('저장소 파일 씀', len(new.split('\n')), '줄')
else:
    open(os.path.join(os.path.dirname(__file__), 'nz_alpha_bnb_dr_ported.jl'), 'w', encoding='utf-8', newline='\n').write(new)
    print('미리보기 씀', len(new.split('\n')), '줄')
