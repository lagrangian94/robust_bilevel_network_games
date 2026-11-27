# gen_abb_dd_patch.py — 저장소 nonzero_sum/nz_alpha_bnb_dr.jl 의 nz_alpha_bnb_dr 함수만 떼어 "꼭짓점 LP 범위 + 쌍대 차이 범위 (DD)" 를
# 허용하도록 고친 재정의 파일 abb_dd_patch.jl 을 만든다 (2026-11-25 밤, 실험용. 저장소 파일은 고치지 않음).
#   기본 판은 range_lp = true 이면 dd = false 여야 함 (L6 에서 범위 MIP 가 노드 시간을 다 써서). 지리 5 전선 진단에서
#   상자 DD 가 상한 큰 상자를 모두 실행 불가능으로 만들었음 (front_dd.jl, 재확인 통과, incumbent 상자 유효성 통과). 루트 DD 한 번은 효과 없음.
#   dd_mode = :full   꼭짓점 범위 뒤 두 성분 모두 안 고정인 쌍 전부를 KKT-MIP 로 (_nz_dr_dranges!). 상자당 150 ~ 190 s.
#   dd_mode = :screen 노드 LP 로 가지치기 못 한 노드에서만, 두 꼭짓점의 최적 쌍대 면에서 차이가 모인 (쌍, s) 를 골라 KKT-MIP 로 인증하고 다시 풂
#                     (front_dd_lazy.jl SCREEN=corner: 상자 8 개 모두 가지치기, 상자당 약 18 s). MIP 하나 dd_mip_tl 초, 시간 상한이면 objective_bound.
#                     활성 좌표의 최대 폭이 dd_maxw (25) 이하인 노드만. 첫 판 (문턱 없음) 은 489 s 에 노드 170 개 (기본 판 5,032 개) 로 위쪽 넓은 상자에서 시간을 다 씀.
#   유효성: 범위 MIP 판의 DD 와 같은 논증 (상자 안 α 의 recourse 최적 쌍대의 차이 범위). 고른 쌍만 조이는 것은 덜 조일 뿐.
#   자식은 조인 dlo·dhi 를 물려받음 (자식 상자 ⊆ 부모 상자).
#   2026-11-26: dd_sigbd (기본 켬) — 선별 DD 의 KKT-MIP 에 상자의 꼭짓점 LP 범위 [lo, hi] 를 H 행 쌍대의 변수 경계로 넣음 (front_ddmip.jl:
#                MIP 평균 0.87 → 0.19 s, bound 는 같거나 0.01 이하로 더 느슨). 범위는 상자 안 α 의 모든 (정규형) 최적 쌍대를 담아 유효.
#                경계를 넣은 MIP 가 실행 불가능이면 경계를 빼고 다시 풀어 확인한 뒤에만 상자를 비움 (허용오차로 인한 거짓 가지치기 방지).
#   2026-11-27: dd_inc (기본 :ratio) — 선별 후보를 dd_batch 개씩 인증하고 다시 풀어 가지치기되면 멈춤 (front_dd_inc.jl: 상자 10 개 MIP 224 → 44, 시간 69 → 27 s). md 11.24.
#   2026-11-27: 재확인 MIP 결과를 경계를 되돌리기 전에 읽도록 고침 (전에는 버려짐, 유효성 무관). 재확인 수 rechk·rechk_inf 보고.
#   2026-11-27: 시간 내역 계수기 _ABB_DD_T (꼭짓점·면 LP 시간·수, MIP 시간, DD 로 다시 푼 뒤 가지친 노드) 를 :dd_screen 에 보고 (md 11.24).
#   2026-11-26: dd_adapt (기본 켬) — 선별 DD 의 성공률 (상한이 내려간 노드 / DD 받은 노드) 이 낮으면 선별 빈도를 1/dd_probe 로.
#                지리 5 막힌 점은 77% (DD 가 기본 판을 이김), 지리 6 막힌 점은 7% (선별 LP 비용으로 처리량 반) 였음 (md 11.14 (9)).
# 사용: python gen_abb_dd_patch.py  → abb_dd_patch.jl. 스크립트에서 nz_benders.jl 을 include 한 뒤 include("abb_dd_patch.jl"), dd = true 로 부름.
import os
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, '..', '..', 'nonzero_sum', 'nz_alpha_bnb_dr.jl')
L = open(src, encoding='utf-8').read().split('\n')
start = next(i for i, l in enumerate(L) if l.startswith('function nz_alpha_bnb_dr(nd::NZData, x̄;'))
end = max(i for i, l in enumerate(L) if l == 'end')
assert end > start
body = '\n'.join(L[start:end + 1])

def rep(a, b, cnt=1):
    global body
    n = body.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:90]!r}')
    body = body.replace(a, b)

# (0) 인자
rep('                         resume=nothing)',
    '                         resume=nothing,\n'
    '                         # 패치: dd = true 일 때 DD 를 구하는 방식 (:screen 기본 | :full), 선별 DD 의 MIP 하나 시간 상한\n'
    '                         #   dd_maxw: 선별 DD 는 활성 좌표의 최대 상자 폭이 이 값 이하인 노드에서만 (넓은 상자에서는 차이가 상수가 아니라 효과 없고 비쌈)\n'
    '                         #   dd_adapt: DD 를 받은 노드가 dd_warm 개를 넘었는데 상한이 내려간 비율이 dd_minrate 보다 낮으면 dd_probe 번째 노드마다만 선별\n'
    '                         dd_mode::Symbol=:screen, dd_mip_tl::Float64=2.0, dd_maxw::Float64=25.0, dd_sigbd::Bool=true,\n'
    '                         dd_adapt::Bool=true, dd_warm::Int=200, dd_minrate::Float64=0.25, dd_probe::Int=10,\n'
    '                         #   dd_inc (2026-11-27, md 11.24): 선별 후보를 나눠 인증하고 다시 풀어 가지치기되면 멈춤. :off (후보 전부, 예전) | :ratio (모임 비율 순, 양쪽 MIP)\n'
    '                         #     | :viol1 (지금 LP 해가 꼭짓점 구간에서 벗어난 양 순, 벗어난 쪽 MIP 만) | :viol1s (:viol1 을 시나리오 하나씩, 가지치기되면 선별도 멈춤). dd_batch = 한 번에 인증할 후보 수\n'
    '                         #   dd_objstop: :viol1·:viol1s 에서 노드 LP 해를 자를 수 없다고 판명되면 MIP 를 멈춤 (BestObjStop = δ̂), dd_gap: DD MIP 의 MIPGap (NaN = 그대로)\n'
    '                         dd_inc::Symbol=:ratio, dd_batch::Int=2, dd_objstop::Bool=false, dd_gap::Float64=NaN)')
# (1) range_lp 와 dd 함께 허용
rep('    range_lp &&(dd || isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&',
    '    range_lp &&(isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&   # 패치: dd 허용')
rep('    n_cap = Threads.Atomic{Int}(0)',
    '    n_cap = Threads.Atomic{Int}(0)\n'
    '    n_ddnode = Threads.Atomic{Int}(0); n_ddmip = Threads.Atomic{Int}(0); n_ddempty = Threads.Atomic{Int}(0); n_ddcut = Threads.Atomic{Int}(0)   # 패치 (screen)\n'
    '    n_ddelig = Threads.Atomic{Int}(0); n_ddskip = Threads.Atomic{Int}(0)   # 패치 (dd_adapt)\n'
    '    for c in (_ABB_DD_T.tlp, _ABB_DD_T.tmip); c[] = 0.0; end; for c in (_ABB_DD_T.nlp, _ABB_DD_T.nprune, _ABB_DD_T.nrechk, _ABB_DD_T.nrechk_inf, _ABB_DD_T.nobjstop); c[] = 0; end   # 패치 (시간 내역)')
# (2) :full — 꼭짓점 범위 뒤 DD 전부
rep('''                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                    fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                elseif dual_range && !(range_mip''',
    '''                    if config.dd && dd_mode == :full          # 패치 (full): 꼭짓점 범위 뒤 상자별 쌍대 차이 범위 (KKT-MIP 전부), 자식에 물려줌
                        wit = Dict{Tuple{Int,Int,Bool},Vector{Float64}}()
                        tg2 = @elapsed ((dlo, dhi) = _nz_dr_dranges!(DR, nd, l, u, lo, hi, ext.dlo, ext.dhi, witp, wit; deadline=t_end))
                        Threads.atomic_add!(t_range, tg2); tn_rg += tg2
                    end
                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                    fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                elseif dual_range && !(range_mip''')
# (3) :screen — 노드 LP 뒤 선별 DD
rep('''                    tr = pre_inf ? 0.0 : @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd)))
                    pre_inf && (z = -Inf)
''', '''                    tr = pre_inf ? 0.0 : @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd)))
                    pre_inf && (z = -Inf)
                    dd_empty = false                          # 패치 (screen): 가지치기 못 한 노드에서 선별 DD → 상자가 비었거나 조여지면 다시 풂
                    if config.dd && dd_mode == :screen && range_lp && !pre_inf && ubp != Inf && isfinite(z) && z > LB[] + tol(LB[]) &&
                       maximum((u[i] - l[i] for i in 1:nh if nd.hU[i] > 1e-9); init=0.0) <= dd_maxw &&
                       dd_inc !== :off && _abb_dd_go(dd_adapt, n_ddelig, n_ddnode, n_ddcut, n_ddskip, dd_warm, dd_minrate, dd_probe)
                        # 패치 (dd_inc): 후보를 dd_batch 개씩 인증 → 조여지면 다시 풂 → 가지치기 (또는 비움) 되면 멈춤. 남은 후보는 풀지 않음 (덜 조일 뿐)
                        Threads.atomic_add!(n_ddnode, 1)
                        # :viol1s 는 시나리오 하나씩 선별 (ζF 곱 위반이 큰 시나리오부터) → 그 시나리오 후보를 :viol1 처럼 인증 → 가지치기되면 남은 시나리오는 선별도 안 함
                        sord = Int[]
                        if dd_inc === :viol1s
                            vv_ = P.O.v; α̂d = value.(vv_[:α]); d̂d = value.(vv_[:d]); ζFd = value.(vv_[:ζF])
                            sord = sortperm([sum(abs(ζFd[i, s_] - α̂d[i] * d̂d[s_]) for i in 1:nh) for s_ in 1:nd.S]; rev=true)
                        end
                        tdd = @elapsed (cands = dd_inc === :viol1s ? Tuple{Float64,Int,Int,Float64,Float64}[] : _abb_dd_cands(CLs[wid], DR, nd, x̄, l, u, lo, hi, dlo, dhi; deadline=t_end))
                        dlo = copy(dlo); dhi = copy(dhi); z0dd = z; nmdd = 0; kdd = 0
                        for s_ in sord
                            (dd_empty || !isfinite(z) || z <= LB[] + tol(LB[]) || time() >= t_end) && break
                            tdd += @elapsed (cs = _abb_dd_cands(CLs[wid], DR, nd, x̄, l, u, lo, hi, dlo, dhi; deadline=t_end, only_s=s_))
                            while !isempty(cs) && !dd_empty && isfinite(z) && z > LB[] + tol(LB[]) && time() < t_end
                                sv = [_abb_dd_lpviol(P, DR, c[2], c[3], c[4], c[5]) for c in cs]
                                od = sortperm([x[1] for x in sv]; rev=true); cs = cs[od]; sv = sv[od]
                                sv[1][1] <= 1e-6 && break                              # 이 시나리오에는 지금 LP 해를 자를 후보 없음
                                bt = cs[1:min(dd_batch, end)]; cs = cs[(length(bt)+1):end]; tgt = false
                                tdd += @elapsed for (c, (_, sd_, δ_)) in zip(bt, sv)
                                    sd_ == 0 && continue
                                    rr_, nm_ = _abb_dd_mip!(DR, nd, l, u, lo, hi, dlo, dhi, c[2], c[3], sd_ > 0; mip_tl=dd_mip_tl, deadline=t_end, sigbd=dd_sigbd,
                                                            objstop=(dd_objstop ? δ_ : NaN), gap=dd_gap)
                                    nmdd += nm_; rr_ === :empty && (dd_empty = true; break); rr_ === :tight && (tgt = true)
                                end
                                if dd_empty
                                    z = -Inf; Threads.atomic_add!(n_ddempty, 1)
                                elseif tgt
                                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                                    tr += @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=child_rounds, maxadd=sep_maxadd)))
                                end
                            end
                        end
                        dd_inc === :ratio && sort!(cands; by=c -> c[1])
                        while kdd < length(cands) && !dd_empty && isfinite(z) && z > LB[] + tol(LB[]) && time() < t_end
                            if dd_inc === :viol1                                   # 지금 LP 해 기준으로 남은 후보를 다시 줄 세움
                                rest = cands[(kdd+1):end]; sv = [_abb_dd_lpviol(P, DR, c[2], c[3], c[4], c[5]) for c in rest]
                                od = sortperm([x[1] for x in sv]; rev=true); cands = vcat(cands[1:kdd], rest[od]); sv = sv[od]
                                sv[1][1] <= 1e-6 && break                          # 지금 LP 해를 자를 후보 없음
                            end
                            bt = cands[(kdd+1):min(kdd + dd_batch, end)]; kdd += length(bt); tgt = false
                            tdd += @elapsed for c in bt
                                _, p_, s_, c1_, c2_ = c
                                sides = (false, true)
                                δ1_ = NaN
                                if dd_inc === :viol1
                                    _, sd_, δ1_ = _abb_dd_lpviol(P, DR, p_, s_, c1_, c2_); sd_ == 0 && continue; sides = (sd_ > 0,)
                                end
                                for sd in sides
                                    rr_, nm_ = _abb_dd_mip!(DR, nd, l, u, lo, hi, dlo, dhi, p_, s_, sd; mip_tl=dd_mip_tl, deadline=t_end, sigbd=dd_sigbd,
                                                            objstop=(dd_objstop && dd_inc === :viol1 ? δ1_ : NaN), gap=dd_gap)
                                    nmdd += nm_; rr_ === :empty && (dd_empty = true; break); rr_ === :tight && (tgt = true)
                                end
                                dd_empty && break
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
                    elseif config.dd && dd_mode == :screen && range_lp && !pre_inf && ubp != Inf && isfinite(z) && z > LB[] + tol(LB[]) &&
                       maximum((u[i] - l[i] for i in 1:nh if nd.hU[i] > 1e-9); init=0.0) <= dd_maxw && dd_inc === :off &&
                       _abb_dd_go(dd_adapt, n_ddelig, n_ddnode, n_ddcut, n_ddskip, dd_warm, dd_minrate, dd_probe)
                        tdd = @elapsed (ddr = _abb_dd_screen!(CLs[wid], DR, nd, x̄, l, u, lo, hi, dlo, dhi; mip_tl=dd_mip_tl, deadline=t_end, sigbd=dd_sigbd))
                        Threads.atomic_add!(t_range, tdd); tn_rg += tdd; Threads.atomic_add!(n_ddnode, 1); Threads.atomic_add!(n_ddmip, ddr[3])
                        if ddr[1] === :empty
                            dd_empty = true; z = -Inf; Threads.atomic_add!(n_ddempty, 1)
                        elseif ddr[1] === :tight
                            dlo, dhi = ddr[2]
                            _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                            tr += @elapsed (z2 = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=child_rounds, maxadd=sep_maxadd)))
                            (!isnan(z2) && z2 < z) && Threads.atomic_add!(n_ddcut, 1)
                            (!isnan(z2) && z2 <= LB[] + tol(LB[])) && Threads.atomic_add!(_ABB_DD_T.nprune, 1)   # 패치 (시간 내역)
                            z = z2
                        end
                    end
''')
rep('''                    if z == -Inf && dual_range
                        fk = pre_inf''', '''                    if z == -Inf && dual_range && !dd_empty      # 패치: DD MIP 가 상자를 비웠으면 (KKT-MIP 실행 불가능) 재확인할 LP 가 없음
                        fk = pre_inf''')
rep(':gc_wait', ':dd_screen => (nodes=n_ddnode[], mips=n_ddmip[], empty=n_ddempty[], cut=n_ddcut[], skip=n_ddskip[], pruned_after=_ABB_DD_T.nprune[], t_lp=round(_ABB_DD_T.tlp[]; digits=1), n_lp=_ABB_DD_T.nlp[], t_mip=round(_ABB_DD_T.tmip[]; digits=1), rechk=_ABB_DD_T.nrechk[], rechk_inf=_ABB_DD_T.nrechk_inf[], objstop=_ABB_DD_T.nobjstop[]), :gc_wait', cnt=body.count(':gc_wait'))

helper = r'''
"패치 (시간 내역, 2026-11-27): 선별 DD 의 꼭짓점·면 LP 시간·수, MIP 시간, DD 로 다시 푼 뒤 가지친 노드. 호출마다 0 으로"
const _ABB_DD_T = (tlp=Threads.Atomic{Float64}(0.0), nlp=Threads.Atomic{Int}(0), tmip=Threads.Atomic{Float64}(0.0), nprune=Threads.Atomic{Int}(0),
                   nrechk=Threads.Atomic{Int}(0), nrechk_inf=Threads.Atomic{Int}(0), nobjstop=Threads.Atomic{Int}(0))   # 재확인 MIP 수 / 재확인에서도 실행 불가능
"패치 (screen): 두 꼭짓점의 최적 쌍대 면에서 차이가 모인 (쌍, s) 를 골라 KKT-MIP 로 차이 범위를 인증. 반환 (:none | :tight | :empty, (dlo, dhi), MIP 수)"
function _abb_dd_screen!(CL, DR, nd, x̄, l, u, lo, hi, dlo0, dhi0; mip_tl=2.0, deadline=Inf, fixtol=1e-4, sigbd::Bool=true)
    S, m, nh = nd.S, nz_m(nd), nz_nh(nd); dlo = copy(dlo0); dhi = copy(dhi0); nmip = 0; tight = false
    for s in 1:S
        time() > deadline && break
        ps = Int[]
        for (p, (a, b)) in enumerate(DR.pairs)
            (hi[a, s] - lo[a, s] > fixtol && hi[b, s] - lo[b, s] > fixtol) || continue
            wl = max(dlo[p, s], lo[a, s] - hi[b, s]); wh = min(dhi[p, s], hi[a, s] - lo[b, s])
            wh - wl > 1.0 && push!(ps, p)
        end
        isempty(ps) && continue
        C = CL[s]; ext = Dict{Int,Tuple{Float64,Float64}}()
        tlp0 = time()
        for αc in (l, u)
            rr = nz_rhs(nd, x̄, αc, s)
            for k in 1:m; nd.hrow[k] > 0 && set_normalized_coefficient(C.opt, C.σ[k], rr[k]); end
            set_normalized_rhs(C.opt, 1e20)
            @objective(C.model, Min, sum(rr[k] * C.σ[k] for k in 1:m if rr[k] != 0)); optimize!(C.model)
            termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_screen!: 꼭짓점 LP (s=$s) $(termination_status(C.model))")
            zc = objective_value(C.model); set_normalized_rhs(C.opt, zc + 1e-7 * max(1.0, abs(zc)))
            for p in ps
                a, b = DR.pairs[p]; e = C.σ[DR.Hk[a]] - C.σ[DR.Hk[b]]
                @objective(C.model, Min, e); optimize!(C.model)
                termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_screen!: 꼭짓점 차이 LP (s=$s) $(termination_status(C.model))")
                mn = objective_value(C.model)
                @objective(C.model, Max, e); optimize!(C.model)
                termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_screen!: 꼭짓점 차이 LP (s=$s) $(termination_status(C.model))")
                mx = objective_value(C.model)
                o = get(ext, p, (Inf, -Inf)); ext[p] = (min(o[1], mn), max(o[2], mx))
            end
        end
        Threads.atomic_add!(_ABB_DD_T.tlp, time() - tlp0); Threads.atomic_add!(_ABB_DD_T.nlp, 2 + 4 * length(ps))
        M = DR.mips[s]; boxset = false
        for p in ps
            a, b = DR.pairs[p]
            wl = max(dlo[p, s], lo[a, s] - hi[b, s]); wh = min(dhi[p, s], hi[a, s] - lo[b, s])
            ext[p][2] - ext[p][1] < 0.5 * (wh - wl) || continue          # 두 꼭짓점의 차이가 함의 폭의 절반 안에 모인 쌍만
            if !boxset
                for i in 1:nh; set_lower_bound(M.α[i], l[i]); set_upper_bound(M.α[i], u[i]); end
                sigbd && _abb_sigbd!(M, DR, nd, lo, hi, s)
                boxset = true
            end
            for ismax in (false, true)
                if time() > deadline
                    set_time_limit_sec(M.model, nothing); sigbd && _abb_sigbd!(M, DR, nd, nothing, nothing, s)
                    return (tight ? :tight : :none), (dlo, dhi), nmip
                end
                @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[DR.Hk[a]] - M.σ[DR.Hk[b]])
                set_time_limit_sec(M.model, max(0.01, min(mip_tl, deadline - time())))
                nmip += 1
                tm0 = time(); okm = _nz_dr_opt!(DR, M, l, u, "dd screen s=$s 쌍 ($(DR.Hk[a]), $(DR.Hk[b]))"); Threads.atomic_add!(_ABB_DD_T.tmip, time() - tm0)
                okm || continue
                st = termination_status(M.model)
                getbd(st) = st == MOI.OPTIMAL ? objective_bound(M.model) : (st == MOI.TIME_LIMIT ? _nz_dr_bound_or_nan(M.model) : NaN)
                bd = getbd(st)
                if sigbd && st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)   # 경계를 빼고 다시 풀어 확인 (허용오차 거짓 판정 방지)
                    _abb_sigbd!(M, DR, nd, nothing, nothing, s); nmip += 1; Threads.atomic_add!(_ABB_DD_T.nrechk, 1)
                    ok = _nz_dr_opt!(DR, M, l, u, "dd screen 재확인 s=$s")
                    # 2026-11-27 고침: 상태·bound 를 경계를 되돌리기 전에 읽음 (되돌리면 JuMP 가 결과를 지워 상태가 OPTIMIZE_NOT_CALLED 가 되어
                    #   재확인 결과가 버려지고 있었음 — 유효성은 그대로, 그 쌍을 덜 조였을 뿐)
                    ok && (st = termination_status(M.model); bd = getbd(st))
                    sigbd && _abb_sigbd!(M, DR, nd, lo, hi, s)
                    ok || continue
                    st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && Threads.atomic_add!(_ABB_DD_T.nrechk_inf, 1)
                end
                if st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)       # 상자 안에 recourse 최적 쌍대를 갖는 α 없음
                    set_time_limit_sec(M.model, nothing); sigbd && _abb_sigbd!(M, DR, nd, nothing, nothing, s)
                    return :empty, (dlo, dhi), nmip
                end
                isfinite(bd) || continue
                if ismax
                    bd + 1e-6 < dhi[p, s] && (dhi[p, s] = bd + 1e-6; tight = true)
                else
                    bd - 1e-6 > dlo[p, s] && (dlo[p, s] = bd - 1e-6; tight = true)
                end
            end
            dhi[p, s] < dlo[p, s] && (dhi[p, s] = dlo[p, s] = 0.5 * (dlo[p, s] + dhi[p, s]))
        end
        boxset && set_time_limit_sec(M.model, nothing)
        boxset && sigbd && _abb_sigbd!(M, DR, nd, nothing, nothing, s)
    end
    return (tight ? :tight : :none), (dlo, dhi), nmip
end

"패치 (dd_inc): 선별만 — _abb_dd_screen! 의 앞부분. 반환 [(모임 비율, 쌍 p, 시나리오 s, 꼭짓점 구간 c1, c2)]"
function _abb_dd_cands(CL, DR, nd, x̄, l, u, lo, hi, dlo, dhi; deadline=Inf, fixtol=1e-4, only_s::Int=0)
    S, m = nd.S, nz_m(nd); out = Tuple{Float64,Int,Int,Float64,Float64}[]
    for s in (only_s > 0 ? (only_s:only_s) : (1:S))
        time() > deadline && break
        ps = Int[]
        for (p, (a, b)) in enumerate(DR.pairs)
            (hi[a, s] - lo[a, s] > fixtol && hi[b, s] - lo[b, s] > fixtol) || continue
            wl = max(dlo[p, s], lo[a, s] - hi[b, s]); wh = min(dhi[p, s], hi[a, s] - lo[b, s])
            wh - wl > 1.0 && push!(ps, p)
        end
        isempty(ps) && continue
        C = CL[s]; ext = Dict{Int,Tuple{Float64,Float64}}(); tlp0 = time()
        for αc in (l, u)
            rr = nz_rhs(nd, x̄, αc, s)
            for k in 1:m; nd.hrow[k] > 0 && set_normalized_coefficient(C.opt, C.σ[k], rr[k]); end
            set_normalized_rhs(C.opt, 1e20)
            @objective(C.model, Min, sum(rr[k] * C.σ[k] for k in 1:m if rr[k] != 0)); optimize!(C.model)
            termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_cands: 꼭짓점 LP (s=$s) $(termination_status(C.model))")
            zc = objective_value(C.model); set_normalized_rhs(C.opt, zc + 1e-7 * max(1.0, abs(zc)))
            for p in ps
                a, b = DR.pairs[p]; e = C.σ[DR.Hk[a]] - C.σ[DR.Hk[b]]
                @objective(C.model, Min, e); optimize!(C.model)
                termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_cands: 꼭짓점 차이 LP (s=$s) $(termination_status(C.model))")
                mn = objective_value(C.model)
                @objective(C.model, Max, e); optimize!(C.model)
                termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_cands: 꼭짓점 차이 LP (s=$s) $(termination_status(C.model))")
                mx = objective_value(C.model)
                o = get(ext, p, (Inf, -Inf)); ext[p] = (min(o[1], mn), max(o[2], mx))
            end
        end
        Threads.atomic_add!(_ABB_DD_T.tlp, time() - tlp0); Threads.atomic_add!(_ABB_DD_T.nlp, 2 + 4 * length(ps))
        for p in ps
            a, b = DR.pairs[p]
            wl = max(dlo[p, s], lo[a, s] - hi[b, s]); wh = min(dhi[p, s], hi[a, s] - lo[b, s])
            c1, c2 = ext[p]
            c2 - c1 < 0.5 * (wh - wl) && push!(out, ((c2 - c1) / (wh - wl), p, s, c1, c2))
        end
    end
    return out
end

"패치 (dd_inc): (쌍 p, 시나리오 s) 차이의 한쪽 (ismax) 을 KKT-MIP 로 인증해 dlo·dhi 를 제자리 조임. 반환 (:empty | :tight | :none, MIP 수)"
function _abb_dd_mip!(DR, nd, l, u, lo, hi, dlo, dhi, p, s, ismax; mip_tl=2.0, deadline=Inf, sigbd::Bool=true, objstop::Float64=NaN, gap::Float64=NaN)
    M = DR.mips[s]; a, b = DR.pairs[p]; nh = nz_nh(nd); n = 0
    time() > deadline && return :none, 0
    for i in 1:nh; set_lower_bound(M.α[i], l[i]); set_upper_bound(M.α[i], u[i]); end
    sigbd && _abb_sigbd!(M, DR, nd, lo, hi, s)
    @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[DR.Hk[a]] - M.σ[DR.Hk[b]])
    set_time_limit_sec(M.model, max(0.01, min(mip_tl, deadline - time())))
    getbd(st) = st == MOI.OPTIMAL ? objective_bound(M.model) : (st in (MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT) ? _nz_dr_bound_or_nan(M.model) : NaN)
    # objstop (dd_objstop): 이 쪽 MIP 가 노드 LP 해를 자르려면 최댓값 < δ̂ (최솟값이면 > δ̂) 여야 함 → 해가 δ̂ 이상 (이하) 이면 자를 수 없으니 멈춤 (BestObjStop).
    #   멈춰도 objective_bound 는 유효한 범위. gap (dd_gap): MIPGap. 둘 다 풀이 뒤 원래 값으로
    ob0 = isfinite(objstop) ? get_optimizer_attribute(M.model, "BestObjStop") : NaN; gp0 = isfinite(gap) ? get_optimizer_attribute(M.model, "MIPGap") : NaN
    isfinite(objstop) && set_optimizer_attribute(M.model, "BestObjStop", objstop); isfinite(gap) && set_optimizer_attribute(M.model, "MIPGap", gap)
    tm0 = time(); n += 1
    ok = _nz_dr_opt!(DR, M, l, u, "dd inc s=$s 쌍 ($(DR.Hk[a]), $(DR.Hk[b]))")
    st = ok ? termination_status(M.model) : MOI.OTHER_ERROR; bd = ok ? getbd(st) : NaN
    if ok && sigbd && st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)     # 경계를 빼고 다시 풀어 확인 (결과는 되돌리기 전에 읽음)
        _abb_sigbd!(M, DR, nd, nothing, nothing, s); n += 1; Threads.atomic_add!(_ABB_DD_T.nrechk, 1)
        ok = _nz_dr_opt!(DR, M, l, u, "dd inc 재확인 s=$s")
        st = ok ? termination_status(M.model) : MOI.OTHER_ERROR; bd = ok ? getbd(st) : NaN
        ok && st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && Threads.atomic_add!(_ABB_DD_T.nrechk_inf, 1)
    end
    Threads.atomic_add!(_ABB_DD_T.tmip, time() - tm0)
    ok && st == MOI.OBJECTIVE_LIMIT && Threads.atomic_add!(_ABB_DD_T.nobjstop, 1)
    isfinite(objstop) && set_optimizer_attribute(M.model, "BestObjStop", ob0); isfinite(gap) && set_optimizer_attribute(M.model, "MIPGap", gp0)
    set_time_limit_sec(M.model, nothing); sigbd && _abb_sigbd!(M, DR, nd, nothing, nothing, s)
    ok || return :none, n
    st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return :empty, n      # 상자 안에 recourse 최적 쌍대를 갖는 α 없음
    isfinite(bd) || return :none, n
    tight = false
    if ismax
        bd + 1e-6 < dhi[p, s] && (dhi[p, s] = bd + 1e-6; tight = true)
    else
        bd - 1e-6 > dlo[p, s] && (dlo[p, s] = bd - 1e-6; tight = true)
    end
    dhi[p, s] < dlo[p, s] && (dhi[p, s] = dlo[p, s] = 0.5 * (dlo[p, s] + dhi[p, s]))
    return (tight ? :tight : :none), n
end

"패치 (dd_inc = :viol1): 노드 LP 해의 (쌍 p, 시나리오 s) follower 쌍대 차이가 꼭짓점 구간 [c1, c2] 에서 벗어난 양 (× d̂_s) 과 쪽 (+1 위, −1 아래, 0 안)"
function _abb_dd_lpviol(P, DR, p, s, c1, c2)
    v = P.O.v; a, b = DR.pairs[p]; ka, kb = DR.Hk[a], DR.Hk[b]; ds = value(v[:d][s])
    ds > 1e-7 || return 0.0, 0, NaN
    δ = (value(v[:π̃][ka, s]) - value(v[:π̃][kb, s])) / ds
    return (δ > c2 ? (δ - c2) * ds : (δ < c1 ? (c1 - δ) * ds : 0.0)), (δ > c2 ? 1 : (δ < c1 ? -1 : 0)), δ
end

"패치 (dd_sigbd): 시나리오 s 의 범위 MIP 에 H 행 쌍대 경계 [max(0, lo), hi] 를 넣거나 (lo = nothing 이면) 원래 경계 [0, ϖᵁ] 로 되돌림"
function _abb_sigbd!(M, DR, nd, lo, hi, s)
    for (a, k) in enumerate(DR.Hk)
        if lo === nothing
            set_lower_bound(M.σ[k], 0.0); isfinite(nd.varpiU[k]) ? set_upper_bound(M.σ[k], nd.varpiU[k]) : (has_upper_bound(M.σ[k]) && delete_upper_bound(M.σ[k]))
        else
            set_lower_bound(M.σ[k], max(0.0, lo[a, s])); set_upper_bound(M.σ[k], max(hi[a, s], max(0.0, lo[a, s])))
        end
    end
end

"패치 (dd_adapt): 이 노드에서 선별 DD 를 돌릴지. MIP 솔버가 분리기 호출 빈도를 성공률로 조절하는 것과 같은 꼴 —
 DD 를 받은 노드가 warm 개 이상인데 상한이 내려간 비율이 minrate 미만이면 자격 있는 노드 probe 개 중 하나만 (비율은 그 노드들로 계속 갱신).
 돌리지 않는 것은 덜 조일 뿐이라 유효성과 무관"
function _abb_dd_go(adapt, nelig, nnode, ncut, nskip, warm, minrate, probe)
    adapt || return true
    k = Threads.atomic_add!(nelig, 1)
    nn = nnode[]; nc = ncut[]
    (nn < warm || nc >= minrate * nn) && return true
    k % probe == 0 && return true
    Threads.atomic_add!(nskip, 1)
    return false
end
'''

hdr = ('# abb_dd_patch.jl — gen_abb_dd_patch.py 가 만든 파일 (손으로 고치지 말 것). 저장소 nz_alpha_bnb_dr 의 재정의:\n'
       '#   꼭짓점 LP 범위 (range_lp) 와 쌍대 차이 범위 (dd = true; dd_mode :screen 기본, :full) 를 함께 쓸 수 있게 함. 나머지는 저장소 판 그대로.\n'
       'println(stderr, "[abb_dd_patch] nz_alpha_bnb_dr 재정의: range_lp + dd 허용 (실험용)")\n')
open(os.path.join(here, 'abb_dd_patch.jl'), 'w', encoding='utf-8').write(hdr + helper + '\n' + body + '\n')
print('ok', start + 1, end + 1)
