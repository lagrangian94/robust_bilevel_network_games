
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
    end
    false
end
UB = Inf; LB = -Inf; xbest = 0; status = :TimeLimit; t_or = 0.0; n_or = 0; certified = Dict{Int,Float64}()   # 점 → oracle 로 확인했을 때의 마스터 값 (값이 오를 때만 다시)
needs_oracle(c) = (evalall!([c]); mval(c) > get(certified, c, -Inf) + 1e-6 * max(1.0, abs(mval(c))))
function upd_inc!(cs)
    for c in cs
        all(b -> blks[b].kind == :aff || haskey(cache, (c, b)), eachindex(blks)) || continue
        (cinc == 0 || mval(c) < mval(cinc) - 1e-9) && (global cinc = c)
    end
end
"oracle (α-B&B) 을 c 에서. 목표는 지금 LB 기준 (끝에 UB − LB ≤ tol 이 되게)"
orLB = Dict{Int,Float64}(); orcap = Dict{Int,Float64}()
function oracle_at!(c; cap=ORCL_TL)
    evalall!([c]); t0v = mval(c) - qv[c]; xb = pts[c]
    tau = isfinite(LB) ? max(t0v + 1e-6 * max(1.0, abs(t0v)), LB + 0.95 * TOL * max(1.0, abs(LB)) - qv[c]) : t0v + 0.9 * TOL * max(1.0, abs(qv[c] + t0v))
    gap_o = min(1e-3, 0.4 * TOL * max(1.0, abs(qv[c] + t0v)) / max(1.0, abs(t0v)))
    lim = max(10.0, min(cap, TL - (time() - wall))); orcap[c] = cap
    to = @elapsed (res = nz_alpha_bnb_dr(nd, xb; nworkers=max(1, NT - 1), time_limit=lim, rel_gap=gap_o, target=tau, verbose=false))
    global t_or += to; global n_or += 1; orLB[c] = max(get(orLB, c, -Inf), res[:LB])
    if qv[c] + res[:UB] < UB; global UB = qv[c] + res[:UB]; global xbest = c; end
    how = :none
    if res[:α] !== nothing && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v))
        vv, dd, r2, a1 = exact_value(xb, res[:α])
        if dd !== nothing
            how = add_and_secure!(c, vv, dd, r2)
            vv <= t0v + 1e-6 * max(1.0, abs(t0v)) && restricted!(c, a1)
        end
    end
    if how == :none && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v)); nogood!(c, res[:LB]); how = :nogood_lb; end
    certified[c] = mval(c)
    @printf("  [orcl] x=%-24s V in [%.2f, %.2f], t0 %.2f, 목표 %.2f (%.0fs, 상한 %.0fs) %s, 블록 %d, LB %.2f UB %.2f (경과 %.0fs)\n", loc_tkp_xstr(nd, xb), res[:LB], res[:UB], t0v, tau, to, lim, how, length(blks), LB, UB, time() - wall); flush(stdout)
    return how
end

# ---- 시작 ----
b1 = add_lpblk!(qh, qh, MU); evalpairs!([(1, b1)]); push!(cols, Set([1]))
bnd = 0.0; tl0 = @elapsed (global bnd = first(pricemip(b1, 1.0, zeros(nx), 0.0, Inf, ROOT)))
global tlow = bnd - 1e-6 * max(1.0, abs(bnd))
@printf("  [시작] t 하한 %.2f (가격 MILP %.1fs)\n", tlow, tl0); flush(stdout)

# ---- 1 단계 (루트, 휴리스틱 열 생성 + 탐색 + oracle): belief 와 UB 모으기 ----
if PH1
    while time() - wall < TL
        r = dwsolve!(ROOT; exact=false); upd_inc!(r[6])
        @printf("  [ph1] 루트 RMP %.2f, 마스터 UB %.2f (x=%s), 블록 %d, LP 누적 %d (경과 %.0fs)\n", r[2], r[4], loc_tkp_xstr(nd, pts[r[5]]), length(blks), n_lp[], time() - wall); flush(stdout)
        trysearch(r[6]) && continue
        oracle_at!(r[5]) == :none && break
    end
end
t_ph1 = time() - wall
@printf("  [1 단계 끝] %.0fs, belief %d, UB %.2f\n", t_ph1, count(B -> B.kind == :lp, blks), UB); flush(stdout)

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
closedq = Tuple{Node,Int}[]       # 정수로 닫은 노드와 그 점 (하한 = max(마스터 값, q + oracle 하한))
closed_lb(e) = max(e[1].lb, mval(e[2]), qv[e[2]] + get(orLB, e[2], -Inf))
global_lb(extra=Inf) = min(extra, minimum((o.lb for o in openq); init=Inf), minimum((closed_lb(e) for e in closedq); init=Inf))
function finish_or_deepen!()
    while time() - wall < TL
        if isempty(openq)
            lbg = isfinite(UB) ? min(UB - TOL * max(1.0, abs(UB)), global_lb()) : global_lb()
            global LB = max(LB, lbg)
            if isfinite(UB) && UB - LB <= TOL * max(1.0, abs(UB)); global status = :Optimal; return; end
            isempty(closedq) && return
            k = argmin([closed_lb(e) for e in closedq]); N, c = closedq[k]
            cap2 = 2 * get(orcap, c, ORCL_TL)
            @printf("  [deepen] 닫힌 노드 %s (점 %s, 하한 %.2f) 이 막음 → oracle 상한 %.0fs 로 다시\n", N.tag, loc_tkp_xstr(nd, pts[c]), closed_lb(closedq[k]), cap2); flush(stdout)
            h = oracle_at!(c; cap=cap2)
            if h != :none
                deleteat!(closedq, k); push!(openq, N)          # 블록이 늘었으니 이 노드를 다시 엶
                return :reopen
            end
            continue
        end
        return :open
    end
end
openq = [ROOT]; nproc = 0; nprune = 0; nexact = 0; nint = 0
gapok(lb) = isfinite(UB) && lb >= UB - TOL * max(1.0, abs(UB))
while time() - wall < TL
while !isempty(openq) && time() - wall < TL
    sort!(openq; by=N -> (N.lb, N.est)); N = popfirst!(openq)
    global LB = max(LB, global_lb(N.lb))
    if isfinite(UB) && UB - LB <= TOL * max(1.0, abs(UB)); global status = :Optimal; break; end
    gapok(N.lb) && (global nprune += 1; continue)
    global nproc += 1
    local r
    while true
        r = dwsolve!(N; exact=false); r[1] == Inf && break
        upd_inc!(r[6]); trysearch(filter(c -> !gapok(mval(c)), r[6])) || break    # 이미 UB − tol 위인 점의 belief 는 가지치기를 못 바꿈
    end
    if r[1] == Inf; global nprune += 1; continue; end            # 실행불가 노드
    v, xv = r[2], r[3]; lbN = N.lb
    integral = all(i -> abs(xv[i] - round(xv[i])) < 1e-6, al)
    if gapok(v) || integral
        r = dwsolve!(N; exact=true, ubglob=UB); global nexact += 1; upd_inc!(r[6])
        lbN = max(lbN, r[1]); v, xv = r[2], r[3]; integral = all(i -> abs(xv[i] - round(xv[i])) < 1e-6, al)
    end
    @printf("  [node %3d] %-34s 깊이 %d: RMP %.2f, LB %.2f, %s, 열린 %d, LB* %.2f UB %.2f, 마스터 inc %.2f, 블록 %d, MILP %d (경과 %.0fs)\n",
            nproc, N.tag, N.depth, v, lbN, integral ? "정수" : "분수", length(openq), LB, UB, cinc > 0 ? mval(cinc) : NaN, length(blks), n_mip[], time() - wall); flush(stdout)
    # oracle: 마스터 incumbent 가 UB 를 고칠 수 있고 지금 블록으로 확인 안 된 경우
    if cinc > 0 && mval(cinc) < UB - 0.1 * TOL * max(1.0, abs(UB)) && needs_oracle(cinc)
        h = oracle_at!(cinc)
        if h != :none; push!(openq, Node(N.f0, N.f1, N.opn, lbN, v, N.depth, N.tag)); continue; end   # 블록이 늘었으니 이 노드를 다시
    end
    gapok(lbN) && (global nprune += 1; continue)
    if integral
        c = idx[key(round.(xv))]; global nint += 1
        if needs_oracle(c)
            h = oracle_at!(c)
            h != :none && (push!(openq, Node(N.f0, N.f1, N.opn, lbN, v, N.depth, N.tag)); continue)
        end
        push!(closedq, (Node(N.f0, N.f1, N.opn, lbN, v, N.depth, N.tag), c))   # 정수 노드: 하한을 남기고 닫음
        continue
    end
    ch = children(N, xv, v)
    ch === nothing && (println("  [!] 분기 변수 없음"); continue)
    push!(openq, ch[1]); push!(openq, ch[2])
end   # while (B&P 노드)
status == :Optimal && break
finish_or_deepen!() == :reopen || break
end   # 바깥
tot = time() - wall
nbel = count(B -> B.kind == :lp, blks)
@printf("RESULT ccgbp g=%d status=%s x=%s LB=%.4f UB=%.4f gap=%.3f%% beliefs=%d 노드 %d (가지침 %d, 정확 %d, 정수 %d, 남은 %d, 닫힌 정수 %d) 안전장치 %d (no-good %d, restricted %d, μ 올림 %d) time=%.1f (1 단계 %.0fs)\n",
        GEO, status, xbest == 0 ? "-" : loc_tkp_xstr(nd, pts[xbest]), LB, UB, 100 * (UB - LB) / abs(UB), nbel, nproc, nprune, nexact, nint, length(openq), length(closedq), n_ng[] + n_rc[], n_ng[], n_rc[], n_mu[], tot, t_ph1)
@printf("  시간: oracle %d 번 %.0fs, 탐색 %d 번 %.0fs, 블록 LP %d 번 %.0fs, 가격 MILP %d 번 %.0fs (노드 %d), RMP %d 번 %.0fs (crossover 재시도 %d), 생성기 구성 %.0fs; 서로 다른 점 %d / %d\n",
        n_or, t_or, n_srch[], t_srch[], n_lp[], t_lp[], n_mip[], t_mip[], nodes_mip[], n_rmp[], t_rmp[], n_rmpfix[], t_build[], length(unique([c for (c, _) in keys(cache)])), NP)
