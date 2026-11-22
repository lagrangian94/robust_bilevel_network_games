
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
    get(idx, key(z), 0)
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
    return v, idx[key(z)]
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
    zs = [idx[key(round.(value.(xb; result=k)))] for k in 1:result_count(Mp)]
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
"노드 N 에서 열 생성. 반환 (lb (exact 일 때만, 아니면 −Inf; 실행불가면 Inf), RMP 값, x̃, 마스터 UB, 그 점, 상위 정수점, 정보)"
function dwsolve!(N; exact=true, ubglob=Inf)
    lb = -Inf; ubm = Inf; cbest = 0; it = 0; nex = 0; topl = Int[]; v = NaN; xv = zeros(nx)
    s0 = seedpt(N); s0 == 0 && return Inf, Inf, xv, Inf, 0, Int[], (it = 0, nex = 0)
    evalall!([s0]); for b in eachindex(blks); push!(cols[b], s0); end
    while true
        it += 1
        v, w, π, σ, supp, xv = rmp(N)
        tolabs = 1e-4 * max(1, abs(v))
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
        # 가격 MILP 멈춤 기준: 가지치기에 필요한 여유 (RMP − (UB − tol)) 를 블록에 나눔. 여유가 없으면 엄격 (tolabs)
        nw = count(b -> blks[b].kind == :lp && w[b] > 1e-9, eachindex(blks))
        need = isfinite(ubglob) ? v - (ubglob - TOL * max(1.0, abs(ubglob))) : 0.0
        allow = (PSTOP && need > 0 && nw > 0) ? max(tolabs, 0.9 * need / nw) : tolabs
        for b in eachindex(blks)
            if blks[b].kind == :lp && w[b] > 1e-9
                bnd, zs, inc = pricemip(b, w[b], π[b], σ[b], σ[b] - allow, N)
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
        tgt = min(ubm, ubglob)
        (added2 == 0 || tgt - lb <= CGTOL * max(1, abs(tgt))) && return lb, v, xv, ubm, cbest, topl, (it = it, nex = nex)
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

