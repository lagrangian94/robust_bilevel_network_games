"""
test_corner_ranges.jl — 꼭짓점 LP 범위 (nz_alpha_bnb_dr 의 range_lp = true, _nz_dr_corner_ranges) 가 범위 MIP (정확) 를 포함하는지 무작위 상자에서 대조 (2026-10-28).

전제 (recourse 값함수의 submodularity) 가 깨지면 꼭짓점 범위가 MIP 범위보다 좁아진다 (= 무효). 그 성분 수를 센다.
인스턴스: pilot_pooled_store.jl 의 mk 와 같은 L 계열. 상자: 중심 균등, 반폭 = w·hU (w ∈ {0.5, 0.2, 0.05, 0.01}), [0, hU] 로 자름.
환경변수: TC_LIST = "25:5:5:1:3:2,3;..." (d:nA:nB:seed:S:x — x 는 쉼표, 빈칸이면 x=∅)  TC_BOX = 6 (조합마다 상자 수)  TC_TL = 120 (MIP 하나 제한)
실행: julia -t 2 nonzero_sum/test_corner_ranges.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
mk1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
LIST = envf("TC_LIST", "25:5:5:1:3:2,3;25:5:5:5:3:1,2;25:5:5:6:3:1,3,4;25:5:5:2:10:1,3,5;20:3:2:1:10:1;15:2:3:3:10:1,2;25:5:5:3:3:")
NBOX = parse(Int, envf("TC_BOX", "6")); TLm = parse(Float64, envf("TC_TL", "120"))
tot = 0; bad = 0; wide = 0; tmip = 0.0; tlp = 0.0; nunk = 0
for spec in split(LIST, ";"; keepempty=false)
    f = split(spec, ":")
    nloc, nA, nB, seed, S = parse.(Int, f[1:5]); xi = parse.(Int, split(f[6], ","; keepempty=false))
    capv = 150.0 * (nloc - nA - nB) / nA
    nd0 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=nA, nB=nB, NB=nB, S=S, seed=seed,
        bA=capv, bB=capv, qB=750.0, dlo=50.0, dhi=150.0, p=0.3, f=0.1, reservation=:pooled, pool_limit=:store, quota=300.0,
        eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    nd1 = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1], lambdaU=2000.0)
    x̄ = zeros(nd1.nx); x̄[xi] .= 1
    nd = nz_with_hU(nd1, nz_presolve_hU(nd1, x̄)[1]); nh = nz_nh(nd)
    P = _nz_build_sep(nd, x̄; optimizer=mk1)
    DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk1, dd=false)
    CL = [_nz_dr_build_corner(nd, x̄, s; optimizer=mk1) for s in 1:S]
    lo0, hi0 = _nz_dr_global(nd, DR)
    rng = MersenneTwister(1000 * seed + S)
    nb = 0; nbad = 0
    for b in 1:NBOX
        w = (0.5, 0.2, 0.05, 0.01)[mod1(b, 4)]
        c = rand(rng, nh) .* nd.hU
        l = max.(c .- w .* nd.hU, 0.0); u = min.(c .+ w .* nd.hU, nd.hU)
        t1 = @elapsed (rg = _nz_dr_ranges(DR, nd, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S);
                                          deadline=Inf, mip_deadline=time() + TLm * 2 * length(DR.Hk) * S))
        rg === nothing && (println("  상자 $b: 실행 불가능 (건너뜀)"); continue)
        t2 = @elapsed ((clo, chi) = _nz_dr_corner_ranges(CL, DR, nd, x̄, l, u, lo0, hi0))
        mlo, mhi = rg[1], rg[2]
        n = length(mlo)
        nar = count((clo .> mlo .+ 1e-5) .| (chi .< mhi .- 1e-5))
        wid = count((clo .< mlo .- 1e-5) .| (chi .> mhi .+ 1e-5))
        global tot += n; global bad += nar; global wide += wid; global tmip += t1; global tlp += t2
        nb += 1; nbad += nar
        nar > 0 && for a in 1:size(mlo, 1), s in 1:S
            (clo[a, s] > mlo[a, s] + 1e-5 || chi[a, s] < mhi[a, s] - 1e-5) &&
                @printf("    위반: d=%d seed=%d S=%d x=%s 상자 %d (w=%.2f) k=%d s=%d: MIP [%.5f, %.5f] 꼭짓점 [%.5f, %.5f]\n",
                        nloc, seed, S, string(xi), b, w, DR.Hk[a], s, mlo[a, s], mhi[a, s], clo[a, s], chi[a, s])
        end
    end
    @printf("d=%d A=%d B=%d seed=%d S=%d x=%s: 상자 %d 개, 위반 성분 %d\n", nloc, nA, nB, seed, S, string(xi), nb, nbad); flush(stdout)
end
@printf("\nRESULT 성분 %d 개: 꼭짓점이 더 좁음 (무효) %d, 더 넓음 %d | 시간 MIP %.1fs, 꼭짓점 LP %.1fs\n", tot, bad, wide, tmip, tlp)
