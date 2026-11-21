# run_nz_belief_ccg.jl — 입지 가치 계열 (loc_tkp) 을 belief 블록 CCG (nz_belief_ccg.jl) 로 푼다 (2026-11-21).
#   환경 변수: GEO (1), MODEL (coupled | belief | data | nominal | product), R (0.2), SS (10), SEED (1), SITES (예 "4,7,11", 비우면 전부),
#             TL (3600), MU (10), ETA (0.1), MU_MAX (1e4), SAFEGUARD (nogood | restricted | both), MGAP (1e-3), EARLY_GAP (0), OBJ_CUT (0), POOL (1), VUB (0), NRAND (40)
#             ALG (explicit | tree: nz_belief_ccg_tree.jl 의 한 나무 판), tree 전용: ROOT_ROUNDS (30), UCUT (0), PAR (1), LAZY_ATTR (0)
#   실행: julia -t T,1 nonzero_sum/run_nz_belief_ccg.jl
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl")); include(joinpath(root, "nonzero_sum", "nz_belief_ccg_tree.jl"))
envf(k, d) = get(ENV, k, d)
g = parse(Int, envf("GEO", "1")); MODEL = envf("MODEL", "coupled"); rr = parse(Float64, envf("R", "0.2"))
eh, et, dl = MODEL == "coupled" ? (rr, rr, rr / 3) : MODEL == "belief" ? (0.0, rr, Inf) : MODEL == "data" ? (rr, 0.0, Inf) :
             MODEL == "nominal" ? (0.0, 0.0, Inf) : MODEL == "product" ? (rr, rr, Inf) : error("MODEL = coupled | belief | data | nominal | product")
nd = make_loc_tkp(; g=g, S=parse(Int, envf("SS", "10")), seed=parse(Int, envf("SEED", "1")), eps_hat=eh, eps_tilde=et, delta=dl, beta=0.4)
K, nB = nd.meta[:K], nd.meta[:nB]
if !isempty(envf("SITES", ""))
    keep = parse.(Int, split(ENV["SITES"], ","))
    for st in 1:nB, k in 1:K; st in keep || (nd.x_allowed[(st - 1) * K + k] = false); end
end
nX = let ns = count(st -> nd.x_allowed[(st - 1) * K + 1], 1:nB); sum(binomial(ns, n) * K^n for n in 0:nd.gamma) end
@printf("--- nz_belief_ccg | g=%d model=%s r=%.2f S=%d |X|=%d 스레드 %d\n", g, MODEL, rr, nd.S, nX, Threads.nthreads()); flush(stdout)
ALG = envf("ALG", "explicit")
res = if ALG == "explicit"
    nz_belief_ccg(nd; time_limit=parse(Float64, envf("TL", "3600")), mu=parse(Float64, envf("MU", "10")), eta=parse(Float64, envf("ETA", "0.1")),
                  mu_max=parse(Float64, envf("MU_MAX", "1e4")), safeguard=Symbol(envf("SAFEGUARD", "nogood")),
                  master_gap=parse(Float64, envf("MGAP", "1e-3")), early_gap=parse(Float64, envf("EARLY_GAP", "0")), obj_cut=envf("OBJ_CUT", "0") == "1", wd=envf("WD", "0") == "1",
                  pool=parse(Int, envf("POOL", "1")), vub=envf("VUB", "0") == "1", nrand=parse(Int, envf("NRAND", "40")))
elseif ALG == "tree"
    nz_belief_ccg_tree(nd; time_limit=parse(Float64, envf("TL", "3600")), mu=parse(Float64, envf("MU", "10")), eta=parse(Float64, envf("ETA", "0.1")),
                       mu_max=parse(Float64, envf("MU_MAX", "1e4")), safeguard=Symbol(envf("SAFEGUARD", "nogood")),
                       master_gap=parse(Float64, envf("MGAP", "1e-3")), wd=envf("WD", "0") == "1", nrand=parse(Int, envf("NRAND", "40")),
                       root_rounds=parse(Int, envf("ROOT_ROUNDS", "30")), ucut_calls=parse(Int, envf("UCUT", "0")), parallel=envf("PAR", "1") == "1", lazy_attr=parse(Int, envf("LAZY_ATTR", "0")))
else
    error("ALG = explicit | tree")
end
if ALG == "tree"
    @printf("TREE tree_beliefs=%d kept=%d callback=%.0fs gen=%.0fs (평가 %d) search=%.0fs (%d 번) lazy_calls=%d lazy_cuts=%d ucut_calls=%d ucuts=%d real_cuts=%d\n",
            res[:tree_beliefs], res[:kept_incumbent], res[:callback_time], res[:gen_time], res[:gen_evals], res[:search_time], res[:searches], res[:lazy_calls], res[:lazy_cuts],
            res[:ucut_calls], res[:ucuts], res[:real_cuts])
end
@printf("RESULT %s g=%d model=%s r=%.2f |X|=%d status=%s x=%s LB=%.4f UB=%.4f gap=%.2e iters=%d beliefs=%d master=%.0fs oracle=%d (%.0fs) nogood=%d restricted=%d mu_raises=%d time=%.1f args=%s\n",
        res[:impl], g, MODEL, rr, nX, res[:status], res[:x] === nothing ? "-" : loc_tkp_xstr(nd, res[:x]), res[:LB], res[:UB], res[:gap],
        res[:iters], res[:beliefs], res[:master_time], res[:oracle_calls], res[:oracle_time], res[:nogood], res[:restricted], res[:mu_raises],
        res[:time], string(res[:args]))
