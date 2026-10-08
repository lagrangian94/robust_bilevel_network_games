"""
nz_paper_instances.jl — 원고 실험 (true_dro/factor_5/run_baseline_batch.jl) 과 같은 max-flow 인스턴스를 NZData 로.

  S = 10, factor_additive 시나리오 (k = 5, seed 42), w = 0.5 γ median(용량), γ = 2, λᵁ = 10,
  v_ks ~ Bernoulli(0.75) (seed 42), q̂ 균등. real-world 망은 모든 arc 가 interdictable 이고
  source/sink cut (출발·도착 arc 를 모두 막지 못함) 을 OMP 카디널리티로 넣는다 (meta[:omp_card]).

network_generator.jl (module NetworkGenerator) 와 true_dro_data.jl 이 먼저 include 되어 있어야 한다.
반환 (nd, td): nd = nz_from_true_dro(td) (+ meta), td = TrueDROData (원고 코드 교차검증용).
"""

using Statistics, Random

function make_paper_maxflow(name::AbstractString; S=10, eps_hat=0.2, eps_tilde=0.2, beta=0.7, lambdaU=10.0, gamma=2,
                            seed=42)
    NG = Main.NetworkGenerator
    if startswith(name, "grid")                       # "grid5x5" (원고), 검증용 "grid3x3", "grid3x4" 등
        mm, nn = parse.(Int, split(name[5:end], "x"))
        net = NG.generate_grid_network(mm, nn; seed=seed)
        intd = net.interdictable_arcs
    else
        gen = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network,
                   "nobel_us" => NG.generate_nobel_us_network, "sioux_falls" => NG.generate_sioux_falls_network)
        net0 = gen[name]()
        intd = fill(true, length(net0.arcs))
        net = NG.RealWorldNetworkData(net0.name, net0.original_node_names, net0.nodes, net0.arcs,
            net0.N, intd, net0.arc_adjacency, net0.node_arc_incidence)
    end
    K = length(net.arcs) - 1
    caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
        interdictable_arcs=intd, seed=seed, num_factors=5)
    intd_idx = findall(intd[1:K])
    w = round(0.5 * gamma * median(caps[intd_idx, :]); digits=4)
    Random.seed!(seed)
    v = zeros(K, S)
    for k in 1:K, s in 1:S
        v[k, s] = intd[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
    end
    td = Main.make_true_dro_data(net, caps, fill(1.0 / S, S), eps_hat, eps_tilde;
        w=w, lambda_U=lambdaU, gamma=gamma, beta=beta, v_scenarios=v)
    nd = nz_from_true_dro(td)
    card = Tuple{Vector{Int},Int}[]
    if !startswith(name, "grid")
        src = [i for i in 1:K if net.arcs[i][1] == "s"]
        snk = [i for i in 1:K if net.arcs[i][2] == "t"]
        length(src) >= 2 && push!(card, (src, length(src) - 1))
        length(snk) >= 2 && push!(card, (snk, length(snk) - 1))
    end
    nd.meta[:omp_card] = card
    nd.meta[:network] = name
    nd.meta[:w] = w
    return nd, td
end

"x 가 OMP 카디널리티를 만족하는가"
nz_card_ok(nd, x) = all(sum(x[i] for i in idx) <= rhs for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]))
