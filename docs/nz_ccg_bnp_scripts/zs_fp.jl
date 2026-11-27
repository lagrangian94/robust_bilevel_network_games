# zero-sum 인스턴스 지문 (2026-11-28): gen_zs.py 의 zs_instance 와 같은 규칙으로 caps·v·w 합. 이 컴퓨터 (Julia 1.11) 와 서버 (1.13) 비교용.
#   julia zs_fp.jl <저장소 경로> <S> <시드 목록 "42,1">
root = ARGS[1]
using Random, Statistics, Printf
include(joinpath(root, "network_generator.jl")); using .NetworkGenerator; const NG = NetworkGenerator
S = parse(Int, ARGS[2])
println("julia ", VERSION)
for sd in parse.(Int, split(ARGS[3], ",")), name in ("grid5x5", "polska", "abilene", "nobel_us", "sioux_falls")
    if name == "grid5x5"
        net = NG.generate_grid_network(5, 5; seed=42); intd = net.interdictable_arcs
    else
        gen = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network,
                   "nobel_us" => NG.generate_nobel_us_network, "sioux_falls" => NG.generate_sioux_falls_network)
        net = gen[name](); intd = fill(true, length(net.arcs))
    end
    K = length(net.arcs) - 1
    caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S; interdictable_arcs=intd, seed=sd, num_factors=5)
    w = round(0.5 * 2 * median(caps[findall(intd[1:K]), :]); digits=4)
    Random.seed!(sd); v = [intd[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0 for k in 1:K, s in 1:S]
    @printf("%s s%d: K %d | caps %.6f | caps2 %.6f | w %.4f | v %d\n", name, sd, K, sum(caps), sum(abs2, caps), w, Int(sum(v)))
end
