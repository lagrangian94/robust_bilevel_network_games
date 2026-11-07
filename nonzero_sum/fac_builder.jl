# 지역 요인 수요 SGB15 빌더 (sgb15_fac.jl 에서 떼어 냄, 2026-11-06). nz_data.jl·Distributions 를 먼저 include/using.
const REGION = Dict("San Diego" => 1, "San Jose" => 1, "San Francisco" => 1, "Vancouver" => 1, "Seattle" => 1, "Tucson" => 1, "Sacramento" => 1,
                    "San Antonio" => 2, "Winnipeg" => 2, "Tulsa" => 2, "Wichita" => 2, "Saint Louis" => 2,
                    "Toronto" => 3, "Washington" => 3, "Toledo" => 3)
const RNAME = ["서부", "중부", "동부"]

"지역 요인 수요 SGB15 (dem = :region) 또는 저장소 그대로 (dem = :iid)"
function make_sgb15_fac(; S=10, seed=1, dem=:region, ρ=0.9, kw...)
    nd = make_sgb15_instance(; S=S, seed=seed, dist=:euclid, dist_res=10.0, quota=300.0, kw...)
    dem == :iid && return nd
    nst = nd.meta[:nst]; ncu = nd.meta[:ncu]; custs = nd.meta[:cities][nst+1:nst+ncu]
    reg = [REGION[c] for c in custs]
    μ = sgb15_mu(sgb15_kuhn().pop)              # 고객 노드 = 표 순서 (nz_data.jl 과 같음)
    rng = MersenneTwister(hash((4242, seed)))
    Z = randn(rng, 3, S); E = randn(rng, ncu, S)
    ξ = [round(μ[j] * (0.5 + cdf(Normal(), ρ * Z[reg[j], s] + sqrt(1 - ρ^2) * E[j, s])); digits=1) for j in 1:ncu, s in 1:S]
    rd = nd.meta[:rdem]
    for s in 1:S, j in 1:ncu; nd.u[rd(j), s] = -ξ[j, s]; end
    nd.meta[:xi] = ξ; nd.meta[:region] = reg
    @assert nd.meta[:nA] * nd.meta[:bA] > maximum(sum(ξ; dims=1)) "A 총용량 > 최대 총수요 필요"
    return nd
end
