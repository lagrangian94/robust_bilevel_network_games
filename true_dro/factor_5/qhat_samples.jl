"""
qhat_samples.jl — Dir(1 ∈ R^S)에서 q̂ 샘플링 (greedy diversity).
  100개 생성 후:
    #1: uniform에서 가장 먼 것
    #2: uniform과 #1 둘 다에서 가장 먼 것 (min TV 기준)
    #3: uniform, #1, #2 셋에서 가장 먼 것
  S와 seed(=123)만 같으면 네트워크 무관하게 동일한 q̂.

Usage:
  include("qhat_samples.jl")
  q_hat = generate_qhat(S, idx)   # idx=0 → uniform, 1/2/3 → Dir(1) greedy
"""

using Random

function generate_qhat(S::Int, idx::Int; seed::Int=123, n_samples::Int=100)
    idx in 0:3 || error("qhat idx must be 0 (uniform), 1, 2, or 3. Got: $idx")
    if idx == 0
        return fill(1.0 / S, S)
    end
    rng = MersenneTwister(seed)
    samples = Vector{Vector{Float64}}(undef, n_samples)
    for i in 1:n_samples
        g = [-log(rand(rng)) for _ in 1:S]
        samples[i] = g ./ sum(g)
    end
    tv(a, b) = 0.5 * sum(abs.(a .- b))

    # Greedy: 순차적으로 기존 선택들과의 min distance가 최대인 샘플 선택
    anchors = [fill(1.0 / S, S)]  # uniform이 첫 anchor
    chosen = Int[]
    for _ in 1:3
        best_i, best_score = -1, -Inf
        for i in 1:n_samples
            i in chosen && continue
            score = minimum(tv(samples[i], a) for a in anchors)
            if score > best_score
                best_i, best_score = i, score
            end
        end
        push!(chosen, best_i)
        push!(anchors, samples[best_i])
    end
    return samples[chosen[idx]]
end
