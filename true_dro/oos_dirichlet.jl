"""
oos_dirichlet.jl — Dirichlet 샘플링 + ε calibration for True-DRO OOS experiment.

TV distance ε를 Dirichlet meta-distribution으로부터 calibrate:
  1. Dir(β·1_K) 에서 n_cal개 샘플 → uniform q̂=(1/K)·1 과의 L1 distance
  2. coverage-th quantile의 절반 = TV distance ε̂
"""

using Distributions
using LinearAlgebra
using Statistics


"""
    sample_dirichlet(K, β; n_samples=1) -> Matrix{Float64}

Dirichlet(β·1_K)에서 샘플링. Returns K × n_samples matrix.
"""
function sample_dirichlet(K::Int, β::Float64; n_samples::Int=1)
    dist = Dirichlet(K, β)
    return rand(dist, n_samples)  # K × n_samples
end


"""
    calibrate_epsilon(K, β; n_cal=10000, coverage=0.95) -> Float64

Dirichlet(β·1_K)로부터 TV distance ε 를 calibrate.

1. n_cal 개 Dir(β·1_K) 샘플 q^(i) 생성
2. 각 q^(i) 와 uniform q̂=(1/K)·1 의 L1 distance 계산
3. coverage-th quantile = L1_quantile
4. ε = L1_quantile / 2  (TV distance = ½ · L1)

Formulation 제약: Σ_s |b_s| ≤ 2ε̂ → L1 ≤ 2ε̂ → TV = ½·L1 ≤ ε̂.
즉 ε̂ 자체가 TV distance radius.

Calibration: L1_quantile (95th percentile) 구한 뒤 ε = L1_quantile / 2 → TV distance.
test_benders.jl의 epsilon_hat과 동일한 scale (2배 차이 없음).
"""
function calibrate_epsilon(K::Int, β::Float64; n_cal::Int=10000, coverage::Float64=0.95)
    @assert K >= 2 "K must be >= 2"
    @assert β > 0.0 "β must be positive"
    @assert 0.0 < coverage < 1.0 "coverage must be in (0,1)"

    q_uniform = fill(1.0 / K, K)

    # Draw n_cal samples from Dir(β·1_K)
    samples = sample_dirichlet(K, β; n_samples=n_cal)  # K × n_cal

    # L1 distances to uniform
    l1_distances = vec(sum(abs.(samples .- q_uniform), dims=1))  # n_cal vector

    # Coverage quantile
    l1_quantile = quantile(l1_distances, coverage)

    # TV distance = ½ · L1
    ε = l1_quantile / 2.0

    return ε
end
