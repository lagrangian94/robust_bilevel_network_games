"""
oos_dirichlet.jl — Dirichlet 샘플링 + ε calibration for True-DRO OOS experiment.

TV distance ε를 Dirichlet meta-distribution으로부터 calibrate:
  1. Dir(β·1_K) 에서 n_cal개 샘플 → uniform q̂=(1/K)·1 과의 L1 distance
  2. coverage-th quantile의 절반 = TV distance ε̂

CSV 표 캐싱:
  - build_epsilon_table() → S × coverage 조합별 ε 계산, CSV 저장
  - lookup_epsilon() → CSV 있으면 읽고, 없으면 build 후 반환
"""

using Distributions
using LinearAlgebra
using Statistics
using DelimitedFiles
using Printf


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


# ── ε table: S × coverage 조합별 CSV 캐싱 ──────────────────────────────────

const DEFAULT_S_VALUES        = [10, 20, 30, 50, 100]
const DEFAULT_COVERAGE_VALUES = [0.80, 0.85, 0.90, 0.95]

"""
    build_epsilon_table(β; S_values, coverage_values, n_cal, csv_path) -> Matrix{Float64}

S × coverage 조합별 calibrated ε 를 계산하고 CSV로 저장.
Returns |S_values| × |coverage_values| matrix.
CSV format:
  S,cov_80,cov_85,cov_90,cov_95
  10,0.123,0.134,...
"""
function build_epsilon_table(β::Float64;
        S_values::Vector{Int}       = DEFAULT_S_VALUES,
        coverage_values::Vector{Float64} = DEFAULT_COVERAGE_VALUES,
        n_cal::Int                  = 100_000,
        csv_path::String            = _default_csv_path(β))

    nS = length(S_values)
    nC = length(coverage_values)
    table = Matrix{Float64}(undef, nS, nC)

    for (i, S) in enumerate(S_values)
        # 한 번만 샘플링해서 여러 quantile 뽑기
        samples = sample_dirichlet(S, β; n_samples=n_cal)
        q_uniform = fill(1.0 / S, S)
        l1_distances = vec(sum(abs.(samples .- q_uniform), dims=1))

        for (j, cov) in enumerate(coverage_values)
            table[i, j] = quantile(l1_distances, cov) / 2.0
        end
    end

    # CSV 저장
    mkpath(dirname(csv_path))
    open(csv_path, "w") do io
        # header
        cov_headers = join(["cov_$(Int(c*100))" for c in coverage_values], ",")
        println(io, "S,", cov_headers)
        # rows
        for (i, S) in enumerate(S_values)
            vals = join([@sprintf("%.8f", table[i,j]) for j in 1:nC], ",")
            println(io, S, ",", vals)
        end
    end
    @info "ε table saved" csv_path size(table)

    return table
end


"""
    lookup_epsilon(K, β; coverage, ...) -> Float64

CSV 표에 (K, coverage) 조합이 있으면 읽어서 반환, 없으면 build 후 반환.
표에 없는 K/coverage 조합이면 fallback으로 직접 calibrate.
"""
function lookup_epsilon(K::Int, β::Float64;
        coverage::Float64           = 0.95,
        S_values::Vector{Int}       = DEFAULT_S_VALUES,
        coverage_values::Vector{Float64} = DEFAULT_COVERAGE_VALUES,
        n_cal::Int                  = 100_000,
        csv_path::String            = _default_csv_path(β))

    # CSV 있으면 로드, 없으면 build
    if isfile(csv_path)
        table, S_col, cov_cols = _load_epsilon_csv(csv_path)
    else
        @info "ε table not found, building..." csv_path
        table = build_epsilon_table(β;
            S_values=S_values, coverage_values=coverage_values,
            n_cal=n_cal, csv_path=csv_path)
        S_col = S_values
        cov_cols = coverage_values
    end

    # 표에서 exact match 찾기
    s_idx = findfirst(==(K), S_col)
    c_idx = findfirst(x -> abs(x - coverage) < 1e-6, cov_cols)

    if s_idx !== nothing && c_idx !== nothing
        return table[s_idx, c_idx]
    end

    # 표에 없는 조합 → 직접 계산
    @warn "ε table miss — computing directly" K coverage
    return calibrate_epsilon(K, β; n_cal=n_cal, coverage=coverage)
end


# ── helpers ─────────────────────────────────────────────────────────────────

function _default_csv_path(β::Float64)
    β_str = replace(@sprintf("%.2f", β), "." => "p")
    return joinpath(@__DIR__, "eps_table_beta_$(β_str).csv")
end

function _load_epsilon_csv(csv_path::String)
    lines = readlines(csv_path)
    # header: "S,cov_80,cov_85,..."
    header = split(lines[1], ",")
    cov_cols = [parse(Float64, replace(h, r"cov_" => "")) / 100.0 for h in header[2:end]]

    nrows = length(lines) - 1
    ncols = length(cov_cols)
    S_col = Vector{Int}(undef, nrows)
    table = Matrix{Float64}(undef, nrows, ncols)

    for (i, line) in enumerate(lines[2:end])
        parts = split(line, ",")
        S_col[i] = parse(Int, parts[1])
        for j in 1:ncols
            table[i, j] = parse(Float64, parts[j+1])
        end
    end

    return table, S_col, cov_cols
end
