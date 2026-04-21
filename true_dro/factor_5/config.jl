"""
factor_3/config.jl — Factor k=3, γ=1(real-world)/3(grid_5x5) 실험 공통 설정.
"""

const F3_NETWORKS = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls]

const F3_NET_CONFIGS = Dict(
    :grid_5x5    => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us    => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene     => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska      => Dict(:type => :real_world, :generator => generate_polska_network),
)

const F3_S = 20
const F3_SEED = 42
const F3_LAMBDA_U = 2.0
const F3_NUM_FACTORS = 3

function f3_compute_gamma(config_key::Symbol, num_interdictable::Int)
    if config_key == :grid_5x5
        return 3
    end
    return 1  # real-world: γ=1
end

function f3_regenerate_network(config_key::Symbol; S::Int=F3_S)
    config = F3_NET_CONFIGS[config_key]
    network = config[:type] == :grid ?
        generate_grid_network(config[:m], config[:n]; seed=F3_SEED) :
        config[:generator]()

    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = f3_compute_gamma(config_key, num_interdictable)

    capacities, F = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=F3_SEED,
        num_factors=F3_NUM_FACTORS)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    w = round(maximum(capacities[interdictable_idx, :]); digits=4)

    return network, capacities, w, γ, F
end

"""
    f3_log_path(subdir, net_key, suffix) → String

logs/ 아래에 타임스탬프 포함 로그 파일 경로 생성.
"""
function f3_log_path(net_key::Symbol, suffix::String)
    ts = Dates.format(now(), "yyyymmdd_HHMMSSs")
    return joinpath(@__DIR__, "logs", "log_$(net_key)_$(ts)_$(suffix).txt")
end

"""
    f3_find_log(net_key, suffix) → Union{String, Nothing}

logs/ 에서 가장 최근 로그 파일 찾기.
"""
function f3_find_log(net_key::Symbol, suffix::String)
    log_dir = joinpath(@__DIR__, "logs")
    !isdir(log_dir) && return nothing
    pattern_prefix = "log_$(net_key)_"
    files = filter(readdir(log_dir)) do f
        startswith(f, pattern_prefix) && endswith(f, "$(suffix).txt")
    end
    isempty(files) && return nothing
    sort!(files)
    return joinpath(log_dir, files[end])
end

"""
    f3_parse_log(filepath) → Dict

로그에서 x*, Z0, status 파싱.
"""
function f3_parse_log(filepath::String)
    lines = readlines(filepath)

    x_star_summary = nothing
    x_star_omp = nothing
    Z0 = NaN
    status = "Unknown"
    iters = 0
    wall_time = NaN

    for line in lines
        m_td = match(r"(True-DRO|Single-layer):\s*status=(\w+),\s*Z₀=([\d.]+),\s*iters=(\d+),\s*time=([\d.]+)s", line)
        if m_td !== nothing
            status = m_td.captures[2]
            Z0 = parse(Float64, m_td.captures[3])
            iters = parse(Int, m_td.captures[4])
            wall_time = parse(Float64, m_td.captures[5])
        end

        m_nom = match(r"Nominal SP:\s*Z₀=([\d.]+),\s*time=([\d.]+)s", line)
        if m_nom !== nothing
            status = "Optimal"
            Z0 = parse(Float64, m_nom.captures[1])
            wall_time = parse(Float64, m_nom.captures[2])
        end

        m_omp = match(r"OMP:.*x=\[([0-9,\s]+)\]", line)
        if m_omp !== nothing
            x_star_omp = parse.(Int, split(m_omp.captures[1], r"[,\s]+"; keepempty=false))
        end

        if !isnan(Z0) && x_star_summary === nothing
            m_x = match(r"^\s*x\*\s*=\s*\[([0-9,\s]+)\]", line)
            if m_x !== nothing
                x_star_summary = parse.(Int, split(m_x.captures[1], r"[,\s]+"; keepempty=false))
            end
        end
    end

    x_star = x_star_summary
    x_source = :summary
    if x_star !== nothing && all(x_star .== 0) && x_star_omp !== nothing
        x_star = x_star_omp
        x_source = :last_omp
    end
    if x_star === nothing && x_star_omp !== nothing
        x_star = x_star_omp
        x_source = :last_omp
    end

    return Dict(:x_star => x_star, :x_source => x_source, :Z0 => Z0,
                :status => status, :iters => iters, :wall_time => wall_time)
end

"""
    f3_tee_solve(f, net_key, suffix) → result

stdout를 로그 파일에 tee하면서 f()를 실행.
"""
function f3_tee_solve(f::Function, net_key::Symbol, suffix::String)
    log_file = f3_log_path(net_key, suffix)
    log_io = open(log_file, "w")
    original_stdout = stdout
    rd, wr = redirect_stdout()
    log_task = @async begin
        try
            while isopen(rd)
                buf = readavailable(rd)
                isempty(buf) && break
                write(original_stdout, buf)
                flush(original_stdout)
                write(log_io, buf)
                flush(log_io)
            end
        catch e
            e isa EOFError || rethrow()
        end
    end

    result = nothing
    try
        println("Log file: $log_file")
        println("Started: $(now())")
        println()
        result = f()
        println("\nFinished: $(now())")
    catch err
        println("\nERROR: $err")
        bt = catch_backtrace()
        Base.showerror(stdout, err, bt)
        rethrow()
    finally
        redirect_stdout(original_stdout)
        close(wr)
        try; wait(log_task); catch; end
        close(log_io)
        println("  Log saved → $log_file")
    end
    return result
end
