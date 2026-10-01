"""
make_computational_table.jl — Parse computational logs and generate LaTeX table
  Scans logs/factor/computational/ for eps_*_beta_* folders (naive, mb_only),
  and logs/factor/eps_*_beta_* for full baseline (double).

Usage:
  julia make_computational_table.jl
"""

using Printf

# ── network metadata (|V| includes s,t; |A| excludes return arc) ──
const NETWORK_ORDER = ["grid5x5", "abilene", "nobel_us", "sioux_falls", "polska"]
const NETWORK_DISPLAY = Dict(
    "grid5x5"     => "Grid \$5\\times5\$",
    "abilene"     => "Abilene",
    "nobel_us"    => "Nobel-US",
    "sioux_falls" => "Sioux Falls",
    "polska"      => "Polska",
)
const NETWORK_SIZE = Dict(
    "grid5x5"     => (27, 47),
    "abilene"     => (12, 30),
    "nobel_us"    => (14, 38),
    "sioux_falls" => (24, 76),
    "polska"      => (12, 36),
)

# ── parse one log file → (Z₀, iters, time) or nothing ──
function parse_log(path::String)
    isfile(path) || return nothing
    z0, iters, wt = NaN, -1, NaN
    max_iter_limit = 1000  # from run scripts
    for line in eachline(path)
        m = match(r"Z₀=([\d.]+),\s*iters=(\d+),\s*time=([\d.]+)s", line)
        if m !== nothing
            z0    = parse(Float64, m.captures[1])
            iters = parse(Int, m.captures[2])
            wt    = parse(Float64, m.captures[3])
        end
    end
    iters < 0 && return nothing
    hit_limit = (iters >= max_iter_limit)
    return (z0=z0, iters=iters, time=wt, hit_limit=hit_limit)
end

# ── parse eps/beta from folder name "eps_0p2_beta_0p4" ──
function parse_eps_beta(folder_name::String)
    m = match(r"eps_([\dp]+)_beta_([\dp]+)", folder_name)
    m === nothing && return nothing
    eps_str  = replace(m.captures[1], "p" => ".")
    beta_str = replace(m.captures[2], "p" => ".")
    return (eps=parse(Float64, eps_str), beta=parse(Float64, beta_str))
end

# ── scan all available (eps, beta) combos from computational/ ──
function scan_combos(base_dir::String)
    comp_dir = joinpath(base_dir, "computational")
    isdir(comp_dir) || error("No computational/ directory found at $comp_dir")
    combos = Tuple{Float64,Float64}[]
    for d in readdir(comp_dir)
        isdir(joinpath(comp_dir, d)) || continue
        eb = parse_eps_beta(d)
        eb === nothing && continue
        push!(combos, (eb.eps, eb.beta))
    end
    sort!(combos; by=x->(x[2], x[1]))  # sort by beta first, then eps
    return combos
end

# ── thousand-separator comma ──
function add_commas(s::String)
    # split at decimal point if present
    parts = split(s, ".")
    int_part = parts[1]
    # insert commas from right
    n = length(int_part)
    result = ""
    for (i, c) in enumerate(int_part)
        result *= c
        remaining = n - i
        if remaining > 0 && remaining % 3 == 0
            result *= ","
        end
    end
    length(parts) == 2 ? result * "." * parts[2] : result
end

# ── format time ──
function fmt_time(t::Float64)
    if t < 100
        return @sprintf("%.1f", t)
    else
        return add_commas(@sprintf("%.0f", t))
    end
end

# ── main ──
function make_table()
    base_dir = joinpath(@__DIR__, "logs", "factor")
    combos = scan_combos(base_dir)

    if isempty(combos)
        println("No (eps, beta) combos found in computational/")
        return
    end

    println("Found combos: ", join(["(ε=$(c[1]), β=$(c[2]))" for c in combos], ", "))

    # ── collect data ──
    # data[(eps,beta)][network] = (naive=..., mb_only=..., full=...)
    data = Dict{Tuple{Float64,Float64}, Dict{String, NamedTuple}}()

    for (eps, beta) in combos
        eps_str  = replace(@sprintf("%.2g", eps), "." => "p")
        beta_str = replace(@sprintf("%.2g", beta), "." => "p")
        folder_name = "eps_$(eps_str)_beta_$(beta_str)"

        comp_folder = joinpath(base_dir, "computational", folder_name)
        full_folder = joinpath(base_dir, folder_name)

        net_data = Dict{String, NamedTuple}()
        for net_name in NETWORK_ORDER
            naive   = parse_log(joinpath(comp_folder, "$(net_name)_double_naive.log"))
            mb_only = parse_log(joinpath(comp_folder, "$(net_name)_double_mb_only.log"))
            full    = parse_log(joinpath(full_folder,  "$(net_name)_double.log"))
            net_data[net_name] = (naive=naive, mb_only=mb_only, full=full)
        end
        data[(eps, beta)] = net_data
    end

    # ── generate LaTeX ──
    L(s) = s  # identity, just for readability
    lines = String[]
    push!(lines, L("\\begin{table}[t]"))
    push!(lines, L("\\centering"))
    push!(lines, L("\\caption{Computational performance of algorithmic enhancements.}"))
    push!(lines, L("\\label{tab:computational}"))
    push!(lines, L("\\small"))
    push!(lines, L("\\begin{tabular}{cl c rr rr rr}"))
    push!(lines, L("\\toprule"))
    push!(lines, L(" & & & \\multicolumn{2}{c}{\\textbf{Plain BD}} & \\multicolumn{2}{c}{\\textbf{Enhanced BD}} & \\multicolumn{2}{c}{\\textbf{Enhanced BD + VIs}} \\\\"))
    push!(lines, L("\\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}"))
    push!(lines, L("\$\\beta\$ & Network & \$\\varepsilon\$ & Time\\,(s) & Iter & Time\\,(s) & Iter & Time\\,(s) & Iter \\\\"))
    push!(lines, L("\\midrule"))

    for (ci, (eps, beta)) in enumerate(combos)
        net_data = data[(eps, beta)]

        for (ni, net_name) in enumerate(NETWORK_ORDER)
            d = net_data[net_name]
            nV, nA = NETWORK_SIZE[net_name]
            net_display = NETWORK_DISPLAY[net_name]

            # β column: only on first row of this combo group
            beta_col = if ni == 1
                nrows = length(NETWORK_ORDER)
                @sprintf("\\multirow{%d}{*}{\$%.1f\$}", nrows, beta)
            else
                ""
            end

            # network column with size info
            net_col = @sprintf("%s {\\scriptsize(\$|\\mathcal{V}|\\!=\\!%d,\\, |\\mathcal{A}|\\!=\\!%d\$)}", net_display, nV, nA)

            # eps column
            eps_col = @sprintf("\$%.1f\$", eps)

            # format each method's results
            function fmt_method(r)
                r === nothing && return ("--", "--")
                t_str = fmt_time(r.time)
                i_str = add_commas(string(r.iters))
                r.hit_limit && (i_str = i_str * "\$^\\dagger\$")
                return (t_str, i_str)
            end

            naive_t, naive_i = fmt_method(d.naive)
            mb_t, mb_i       = fmt_method(d.mb_only)
            full_t, full_i   = fmt_method(d.full)

            row = "$beta_col & $net_col & $eps_col & $naive_t & $naive_i & $mb_t & $mb_i & $full_t & $full_i \\\\"
            push!(lines, row)
        end

        # midrule between combo groups (but not after the last one)
        if ci < length(combos)
            push!(lines, L("\\midrule"))
        end
    end

    push!(lines, L("\\bottomrule"))
    push!(lines, L("\\end{tabular}"))
    push!(lines, L("\\vspace{2pt}"))
    push!(lines, L("{\\footnotesize \$^\\dagger\$\\,Maximum iteration limit reached (not converged).}"))
    push!(lines, L("\\end{table}"))

    tex = join(lines, "\n")

    # ── write to file ──
    out_path = joinpath(@__DIR__, "tables", "table_computational.tex")
    open(out_path, "w") do io
        write(io, tex)
    end
    println("\nSaved: $out_path")
    println()
    println(tex)
end

function make_geomean_table()
    base_dir = joinpath(@__DIR__, "logs", "factor")
    combos = scan_combos(base_dir)
    isempty(combos) && return

    # ── collect data (same as make_table) ──
    data = Dict{Tuple{Float64,Float64}, Dict{String, NamedTuple}}()
    for (eps, beta) in combos
        eps_str  = replace(@sprintf("%.2g", eps), "." => "p")
        beta_str = replace(@sprintf("%.2g", beta), "." => "p")
        folder_name = "eps_$(eps_str)_beta_$(beta_str)"
        comp_folder = joinpath(base_dir, "computational", folder_name)
        full_folder = joinpath(base_dir, folder_name)
        net_data = Dict{String, NamedTuple}()
        for net_name in NETWORK_ORDER
            naive   = parse_log(joinpath(comp_folder, "$(net_name)_double_naive.log"))
            mb_only = parse_log(joinpath(comp_folder, "$(net_name)_double_mb_only.log"))
            full    = parse_log(joinpath(full_folder,  "$(net_name)_double.log"))
            net_data[net_name] = (naive=naive, mb_only=mb_only, full=full)
        end
        data[(eps, beta)] = net_data
    end

    # ── geometric mean ──
    function geomean(vals)
        isempty(vals) && return NaN
        return exp(sum(log.(vals)) / length(vals))
    end

    L(s) = s
    lines = String[]
    push!(lines, L("\\begin{table}[t]"))
    push!(lines, L("\\centering"))
    push!(lines, L("\\caption{Geometric mean of computational metrics across \$(\\beta, \\varepsilon)\$ configurations.}"))
    push!(lines, L("\\label{tab:computational_geom}"))
    push!(lines, L("\\small"))
    push!(lines, L("\\begin{tabular}{l rr rr rr}"))
    push!(lines, L("\\toprule"))
    push!(lines, L(" & \\multicolumn{2}{c}{\\textbf{Plain BD}} & \\multicolumn{2}{c}{\\textbf{Enhanced BD}} & \\multicolumn{2}{c}{\\textbf{Enhanced BD + VIs}} \\\\"))
    push!(lines, L("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}"))
    push!(lines, L("Network & Time\\,(s) & Iter & Time\\,(s) & Iter & Time\\,(s) & Iter \\\\"))
    push!(lines, L("\\midrule"))

    for (ni, net_name) in enumerate(NETWORK_ORDER)
        nV, nA = NETWORK_SIZE[net_name]
        net_display = NETWORK_DISPLAY[net_name]

        naive_times, naive_iters = Float64[], Float64[]
        mb_times, mb_iters = Float64[], Float64[]
        full_times, full_iters = Float64[], Float64[]

        for (eps, beta) in combos
            d = data[(eps, beta)][net_name]
            if d.naive !== nothing
                push!(naive_times, d.naive.time); push!(naive_iters, Float64(d.naive.iters))
            end
            if d.mb_only !== nothing
                push!(mb_times, d.mb_only.time); push!(mb_iters, Float64(d.mb_only.iters))
            end
            if d.full !== nothing
                push!(full_times, d.full.time); push!(full_iters, Float64(d.full.iters))
            end
        end

        net_col = @sprintf("%s {\\scriptsize(\$|\\mathcal{V}|\\!=\\!%d,\\, |\\mathcal{A}|\\!=\\!%d\$)}", net_display, nV, nA)

        function fmt_geom(times, iters)
            isempty(times) && return ("--", "--")
            return (fmt_time(geomean(times)), add_commas(@sprintf("%.0f", geomean(iters))))
        end

        nt, ni_s = fmt_geom(naive_times, naive_iters)
        mt, mi = fmt_geom(mb_times, mb_iters)
        ft, fi = fmt_geom(full_times, full_iters)

        push!(lines, "$net_col & $nt & $ni_s & $mt & $mi & $ft & $fi \\\\")
    end

    push!(lines, L("\\bottomrule"))
    push!(lines, L("\\end{tabular}"))
    push!(lines, L("\\end{table}"))

    tex = join(lines, "\n")
    out_path = joinpath(@__DIR__, "tables", "table_computational_geom.tex")
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        write(io, tex)
    end
    println("\nSaved: $out_path")
    println()
    println(tex)
end

make_table()
make_geomean_table()
