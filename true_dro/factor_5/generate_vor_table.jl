"""
generate_vor_table.jl — compute_vor/vof 로그에서 LaTeX 테이블 생성
  음수 → 0.00 clamp, 각 (network, β) row에서 ε별 최대값 bold

Usage:
  julia generate_vor_table.jl                          # combined VoR+VoF
  julia generate_vor_table.jl <vor_log> <vof_log>      # explicit paths
"""

using Printf

vor_log = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "logs", "compute_vor.log")
vof_log = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "logs", "compute_vof.log")

# ── parse log → lookup dict ──
function parse_value_log(path)
    isfile(path) || error("Log not found: $path")
    lookup = Dict{Tuple{String,Float64,Float64}, Float64}()
    for line in readlines(path)
        m = match(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+\[[\d,]+\]\s+[\d.]+\s+(?:[\d.]+|FAIL)\s+([-\d.]+|N/A)", line)
        m === nothing && continue
        net = m.captures[1]
        eps = parse(Float64, m.captures[2])
        beta = parse(Float64, m.captures[3])
        v_str = m.captures[4]
        v = v_str == "N/A" ? 0.0 : max(0.0, parse(Float64, v_str))
        lookup[(net, beta, eps)] = v
    end
    @printf("Parsed %d entries from %s\n", length(lookup), path)
    return lookup
end

vor_lookup = parse_value_log(vor_log)
vof_lookup = parse_value_log(vof_log)

# ── constants ──
networks_order = ["grid5x5", "polska", "abilene", "nobel_us", "sioux_falls"]
display_names = Dict(
    "grid5x5" => "Grid 5\$\\times\$5",
    "polska" => "Polska",
    "abilene" => "Abilene",
    "nobel_us" => "Nobel-US",
    "sioux_falls" => "Sioux Falls",
)
betas = [0.0, 0.05, 0.4, 0.7]
epsilons = [0.1, 0.2, 0.5]

# ── helper: format 3 cells with row-wise bold ──
function fmt_cells(lookup, net, beta)
    vals = [get(lookup, (net, beta, eps), 0.0) for eps in epsilons]
    max_val = maximum(vals)
    cells = String[]
    for v in vals
        s = @sprintf("%.2f", v)
        if v == max_val && max_val > 0.0
            push!(cells, "\\textbf{$s}")
        else
            push!(cells, s)
        end
    end
    return cells
end

# ── generate combined LaTeX ──
lines = String[]
push!(lines, "\\begin{table}[htbp]")
push!(lines, "\\centering")
push!(lines, "\\caption{Value of Robustness and Value of Follower Ambiguity (\\%) by network, \$\\beta\$, and \$\\varepsilon\$.}")
push!(lines, "\\label{tab:vor_vof}")
push!(lines, "\\small")
push!(lines, "\\begin{tabular}{ll rrr rrr}")
push!(lines, "\\toprule")
push!(lines, " & & \\multicolumn{3}{c}{VoR (\\%)} & \\multicolumn{3}{c}{VoF (\\%)} \\\\")
push!(lines, "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}")
push!(lines, "Network & \$\\beta\$ & \$\\varepsilon\\!=\\!0.1\$ & \$\\varepsilon\\!=\\!0.2\$ & \$\\varepsilon\\!=\\!0.5\$ & \$\\varepsilon\\!=\\!0.1\$ & \$\\varepsilon\\!=\\!0.2\$ & \$\\varepsilon\\!=\\!0.5\$ \\\\")
push!(lines, "\\midrule")

for (ni, net) in enumerate(networks_order)
    for (bi, beta) in enumerate(betas)
        vor_cells = fmt_cells(vor_lookup, net, beta)
        vof_cells = fmt_cells(vof_lookup, net, beta)

        prefix = bi == 1 ? @sprintf("\\multirow{%d}{*}{%s}", length(betas), display_names[net]) : ""
        beta_str = @sprintf("%.2f", beta)

        push!(lines, " $prefix & $beta_str & $(vor_cells[1]) & $(vor_cells[2]) & $(vor_cells[3]) & $(vof_cells[1]) & $(vof_cells[2]) & $(vof_cells[3]) \\\\")
    end
    if ni < length(networks_order)
        push!(lines, "\\midrule")
    end
end

push!(lines, "\\bottomrule")
push!(lines, "\\end{tabular}")
push!(lines, "\\end{table}")

tex_str = join(lines, "\n")

out_path = joinpath(@__DIR__, "tables", "vor_vof_table.tex")
mkpath(dirname(out_path))
open(out_path, "w") do io
    print(io, tex_str)
end
@printf("Saved: %s\n", out_path)
println("\n" * tex_str)
