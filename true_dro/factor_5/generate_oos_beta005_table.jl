# beta_risk=0.05에서 DRO solution이 다른 네트워크(Polska, Abilene)만 대상으로
# JLS → CSV + LaTeX summary table 생성
using Serialization, Statistics, Printf, DataFrames, CSV

base_dir = joinpath(@__DIR__, "plots", "test_seed43", "beta_0p05_diff")
out_dir = joinpath(@__DIR__, "tables")

networks = [
    ("polska",  "Polska",  0.2),
    ("abilene", "Abilene", 0.2),
]

sol_keys = ["nominal", "single_l", "double"]
sol_display = ["Nominal SP", "Partial DR", "Full DR"]

# ── 1. JLS → CSV ──
rows = []
for (net, label, eps) in networks
    jls_path = joinpath(base_dir, "$(net)_factor_phaseAB.jls")
    data = deserialize(jls_path)
    β_risk = data[:β_risk]
    sol_labels = data[:sol_labels]

    for β_dir in sort(collect(keys(data[:results])))
        phases = data[:results][β_dir]
        for pkey in sort(collect(keys(phases)))
            costs = phases[pkey]
            # phase_type
            phase_type = startswith(pkey, "β") ? "A" : "B"
            phase_name = if phase_type == "A"
                @sprintf("gamma=%.1f", β_dir)
            else
                pkey  # e.g. "ζ=0.5"
            end

            nom = costs[sol_labels[1]]
            par = costs[sol_labels[2]]
            ful = costs[sol_labels[3]]

            push!(rows, (
                network=net, beta_risk=β_risk, eps=eps, beta_dir=β_dir,
                phase=phase_name, phase_type=phase_type,
                nom_q05=quantile(nom, 0.05), nom_q95=quantile(nom, 0.95),
                nom_interval=quantile(nom, 0.95)-quantile(nom, 0.05),
                nom_mean=mean(nom),
                partial_q05=quantile(par, 0.05), partial_q95=quantile(par, 0.95),
                partial_interval=quantile(par, 0.95)-quantile(par, 0.05),
                partial_mean=mean(par),
                full_q05=quantile(ful, 0.05), full_q95=quantile(ful, 0.95),
                full_interval=quantile(ful, 0.95)-quantile(ful, 0.05),
                full_mean=mean(ful),
            ))
        end
    end
end

df = DataFrame(rows)
csv_path = joinpath(out_dir, "oos_test_beta005.csv")
CSV.write(csv_path, df)
println("CSV written: $csv_path")

# ── 2. LaTeX summary table ──
fmt2(x) = @sprintf("%.2f", x)

function make_row(sub_df, label, eps)
    nom_q05 = mean(sub_df.nom_q05)
    nom_q95 = mean(sub_df.nom_q95)
    nom_int = mean(sub_df.nom_interval)
    par_q05 = mean(sub_df.partial_q05)
    par_q95 = mean(sub_df.partial_q95)
    par_int = mean(sub_df.partial_interval)
    ful_q05 = mean(sub_df.full_q05)
    ful_q95 = mean(sub_df.full_q95)
    ful_int = mean(sub_df.full_interval)

    ints = [nom_int, par_int, ful_int]
    min_int = minimum(ints)
    q95s = [nom_q95, par_q95, ful_q95]
    min_q95 = minimum(q95s)

    function fmt_bracket(q05, q95, is_min)
        q95_s = is_min ? "\\mathbf{$(fmt2(q95))}" : fmt2(q95)
        "\$[$(fmt2(q05)), $q95_s]\$"
    end
    function fmt_int(val, is_min)
        s = fmt2(val)
        is_min ? "\$\\mathbf{$s}\$" : "\$$s\$"
    end

    eps_str = "\$$(fmt2(eps))\$"
    nb = fmt_bracket(nom_q05, nom_q95, nom_q95 ≈ min_q95)
    pb = fmt_bracket(par_q05, par_q95, par_q95 ≈ min_q95)
    fb = fmt_bracket(ful_q05, ful_q95, ful_q95 ≈ min_q95)
    ni = fmt_int(nom_int, nom_int ≈ min_int)
    pi = fmt_int(par_int, par_int ≈ min_int)
    fi = fmt_int(ful_int, ful_int ≈ min_int)

    return "$label & $eps_str & $nb & $ni & $pb & $pi & $fb & $fi \\\\"
end

rows_tex = String[]
for (net, label, eps) in networks
    sub = filter(row -> row.network == net, df)
    push!(rows_tex, make_row(sub, label, eps))
end

tex = """
\\begin{table}[htbp]
\\centering
\\caption{Out-of-sample summary (\$\\beta=0.05\$): networks where DRO solution differs from nominal}
\\label{tab:oos-summary-beta005}
\\small
\\setlength{\\tabcolsep}{6pt}
\\begin{tabular}{l c cc cc cc}
\\toprule
& & \\multicolumn{2}{c}{\\textbf{Nominal SP}} & \\multicolumn{2}{c}{\\textbf{Partial DR}} & \\multicolumn{2}{c}{\\textbf{Full DR}} \\\\
\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}
Network & \$\\varepsilon^{*}\$ & \$[q_{05}, q_{95}]\$ & \$q_{95} \\!-\\! q_{05}\$ & \$[q_{05}, q_{95}]\$ & \$q_{95} \\!-\\! q_{05}\$ & \$[q_{05}, q_{95}]\$ & \$q_{95} \\!-\\! q_{05}\$ \\\\
\\midrule
$(join(rows_tex, "\n"))
\\bottomrule
\\end{tabular}
\\end{table}
"""

tex_path = joinpath(out_dir, "oos_summary_beta005.tex")
open(tex_path, "w") do io; write(io, tex); end
println("LaTeX written: $tex_path")
println(tex)
