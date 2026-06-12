using CSV, DataFrames, Printf, Statistics

# ─── 공통 설정 ───
const NETWORKS_ORDER = ["grid5x5", "polska", "sioux_falls", "abilene", "nobel_us"]
const NETWORK_LABELS = Dict(
    "grid5x5"     => raw"Grid $5 \times 5$",
    "polska"      => "Polska",
    "sioux_falls" => "Sioux Falls",
    "abilene"     => "Abilene",
    "nobel_us"    => "Nobel-US"
)
const NETWORK_EPS = Dict(
    "grid5x5"     => 0.1,
    "polska"      => 0.1,
    "sioux_falls" => 0.1,
    "abilene"     => 0.2,
    "nobel_us"    => 0.2
)

# ─── 한 행 포맷 (네트워크 데이터 → LaTeX row) ───
function format_row(sub::AbstractDataFrame, net::String)
    eps_val = NETWORK_EPS[net]

    nom_q05_avg = mean(sub.nom_q05)
    nom_q95_avg = mean(sub.nom_q95)
    nom_int_avg = mean(sub.nom_interval)

    partial_q05_avg = mean(sub.partial_q05)
    partial_q95_avg = mean(sub.partial_q95)
    partial_int_avg = mean(sub.partial_interval)

    full_q05_avg = mean(sub.full_q05)
    full_q95_avg = mean(sub.full_q95)
    full_int_avg = mean(sub.full_interval)

    # bold: 최소 interval, 최소 q95
    intervals = [nom_int_avg, partial_int_avg, full_int_avg]
    min_int = minimum(intervals)
    q95s = [nom_q95_avg, partial_q95_avg, full_q95_avg]
    min_q95 = minimum(q95s)

    fmt2(x) = @sprintf("%.2f", x)

    function fmt_bracket(q05, q95, q95_is_min)
        q05_s = fmt2(q05)
        q95_s = q95_is_min ? "\\mathbf{$(fmt2(q95))}" : fmt2(q95)
        "\$[$q05_s, $q95_s]\$"
    end
    function fmt_int(val, is_min)
        s = fmt2(val)
        is_min ? "\$\\mathbf{$s}\$" : "\$$s\$"
    end

    eps_str = eps_val == 0.1 ? "\$0.1\$" : "\$0.2\$"

    nom_bracket = fmt_bracket(nom_q05_avg, nom_q95_avg, nom_q95_avg ≈ min_q95)
    partial_bracket = fmt_bracket(partial_q05_avg, partial_q95_avg, partial_q95_avg ≈ min_q95)
    full_bracket = fmt_bracket(full_q05_avg, full_q95_avg, full_q95_avg ≈ min_q95)

    nom_int_str = fmt_int(nom_int_avg, nom_int_avg ≈ min_int)
    partial_int_str = fmt_int(partial_int_avg, partial_int_avg ≈ min_int)
    full_int_str = fmt_int(full_int_avg, full_int_avg ≈ min_int)

    label = NETWORK_LABELS[net]
    return "$label & $eps_str & $nom_bracket & $nom_int_str & $partial_bracket & $partial_int_str & $full_bracket & $full_int_str \\\\"
end

# ─── LaTeX table 조립 ───
function build_table(rows_tex::Vector{String}; caption::String, label::String)
    return """
\\begin{table}[htbp]
\\centering
\\caption{$caption}
\\label{$label}
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
end

# ─── (1) 전체 평균 summary ───
function generate_oos_summary_table(;
    input_csv="tables/oos_test.csv",
    output_tex="tables/oos_summary_table.tex"
)
    df = CSV.read(input_csv, DataFrame)
    rows_tex = String[]

    for net in NETWORKS_ORDER
        sub = filter(row -> row.network == net, df)
        nrow(sub) == 0 && continue
        push!(rows_tex, format_row(sub, net))
    end

    tex = build_table(rows_tex;
        caption="Out-of-sample summary: average 90\\% quantile interval across all interdiction phases",
        label="tab:oos-summary")

    open(output_tex, "w") do io; write(io, tex); end
    println("Written to $output_tex")
    println(tex)
end

# ─── (2) Phase별 5개 테이블 ───
function generate_oos_per_phase_tables(;
    input_csv="tables/oos_test.csv",
    output_dir="tables"
)
    df = CSV.read(input_csv, DataFrame)

    # 5개 phase 슬롯: A(gamma), B(zeta=0.5), B(zeta=1.0), B(zeta=2.0), B(zeta=5.0)
    phase_filters = [
        ("A",     row -> row.phase_type == "A",               "gamma",   "Phase A (\$\\gamma\$)"),
        ("B_z05", row -> row.phase_type == "B" && contains(row.phase, "zeta=0.5"), "zeta05",  "Phase B (\$\\zeta=0.5\$)"),
        ("B_z10", row -> row.phase_type == "B" && contains(row.phase, "zeta=1.0"), "zeta10",  "Phase B (\$\\zeta=1.0\$)"),
        ("B_z20", row -> row.phase_type == "B" && contains(row.phase, "zeta=2.0"), "zeta20",  "Phase B (\$\\zeta=2.0\$)"),
        ("B_z50", row -> row.phase_type == "B" && contains(row.phase, "zeta=5.0"), "zeta50",  "Phase B (\$\\zeta=5.0\$)"),
    ]

    all_tex = String[]
    for (suffix, filt, label_short, phase_desc) in phase_filters
        rows_tex = String[]
        for net in NETWORKS_ORDER
            sub = filter(row -> row.network == net && filt(row), df)
            nrow(sub) == 0 && continue
            push!(rows_tex, format_row(sub, net))
        end

        tex = build_table(rows_tex;
            caption="Out-of-sample 90\\% quantile interval — $phase_desc (averaged over \$\\beta\$-directions)",
            label="tab:oos-$label_short")
        push!(all_tex, tex)
    end

    combined = join(all_tex, "\n\n")
    outfile = joinpath(output_dir, "oos_summary_per_phase.tex")
    open(outfile, "w") do io; write(io, combined); end
    println("Written to $outfile")
    println(combined)
end

# ─── (3) Validation: eps 선택 당위 테이블 (beta=0.4, phase A only) ───
function generate_eps_selection_table(;
    input_csv="tables/oos_validation.csv",
    output_tex="tables/oos_eps_selection.tex"
)
    df = CSV.read(input_csv, DataFrame)
    df04 = filter(row -> row.beta_risk == 0.40 && row.phase_type == "A", df)

    # 선택된 eps
    selected_eps = Dict(
        "grid5x5"     => 0.1,
        "polska"      => 0.1,
        "sioux_falls" => 0.1,
        "abilene"     => 0.2,
        "nobel_us"    => 0.1
    )

    eps_candidates = [0.1, 0.2, 0.5]
    fmt2(x) = @sprintf("%.2f", x)

    rows_tex = String[]
    for net in NETWORKS_ORDER
        label = NETWORK_LABELS[net]
        sel = selected_eps[net]

        parts = String[]
        for eps in eps_candidates
            sub = filter(row -> row.network == net && row.eps == eps, df04)
            nrow(sub) == 0 && continue

            q95_val = mean(sub.full_q95)
            int_val = mean(sub.full_interval)

            # 선택된 eps는 bold
            if eps ≈ sel
                push!(parts, "\$\\mathbf{$(fmt2(q95_val))}\$ & \$\\mathbf{$(fmt2(int_val))}\$")
            else
                push!(parts, "\$$(fmt2(q95_val))\$ & \$$(fmt2(int_val))\$")
            end
        end

        sel_str = sel == 0.1 ? "\$0.1\$" : (sel == 0.2 ? "\$0.2\$" : "\$0.5\$")
        row_str = "$label & $(join(parts, " & ")) & $sel_str \\\\"
        push!(rows_tex, row_str)
    end

    tex = """
\\begin{table}[htbp]
\\centering
\\caption{Validation-set radius selection (\$\\beta=0.4\$, Phase~A only): Full DR 95th-percentile and interval width, averaged over \$\\gamma\\in\\{0.1,0.3,0.5,1.0\\}\$}
\\label{tab:eps-selection}
\\small
\\setlength{\\tabcolsep}{5pt}
\\begin{tabular}{l cc cc cc c}
\\toprule
& \\multicolumn{2}{c}{\$\\varepsilon=0.1\$} & \\multicolumn{2}{c}{\$\\varepsilon=0.2\$} & \\multicolumn{2}{c}{\$\\varepsilon=0.5\$} & \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}
Network & \$q_{95}\$ & Interval & \$q_{95}\$ & Interval & \$q_{95}\$ & Interval & \$\\varepsilon^{*}\$ \\\\
\\midrule
$(join(rows_tex, "\n"))
\\bottomrule
\\end{tabular}
\\end{table}
"""

    open(output_tex, "w") do io; write(io, tex); end
    println("Written to $output_tex")
    println(tex)
end

# ─── 실행 ───
println("="^60)
println("  전체 평균 summary")
println("="^60)
generate_oos_summary_table()

println("\n\n")
println("="^60)
println("  Phase별 5개 테이블")
println("="^60)
generate_oos_per_phase_tables()

println("\n\n")
println("="^60)
println("  Epsilon 선택 테이블 (validation)")
println("="^60)
generate_eps_selection_table()
