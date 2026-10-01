"""
generate_oos_table.jl — JLS에서 OOS 통계 CSV 생성
  (1) Validation: plots/eps_*_beta_*/ 폴더들
  (2) Test: plots/test_seed43/ 폴더

  칼럼: network, beta_risk, eps, beta_dir, phase, phase_type,
        nom_q05, nom_q95, nom_interval, nom_wc, nom_mean, nom_median,
        partial_q05, ..., full_q05, ...

Usage:
  julia generate_oos_table.jl
"""

using Serialization, Statistics, Printf

# ── helper ──
function compute_stats(data)
    q05 = quantile(data, 0.05)
    q95 = quantile(data, 0.95)
    return (q05=q05, q95=q95, interval=q95 - q05, wc=q95,
            mean=mean(data), median=median(data))
end

# ── scan folders ──
plots_dir = joinpath(@__DIR__, "plots")
sol_order = ["nominal", "single_l", "double"]

val_folders = String[]
for d in readdir(plots_dir; join=false)
    m = match(r"^eps_(.+)_beta_(.+)$", d)
    m !== nothing && d != "eps_0p2_beta_0p4_single_l" && push!(val_folders, d)
end
sort!(val_folders)

test_folder = "test_seed43"

# ── collect rows from JLS ──
function collect_csv_rows(jls_path)
    d = deserialize(jls_path)
    network = d[:network]
    β_risk = d[:β_risk]
    ε_val = get(d, :ε, NaN)
    sol_labels = d[:sol_labels]
    β_dirs = d[:β_dirs]
    noises = get(d, :noises, [0.5, 1.0, 2.0, 5.0])
    results = d[:results]

    rows = []
    for β_dir in β_dirs
        β_result = results[β_dir]

        first_key = first(keys(β_result))
        if startswith(first_key, "PhaseA") || startswith(first_key, "n=")
            jls_keys = ["PhaseA"; [@sprintf("n=%.1f", n) for n in noises]]
            display_keys = [@sprintf("gamma=%.1f", β_dir); [@sprintf("zeta_dir=%.1f", n) for n in noises]]
        else
            jls_keys = [@sprintf("β=%.1f", β_dir); [@sprintf("ζ=%.1f", n) for n in noises]]
            display_keys = [@sprintf("gamma=%.1f", β_dir); [@sprintf("zeta_dir=%.1f", n) for n in noises]]
        end

        for (ki, jkey) in enumerate(jls_keys)
            !haskey(β_result, jkey) && continue
            phase_dict = β_result[jkey]
            dkey = display_keys[ki]
            ptype = ki == 1 ? "A" : "B"

            stats = Dict{String, Any}()
            for sol in sol_order
                haskey(phase_dict, sol) || continue
                stats[sol] = compute_stats(phase_dict[sol])
            end
            haskey(stats, "nominal") && haskey(stats, "single_l") && haskey(stats, "double") || continue

            n = stats["nominal"]; p = stats["single_l"]; f = stats["double"]
            push!(rows, (
                network=network, beta_risk=β_risk, eps=ε_val,
                beta_dir=β_dir, phase=dkey, phase_type=ptype,
                nom_q05=n.q05, nom_q95=n.q95, nom_interval=n.interval, nom_wc=n.wc, nom_mean=n.mean, nom_median=n.median,
                partial_q05=p.q05, partial_q95=p.q95, partial_interval=p.interval, partial_wc=p.wc, partial_mean=p.mean, partial_median=p.median,
                full_q05=f.q05, full_q95=f.q95, full_interval=f.interval, full_wc=f.wc, full_mean=f.mean, full_median=f.median,
            ))
        end
    end
    return rows
end

function write_csv(rows, out_path)
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        # header
        println(io, "network,beta_risk,eps,beta_dir,phase,phase_type," *
            "nom_q05,nom_q95,nom_interval,nom_wc,nom_mean,nom_median," *
            "partial_q05,partial_q95,partial_interval,partial_wc,partial_mean,partial_median," *
            "full_q05,full_q95,full_interval,full_wc,full_mean,full_median")
        for r in rows
            @printf(io, "%s,%.2f,%.2f,%.1f,%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                r.network, r.beta_risk, r.eps, r.beta_dir, r.phase, r.phase_type,
                r.nom_q05, r.nom_q95, r.nom_interval, r.nom_wc, r.nom_mean, r.nom_median,
                r.partial_q05, r.partial_q95, r.partial_interval, r.partial_wc, r.partial_mean, r.partial_median,
                r.full_q05, r.full_q95, r.full_interval, r.full_wc, r.full_mean, r.full_median)
        end
    end
    @printf("Saved: %s (%d rows)\n", out_path, length(rows))
end

# ── Collect validation ──
val_rows = []
for folder in val_folders
    jls_files = filter(f -> endswith(f, ".jls"), readdir(joinpath(plots_dir, folder)))
    for jf in jls_files
        append!(val_rows, collect_csv_rows(joinpath(plots_dir, folder, jf)))
    end
end

# ── Collect test ──
test_rows = []
test_dir = joinpath(plots_dir, test_folder)
if isdir(test_dir)
    jls_files = filter(f -> endswith(f, ".jls"), readdir(test_dir))
    for jf in jls_files
        append!(test_rows, collect_csv_rows(joinpath(test_dir, jf)))
    end
end

# ── Write CSVs ──
write_csv(val_rows, joinpath(@__DIR__, "tables", "oos_validation.csv"))
write_csv(test_rows, joinpath(@__DIR__, "tables", "oos_test.csv"))

println("\nDone!")
