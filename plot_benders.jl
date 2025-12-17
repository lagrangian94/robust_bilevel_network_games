using Plots
function plot_benders_convergence(result::Dict)
    # Ensure Plots is loaded, and use Plots
    iter_when_center_changed = result[:tr_info][:major_iter]
    iter_when_region_expanded = result[:tr_info][:bin_B_steps]

    past_lower_bound = result[:past_lower_bound]
    past_upper_bound = result[:past_upper_bound]

    n_lower_bound = length(past_lower_bound)
    n_upper_bound = length(past_upper_bound)
    step_lower_bound = max(1, Int(round(n_lower_bound / 10)))
    step_upper_bound = max(1, Int(round(n_upper_bound / 10)))
    marker_idx_lower_bound = 1:step_lower_bound:n_lower_bound
    marker_idx_upper_bound = 1:step_upper_bound:n_upper_bound

    # 실제로 플롯에서 marker_idx_obj, marker_idx_sub를 쓰려면
    # 주곡선은 선만 그리고, marker만 따로 플롯
    # (즉, markersize/모양은 아래서 'marker' 없애고, 따로 scatter로!)

    global_lower_bound_iter = 1:n_lower_bound
    global_upper_bound_iter = 1:n_upper_bound

    # markers만 보여줄 데이터
    lower_bound_marker_iter = global_lower_bound_iter[marker_idx_lower_bound]
    lower_bound_marker_vals = past_lower_bound[marker_idx_lower_bound]

    upper_bound_marker_iter = global_upper_bound_iter[marker_idx_upper_bound]
    upper_bound_marker_vals = past_upper_bound[marker_idx_upper_bound]

    plt = plot(1:length(past_lower_bound), past_lower_bound; 
        label="(Local)Lower Bound", 
        color=:blue, 
        linestyle=:solid,
        linewidth=1,
        xlabel="Iteration", 
        ylabel="Objective Value", 
        title="Convergence Plot"
    )
    scatter!(plt, lower_bound_marker_iter, lower_bound_marker_vals;
        label="",
        color=:blue,
        marker=:circle,
        markersize=3
    )
    plot!(plt, 1:length(past_upper_bound), past_upper_bound; 
        label="Upper Bound", 
        color=:red, 
        linestyle=:solid
    )
    scatter!(plt, upper_bound_marker_iter, upper_bound_marker_vals;
        label="",
        color=:red,
        marker=:x,
        markersize=3
    )
    # iter_when_center_changed와 iter_when_region_expanded에 대해 marker 추가
    if !isempty(iter_when_center_changed)
        scatter!(
            plt, 
            iter_when_center_changed, 
            past_lower_bound[iter_when_center_changed], 
            marker=:diamond, 
            color=:green, 
            label="center changed",
            markersize=5
        )
    end
    if !isempty(iter_when_region_expanded)
        scatter!(
            plt, 
            iter_when_region_expanded, 
            past_lower_bound[iter_when_region_expanded], 
            marker=:star5, 
            color=:orange, 
            label="region expanded",
            markersize=7
        )
    end
    # "global optimal found"를 marker로 scatter!하지 않고, 해당 x에 세로 실선(vline!)을 그림
    if !(isnothing(result[:iter_when_global_optimal]))
        vline!(
            plt, 
            [result[:iter_when_global_optimal]], 
            color=:purple, 
            linewidth=2, 
            linestyle=:dot,  # 점선으로 더 dash가 많은 스타일
            label="global optimal found"
        )
    end
    display(plt)
end