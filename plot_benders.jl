using Plots
function plot_benders_convergence(result::Dict)
    # Ensure Plots is loaded, and use Plots

    past_obj = result[:past_obj]
    past_subprob_obj = result[:past_subprob_obj]

    n_obj = length(past_obj)
    n_subprob = length(past_subprob_obj)
    step_obj = max(1, Int(round(n_obj / 10)))
    step_sub = max(1, Int(round(n_subprob / 10)))
    marker_idx_obj = 1:step_obj:n_obj
    marker_idx_sub = 1:step_sub:n_subprob

    # 실제로 플롯에서 marker_idx_obj, marker_idx_sub를 쓰려면
    # 주곡선은 선만 그리고, marker만 따로 플롯
    # (즉, markersize/모양은 아래서 'marker' 없애고, 따로 scatter로!)

    global_obj_iter = 1:n_obj
    global_sub_iter = 1:n_subprob

    # markers만 보여줄 데이터
    obj_marker_iter = global_obj_iter[marker_idx_obj]
    obj_marker_vals = past_obj[marker_idx_obj]

    sub_marker_iter = global_sub_iter[marker_idx_sub]
    sub_marker_vals = past_subprob_obj[marker_idx_sub]

    plt = plot(1:length(past_obj), past_obj; 
        label="Outer Problem Objective", 
        color=:blue, 
        linestyle=:solid,
        linewidth=1,
        xlabel="Iteration", 
        ylabel="Objective Value", 
        title="Convergence Plot"
    )
    scatter!(plt, obj_marker_iter, obj_marker_vals;
        label="",
        color=:blue,
        marker=:circle,
        markersize=3
    )
    plot!(plt, 1:length(past_subprob_obj), past_subprob_obj; 
        label="Inner Problem Objective", 
        color=:red, 
        linestyle=:solid
    )
    scatter!(plt, sub_marker_iter, sub_marker_vals;
        label="",
        color=:red,
        marker=:x,
        markersize=3
    )

    display(plt)
end