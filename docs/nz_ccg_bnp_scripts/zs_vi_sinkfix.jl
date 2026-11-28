# zs_vi_sinkfix.jl (2026-11-28 아침): 저장소 true_dro_mincut_vi.jl 의 add_phase1_mincut_vi! 를 실행 사본에서만 덮어쓰는 판.
#   저장소 판은 sink 를 Ny 의 마지막 행 (nv1) 으로 가정한다 (τ[nv1, j] == 1). grid5x5·Sioux Falls 는 t 가 마지막 행이라 맞지만
#   Polska·Abilene·Nobel-US 는 t 가 9 행 (nv1 = 11, 11, 13) 이라 VI 가 t 가 아닌 마디까지의 최소 cut 을 하한으로 넣는다 (Polska s4 {7,18}: 우변 28.07 > V 21.38).
#   여기서는 sink 행을 더미 호 (t → s) 의 열 Nts 에서 읽는다 (Nts 의 비영 원소는 t 행의 +1 하나). 나머지 식은 저장소 판과 같다.
#   저장소 파일을 include 한 뒤 이 파일을 include 해야 덮어쓴다.
function add_phase1_mincut_vi!(omp_model, omp_vars, td::TrueDROData)
    K = td.num_arcs
    S = td.S
    nv1 = td.nv1
    nzr = findall(!iszero, td.Nts)
    (length(nzr) == 1 && td.Nts[nzr[1]] == 1.0) || error("add_phase1_mincut_vi! (sink 수정판): Nts 의 비영 원소가 +1 하나가 아님 ($(nzr), $(td.Nts[nzr]))")
    tsk = nzr[1]
    arc_topo = extract_arc_topology(td.Ny, nv1)
    x = omp_vars[:x]
    t_0 = omp_vars[:t_0]
    δ = @variable(omp_model, [1:K, 1:S], lower_bound=0, upper_bound=1, base_name="mc_p1_δ")
    τ = @variable(omp_model, [1:nv1, 1:S], lower_bound=0, upper_bound=1, base_name="mc_p1_τ")
    w = @variable(omp_model, [1:K, 1:S], lower_bound=0, upper_bound=1, base_name="mc_p1_w")
    for j in 1:S
        @constraint(omp_model, τ[tsk, j] == 1)                       # DC-2: τ_t = 1 (t 의 행)
        for k in 1:K
            τ_head = arc_topo.head[k] == 0 ? 0.0 : τ[arc_topo.head[k], j]
            τ_tail = arc_topo.tail[k] == 0 ? 0.0 : τ[arc_topo.tail[k], j]
            @constraint(omp_model, δ[k, j] >= τ_head - τ_tail)       # DC-1
            @constraint(omp_model, w[k, j] <= x[k])                   # MC-1
            @constraint(omp_model, w[k, j] <= δ[k, j])                # MC-2
            @constraint(omp_model, w[k, j] >= δ[k, j] - (1 - x[k]))   # MC-3
        end
    end
    link_expr = sum(td.q_hat[j] * sum(td.xi_bar[k, j] * (δ[k, j] - td.v[k, j] * w[k, j]) for k in 1:K) for j in 1:S)
    @constraint(omp_model, t_0 >= link_expr)
    return nothing
end
