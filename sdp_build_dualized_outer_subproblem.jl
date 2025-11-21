using JuMP
using LinearAlgebra
using SparseArrays
using Infiltrator
using Pajarito
using Gurobi
using Mosek, MosekTools
using Hypatia, HiGHS
"""
@infiltrate 지점에서 멈추고 변수 확인 가능
@locals 입력하면 모든 변수 확인
@continue 또는 @exit로 계속 진행
"""
# Load network generator
include("network_generator.jl")
using .NetworkGenerator
"""
Build the full 2DRNDP model (14) without COP constraints (14f, 14i)

Arguments:
- network: Network structure from NetworkGenerator
- S: Number of scenarios
- ϕU: Upper bound on interdiction effectiveness
- γ: Interdiction budget
- w: Budget weight parameter
- v: Interdiction effectiveness parameter (used in COP matrix structure)
- uncertainty_set: Dictionary containing uncertainty set
- optimizer: JuMP optimizer (e.g., Gurobi.Optimizer)
- λ,x,h: Given master problem solution

Returns:
- model: JuMP model
- vars: Dictionary containing all decision variables

Note: 
- ν (nu) is a DECISION VARIABLE (appears in objective and constraint 14l)
- v is a PARAMETER (appears in COP matrix Φ - vW)
"""
function build_dualized_outer_subproblem(network, S, ϕU, γ, w, v, uncertainty_set, λ, x, h, ψ0; optimizer=nothing)
    # Extract network dimensions
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    num_interdictable = sum(network.interdictable_arcs)
    
    # Node-arc incidence matrix (excluding source row)
    N = network.N
    N_y = N[:, 1:num_arcs]  # Regular arcs: (|V|-1) × |A|
    N_ts = N[:, end]         # Dummy arc: (|V|-1) × 1
    R, r_dict, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:epsilon]
    xi_bar = Dict(s=>r_dict[s][2:num_arcs+1] for s in 1:S)
    # Dummy arc index (t,s)
    dummy_arc_idx = findfirst(arc -> arc == ("t", "s"), network.arcs)
    E = ones(num_arcs, num_arcs+1) # num_arcs × num_arcs+1 matrix of ones
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]
    # Create model
    model = Model(Mosek.Optimizer)
    if !isnothing(optimizer)
        set_optimizer(model, optimizer)
    end
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    println("Building dualized outer subproblem...")
    println("  Nodes: $num_nodes, Arcs: $num_arcs, Scenarios: $S")
    println("  Interdictable arcs: $num_interdictable")
    println("  Dummy arc index: $dummy_arc_idx")
    println("  Parameters: ϕU = $ϕU, γ = $γ, w = $w, v = $v")
    
    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================
    
    # --- Scalar variables ---
    @variable(model, α[1:num_arcs] >= 0)
    # --- Vector variables ---
    dim_Λhat1_rows = (num_arcs + 1) + (num_nodes - 1) + num_arcs ## equal to dim_Λhat1_rows in full model
    dim_Λhat2_rows = num_arcs ## equal to dim_Λhat2_rows in full model
    dim_Λtilde1_rows = num_arcs+1 + (num_nodes - 1) + num_arcs + num_nodes-1 + num_arcs + num_arcs ## equal to dim_Λtilde1_rows in full model
    dim_Λtilde2_rows = num_arcs ## equal to dim_Λtilde2_rows in full model
    @variable(model, βhat1[s=1:S,1:dim_Λhat1_rows]>=0)
    @variable(model, βhat2[s=1:S,1:dim_Λhat2_rows]>=0)
    @variable(model, βtilde1[s=1:S,1:dim_Λtilde1_rows]>=0)
    @variable(model, βtilde2[s=1:S,1:dim_Λtilde2_rows]>=0)

    βhat1_1 = βhat1[:,1:num_arcs+1]
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    βhat1_2 = βhat1[:,block2_start:block3_start-1]
    βhat1_3 = βhat1[:,block3_start:dim_Λhat1_rows]
    block2_start, block3_start= -1, -1 ## 이후에 다시 쓰이는데 초기화

    #βtilde1 block 분리
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    block4_start = block3_start + num_arcs
    block5_start = block4_start + num_nodes-1
    block6_start = block5_start + num_arcs
    βtilde1_1 = βtilde1[:,1:num_arcs+1] 
    βtilde1_2 = βtilde1[:,block2_start:block3_start-1]
    βtilde1_3 = βtilde1[:,block3_start:block4_start-1]
    βtilde1_4 = βtilde1[:,block4_start:block5_start-1]
    βtilde1_5 = βtilde1[:,block5_start:block6_start-1]
    βtilde1_6 = βtilde1[:,block6_start:dim_Λtilde1_rows]
    block2_start, block3_start, block4_start, block5_start, block6_start= -1, -1, -1, -1, -1 ## 이후에 다시 쓰이는데 초기화
    # check if the blocks are correct (block들 column dimension 합이 dim_Λtilde1_rows와 같은지 확인)
    # @assert sum([size(βtilde1_1,2), size(βtilde1_2,2), size(βtilde1_3,2), size(βtilde1_4,2), size(βtilde1_5,2), size(βtilde1_6,2)]) == dim_Λtilde1_rows
    # --- Matrix variables ---
    @variable(model, Mhat[s=1:S,1:num_arcs+1,1:num_arcs+1])
    @variable(model, Mtilde[s=1:S,1:num_arcs+1,1:num_arcs+1])
    @variable(model, Uhat1[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Uhat2[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Uhat3[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Utilde1[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Utilde2[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Utilde3[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    dim_R_cols = size(R,2)
    @variable(model, Zhat1[s=1:S,1:dim_Λhat1_rows,1:dim_R_cols])
    @variable(model, Zhat2[s=1:S,1:dim_Λhat2_rows,1:dim_R_cols])
    @variable(model, Ztilde1[s=1:S,1:dim_Λtilde1_rows,1:dim_R_cols])
    @variable(model, Ztilde2[s=1:S,1:dim_Λtilde2_rows,1:dim_R_cols])

    # Zhat1도 3개 블록으로 분리, sdp_build_full_model.jl 참고
    Zhat1_1 = Zhat1[:,1:num_arcs+1,:]
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    Zhat1_2 = Zhat1[:,block2_start:block3_start-1,:]
    Zhat1_3 = Zhat1[:,block3_start:dim_Λhat1_rows,:]
    block2_start, block3_start= -1, -1 ## 이후에 다시 쓰이는데 초기화
    # check if the blocks are correct (block들 column dimension 합이 dim_Λhat1_rows와 같은지 확인)
    @assert sum([size(Zhat1_1,2), size(Zhat1_2,2), size(Zhat1_3,2)]) == dim_Λhat1_rows
    # Ztilde1도 6개 블록으로 분리, sdp_build_full_model.jl 참고
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    block4_start = block3_start + num_arcs
    block5_start = block4_start + num_nodes-1
    block6_start = block5_start + num_arcs
    Ztilde1_1 = Ztilde1[:,1:num_arcs+1,:]
    Ztilde1_2 = Ztilde1[:,block2_start:block3_start-1,:]
    Ztilde1_3 = Ztilde1[:,block3_start:block4_start-1,:]
    Ztilde1_4 = Ztilde1[:,block4_start:block5_start-1,:]
    Ztilde1_5 = Ztilde1[:,block5_start:block6_start-1,:]
    Ztilde1_6 = Ztilde1[:,block6_start:dim_Λtilde1_rows,:]
    block2_start, block3_start, block4_start, block5_start, block6_start= -1, -1, -1, -1, -1 ## 이후에 다시 쓰이는데 초기화
    # check if the blocks are correct (block들 column dimension 합이 dim_Λtilde1_rows와 같은지 확인)
    @assert sum([size(Ztilde1_1,2), size(Ztilde1_2,2), size(Ztilde1_3,2), size(Ztilde1_4,2), size(Ztilde1_5,2), size(Ztilde1_6,2)]) == dim_Λtilde1_rows
    #dim_Λhat1_cols = size(R,1)
    @variable(model, Γhat1[s=1:S, 1:dim_Λhat1_rows, 1:size(R,1)])
    @variable(model, Γhat2[s=1:S, 1:dim_Λhat2_rows, 1:size(R,1)])
    @variable(model, Γtilde1[s=1:S, 1:dim_Λtilde1_rows, 1:size(R,1)])
    @variable(model, Γtilde2[s=1:S, 1:dim_Λtilde2_rows, 1:size(R,1)])

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================
    diag_x_E = Diagonal(x) * E  # diag(x)E
    diag_λ_ψ = Diagonal(λ.-v.*ψ0)
    # s에 대해 summing이 필요하다면 sum over s 추가
    # matrix inner product: sum(M .* N)
    obj_term1 = [-ϕU * sum((Uhat1[s, :, :] + Utilde1[s, :, :]) .* diag_x_E) for s=1:S]
    obj_term2 = [-ϕU * sum((Uhat3[s, :, :] + Utilde3[s, :, :]) .* (E-diag_x_E)) for s=1:S]
    obj_term3 = [d0'* βhat1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible
    obj_term4 = [sum(Ztilde1_3[s, :, :] .* diag_λ_ψ) for s=1:S]
    obj_term5 = [λ*d0'* βtilde1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible
    obj_term6 = [-h'* βtilde1_3[s,:] for s=1:S]
    # @objective(model, Max, sum(obj_term1) + sum(obj_term2) + sum(obj_term3) + sum(obj_term5) + sum(obj_term6))
    @objective(model, Max, sum(obj_term3))
    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # --- Semi-definite cone constraints ---
    @constraint(model, [s=1:S], Mhat[s,:,:] in PSDCone())
    @constraint(model, [s=1:S], Mtilde[s,:,:] in PSDCone())
    # --- Second order cone constraints ---
    @constraint(model, [s=1:S, i=1:dim_Λhat1_rows], Γhat1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λhat2_rows], Γhat2[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λtilde1_rows], Γtilde1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λtilde2_rows], Γtilde2[s, i, :] in SecondOrderCone())

    # Scalar constraints
    @constraint(model, [s=1:S], Mhat[s, num_arcs+1, num_arcs+1] == 1/S)
    @constraint(model, [s=1:S], Mtilde[s, num_arcs+1, num_arcs+1] == 1/S)
    @constraint(model, sum(α) == w*(1/S))
    @constraint(model, [s=1:S], tr(Mhat[s, 1:num_arcs, 1:num_arcs]) - 2* Mhat[s,1:num_arcs,num_arcs+1]'*xi_bar[s] - Mhat[s,num_arcs+1,num_arcs+1]*((epsilon^2 - sum(xi_bar[s].^2))) <= 0)
    @constraint(model, [s=1:S], tr(Mtilde[s, 1:num_arcs, 1:num_arcs]) - 2* Mtilde[s,1:num_arcs,num_arcs+1]'*xi_bar[s] - Mtilde[s,num_arcs+1,num_arcs+1]*((epsilon^2 - sum(xi_bar[s].^2))) <= 0)

    # Matrix Constraints
    # --- From Φhat ---
    @constraint(model, [s=1:S], -Mhat[s,1:num_arcs, 1:num_arcs+1] + Uhat2[s,:,:] - Uhat3[s,:,:]
    + hcat(-1*(I_0*Zhat1_1[s,:,:] + Zhat1_3[s,:,:]) + Zhat2[s,:,:], I_0*βhat1_1[s,:] + βhat1_3[s,:] - βhat2[s,:]) .== 0)
    # --- From Φhat
    @constraint(model, [s=1:S], v*Mhat[s,1:num_arcs, 1:num_arcs+1] - Uhat1[s,:,:] - Uhat2[s,:,:] + Uhat3[s,:,:] .== 0)
    # --- From Φtilde ---
    @constraint(model, [s=1:S], -Mtilde[s,1:num_arcs, 1:num_arcs+1] + Utilde2[s,:,:] - Utilde3[s,:,:]
    + hcat(-I_0*Ztilde1_1[s,:,:] - Ztilde1_5[s,:,:] + Ztilde2[s,:,:], I_0*βtilde1_1[s,:] + βtilde1_5[s,:] - βtilde2[s,:]) .== 0)
    # --- From Ψtilde ---
    @constraint(model, [s=1:S], v*Mtilde[s,1:num_arcs, 1:num_arcs+1] - Utilde1[s,:,:] - Utilde2[s,:,:] + Utilde3[s,:,:] .== 0)
    # --- From Ytilde_ts ---
    @constraint(model, [s=1:S], Mtilde[s,:, num_arcs+1]' + hcat(N_ts'* Ztilde1_2[s,:,:], -N_ts'* βtilde1_2[s,:]) .== 0)
    # --- From μhat ---
    @constraint(model, [s=1:S, k=1:num_arcs], βhat2[s,k] == α[k])
    # --- From μtilde ---
    @constraint(model, [s=1:S, k=1:num_arcs], βtilde2[s,k] == α[k])
    # --- From Πhat ---
    @constraint(model, [s=1:S], hcat(-N*Zhat1_1[s,:,:]-Zhat1_2[s,:,:], N*βhat1_1[s,:]+ βhat1_2[s,:]) .==0.0)
    # --- From Πtilde ---
    @constraint(model, [s=1:S], hcat(-N*Ztilde1_1[s,:,:]-Ztilde1_4[s,:,:], N*βtilde1_1[s,:]+ βtilde1_4[s,:]) .==0.0)
    # --- From Ytilde ---
    @constraint(model, [s=1:S], hcat(N_y'*Ztilde1_2[s,:,:]+Ztilde1_3[s,:,:]-Ztilde1_6[s,:,:], -N_y'*βtilde1_2[s,:]-βtilde1_3[s,:]+βtilde1_6[s,:]) .==0.0)
    # --- From Λhat1 ---
    @constraint(model, [s=1:S], Zhat1[s,:,:]*R' + βhat1[s,:]*r_dict[s]' + Γhat1[s,:,:] .== 0.0)
    # --- From Λhat2 ---
    @constraint(model, [s=1:S], Zhat2[s,:,:]*R' + βhat2[s,:]*r_dict[s]' + Γhat2[s,:,:] .== 0.0)
    # --- From Λtilde1 ---
    @constraint(model, [s=1:S], Ztilde1[s,:,:]*R' + βtilde1[s,:]*r_dict[s]' + Γtilde1[s,:,:] .== 0.0)
    # --- From Λtilde2 ---
    @constraint(model, [s=1:S], Ztilde2[s,:,:]*R' + βtilde2[s,:]*r_dict[s]' + Γtilde2[s,:,:] .== 0.0)
    optimize!(model)
    t_status = termination_status(model)
    p_status = primal_status(model)
    if t_status == MOI.OPTIMAL || t_status == MOI.FEASIBLE_POINT
        obj_value = objective_value(model)
        println("\nOptimal objective value: ", obj_value)
    end
    @infiltrate
    return model, vars
end


# test model construction
network = generate_grid_network(3, 3, seed=42)
