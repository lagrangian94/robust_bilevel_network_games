"""
MatrixBuilder.jl

Robust Bi-level Network Interdiction Problem의 
Finite Reformulation에 필요한 matrix들을 생성하는 module

NOTE: Uncertainty set matrices (R, r)은 build_uncertainty_set.jl 참조

연구 노트 참조:
- Section 2: Finite Reformulations (E, Q, diagonal matrices)

References:
- Sadana & Delage (2022): Network structure, McCormick reformulation
- Lei & Song (2018): Node-arc incidence matrix formulation
"""

module MatrixBuilder

using LinearAlgebra

export build_E_matrix
export build_diagonal_matrices
export verify_E_matrix_properties

# ============================================================================
# McCORMICK REFORMULATION MATRICES
# ============================================================================

"""
    build_E_matrix(num_regular_arcs::Int)

Matrix of ones E ∈ ℝ^{|A| × (|A|+1)} 생성

연구 노트 식 (11g), (11l)의 McCormick reformulation:
    Ŵ^s ≤ φ^U diag(x)E
    W̃^s ≤ φ^U diag(x)E
    
여기서 E = [e, e, ..., e]는 모든 column이 ones vector인 matrix

# Arguments
- `num_regular_arcs::Int`: regular arcs 개수 |A| (dummy arc 제외)

# Returns
- `E::Matrix{Float64}`: matrix of ones (|A| × (|A|+1))

# Mathematical Purpose
McCormick envelope의 upper bound를 표현:
- Scalar form: ŵ^s_{k,l} ≤ φ^U x_k  ∀l, k ∈ A
- Matrix form: Ŵ^s ≤ φ^U diag(x)E

diag(x)E의 k-th row는 x_k가 모든 |A|+1 columns에 복제됨

# Example
```julia
E = build_E_matrix(5)  # 5×6 matrix of ones
x = [1.0, 0.0, 1.0, 1.0, 0.0]
phi_U = 10.0
upper_bound = phi_U * Diagonal(x) * E
# upper_bound[k, :] = phi_U * x_k * [1,1,1,1,1,1]
```
"""
function build_E_matrix(num_regular_arcs::Int)
    dim_xi = num_regular_arcs + 1  # ξ = (ζ; τ) dimension
    E = ones(Float64, num_regular_arcs, dim_xi)
    return E
end


end # module MatrixBuilder