"""
compact_ldr_utils.jl

Dictionary-indexed compact LDR 유틸리티 모듈.

## 배경
LDR (Linear Decision Rule)에서 arc k의 recourse 결정은 인접 arc (공통 노드를 공유하는 arc)의
불확실성에만 의존한다. 따라서 LDR 계수 행렬의 비인접 항목은 구조적으로 0이다.

이전 fix() 접근법은 변수를 0으로 고정할 뿐 변수 자체를 제거하지 않아 성능 이득이 없었다.
이 모듈은 JuMP dictionary-indexed 변수를 위한 **인덱스 셋 생성기**를 제공하여,
비인접 변수를 아예 생성하지 않는 진짜 compact 구현을 지원한다.

## 제공 함수
- 인덱스 셋 생성기: arc_adj_pairs, arc_full_pairs, node_arc_inc_pairs, node_arc_full_pairs
- Compact → Dense 변환: compact_to_dense_arc, compact_to_dense_node
- 통계 출력: print_compact_ldr_stats

## 사용법
    include("compact_ldr_utils.jl")
    afp = arc_full_pairs(network)   # U, P_Φ, P_Y 변수 인덱싱용
    nfp = node_arc_full_pairs(network)  # P_Π 변수 인덱싱용
"""

using JuMP


# =============================================================================
# 1. 인덱스 셋 생성기
#    LDR 행렬의 _L 열 (slope 부분)과 intercept 열에 대한 인덱스 쌍을 생성한다.
#    _L 열: 인접/incident 쌍에 대해서만 변수 생성
#    intercept 열 (j = num_arcs+1): 모든 행에 대해 항상 생성
# =============================================================================

"""
    arc_adj_pairs(network) -> Vector{Tuple{Int,Int}}

LDR 행렬의 _L 열에 대한 인접 arc 쌍을 반환한다.
arc_adjacency[i,j] == true인 (i,j) 쌍만 포함.

수학적 의미: Φ̂_L[i,j], Ψ̂_L[i,j] 등의 slope 계수가 0이 아닌 위치.
arc i의 recourse가 arc j의 불확실성 ξ_j에 의존하려면 i,j가 인접해야 한다.
"""
function arc_adj_pairs(network)
    num_arcs = length(network.arcs) - 1
    return [(i,j) for i in 1:num_arcs for j in 1:num_arcs if network.arc_adjacency[i,j]]
end

"""
    arc_full_pairs(network) -> Vector{Tuple{Int,Int}}

인접 arc 쌍 + intercept 열 (j = num_arcs+1)을 포함한 전체 인덱스 셋.
U, P_Φ, P_Y 변수 인덱싱에 사용된다.

구조: [aap..., (1,na1), (2,na1), ..., (num_arcs,na1)]
- aap: arc_adj_pairs (slope 열, 인접 쌍만)
- intercept: 모든 행 i에 대해 (i, num_arcs+1) 추가
"""
function arc_full_pairs(network)
    num_arcs = length(network.arcs) - 1
    aap = arc_adj_pairs(network)
    # intercept 열: LDR의 상수항으로, 모든 arc에 대해 존재
    intercept_pairs = [(i, num_arcs+1) for i in 1:num_arcs]
    return vcat(aap, intercept_pairs)
end

"""
    node_arc_inc_pairs(network) -> Vector{Tuple{Int,Int}}

Π (node price) LDR 행렬의 _L 열에 대한 incident (node, arc) 쌍.
node_arc_incidence[i,j] == true인 (i,j) 쌍만 포함.

수학적 의미: Π̂_L[i,j]의 slope 계수가 0이 아닌 위치.
node i의 dual price가 arc j의 불확실성에 의존하려면 node i가 arc j에 incident해야 한다.
"""
function node_arc_inc_pairs(network)
    num_arcs = length(network.arcs) - 1
    num_nodes = length(network.nodes)
    # num_nodes-1: 마지막 노드(sink)의 dual은 0으로 normalize
    return [(i,j) for i in 1:num_nodes-1 for j in 1:num_arcs if network.node_arc_incidence[i,j]]
end

"""
    node_arc_full_pairs(network) -> Vector{Tuple{Int,Int}}

Incident 쌍 + intercept 열을 포함한 전체 인덱스 셋.
P_Π 변수 인덱싱에 사용된다.
"""
function node_arc_full_pairs(network)
    num_arcs = length(network.arcs) - 1
    num_nodes = length(network.nodes)
    nap = node_arc_inc_pairs(network)
    intercept_pairs = [(i, num_arcs+1) for i in 1:num_nodes-1]
    return vcat(nap, intercept_pairs)
end


# =============================================================================
# 2. Compact → Dense 변환 헬퍼
#    evaluate_master_opt_cut에서 outer loop의 cut 구성에 필요.
#    dict-indexed 값을 원본 코드와 호환되는 dense 행렬로 변환한다.
# =============================================================================

"""
    compact_to_dense_arc(vals, afp, num_arcs) -> Matrix{Float64}

Dict-indexed 값 (afp 튜플로 키잉)을 dense (num_arcs × num_arcs+1) 행렬로 변환.
비인접 항목은 0으로 유지된다.
"""
function compact_to_dense_arc(vals, afp, num_arcs)
    dense = zeros(num_arcs, num_arcs+1)
    for (i,j) in afp
        dense[i,j] = vals[(i,j)]
    end
    return dense
end

"""
    compact_to_dense_node(vals, nfp, num_nodes_minus1, num_arcs) -> Matrix{Float64}

Dict-indexed 값 (nfp 튜플로 키잉)을 dense (num_nodes-1 × num_arcs+1) 행렬로 변환.
비incident 항목은 0으로 유지된다.
"""
function compact_to_dense_node(vals, nfp, num_nodes_minus1, num_arcs)
    dense = zeros(num_nodes_minus1, num_arcs+1)
    for (i,j) in nfp
        dense[i,j] = vals[(i,j)]
    end
    return dense
end


# =============================================================================
# 3. 통계 출력
# =============================================================================

"""
    print_compact_ldr_stats(network)

Dictionary-indexed compact LDR의 변수 절감 통계를 출력한다.
Full dimension 대비 실제 생성되는 변수 수를 비교.
"""
function print_compact_ldr_stats(network)
    num_arcs = length(network.arcs) - 1
    num_nodes = length(network.nodes)

    aap = arc_adj_pairs(network)
    nap = node_arc_inc_pairs(network)
    afp = arc_full_pairs(network)
    nfp = node_arc_full_pairs(network)

    full_arc = num_arcs * (num_arcs + 1)
    full_node = (num_nodes - 1) * (num_arcs + 1)

    println("="^80)
    println("COMPACT LDR STATISTICS (dictionary-indexed)")
    println("="^80)
    println("Network: $(num_nodes) nodes, $(num_arcs) arcs")
    println()
    println("Arc-indexed variables (U, P_Φ, P_Y):")
    println("  Full:    $(full_arc) per matrix")
    println("  Compact: $(length(afp)) per matrix ($(length(aap)) adj + $(num_arcs) intercept)")
    println("  Saved:   $(full_arc - length(afp)) per matrix ($(round(100*(full_arc-length(afp))/full_arc, digits=1))%)")
    println()
    println("Node-arc-indexed variables (P_Π):")
    println("  Full:    $(full_node) per matrix")
    println("  Compact: $(length(nfp)) per matrix ($(length(nap)) inc + $(num_nodes-1) intercept)")
    println("  Saved:   $(full_node - length(nfp)) per matrix ($(round(100*(full_node-length(nfp))/full_node, digits=1))%)")
    println("="^80)
end
