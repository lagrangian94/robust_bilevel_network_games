using Base.Threads: @threads, nthreads

"""
    solve_scenarios(f, S; parallel=false)

시나리오별 독립 JuMP 문제를 순차/병렬로 solve하는 헬퍼.
`f(s)` → `(ok::Bool, result)` 형태의 closure를 받아 pre-allocated 배열에 저장.

Usage:
    results, all_ok = solve_scenarios(S; parallel=true) do s
        # ... solve scenario s ...
        return (true, some_result)
    end
"""
function solve_scenarios(f::Function, S::Int; parallel::Bool=false)
    results = Vector{Any}(undef, S)
    statuses = Vector{Bool}(undef, S)
    if parallel
        nthreads() > 1 || error("parallel=true인데 nthreads()=$(nthreads()). `julia -t auto` 또는 JULIA_NUM_THREADS 설정 필요.")
        @threads for s in 1:S
            (ok, res) = f(s)
            statuses[s] = ok
            results[s] = res
        end
    else
        for s in 1:S
            (ok, res) = f(s)
            statuses[s] = ok
            results[s] = res
        end
    end
    return results, all(statuses)
end
