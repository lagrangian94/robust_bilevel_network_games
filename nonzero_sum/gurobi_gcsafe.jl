"""
gurobi_gcsafe.jl — Gurobi 의 풀이 호출 (GRBoptimize) 을 Julia 의 **GC-safe 구간** 에서 하도록 바꾼다.

왜 필요한가 (2026-10-13 측정, α-B&B worker 12 개, S = 20, 120 s)
  Julia 의 GC 는 stop-the-world 라, 한 스레드가 GC 를 시작하면 **모든 스레드가 safepoint 에 닿을 때까지 기다린다**.
  ccall 안에 있는 스레드는 (기본으로는) safepoint 가 아니어서, worker 하나가 Gurobi LP 를 푸는 동안에는 GC 가 시작되지 못하고,
  그 사이 LP 를 끝낸 다른 worker 들은 메모리 할당 지점에서 멈춰 기다린다. worker 12 개가 번갈아 LP 를 풀면 이 대기가 겹쳐
  wall 120 s 중 82 s 가 "safepoint 대기" 였다 (Base.gc_num().total_time_to_safepoint) — worker 12 개가 실제로는 4 개 남짓으로 돈 셈.
  GRBoptimize 를 GC-safe 구간에서 부르면 그 스레드는 GC 를 막지 않는다: 같은 실행에서 대기 0.3 s, 처리 노드 373 → 860.

안전한가
  GC-safe 구간에서는 그 스레드가 Julia 힙을 건드리면 안 된다. GRBoptimize 는 순수 C 호출이고, 인자 (모델 포인터) 는 구간에 들어가기 전에 꺼낸다.
  Gurobi 가 Julia callback 을 부르는 경우 (Gurobi.jl 은 Ctrl-C 처리를 위해 빈 callback 을 건다) 는 cfunction 진입 시 Julia 가 GC 상태를 되돌린다.
  Julia 1.12 의 `@ccall gc_safe=true` 와 같은 일을 1.11 에서 내부 함수 (jl_gc_safe_enter / jl_gc_safe_leave) 로 하는 것이다.
  끄려면 환경변수 NZ_GCSAFE=0. 함수가 없는 Julia 버전에서는 아무것도 바꾸지 않고 알린다.

이 파일은 Gurobi 모듈의 GRBoptimize 메서드를 덮어쓴다 (같은 프로세스의 모든 Gurobi 풀이에 적용). nz_alpha_bnb_dr.jl 이 include 한다.
"""

import Gurobi

const _NZ_GCSAFE_STATE = Ref(:unset)

function _nz_gcsafe_install!()
    _NZ_GCSAFE_STATE[] == :unset || return _NZ_GCSAFE_STATE[]
    if get(ENV, "NZ_GCSAFE", "1") == "0"
        println(stderr, "gurobi_gcsafe: NZ_GCSAFE=0 — GRBoptimize 를 그대로 둠 (worker 병렬 실행에서 GC 대기가 커질 수 있음)")
        return _NZ_GCSAFE_STATE[] = :off
    end
    ok = try
        st = ccall(:jl_gc_safe_enter, Int8, ()); ccall(:jl_gc_safe_leave, Cvoid, (Int8,), st); true
    catch
        false
    end
    if !ok || !isdefined(Gurobi, :GRBoptimize) || !isdefined(Gurobi, :libgurobi)
        println(stderr, "gurobi_gcsafe: 이 Julia / Gurobi.jl 에서는 GC-safe 호출을 쓸 수 없음 — GRBoptimize 를 그대로 둠")
        return _NZ_GCSAFE_STATE[] = :unavailable
    end
    @eval Gurobi function GRBoptimize(model)
        GC.@preserve model begin
            p = Base.unsafe_convert(Ptr{Cvoid}, Base.cconvert(Ptr{Cvoid}, model))
            st = ccall(:jl_gc_safe_enter, Int8, ())
            ret = ccall((:GRBoptimize, libgurobi), Cint, (Ptr{Cvoid},), p)
            ccall(:jl_gc_safe_leave, Cvoid, (Int8,), st)
            return ret
        end
    end
    return _NZ_GCSAFE_STATE[] = :on
end

_nz_gcsafe_install!()
