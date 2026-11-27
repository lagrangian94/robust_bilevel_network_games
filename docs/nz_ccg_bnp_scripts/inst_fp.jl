# 인스턴스 지문 (2026-11-27 밤): 지리 여러 개 × S 의 자료 합. 서버 사본 (Julia 판이 다르면 시드 hash 를 고친 loc_tkp_srv.jl) 과 이 컴퓨터 결과가 같은지 비교.
#   julia inst_fp.jl <loc_tkp 파일 경로> <S> <지리 목록 "1,2,3">
using JuMP, Printf, LinearAlgebra, Random
include(ARGS[1])
S = parse(Int, ARGS[2]); geos = parse.(Int, split(ARGS[3], ","))
println("julia ", VERSION)
for g in geos
    nd = make_loc_tkp(; g=g, S=S, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=1e4)
    @printf("g=%d S=%d nx %d nh %d | u %.6f | q %.6f | hU %.3f | c %.6f | ell %.6f | g %.6f | A %.6f | W %.6f | c0 %.6f\n", g, S, nd.nx, length(nd.hU),
            sum(nd.u), sum(nd.q_hat), sum(nd.hU), sum(nd.c), sum(nd.ell), sum(nd.g), sum(nd.A), sum(nd.W), sum(nd.c0))
end
