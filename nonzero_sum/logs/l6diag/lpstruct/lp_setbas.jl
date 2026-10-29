#= lp_setbas.jl — GRBsetintattrarray 로 넣은 VBasis/CBasis 를 Gurobi 가 쓰는가: (1) 새 모델 (2) 이미 다른 basis 로 푼 뒤 계수 바꾼 모델 =#
ENV["LS_NOMAIN"] = "1"; include(raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpstruct\lp_struct.jl")
p = raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpdump\S20\ok_300.mps"; q = raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpdump\S20\ok_750.mps"
it(m) = Int(getd(m, "IterCount"))
A = load(p); prm!(A); C.GRBread(A, replace(p, ".mps" => ".bas")); C.GRBoptimize(A)
nr, nc = geti(A, "NumConstrs"), geti(A, "NumVars"); cb = iarr(A, "CBasis", nr); vb = iarr(A, "VBasis", nc)
@printf("A (ok_300, 자기 .bas): %s %d 반복\n", st(A), it(A))
B = load(p); prm!(B); C.GRBoptimize(B); @printf("B cold: %d 반복\n", it(B))
B2 = load(p); prm!(B2); r1 = C.GRBsetintattrarray(B2, "VBasis", 0, nc, vb); r2 = C.GRBsetintattrarray(B2, "CBasis", 0, nr, cb)
C.GRBoptimize(B2); @printf("새 모델 + 최적 basis 넣음 (rc %d %d): %d 반복\n", r1, r2, it(B2))
# 다른 노드 (ok_750) 의 모델: 크기가 같으면 먼저 풀고 (다른 basis), ok_300 의 상한·계수로 바꾸는 대신 그냥 같은 모델에서 RHS 를 흔든 뒤 basis 넣기
D = load(p); prm!(D); C.GRBoptimize(D)
rhs = darr(D, "RHS", nr); k = argmax(abs.(rhs)); C.GRBsetdblattrelement(D, "RHS", k - 1, rhs[k] * 1.01); C.GRBupdatemodel(D); C.GRBoptimize(D)
@printf("D: 푼 뒤 RHS 하나 바꿔 다시: %d 반복\n", it(D))
C.GRBsetdblattrelement(D, "RHS", k - 1, rhs[k]); C.GRBupdatemodel(D)
r1 = C.GRBsetintattrarray(D, "VBasis", 0, nc, vb); r2 = C.GRBsetintattrarray(D, "CBasis", 0, nr, cb); C.GRBoptimize(D)
@printf("D: RHS 되돌리고 원래 최적 basis 넣음 (rc %d %d): %d 반복 (안 쓰면 이어 풀기 반복)\n", r1, r2, it(D))
D2 = load(p); prm!(D2); C.GRBoptimize(D2); C.GRBsetdblattrelement(D2, "RHS", k - 1, rhs[k] * 1.01); C.GRBupdatemodel(D2); C.GRBoptimize(D2)
C.GRBsetdblattrelement(D2, "RHS", k - 1, rhs[k]); C.GRBupdatemodel(D2); C.GRBoptimize(D2)
@printf("D2: 같은 과정, basis 안 넣음: %d 반복\n", it(D2))
# 확실한 시험: 목적을 바꿔 푼 (다른 basis 를 가진) 모델에서 목적을 되돌린 뒤 최적 basis 를 넣는가 / 안 넣는가
for put in (false, true)
    E = load(p); prm!(E); ob = darr(E, "Obj", nc)
    C.GRBsetdblattrarray(E, "Obj", 0, nc, -ob); C.GRBupdatemodel(E); C.GRBoptimize(E); i1 = it(E)
    C.GRBsetdblattrarray(E, "Obj", 0, nc, ob); C.GRBupdatemodel(E)
    put && (C.GRBsetintattrarray(E, "VBasis", 0, nc, vb); C.GRBsetintattrarray(E, "CBasis", 0, nr, cb))
    C.GRBoptimize(E)
    @printf("E: 목적 반대로 푼 뒤 (%d 반복) 되돌리고 최적 basis %s: %s %d 반복\n", i1, put ? "넣음" : "안 넣음", st(E), it(E))
end
