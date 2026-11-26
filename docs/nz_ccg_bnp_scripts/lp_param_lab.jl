# lp_param_lab.jl — 저장한 dive 노드 LP (abb_lpdump_patch 의 dv_k.mps + dv_k_start.bas) 를 같은 시작 basis 에서 Gurobi 설정별로 다시 풀어
#   반복·시간·목적값을 비교 (2026-11-27, α-B&B 완화 LP 튜닝 실험실, 스레드 1). docs/nz_ccg_bnp_2026-11-23.md 11.22.
#  환경 변수: DUMP (dv_*.mps 가 있는 폴더), VARS (쉼표로 고른 변형 이름, 기본 전부), REP (같은 설정 반복 횟수, 기본 1)
#  기준 설정 = 노드 LP 설정 (dual simplex, Presolve 0, Feasibility/OptimalityTol 1e-7, NumericFocus 0, InfUnbdInfo 1, Threads 1)
using Gurobi, Printf, Statistics
const C = Gurobi
dir = get(ENV, "DUMP", "")
isdir(dir) || error("DUMP 폴더 없음: $dir")
files = sort([f for f in readdir(dir) if occursin(r"^dv_\d+\.mps$", f)]; by=f -> parse(Int, match(r"\d+", f).match))
isempty(files) && error("dv_*.mps 없음")
env = C.Env()
setp(m, k, v::Integer) = C.GRBsetintparam(C.GRBgetenv(m), k, v) == 0 || error("param $k")
setp(m, k, v::Real) = C.GRBsetdblparam(C.GRBgetenv(m), k, Float64(v)) == 0 || error("param $k")
geti(m, a) = (r = Ref{Cint}(); C.GRBgetintattr(m, a, r) == 0 ? Int(r[]) : -1)
getd(m, a) = (r = Ref{Cdouble}(); C.GRBgetdblattr(m, a, r) == 0 ? r[] : NaN)
function load(path)
    p = Ref{Ptr{Cvoid}}()
    C.GRBreadmodel(env, path, p) == 0 || error("read $path")
    return p[]
end
# 변형: (이름, 시작 basis 사용, 추가 설정)
VARIANTS = [
    ("base",      true,  Pair{String,Any}[]),
    ("cold",      false, Pair{String,Any}[]),
    ("cold_pre",  false, ["Presolve" => -1]),
    ("price0",    true,  ["SimplexPricing" => 0]),
    ("price1",    true,  ["SimplexPricing" => 1]),
    ("price2",    true,  ["SimplexPricing" => 2]),
    ("price3",    true,  ["SimplexPricing" => 3]),
    ("primal",    true,  ["Method" => 0]),
    ("norm0",     true,  ["NormAdjust" => 0]),
    ("norm1",     true,  ["NormAdjust" => 1]),
    ("norm2",     true,  ["NormAdjust" => 2]),
    ("norm3",     true,  ["NormAdjust" => 3]),
    ("scale0",    true,  ["ScaleFlag" => 0]),
    ("scale2",    true,  ["ScaleFlag" => 2]),
    ("perturb0",  true,  ["PerturbValue" => 0.0]),
    ("barrier",   false, ["Method" => 2]),
    ("s2p1",      true,  Pair{String,Any}["ScaleFlag" => 2, "SimplexPricing" => 1]),
    ("s2pt0",     true,  Pair{String,Any}["ScaleFlag" => 2, "PerturbValue" => 0.0]),
    ("s3",        true,  ["ScaleFlag" => 3]),
    # 기준값용: cold, NumericFocus 3, 허용오차 1e-9 (노드 LP 값이 맞는지 보는 기준)
    ("ref",       false, Pair{String,Any}["NumericFocus" => 3, "FeasibilityTol" => 1e-9, "OptimalityTol" => 1e-9]),
    ("ref_quad",  false, Pair{String,Any}["NumericFocus" => 3, "FeasibilityTol" => 1e-9, "OptimalityTol" => 1e-9, "Quad" => 1]),
]
sel = get(ENV, "VARS", "")
isempty(sel) || (VARIANTS = [v for v in VARIANTS if v[1] in split(sel, ",")])
nrep = parse(Int, get(ENV, "REP", "1"))
m0 = load(joinpath(dir, files[1]))
@printf("LP %d 개 (%s), 첫 LP 행 %d 열 %d 비영 %d\n", length(files), dir, geti(m0, "NumConstrs"), geti(m0, "NumVars"), geti(m0, "NumNZs"))
C.GRBfreemodel(m0)
res = Dict{String,Vector{NTuple{4,Float64}}}()   # 변형 → (반복, 시간, 목적값, 상태)
for (nm, usebas, extra) in VARIANTS
    out = NTuple{4,Float64}[]
    for f in files
        path = joinpath(dir, f); bas = replace(path, ".mps" => "_start.bas")
        best = (Inf, Inf, NaN, -1.0)
        for r in 1:nrep
            m = load(path)
            setp(m, "OutputFlag", 0); setp(m, "Threads", 1); setp(m, "Method", 1); setp(m, "Presolve", 0)
            setp(m, "FeasibilityTol", 1e-7); setp(m, "OptimalityTol", 1e-7); setp(m, "NumericFocus", 0); setp(m, "InfUnbdInfo", 1)
            for (k, v) in extra; setp(m, k, v); end
            usebas && (C.GRBread(m, bas) == 0 || error("basis 읽기 실패 $bas"))
            C.GRBoptimize(m)
            it = getd(m, "IterCount"); t = getd(m, "Runtime"); st = geti(m, "Status")
            z = st == 2 ? getd(m, "ObjVal") : NaN
            nm == "barrier" && (it = getd(m, "IterCount") + geti(m, "BarIterCount"))
            t < best[2] && (best = (it, t, z, Float64(st)))
            C.GRBfreemodel(m)
        end
        push!(out, best)
    end
    res[nm] = out
    its = [o[1] for o in out]; ts = [o[2] for o in out]; nopt = count(o -> o[4] == 2.0, out)
    zb = haskey(res, "base") ? [o[3] for o in res["base"]] : fill(NaN, length(out))
    dz = maximum([abs(out[i][3] - zb[i]) / max(1.0, abs(zb[i])) for i in eachindex(out) if isfinite(out[i][3]) && isfinite(zb[i])]; init=0.0)
    @printf("%-9s 최적 %2d/%d  반복 합 %8.0f (중앙 %6.0f)  시간 합 %7.2f s (중앙 %.3f, 최대 %.3f)  기준 대비 시간 %5.2f  목적 차 최대 %.1e\n",
            nm, nopt, length(out), sum(its), median(its), sum(ts), median(ts), maximum(ts),
            haskey(res, "base") ? sum(ts) / sum(o[2] for o in res["base"]) : NaN, dz)
    flush(stdout)
end
# DETAIL=1: LP 마다 ref (cold, NumericFocus 3, 허용오차 1e-9) 대비. 노드 LP 상한으로 쓰려면 변형 값 + zpad 여유 (상대 1e-6) ≥ ref 여야 함.
#   낮음 = 변형 < ref − 1e-6·max(1,|ref|) (상한이 실제보다 낮게 나옴), 판정 = 저장 때 문턱 thr (= LB + tol) 기준 가지치기 판정이 ref 와 다름
if get(ENV, "DETAIL", "0") == "1"
    names = [v[1] for v in VARIANTS]
    "ref" in names || error("DETAIL 은 ref 변형이 필요")
    thr = [parse(Float64, match(r"thr (\S+)", read(joinpath(dir, replace(f, ".mps" => ".txt")), String)).captures[1]) for f in files]
    @printf("
변형별: ref 보다 여유 넘게 낮은 LP 수, 가지치기 판정이 ref 와 다른 LP 수, ref 대비 최대 절대 차 (|ref| < 1e5 인 LP 만, 문턱 근처)
")
    for nm in names
        nm == "ref" && continue
        lo = 0; flip = 0; dmax = 0.0; dlist = String[]
        for i in eachindex(files)
            zr = res["ref"][i][3]; z = res[nm][i][3]
            (isfinite(zr) && isfinite(z)) || continue
            z < zr - 1e-6 * max(1.0, abs(zr)) && (lo += 1; push!(dlist, @sprintf("%s:%+.3g", files[i], z - zr)))
            ((z <= thr[i]) != (zr <= thr[i])) && (flip += 1)
            abs(zr) < 1e5 && (dmax = max(dmax, abs(z - zr)))
        end
        @printf("%-9s 낮음 %2d  판정 다름 %2d  최대 절대 차 %.3g  %s
", nm, lo, flip, dmax, join(first(dlist, 6), " "))
    end
    println("
LP 별 (ref 값, 문턱, 변형 − ref)")
    for (i, f) in enumerate(files)
        zr = res["ref"][i][3]
        @printf("%-10s ref %14.6f thr %10.3f |", f, zr, thr[i])
        for nm in names
            nm == "ref" && continue
            @printf(" %s %+.2e", nm, res[nm][i][3] - zr)
        end
        println()
    end
end
