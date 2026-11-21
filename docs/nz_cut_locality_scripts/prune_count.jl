include(joinpath(@__DIR__, "lovasz_defs.jl"))
function value_at(xv)
    for i in 1:nx; fix(x[i], xv[i]; force=true); end
    lev = zeros(Int, nB); for i in 1:nB, k in 1:K; xv[(i - 1) * K + k] > 0.5 && (lev[i] = k); end
    c = @constraint(M, Fexpr >= VF(lev)); optimize!(M); v = objective_value(M) - dot(nd.qx, xv)
    delete(M, c); for i in 1:nx; unfix(x[i]); set_lower_bound(x[i], 0.0); set_upper_bound(x[i], 1.0); end
    return v
end
key(sel) = Tuple(sort(sel))
xof(sel) = (xv = zeros(nx); for (i, k) in sel; xv[(i - 1) * K + k] = 1; end; xv)
h = Dict{Any,Float64}(); h[()] = 0.0
t = @elapsed begin
    for i in 1:nB, k in 1:K; h[key([(i, k)])] = value_at(xof([(i, k)])); end
    for a in 1:nB, b in a+1:nB, ka in 1:K, kb in 1:K; h[key([(a, ka), (b, kb)])] = value_at(xof([(a, ka), (b, kb)])); end
    for a in 1:nB, b in a+1:nB, c in b+1:nB, ka in 1:K, kb in 1:K, kc in 1:K
        h[key([(a, ka), (b, kb), (c, kc)])] = value_at(xof([(a, ka), (b, kb), (c, kc)]))
    end
end
q(sel) = sum(nd.qx[(i - 1) * K + k] for (i, k) in sel; init=0.0)
f = Dict(k => q(collect(k)) + v for (k, v) in h); fbest = minimum(values(f)); kbest = argmin(f)
@printf("모든 x 의 값 %d 개 (%.0fs). 최적 %.2f at %s\n", length(h), t, fbest, string(kbest))
# 유효성: 세 곳 값 ≥ 두 곳 값 + 남은 한 곳 값 (모든 분할), 두 곳 ≥ 한 곳 합
v2 = 0; v3 = 0
for (k, v) in h
    n = length(k); n <= 1 && continue
    if n == 2; v < h[(k[1],)] + h[(k[2],)] - 1e-6 * abs(v) && (global v2 += 1); end
    if n == 3; for c in 1:3; ab = Tuple(k[j] for j in 1:3 if j != c); v < h[ab] + h[(k[c],)] - 1e-6 * abs(v) && (global v3 += 1); end; end
end
@printf("유효성: 두 곳 ≥ 한 곳 합 위반 %d / 702, 세 곳 ≥ 두 곳 + 한 곳 위반 %d / %d\n", v2, v3, 3 * 7722)
thr = fbest + 5e-3 * abs(fbest)
bound1(k) = q(collect(k)) + sum(h[(e,)] for e in k; init=0.0)
bound2(k) = length(k) < 3 ? q(collect(k)) + h[k] : q(collect(k)) + maximum(h[Tuple(k[j] for j in 1:3 if j != c)] + h[(k[c],)] for c in 1:3)
for (nm, B) in (("한 곳 값의 합", bound1), ("두 곳 값 + 한 곳 값", bound2))
    surv = [k for k in keys(h) if B(k) < thr]
    near = count(k -> f[k] < thr, keys(h))
    @printf("PRUNE %-22s 남는 x %5d / %d  (두 곳 %d, 세 곳 %d) | 실제로 최적의 0.5%% 안 %d\n", nm, length(surv), length(h),
            count(k -> length(k) == 2, surv), count(k -> length(k) == 3, surv), near)
end
gaps = sort([(f[k] - fbest) / abs(fbest) for k in keys(h)])
@printf("최적 대비 격차 분포: 0.5%% 안 %d, 1%% 안 %d, 5%% 안 %d, 10%% 안 %d, 20%% 안 %d\n", count(<=(0.005), gaps), count(<=(0.01), gaps), count(<=(0.05), gaps), count(<=(0.1), gaps), count(<=(0.2), gaps))
