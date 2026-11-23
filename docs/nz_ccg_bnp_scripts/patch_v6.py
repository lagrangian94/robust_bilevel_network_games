s = open('ccgbp_v5nb.jl', encoding='utf-8').read()
def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    assert n == cnt, (a[:70], n)
    s = s.replace(a, b)

old = s[s.index('pts = [zeros(nx)]\n'):s.index('nbrs_cache = Dict{Int,Vector{Int}}()')]
new = '''# 점 번호는 만난 점에만 붙인다 (v6, 2026-11-23): X 를 미리 나열하지 않음. |X| 는 보고용으로 공식으로만 계산
pts = Vector{Vector{Float64}}(); qv = Float64[]; idx = Dict{Tuple,Int}()
key(z) = Tuple(findall(>(0.5), z))
"점 z 의 번호 (처음 만나면 등록). z 는 X 의 점이어야 함 (호출하는 곳이 실행가능하게 만듦)"
function pid(z)
    k = key(z); c = get(idx, k, 0); c > 0 && return c
    push!(pts, Float64.(z .> 0.5)); push!(qv, dot(nd.qx, pts[end])); idx[k] = length(pts)
    return length(pts)
end
pid(zeros(nx))                                                   # 빈 x = 번호 1
NP = sum(binomial(length(sites), n) * K^n for n in 0:nd.gamma)   # |X| (보고용)
'''
s = s.replace(old, new)
rep('put(conf) = (w = zeros(nx); for (s, k) in conf; w[(s - 1) * K + k] = 1; end; get(idx, key(w), 0))',
    'put(conf) = (w = zeros(nx); for (s, k) in conf; w[(s - 1) * K + k] = 1; end; pid(w))')
rep('    get(idx, key(z), 0)\nend', '    pid(z)\nend')
rep('    return v, idx[key(z)]\nend', '    return v, pid(z)\nend')
rep('zs = [idx[key(round.(value.(xb; result=k)))] for k in 1:result_count(Mp)]', 'zs = [pid(round.(value.(xb; result=k))) for k in 1:result_count(Mp)]')
rep('c = idx[key(round.(xv))]; global nint += 1', 'c = pid(round.(xv)); global nint += 1')
rep('--- ccgbp v5 |', '--- ccgbp v6 (X 미리 나열 안 함) |')
assert 'idx[key' not in s and 'get(idx, key' not in s.replace('c = get(idx, k, 0)', '')
# 열 수 기록 (RMP 에 실제로 들어간 열)
s = s.rstrip('\n') + '\n@printf("  열: RMP 에 들어간 열 (블록 합) %d, 서로 다른 열 %d, 번호를 붙인 점 %d / |X| %d\\n", sum(length.(cols)), length(union(cols...)), length(pts), NP)\n'
open('ccgbp_v6.jl', 'w', encoding='utf-8').write(s)
print('ok')
