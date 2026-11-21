# -*- coding: utf-8 -*-
import io, itertools, statistics as st
V = {}; SOLD = {}
for line in io.open('vtable_g1_nominal.tsv', encoding='utf8').read().splitlines()[1:]:
    p = line.split('\t')
    key = tuple(sorted(tuple(map(int, t.split(':'))) for t in p[0].split(';'))) if p[0] else ()
    V[key] = float(p[2]); sd = {}
    if len(p) > 4 and p[4]:
        for t in p[4].split(';'):
            i, vals = t.split(':'); sd[int(i)] = tuple(map(float, vals.split(',')))
    SOLD[key] = sd
tiers = [200, 300, 400]
print("[4] 총 판매량 (약정 ΣR, 현장 ΣSp) 의 분포: 거점 수·크기별")
for n in (1, 2, 3):
    keys = [k for k in V if len(k) == n]
    R = [sum(v[0] for v in SOLD[k].values()) for k in keys]; Sp = [sum(v[1] for v in SOLD[k].values()) for k in keys]
    cap = [sum(tiers[kk - 1] for (_, kk) in k) for k in keys]
    print(f"  {n} 곳 ({len(keys)}): ΣR 범위 [{min(R):.0f}, {max(R):.0f}] 중앙 {st.median(R):.0f} | ΣSp 범위 [{min(Sp):.0f}, {max(Sp):.0f}] | 총용량 범위 [{min(cap)}, {max(cap)}]")
# 약정 총량의 상한이 있는가: ΣR 이 용량 대비 어디서 꺾이는가
print("\n[5] 총용량별 ΣR 최대·최소 (두 곳·세 곳)")
from collections import defaultdict
byc = defaultdict(list)
for k in V:
    if len(k) >= 2:
        byc[sum(tiers[kk - 1] for (_, kk) in k)].append((sum(v[0] for v in SOLD[k].values()), sum(v[1] for v in SOLD[k].values()), V[k]))
for c in sorted(byc):
    L = byc[c]
    print(f"  용량 {c:5d} (n={len(L):4d}): ΣR [{min(t[0] for t in L):6.0f}, {max(t[0] for t in L):6.0f}]  ΣSp [{min(t[1] for t in L):5.0f}, {max(t[1] for t in L):5.0f}]  V [{min(t[2] for t in L):7.0f}, {max(t[2] for t in L):7.0f}]")
