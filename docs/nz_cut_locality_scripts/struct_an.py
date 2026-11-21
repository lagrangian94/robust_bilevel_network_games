# -*- coding: utf-8 -*-
import io, itertools, statistics as st, math
V = {}; SOLD = {}
for line in io.open('vtable_g1_nominal.tsv', encoding='utf8').read().splitlines()[1:]:
    p = line.split('\t')
    key = tuple(sorted(tuple(map(int, t.split(':'))) for t in p[0].split(';'))) if p[0] else ()
    V[key] = float(p[2])
    sd = {}
    if len(p) > 4 and p[4]:
        for t in p[4].split(';'):
            i, vals = t.split(':'); sd[int(i)] = tuple(map(float, vals.split(',')))
    SOLD[key] = sd
geo = {}; D = []
for line in io.open('geo_g1.tsv', encoding='utf8').read().splitlines()[1:]:
    if line.startswith('#'): continue
    if len(line.split('\t')) == 4:
        s, c, lon, reg = line.split('\t'); geo[int(s)] = (c, float(lon), int(reg))
    else:
        D.append(list(map(int, line.split())))
nB = 13; K = 3; tiers = [200, 300, 400]
print("거점: " + ", ".join(f"{i}:{geo[i][0].split(',')[0]}(R{geo[i][2]})" for i in range(1, nB + 1)))
# 1) 한 곳: 판매 구성
print("\n[1] 한 곳 값과 판매 (약정 R, 현장 Sp, 버림 W; r 가중 평균 단위)")
for i in range(1, nB + 1):
    row = []
    for k in range(1, K + 1):
        v = V[((i, k),)]; R, Sp, W = SOLD[((i, k),)][i]
        row.append(f"{tiers[k-1]}: {v:8.1f} [R {R:5.0f} Sp {Sp:4.0f} W {W:4.0f}]")
    print(f"  {i:2d} {geo[i][0][:16]:16s} R{geo[i][2]}  " + " | ".join(row))
# 2) 두 곳 상호작용 I = V(ab) − V(a) − V(b)  (≥ 0: 덜 내려감 = 잠식)
I = {}
for a, b in itertools.combinations(range(1, nB + 1), 2):
    for ka in range(1, K + 1):
        for kb in range(1, K + 1):
            I[(a, ka, b, kb)] = V[((a, ka), (b, kb))] - V[((a, ka),)] - V[((b, kb),)]
vals = list(I.values())
print(f"\n[2] 두 곳 상호작용 I (702 개): 0 (|I| < 1) {sum(abs(v) < 1 for v in vals)}, 양수 {sum(v >= 1 for v in vals)}, 음수 {sum(v <= -1 for v in vals)}, 최대 {max(vals):.0f}, 중앙 (양수 중) {st.median([v for v in vals if v >= 1]) if any(v>=1 for v in vals) else 0:.0f}")
# 거점 쌍별 (크기 9 조합 중 최대 상호작용) 과 지역·거리
print("  거점 쌍별 최대 I (지역 같음 *, 거리):")
pairs = []
for a, b in itertools.combinations(range(1, nB + 1), 2):
    mx = max(I[(a, ka, b, kb)] for ka in range(1, K + 1) for kb in range(1, K + 1))
    pairs.append((mx, a, b, geo[a][2] == geo[b][2], D[a - 1][b - 1 + 0] if False else None))
same = [p for p in pairs if p[3]]; diff = [p for p in pairs if not p[3]]
print(f"   같은 지역 쌍 {len(same)}: 상호작용 있음 (>1) {sum(p[0] > 1 for p in same)}, 다른 지역 쌍 {len(diff)}: 있음 {sum(p[0] > 1 for p in diff)}")
nz = sorted([p for p in pairs if p[0] > 1], reverse=True)
print("   상호작용 큰 쌍: " + ", ".join(f"({p[1]},{p[2]}{'*' if p[3] else ''}) {p[0]:.0f}" for p in nz[:15]))
print(f"   상호작용 없는 쌍 수: {sum(p[0] <= 1 for p in pairs)} / {len(pairs)}")
# 크기에 따른 상호작용: 200+200 / 400+400
for (ka, kb) in [(1, 1), (2, 2), (3, 3)]:
    v = [I[(a, ka, b, kb)] for a, b in itertools.combinations(range(1, nB + 1), 2)]
    print(f"   크기 {tiers[ka-1]}+{tiers[kb-1]}: 상호작용 있는 쌍 {sum(x > 1 for x in v)} / 78, 최대 {max(v):.0f}")
# 3) 세 곳 상호작용 (포함배제 3 차 항)
J = []
for a, b, c in itertools.combinations(range(1, nB + 1), 3):
    for ka in range(1, K + 1):
        for kb in range(1, K + 1):
            for kc in range(1, K + 1):
                A_, B_, C_ = (a, ka), (b, kb), (c, kc)
                j = V[(A_, B_, C_)] - V[(A_, B_)] - V[(A_, C_)] - V[(B_, C_)] + V[(A_,)] + V[(B_,)] + V[(C_,)]
                J.append((j, A_, B_, C_))
jv = [t[0] for t in J]
print(f"\n[3] 세 곳 3 차 상호작용 J (7,722 개): |J| < 1 {sum(abs(v) < 1 for v in jv)}, 양수 {sum(v >= 1 for v in jv)}, 음수 {sum(v <= -1 for v in jv)}, 범위 [{min(jv):.0f}, {max(jv):.0f}]")
# 2 차 근사 V ≈ Σ 한 곳 + Σ 쌍 상호작용 의 오차
err = [abs(t[0]) / abs(V[(t[1], t[2], t[3])]) for t in J]
print(f"   2 차 근사의 상대 오차: 중앙 {st.median(err):.4f}, 90% {sorted(err)[int(0.9*len(err))]:.4f}, 최대 {max(err):.3f}, 0.5% 안 {sum(e <= 0.005 for e in err)} / {len(err)}")
# 3 차 항이 0 이 아닌 경우: 세 거점이 서로 모두 상호작용하는가
def has(a, b): return max(I[(min(a,b), ka, max(a,b), kb)] if a < b else I[(b, kb, a, ka)] for ka in range(1, K+1) for kb in range(1, K+1)) > 1
cnt = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}
for t in J:
    a, b, c = t[1][0], t[2][0], t[3][0]
    n = has(a, b) + has(a, c) + has(b, c)
    cnt[n][0] += 1; cnt[n][1] += abs(t[0]) >= 1
print("   상호작용하는 쌍의 수별 (세 곳 중): " + ", ".join(f"{n} 쌍: {cnt[n][1]}/{cnt[n][0]} 가 3 차 항 있음" for n in range(4)))
