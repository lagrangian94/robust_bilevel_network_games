# 기저 파일 기준으로 N1/Ffeas 중 느슨한 행과 ŷ/ȳ 중 비기저 열을 뺀 MPS 를 쓴다 (oracle 활성 집합)
# mode: rows (행만 뺌 = 완화) | both (행+열)
import sys, re
sys.stdout.reconfigure(encoding='utf-8')
mps, bas, out, mode = sys.argv[1:5]
basic_cols = set(); nb_rows = set(); rnames = set()
lines = open(mps, encoding='utf-8').read().splitlines()
sec = None
for l in lines:
    if not l.startswith(' ') and not l.startswith('*'): sec = l.split()[0]; continue
    if sec == 'ROWS': rnames.add(l.split()[1])
for l in open(bas, encoding='utf-8'):
    t = l.split()
    if not t or t[0] in ('NAME', 'ENDATA'): continue
    if t[0] in ('XU', 'XL'): basic_cols.add(t[1]); nb_rows.add(t[2])
    elif t[0] in ('UL', 'LL') and t[1] in rnames: nb_rows.add(t[1])
isrow = lambda r: re.match(r'(N1|Ffeas)\[', r) is not None
iscol = lambda c: re.match(r'(ŷ|ȳ)\[', c) is not None
drop_r = {r for r in rnames if isrow(r) and r not in nb_rows}
drop_c = set()
sec = None; outl = []
for l in lines:
    if not l.startswith(' ') and not l.startswith('*'):
        sec = l.split()[0]; outl.append(l); continue
    t = l.split()
    if sec == 'ROWS':
        if t[1] in drop_r: continue
    elif sec == 'COLUMNS':
        if mode == 'both' and iscol(t[0]) and t[0] not in basic_cols: drop_c.add(t[0]); continue
        keep = [(t[k], t[k+1]) for k in range(1, len(t) - 1, 2) if t[k] not in drop_r]
        if t[0] == 'MARKER' or not keep: 
            if t[0] == 'MARKER': outl.append(l)
            continue
        l = '    ' + t[0] + ' ' + ' '.join(f'{a} {b}' for a, b in keep)
    elif sec in ('RHS', 'RANGES'):
        keep = [(t[k], t[k+1]) for k in range(1, len(t) - 1, 2) if t[k] not in drop_r]
        if not keep: continue
        l = '    ' + t[0] + ' ' + ' '.join(f'{a} {b}' for a, b in keep)
    elif sec == 'BOUNDS':
        if t[2] in drop_c: continue
    outl.append(l)
open(out, 'w', encoding='utf-8').write('\n'.join(outl) + '\n')
print(f'{out}: rows dropped {len(drop_r)}, cols dropped {len(drop_c)}')
