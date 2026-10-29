import sys, re, collections
sys.stdout.reconfigure(encoding='utf-8')
mps, bas = sys.argv[1], sys.argv[2]
def pref(n):
    p = re.sub(r'\[.*', '', n)
    if re.fullmatch(r'R\d+', p): return None
    if p.startswith('rltc_'): return 'rltc(sticky)'
    return p
rows = {}; rtype = {}; order = []
cols = collections.OrderedDict()   # col -> list of (row, val)
sec = None
with open(mps, encoding='utf-8') as f:
    for line in f:
        if line.startswith('*'): continue
        if not line.startswith(' '):
            sec = line.split()[0]; continue
        t = line.split()
        if sec == 'ROWS':
            rtype[t[1]] = t[0]; order.append(t[1])
        elif sec == 'COLUMNS':
            if t[0] == 'MARKER' or (len(t) > 1 and t[1] == "'MARKER'"): continue
            c = t[0]; L = cols.setdefault(c, [])
            for k in range(1, len(t) - 1, 2): L.append((t[k], float(t[k+1])))
colpref = {}
for c in cols:
    p = pref(c); colpref[c] = p if p else 'C(anon)'
rowcols = collections.defaultdict(list)
for c, L in cols.items():
    for r, v in L:
        if r != 'OBJ': rowcols[r].append(c)
def rgroup(r):
    p = pref(r)
    if p: return p
    s = sorted(set(colpref[c] for c in rowcols[r]))
    return 'anon{' + ','.join(s) + '}'
# basis
basic_cols = set(); nonbasic_rows = set()
with open(bas, encoding='utf-8') as f:
    for line in f:
        t = line.split()
        if not t or t[0] in ('NAME', 'ENDATA'): continue
        if t[0] in ('XU', 'XL'): basic_cols.add(t[1]); nonbasic_rows.add(t[2])
        elif t[0] in ('UL', 'LL') and t[1] in rtype: nonbasic_rows.add(t[1])
G = collections.defaultdict(lambda: [0, 0, 0, ''])
for r in order:
    if rtype[r] == 'N': continue
    g = rgroup(r); G[g][0] += 1; G[g][1] += len(rowcols[r]); G[g][2] += (r in nonbasic_rows); G[g][3] = rtype[r]
print('ROWS: group | sense | count | nnz | binding(nonbasic slack)')
for g, (n, z, b, s) in sorted(G.items(), key=lambda x: -x[1][0]):
    print(f'  {g:60s} {s} {n:7d} {z:8d} {b:7d}')
C = collections.defaultdict(lambda: [0, 0, 0])
for c, L in cols.items():
    p = colpref[c]; C[p][0] += 1; C[p][1] += len(L); C[p][2] += (c in basic_cols)
print('COLS: prefix | count | nnz | basic')
for p, (n, z, b) in sorted(C.items(), key=lambda x: -x[1][0]):
    print(f'  {p:20s} {n:7d} {z:8d} {b:7d}')
print('total rows', sum(v[0] for v in G.values()), 'cols', len(cols), 'nnz', sum(v[1] for v in C.values()), 'basic cols', len(basic_cols), 'nonbasic rows', len(nonbasic_rows))
