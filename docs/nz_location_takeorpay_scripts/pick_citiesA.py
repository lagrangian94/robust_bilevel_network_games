# variant of geo/pick_cities.py: same 15 cities and same first two A cities, plus extra A cities (stronger incumbent). usage: pick_citiesA.py <g> <nA>
import random, sys
R = "C:/Users/user/robust_bilevel_network_games/nonzero_sum/data/sgb128/sgb128_name.txt"
names = [l.strip() for l in open(R, encoding="utf-8") if l.strip() and not l.startswith("#")]
g = int(sys.argv[1]); nA = int(sys.argv[2])
rng = random.Random(g); C = rng.sample(names, 15); A = rng.sample(C, 2)
rest = [c for c in C if c not in A]; A = A + random.Random(10000 + g).sample(rest, nA - 2)
print('GEO_CITIES="%s" GEO_A="%s"' % (";".join(C), ";".join(A)))
