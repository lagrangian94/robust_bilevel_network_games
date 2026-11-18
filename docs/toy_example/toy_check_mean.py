# -*- coding: utf-8 -*-
# toy with low/high demand: demand 1 (low) or 3 (high), pi = P(high).  capacity 3 at one of two sites.
# follower orders h in [0, W] in advance at price f, and the shortfall (xi - h)^+ later at price p_i, of which the leader receives m.  pessimistic ties.
import numpy as np
DL, DH, W = 1.0, 3.0, 2.0
def short(h, pi): return (1 - pi) * max(DL - h, 0) + pi * max(DH - h, 0)
def resp(pt, f, p):
    H = [0.0, DL, W]; c = [f * h + p * short(h, pt) for h in H]; return [h for h, v in zip(H, c) if v <= min(c) + 1e-9]
def profit(pi, pt, f, p, m, kap): return min(f * h + m * short(h, pi) - kap for h in resp(pt, f, p))
S = {"site 1": dict(f=9.0, p=15.0, m=10.0, kap=19.0), "site 2": dict(f=9.0, p=30.0, m=10.0, kap=22.0)}
for nm, k in S.items():
    t = {(a, b): round(profit(a, b, **k), 2) for a in (0.7, 0.5) for b in (0.7, 0.5)}; b0 = t[(0.7, 0.7)]
    print(nm, "threshold %.2f" % (k["f"] / k["p"]), "h at belief 0.7 / 0.5:", resp(0.7, k["f"], k["p"]), resp(0.5, k["f"], k["p"]), t,
          "| truth only %+.1f, belief only %+.1f, both %+.1f, extra %+.1f" % (t[(0.5, 0.7)] - b0, t[(0.7, 0.5)] - b0, t[(0.5, 0.5)] - b0, t[(0.5, 0.5)] - t[(0.5, 0.7)] - t[(0.7, 0.5)] + b0))
G = np.linspace(0.5, 0.9, 81)
for lab, D in (("nominal", [(0.7, 0.7)]), ("truth only", [(a, 0.7) for a in G]), ("belief only", [(0.7, b) for b in G]), ("both", [(a, b) for a in G for b in G])):
    print("%-12s" % lab, {nm: round(min(profit(a, b, **k) for a, b in D), 2) for nm, k in S.items()})
