# -*- coding: utf-8 -*-
# toy v6 (mean): one price f for advance and later orders; later orders cost the follower an extra rush cost t_i.  leader's income = f * max(h, demand).
# demand 1 (low) or 3 (high), p = P(high); capacity 3 at one of two sites; advance orders h in [0, 2].  pessimistic ties.
import numpy as np
DL, DH, W = 1.0, 3.0, 2.0
def resp(pt, f, t):
    H = [0.0, DL, W]; c = [f * h + (f + t) * ((1 - pt) * max(DL - h, 0) + pt * max(DH - h, 0)) for h in H]; return [h for h, v in zip(H, c) if v <= min(c) + 1e-9]
def profit(p, pt, f, t, kap): return min((1 - p) * f * max(h, DL) + p * f * max(h, DH) - kap for h in resp(pt, f, t))
S = {"site 1": dict(f=20.0, t=20.0, kap=40.0), "site 2": dict(f=20.0, t=60.0, kap=47.0)}
for nm, k in S.items():
    t = {(a, b): round(profit(a, b, **k), 2) for a in (0.7, 0.4) for b in (0.7, 0.4)}; b0 = t[(0.7, 0.7)]
    print(nm, "threshold %.2f" % (k["f"] / (k["f"] + k["t"])), "h at belief 0.7 / 0.5 / 0.49 / 0.4:", [resp(b, k["f"], k["t"]) for b in (0.7, 0.5, 0.49, 0.4)])
    print("   outcomes (low, high) h=2:", (k["f"] * 2 - k["kap"], k["f"] * 3 - k["kap"]), "h=1:", (k["f"] * 1 - k["kap"], k["f"] * 3 - k["kap"]))
    print("   ", t, "| truth only %+.0f, belief only %+.0f, both %+.0f, extra %+.0f" % (t[(0.4, 0.7)] - b0, t[(0.7, 0.4)] - b0, t[(0.4, 0.4)] - b0, t[(0.4, 0.4)] - t[(0.4, 0.7)] - t[(0.7, 0.4)] + b0))
G = np.linspace(0.4, 1.0, 121)
for lab, D in (("nominal", [(0.7, 0.7)]), ("truth only", [(a, 0.7) for a in G]), ("belief only", [(0.7, b) for b in G]), ("both", [(a, b) for a in G for b in G])):
    print("%-12s" % lab, {nm: round(min(profit(a, b, **k) for a, b in D), 2) for nm, k in S.items()})
better2 = [(a, b) for a in G for b in G if profit(a, b, **S["site 2"]) > profit(a, b, **S["site 1"]) + 1e-9]
print("site 2 better: share of the square %.3f, truth range [%.3f, %.3f], belief range [%.3f, %.3f]" % (len(better2) / len(G) ** 2, min(a for a, b in better2), max(a for a, b in better2), min(b for a, b in better2), max(b for a, b in better2)))
