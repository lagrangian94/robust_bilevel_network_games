# -*- coding: utf-8 -*-
# toy with a risk-averse leader: tail profit = average profit over the worst (1-beta) share of outcomes (CVaR of the loss at level beta), beta = 0.4.
# demand 1 (low) or 3 (high), pi = P(high); capacity 3 at one of two sites; advance orders h in [0, 2] at price f; later orders cost the follower m + t_i, the leader receives m.
import numpy as np
DL, DH, W, BETA = 1.0, 3.0, 2.0, 0.4
def resp(pt, f, p):
    H = [0.0, DL, W]; c = [f * h + p * ((1 - pt) * max(DL - h, 0) + pt * max(DH - h, 0)) for h in H]; return [h for h, v in zip(H, c) if v <= min(c) + 1e-9]
def outcomes(h, f, m, kap): return f * h + m * max(DL - h, 0) - kap, f * h + m * max(DH - h, 0) - kap          # profit when demand is low, high
def tail(pi, h, f, m, kap):
    gl, gh = outcomes(h, f, m, kap); lo, hi = (gl, 1 - pi), (gh, pi)
    (g1, p1), (g2, p2) = sorted([lo, hi]); a = 1 - BETA                                                   # worst outcome first
    return g1 if p1 >= a else (p1 * g1 + (a - p1) * g2) / a
def val(pi, pt, f, t, m, kap): return min(tail(pi, h, f, m, kap) for h in resp(pt, f, m + t))
S = {"site 1": dict(f=9.0, t=5.0, m=10.0, kap=8.0), "site 2": dict(f=9.0, t=20.0, m=10.0, kap=14.0)}
for nm, k in S.items():
    print(nm, "threshold %.2f" % (k["f"] / (k["m"] + k["t"])), "outcomes (low, high) with h=2:", outcomes(2.0, k["f"], k["m"], k["kap"]), "with h=1:", outcomes(1.0, k["f"], k["m"], k["kap"]))
    t = {(a, b): round(val(a, b, **k), 2) for a in (0.7, 0.4) for b in (0.7, 0.4)}; b0 = t[(0.7, 0.7)]
    print("   tail profit", t, "| truth only %+.1f, belief only %+.1f, both %+.1f, extra %+.1f" % (t[(0.4, 0.7)] - b0, t[(0.7, 0.4)] - b0, t[(0.4, 0.4)] - b0, t[(0.4, 0.4)] - t[(0.4, 0.7)] - t[(0.7, 0.4)] + b0))
    print("   mean profit", {(a, b): round(min((1 - a) * outcomes(h, k["f"], k["m"], k["kap"])[0] + a * outcomes(h, k["f"], k["m"], k["kap"])[1] for h in resp(b, k["f"], k["m"] + k["t"])), 2) for a in (0.7, 0.4) for b in (0.7, 0.4)})
G = np.linspace(0.4, 1.0, 121)
for lab, D in (("nominal", [(0.7, 0.7)]), ("truth only", [(a, 0.7) for a in G]), ("belief only", [(0.7, b) for b in G]), ("both", [(a, b) for a in G for b in G])):
    print("%-12s" % lab, {nm: round(min(val(a, b, **k) for a, b in D), 2) for nm, k in S.items()})
