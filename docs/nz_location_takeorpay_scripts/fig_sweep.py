# -*- coding: utf-8 -*-
# figure: radius sweep (ledger 4.31).  Mean cost paid (x) against tail gain (y, reduction of the q95 of total cost) for Belief-only, Data-DR, Coupled at radius 0.1, 0.2, 0.3.
#   Input agg_sweep_out.json (agg_sweep.py), 150 runs, c/f 0.4, 30 cells, % of Nominal's training cost.  Panels: (a) truth unknown, belief follows the truth (rho 1); (b) truth known (K).
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/radius_frontier.png"
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
MOD = (("B", "Belief-only", BLUE), ("D", "Data-DR", GRAY), ("C", "Coupled", ORANGE)); RAD = ("0.1", "0.2", "0.3")
D = json.load(open("agg_sweep_out.json"))
fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor=SURF, sharey=True); fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.12, wspace=0.08)
fig.suptitle("반지름을 키우면: Coupled 는 같은 평균 대가에서 Data-DR 보다 꼬리를 더 줄인다", x=0.07, ha="left", fontsize=14.5, color=INK, fontweight="bold")
fig.text(0.07, 0.895, "가격 기준, 지리 15 × 시드 10 = 150 건, 용량비 0.4, 평가 30 칸 평균. 점 옆 숫자는 반지름, 세로 막대는 꼬리 이득의 표준오차", fontsize=10, color=INK2)
def style(ax):
    ax.set_facecolor(SURF); ax.tick_params(colors=MUTED, labelsize=9, length=0); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for k in ("top", "right"): ax.spines[k].set_visible(False)
    for k in ("left", "bottom"): ax.spines[k].set_color(GRID)
OFF = {0: {"B": [(-24, -4), (-12, 9), (4, 9)], "D": [(8, -13), (-2, -15), (8, -4)], "C": [(-22, 6), (-22, 6), (8, 2)]},
       1: {"B": [(-24, -4), (-24, -3), (9, -3)], "D": [(8, 6), (8, 6), (8, 2)], "C": [(8, 6), (8, 6), (8, 2)]}}
END = {0: {"B": (-27, 36), "D": (14, -22), "C": (-10, 24)}, 1: {"B": (12, 22), "D": (10, -24), "C": (14, 22)}}
for iw, (ax, ttl) in enumerate(zip(axs, ("(가) 수요 분포도 모르는 조건 (belief 가 진짜 분포를 따라감)", "(나) 수요 분포를 아는 조건 (belief 만 다름)"))):
    style(ax); k = 1 if iw == 0 else 0
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
    ax.scatter([0], [0], s=90, color=INK, zorder=5, edgecolor=SURF, linewidth=2); ax.annotate("Nominal", (0, 0), xytext=(8, -14), textcoords="offset points", fontsize=10, color=INK)
    for p, nm, col in MOD:
        x = [D[p + r]["mean"][k][0] for r in RAD]; y = [-D[p + r]["q95"][k][0] for r in RAD]; e = [D[p + r]["q95"][k][1] for r in RAD]
        ax.plot([0] + x, [0] + y, color=col, lw=2, zorder=3); ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col, elinewidth=1, alpha=0.55, zorder=2)
        ax.scatter(x, y, s=70, color=col, edgecolor=SURF, linewidth=2, zorder=4)
        for i, r in enumerate(RAD): ax.annotate(r, (x[i], y[i]), xytext=OFF[iw][p][i], textcoords="offset points", fontsize=9, color=INK2)
        ax.annotate(nm, (x[-1], y[-1]), xytext=END[iw][p], textcoords="offset points", fontsize=11, color=INK, fontweight="bold", ha="left")
    ax.set_xlabel("평균 비용 증가, Nominal 대비 % (오른쪽일수록 대가가 큼)", fontsize=10, color=INK2); ax.set_title(ttl, loc="left", fontsize=11.5, color=INK, pad=8)
    ax.set_xlim(-0.4, 5.6)
axs[0].set_ylabel("꼬리 이득: 나쁜 쪽 5% 비용 (q95) 의 감소, % (위일수록 좋음)", fontsize=10, color=INK2); axs[0].set_ylim(-2.2, 5.4)
fig.savefig(OUT, dpi=150, facecolor=SURF); print(OUT)
