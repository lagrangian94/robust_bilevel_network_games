# -*- coding: utf-8 -*-
# figure: Nominal / Belief-only / Coupled (Data-DR as a gray reference) in the world where the leader does not know the demand distribution.
# Input: agg_zss3m_out.txt ("all cells" lines with SE) and agg_zss3m_cells_out.txt (per-cell tables), c/f 0.4 section.   python fig_three.py [rho]
import re, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RHO = sys.argv[1] if len(sys.argv) > 1 else "1.0"
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/three_models_rho%s.png" % RHO.replace(".", "")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
MOD = {"B0.1": ("Belief-only (0.1)", BLUE, True), "B0.2": ("Belief-only (0.2)", BLUE, False), "C": ("Coupled (0.1)", ORANGE, True), "D0.1": ("Data-DR (0.1)", GRAY, True), "D0.2": ("Data-DR (0.2)", GRAY, False)}
# ---- all-cells averages with SE
L = open("agg_zss3m_out.txt", encoding="utf-8").read().splitlines(); i0 = next(i for i, l in enumerate(L) if "c/f 0.4" in l); world = None; ALLC = {}
for l in L[i0:]:
    if l.startswith("---- world"): world = "K" if "world K" in l else l.split("rho = ")[1].split()[0]
    m = re.match(r"\s+all cells\s+(\S+)\s+- N: mean\s+([-+\d.]+) \+- ([\d.]+) \| q05\s+([-+\d.]+) \+- ([\d.]+) \| q95\s+([-+\d.]+) \+- ([\d.]+)", l)
    if m and world == RHO: ALLC[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(6)), float(m.group(7)))
# ---- per-cell tables
L = open("agg_zss3m_cells_out.txt", encoding="utf-8").read().splitlines(); i0 = next(i for i, l in enumerate(L) if "c/f 0.4" in l); cur = None; T = {}
for l in L[i0:]:
    if l.startswith("-- world"): cur = l[3:].split("(rows")[0].strip(); T[cur] = {}
    elif l.strip().startswith("gamma") and cur: p = l.split(); T[cur][float(p[1])] = [float(re.split("±", x)[0]) for x in p[2:]]
GAM = [0.1, 0.3, 0.5, 1.0, 3.0, 10.0]
def tab(w, stat): return np.array([T["world U rho = %s | %s - N | %s" % (RHO, w, stat)][g] for g in GAM])
fig = plt.figure(figsize=(13.5, 6.4), facecolor=SURF)
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], hspace=0.30, wspace=0.22, left=0.07, right=0.87, top=0.84, bottom=0.10)
fig.suptitle("수요 분포도 모르는 조건: Coupled 는 Data-DR 과 같은 평균 대가로 나쁜 쪽 꼬리를 훨씬 더 줄인다", x=0.07, ha="left", fontsize=14.5, color=INK, fontweight="bold")
fig.text(0.07, 0.905, "Nominal 대비 비용 차 (%%). 음수면 Nominal 보다 낫다. 가격 기준, 지리 15 × 시드 10 = 150 건, 용량비 0.4, 소매상 belief 와 진짜 분포의 묶임 ρ = %s" % RHO, fontsize=10, color=INK2)
# ---- (a) price of protection
ax = fig.add_subplot(gs[:, 0]); ax.set_facecolor(SURF)
ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
ax.scatter([0], [0], s=90, color=INK, zorder=4); ax.annotate("Nominal", (0, 0), xytext=(8, 8), textcoords="offset points", fontsize=10.5, color=INK)
OFF = {"B0.1": (10, 6), "B0.2": (14, -6), "C": (10, -4), "D0.1": (10, 6), "D0.2": (10, 6)}
for w, (nm, col, fill) in MOD.items():
    if w not in ALLC: continue
    mx, sx, my, sy = ALLC[w]
    ax.errorbar([mx], [my], xerr=[sx], yerr=[sy], color=col, lw=1.2, capsize=0, zorder=3)
    ax.scatter([mx], [my], s=110, facecolor=col if fill else SURF, edgecolor=col, linewidth=2, zorder=4)
    ax.annotate(nm, (mx, my), xytext=OFF[w], textcoords="offset points", fontsize=10.5, color=INK, va="center")
ax.set_xlabel("평균 비용의 차 (오른쪽일수록 평균을 더 냄)", fontsize=10, color=INK2); ax.set_ylabel("나쁜 쪽 5% 비용 (q95) 의 차 (아래일수록 꼬리를 더 줄임)", fontsize=10, color=INK2)
ax.set_title("(가) 평균 대가와 꼬리 이득, 평가 격자 30 칸 평균 (막대는 표준오차)", loc="left", fontsize=11.5, color=INK, pad=8)
ax.tick_params(colors=MUTED, labelsize=9, length=0); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for k in ("top", "right"): ax.spines[k].set_visible(False)
for k in ("left", "bottom"): ax.spines[k].set_color(GRID)
xl = ax.get_xlim(); ax.set_xlim(min(xl[0], -0.4), xl[1] + 0.9)
# ---- (b), (c) by gamma
for row, (stat, ttl) in enumerate((("q95", "(나) 나쁜 쪽 5% 비용의 차, 진짜 분포가 추정에서 벗어나는 정도별"), ("mean", "(다) 평균 비용의 차"))):
    ax = fig.add_subplot(gs[row, 1]); ax.set_facecolor(SURF); ax.axhline(0, color=INK2, lw=0.8)
    for w in ("D0.1", "B0.1", "C"):
        nm, col, fill = MOD[w]; v = tab(w, stat).mean(1)
        ax.plot(range(6), v, color=col, lw=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=1.5, zorder=3 if w != "D0.1" else 2)
        dy = (0.07 if w == "D0.1" else -0.07 if w == "C" else 0.0) if stat == "mean" else 0.0
        ax.text(5.15, v[-1] + dy, nm, color=INK, fontsize=9.5, va="center")
    ax.set_xticks(range(6)); ax.set_xticklabels(["%g" % g for g in GAM], fontsize=9, color=INK2); ax.set_xlim(-0.3, 5.3)
    ax.tick_params(length=0); ax.tick_params(axis="y", labelsize=8.5, colors=MUTED); ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for k in ("top", "right", "left"): ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.set_title(ttl, loc="left", fontsize=11.5, color=INK, pad=6)
    if row == 1: ax.set_xlabel("γ  (왼쪽: 진짜 분포가 추정에서 크게 벗어남,  오른쪽: 추정 근처)", fontsize=9.5, color=INK2)
fig.savefig(OUT, dpi=150, facecolor=SURF); print(OUT)
for w in ("B0.1", "B0.2", "C", "D0.1", "D0.2"): print(w, "q95 by gamma", np.round(tab(w, "q95").mean(1), 2).tolist(), "mean", np.round(tab(w, "mean").mean(1), 2).tolist(), "calm cell q95 %+.2f mean %+.2f" % (tab(w, "q95")[5][0], tab(w, "mean")[5][0]))
