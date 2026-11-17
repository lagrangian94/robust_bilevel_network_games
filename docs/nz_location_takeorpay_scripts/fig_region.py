# -*- coding: utf-8 -*-
# figure: the regional-factor demand scenarios of one geography (default g1, seed 1). Input geo/mech/xid_g<g>_s<s>.csv (xidW.jl).
import csv, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
g, s = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("1", "1")
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/region_factor_g%s_s%s.png" % (g, s)
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
RC = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}; RN = {1: "서부", 2: "중부", 3: "동부"}
rows = list(csv.DictReader(open(O + "xid_g%s_s%s.csv" % (g, s), encoding="utf-8"), delimiter=";"))
A = [r["name"] for r in rows if r["kind"] == "A"]; C = [r for r in rows if r["kind"] == "C"]
name = [r["name"].split(",")[0] for r in C]; full = [r["name"] for r in C]; reg = np.array([int(r["region"]) for r in C]); mu = np.array([float(r["mu"]) for r in C])
x = np.array([float(r["x"]) for r in C]); y = np.array([float(r["y"]) for r in C]); xi = np.array([[float(r["xi%d" % k]) for k in range(1, 11)] for r in C])
tot = xi.sum(0); so = np.argsort(tot)                                   # scenarios, lowest total demand first
co = sorted(range(len(C)), key=lambda j: (reg[j], -mu[j]))              # customers grouped by region, largest first
fig = plt.figure(figsize=(13.5, 8.2), facecolor=SURF)
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.55, 1.0], hspace=0.16, wspace=0.30, left=0.05, right=0.885, top=0.86, bottom=0.08)
fig.suptitle("지역 요인 수요: 같은 지역의 고객은 함께 움직이고, 지역끼리는 따로 움직인다", x=0.05, ha="left", fontsize=15, color=INK, fontweight="bold")
fig.text(0.05, 0.915, "지리 %s, 시드 %s. 고객 15 곳, 시나리오 10 개 (확률 0.1 씩). 수요 = 평균 수요 × (0.5 ~ 1.5), 지역 안 상관 0.9, 지역 사이는 독립" % (g, s), fontsize=10.5, color=INK2)
# ---- (a) map
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(SURF)
for r in (1, 2, 3):
    m = reg == r
    if not m.any(): continue
    ax.scatter(x[m], y[m], s=mu[m] * 5.5, color=RC[r], edgecolor=SURF, linewidth=2, zorder=3)
    ax.text(x[m].mean(), y[m].max() + 170, "%s (%d 곳)" % (RN[r], m.sum()), color=INK, fontsize=11.5, ha="center", fontweight="bold")
for j in range(len(C)):
    if full[j] in A: ax.scatter([x[j]], [y[j]], s=mu[j] * 5.5 + 190, facecolor="none", edgecolor=INK, linewidth=1.4, zorder=4)
ax.set_aspect("equal", adjustable="datalim"); ax.margins(0.10, 0.22); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_color(GRID)
ax.set_title("(가) 고객 도시와 지역", loc="left", fontsize=12, color=INK, pad=8)
ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=11, label="기존 사업자의 거점"),
           Line2D([], [], marker="o", ls="", color=MUTED, markersize=6, label="원의 크기 = 평균 수요")], loc="lower left", frameon=False, fontsize=9, labelcolor=INK2, ncol=2)
axt = fig.add_subplot(gs[1, 0]); axt.axis("off")
axt.text(0.0, 0.98, "시나리오 하나는 이렇게 만든다", fontsize=12, color=INK, va="top", fontweight="bold")
T = ["1. 지역마다 주사위를 하나씩 던진다 (지역 요인 3 개, 서로 독립).",
     "2. 고객마다 작은 주사위를 하나 더 던진다 (고객 고유 요인).",
     "3. 두 값을 0.9 대 0.44 의 비중으로 섞어 0.5 ~ 1.5 사이의 배수로 바꾼다.",
     "4. 고객 수요 = 평균 수요 × 그 배수.",
     "",
     "→ (나) 같은 지역의 고객은 한 시나리오에서 같은 색이다.",
     "→ (다) 큰 지역이 낮은 시나리오가 총수요도 낮다. 전체를 함께",
     "    움직이는 요인은 없지만 지역이 3 개뿐이라 총수요가 크게 갈린다."]
for k, t in enumerate(T): axt.text(0.0, 0.84 - 0.105 * k, t, fontsize=10.5, color=INK2, va="top")
# ---- (b) heat map of demand / mean demand
ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(SURF)
cm = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a88", "#b02a2a"])
rel = xi[co][:, so] / mu[co][:, None]; ypos = []; yy = 0.0; prev = None
for j in co:
    if prev is not None and reg[j] != prev: yy += 0.45
    ypos.append(yy); yy += 1.0; prev = reg[j]
for a, j in enumerate(co):
    for b in range(10):
        ax.add_patch(plt.Rectangle((b + 0.04, ypos[a] + 0.04), 0.92, 0.92, color=cm((rel[a, b] - 0.5) / 1.0), lw=0))
ax.set_xlim(0, 10); ax.set_ylim(yy, 0); ax.set_yticks([p + 0.5 for p in ypos]); ax.set_yticklabels([name[j] for j in co], fontsize=8, color=INK2)
ax.set_xticks([]); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
for r in (1, 2, 3):
    idx = [a for a, j in enumerate(co) if reg[j] == r]
    if not idx: continue
    y0, y1 = ypos[idx[0]], ypos[idx[-1]] + 1
    ax.plot([10.12, 10.12], [y0 + 0.08, y1 - 0.08], color=RC[r], lw=4, solid_capstyle="butt", clip_on=False)
    ax.text(10.3, (y0 + y1) / 2, RN[r], color=INK, fontsize=10.5, va="center", ha="left", fontweight="bold")
ax.set_title("(나) 고객별 수요 ÷ 평균 수요   (시나리오를 총수요가 낮은 순으로 줄 세움)", loc="left", fontsize=12, color=INK, pad=8)
cax = fig.add_axes([0.955, 0.50, 0.011, 0.30]); sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0.5, 1.5)); cb = fig.colorbar(sm, cax=cax, ticks=[0.5, 1.0, 1.5])
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5, colors=INK2, length=0); cb.ax.set_title("배", fontsize=9, color=INK2, pad=4)
# ---- (c) total demand per scenario, stacked by region
ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor(SURF); bot = np.zeros(10)
for r in (1, 2, 3):
    m = reg == r
    if not m.any(): continue
    v = xi[m].sum(0)[so]; ax.bar(np.arange(10) + 0.5, v - 6, bottom=bot + 3, width=0.62, color=RC[r], lw=0, label=RN[r])
    ax.text(10.05, bot[-1] + v[-1] / 2, RN[r], color=INK, fontsize=10, va="center", ha="left"); bot += v
for b in range(10): ax.text(b + 0.5, bot[b] + 18, "%.0f" % bot[b], ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.axhline(tot.mean(), color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0); ax.text(0.05, bot.max() * 1.08, "평균 %.0f" % tot.mean(), fontsize=8.5, color=MUTED, va="bottom")
ax.set_xlim(0, 10); ax.set_ylim(0, bot.max() * 1.16); ax.set_xticks(np.arange(10) + 0.5); ax.set_xticklabels(["s%d" % (k + 1) for k in so], fontsize=9, color=INK2)
ax.tick_params(length=0); ax.set_yticks([0, 400, 800, 1200]); ax.tick_params(axis="y", labelsize=8.5, colors=MUTED); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for k in ("top", "right", "left"): ax.spines[k].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_title("(다) 시나리오별 총수요 (지역별로 쌓음)", loc="left", fontsize=12, color=INK, pad=8)
ax.set_xlabel("시나리오 (총수요가 낮은 순)", fontsize=9.5, color=INK2)
fig.savefig(OUT, dpi=150, facecolor=SURF); print(OUT)
r_tot = [xi[reg == r].sum(0) for r in (1, 2, 3) if (reg == r).any()]
print("total min/max %.2f, CV %.3f; regional corr:" % (tot.min() / tot.max(), tot.std() / tot.mean()), np.round(np.corrcoef(r_tot), 2).tolist())
print("within-region mean pairwise corr:", [round(float(np.mean([np.corrcoef(xi[i], xi[j])[0, 1] for i in np.where(reg == r)[0] for j in np.where(reg == r)[0] if i < j])), 2) for r in (2, 3) if (reg == r).sum() > 1])
