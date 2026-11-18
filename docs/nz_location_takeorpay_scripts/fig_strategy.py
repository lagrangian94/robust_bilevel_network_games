# -*- coding: utf-8 -*-
# figure: hedging strategies of the four leaders (Nominal, Belief-only 0.1, Data-DR 0.1, Coupled).  Input agg_mechc_out.txt (parts 1a and 2), c/f 0.4, 150 runs.
#   (a) the two levers: slack capacity (x) and the commitment lost under the 10 worst beliefs (y)
#   (b) the result: damage when only the demand distribution is wrong (x) and when only the belief is wrong (y); label carries the damage when both are wrong
import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "C:/Users/user/robust_bilevel_network_games/docs/figs/nz_location_takeorpay/four_strategies.png"
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
MOD = {"N": ("Nominal", INK), "B0.1": ("Belief-only", BLUE), "D0.1": ("Data-DR", GRAY), "C": ("Coupled", ORANGE)}
L = open("agg_mechc_out.txt", encoding="utf-8").read().splitlines(); part = None; LV = {}; DM = {}
for l in L:
    if l.startswith("-- part"): part = l.split(":")[0][3:].strip()
    p = [x.strip() for x in l.split("|")]
    if part == "part 1a" and p[0] in MOD and len(p) >= 2: a = p[1].split(); LV[p[0]] = (float(a[0]), float(a[1]), float(a[2]))          # cap, commit, slack
    if part == "part 2" and p[0] in MOD and len(p) >= 5: DM[p[0]] = (float(p[1].split()[1]), float(p[2].split()[1]), float(p[3].split()[1]), -float(p[4].split()[0]))   # data, belief, both (worst 10), commitment lost
print(LV); print(DM)
fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.0), facecolor=SURF); fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.12, wspace=0.22)
fig.suptitle("네 리더의 헤징 전략: Coupled 는 거점과 여유 용량을 함께 쓴다", x=0.07, ha="left", fontsize=14.5, color=INK, fontweight="bold")
fig.text(0.07, 0.895, "가격 기준, 반지름 0.1, 지리 15 × 시드 10 = 150 건 평균, 용량비 0.4. 화살표는 Nominal 의 결정에서 움직인 방향", fontsize=10, color=INK2)
def style(ax):
    ax.set_facecolor(SURF); ax.tick_params(colors=MUTED, labelsize=9, length=0); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for k in ("top", "right"): ax.spines[k].set_visible(False)
    for k in ("left", "bottom"): ax.spines[k].set_color(GRID)
def draw(ax, P, off, extra=None):
    x0, y0 = P["N"]
    for w, (nm, col) in MOD.items():
        x, y = P[w]
        if w != "N": ax.annotate("", xy=(x, y), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, shrinkA=7, shrinkB=7), zorder=2)
        ax.scatter([x], [y], s=130, color=col, edgecolor=SURF, linewidth=2, zorder=4)
        ax.annotate(nm + (extra[w] if extra else ""), (x, y), xytext=off[w], textcoords="offset points", fontsize=10.5, color=INK, va="center", ha="left" if off[w][0] > 0 else "right")
# ---- (a) the two levers
ax = axs[0]; style(ax); P = {w: (LV[w][2], DM[w][3]) for w in MOD}
draw(ax, P, {"N": (12, 0), "B0.1": (12, -2), "D0.1": (12, 4), "C": (12, -8)}, {w: "\n여유 %.0f, 빠지는 약정 %.1f" % P[w] for w in MOD})
ax.set_xlabel("여유 용량: 총용량에서 약정을 뺀 것 (왼쪽일수록 여유 용량을 줄임)", fontsize=10, color=INK2)
ax.set_ylabel("belief 가 틀릴 때 빠지는 약정 (아래일수록 약정이 덜 흔들리는 거점)", fontsize=10, color=INK2)
ax.set_title("(가) 전략: 두 손잡이를 얼마나 썼나", loc="left", fontsize=11.5, color=INK, pad=8)
xs = [p[0] for p in P.values()]; ys = [p[1] for p in P.values()]; ax.set_xlim(min(xs) - 8, max(xs) + 26); ax.set_ylim(min(ys) - 2.5, max(ys) + 2.5)
# ---- (b) the result
ax = axs[1]; style(ax); P = {w: (DM[w][0], DM[w][1]) for w in MOD}
draw(ax, P, {"N": (12, 0), "B0.1": (12, -2), "D0.1": (4, 24), "C": (12, -8)}, {w: "\n함께 틀리면 %.1f" % DM[w][2] for w in MOD})
ax.set_xlabel("수요 분포만 틀릴 때의 피해, % (왼쪽일수록 작음)", fontsize=10, color=INK2); ax.set_ylabel("소매상의 belief 만 틀릴 때의 피해, % (아래일수록 작음)", fontsize=10, color=INK2)
ax.set_title("(나) 결과: 두 오류에 얼마나 약한가", loc="left", fontsize=11.5, color=INK, pad=8)
xs = [p[0] for p in P.values()]; ys = [p[1] for p in P.values()]; ax.set_xlim(min(xs) - 0.4, max(xs) + 1.1); ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
fig.savefig(OUT, dpi=150, facecolor=SURF); print(OUT)
