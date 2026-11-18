# -*- coding: utf-8 -*-
# correction: slack capacity does not protect against a belief error (check chk_slack_role.py).  Fix report 4.4, 4.6, conclusion 6; hypothesis document 9.30; memory.
import io, shutil
D = "C:/Users/user/robust_bilevel_network_games/docs/"; SP = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad/"
p = SP + "report_short_main.md"; s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:40]; s = s.replace(a, b)
def span(a, b, new):   # replace from the start of a up to (not including) b
    global s
    assert s.count(a) == 1 and b in s[s.index(a):], a[:40]; i = s.index(a); j = s.index(b, i); s = s[:i] + new + s[j:]
# ---- 4.4
rep("- 여유 용량은 belief 가 틀렸을 때의 보험이다. 덜 약정한 소매상은 수요가 있을 때 모자란 만큼을 정가로 산다.",
    "- 덜 약정한 소매상은 수요가 있을 때 모자란 만큼을 정가로 산다. 약정이 줄면 그만큼 용량이 비므로 이것은 여유 용량이 없어도 일어난다 (4.6).")
rep(" 여유 용량이 남아 있어 덜 약정한 소매상의 주문을 정가로 받기 때문이다.", " 이유는 4.6 에서 다시 본다.")
# ---- conclusion row 6
span("| 6 | **Coupled 는 두 손잡이를 나눠 쓴다.**", "\n", "| 6 | **두 오류의 피해는 더해지고 손잡이는 따로다. Coupled 는 둘을 함께 쓴다.** 수요 분포가 틀리면 추가 주문 수입이, belief 가 틀리면 약정 수입이 흔들린다. 여유 용량은 앞의 피해만 바꾸고 거점은 뒤의 피해를 바꾼다. Belief-only 는 거점만, Data-DR 은 여유 용량만, Coupled 는 둘 다 쓴다 (4.6) | 여유 용량 Nominal 106, Belief-only 104, Data-DR 77, Coupled 83. 피해 (수요 분포만 / belief 만 / 둘 다) Nominal 5.6 / 5.1 / 11.1, Belief-only 5.6 / 3.3 / 9.5, Data-DR 3.9 / 5.1 / 10.5, Coupled 4.4 / 3.3 / 7.9 |")
# ---- 4.6 opening
span("**여유 용량은 두 가지 일을 한다.** 이것이 네 전략을 가른다.", "**전략.**", """**두 오류는 수입의 서로 다른 부분을 건드리고, 손잡이도 따로다.**

- 수요 분포가 틀리면 여유 용량에서 나오는 추가 주문 수입이 흔들린다. 이 피해는 여유 용량으로 바뀐다.
- belief 가 틀리면 약정 수입이 흔들린다. 이 피해는 거점으로 바뀐다. 여유 용량을 줄여도 바뀌지 않는다.

한쪽만 헤징하는 리더는 한 손잡이만 쓴다. 둘 다 모른다고 두는 Coupled 는 둘을 함께 쓴다.

""")
rep("- **Data-DR 은 여유 용량만 줄인다.** 거점은 belief 를 보지 않고 고르므로 약정이 오히려 더 흔들리는 조합으로 간다 (26.5 → 32.7).",
    "- **Data-DR 은 여유 용량만 줄인다.** 거점은 belief 를 보지 않고 고른다. 빠지는 약정은 더 많지만 (26.5 → 32.7) belief 오류의 피해는 Nominal 과 같다 (아래 표, 5.1).")
rep("약정이 가장 덜 흔들리는 거점 조합을 고르고 (22.0), 여유 용량은 Data-DR 보다 덜 줄인다. 결정이 Nominal 과 다른 건에서 Data-DR 은 여유 용량을 88 줄이고 Coupled 는 43 줄인다.",
    "약정이 가장 덜 흔들리는 거점 조합을 고르고 (22.0), 여유 용량도 줄인다 (83). 결정이 Nominal 과 다른 건에서 Data-DR 은 여유 용량을 88 줄이고 Coupled 는 43 줄인다. 다만 Coupled 는 더 자주 바꾸므로 (55% 대 33%) 전체 평균으로는 비슷하게 줄인다 (83 대 77).")
# ---- 4.6 result bullets
rep("- 두 오류가 함께 오면 피해가 더해진다 (Nominal 11.1 ≈ 5.6 + 5.1). 이때 Coupled 가 줄이는 피해 ($-3.1$) 는 두 헤징을 따로 한 것 ($-1.5$, $-0.6$) 을 더한 것보다 크다.",
    "- 두 오류가 함께 오면 피해가 더해진다 (Nominal 11.1 ≈ 5.6 + 5.1). Coupled 는 두 쪽 보호를 그대로 더해 얻는다 ($-1.2$ 와 $-1.8$ 을 더하면 $-3.0$, 실제 $-3.1$).\n- Data-DR 만은 함께 올 때의 보호 ($-0.6$) 가 수요 쪽 보호 ($-1.7$) 에 못 미친다. 피해가 두 피해의 합보다 1.5 크다 (10.5 대 3.9 + 5.1). 빠지는 약정이 가장 많은 결정이고, 수요가 낮은 쪽이 진짜이면 빠진 약정을 추가 주문으로 되찾지 못하기 때문으로 읽힌다 (확인하지 않음).")
# ---- 4.6 natural insurance block
rep("**여유 용량을 다 줄이지 않는 이유.** 수요 분포는", "**belief 오류의 피해는 꼬리에만 남는다.** 수요 분포는")
span("- Data-DR 의 결정은 belief 오류에 가장 약하다 ($+3.09$).", "\n\n", """- 되찾는 몫은 여유 용량과 무관하다. 되찾는 것은 줄어든 약정을 넘는 수요이고, 약정이 줄면 그만큼 용량이 빈다. 여유 용량이 작은 Data-DR 의 결정도 같은 몫을 되찾는다 (27%, 112%). 마지막 열의 결정 간 차이는 약정이 줄어드는 섭동의 집합이 결정마다 달라 직접 견줄 수 없다.

**여유 용량은 belief 오류를 막아 주지 않는다 (점검).** 결정이 Nominal 과 다른 건에서 여유 용량의 변화와 피해의 변화를 견주었다.

| 결정 | 건수 | 여유 용량 변화 | 약정 변화 | 수요 쪽 피해 변화 | belief 쪽 피해 변화 |
|---|---|---|---|---|---|
| Belief-only (0.1) | 70 | $-6$ | $-8$ | $+0.03 \\pm 0.53$ | $-3.72 \\pm 0.49$ |
| Data-DR (0.1) | 50 | $-88$ | $+4$ | $-5.17 \\pm 0.47$ | $+0.10 \\pm 1.13$ |
| Coupled | 82 | $-43$ | $-19$ | $-2.19 \\pm 0.48$ | $-3.31 \\pm 0.53$ |
| Data-DR (0.1), 같은 거점에서 용량만 줄인 건 | 12 | $-111$ | $-6$ | $-7.17 \\pm 0.79$ | $-0.65 \\pm 0.71$ |

여유 용량의 변화는 수요 쪽 피해의 변화와 함께 움직이고 (건별 상관 0.73~0.89) belief 쪽 피해의 변화와는 무관하다 (상관 $-0.02 \\sim +0.09$). Data-DR 이 여유 용량을 88 줄여도 belief 쪽 피해는 그대로다.""")
# ---- 4.6 paper paragraph
span("**원고에 쓸 요지.**", "\n\n", "**원고에 쓸 요지.** 한쪽 불확실성만 헤징하는 리더는 한 손잡이만 쓴다. Belief-only 는 거점을 옮겨 약정 수입을 지키고, Data-DR 은 여유 용량을 줄여 추가 주문 수입의 위험을 던다. 각 손잡이는 한쪽 피해만 줄이고, 두 오류가 함께 오면 피해는 더해진다. 둘 다 모른다고 두는 Coupled 는 두 손잡이를 함께 써서 두 쪽 보호를 모두 얻고, 두 오류가 함께 올 때의 피해를 가장 많이 줄인다.")
rep("### 4.6 네 리더의 헤징 전략: Coupled 는 두 손잡이를 나눠 쓴다", "### 4.6 네 리더의 헤징 전략: Coupled 는 두 손잡이를 함께 쓴다")
io.open(p, "w", encoding="utf-8", newline="").write(s); shutil.copyfile(p, D + "nz_location_takeorpay_scripts/report_short_main.md")
for f in ("chk_slack_role.py", "rec_930.py"): shutil.copyfile(SP + f, D + "nz_location_takeorpay_scripts/" + f)
H = """
### 9.30 정정: 여유 용량은 belief 오류를 막아 주지 않는다 (2026-11-18 낮, 150 건, `chk_slack_role.py`)

9.25 의 다섯째 항목 ("여유 용량은 약정이 줄었을 때 정가로 되찾는 길"), 9.27 의 넷째 항목 ("자연 보험은 여유 용량이 있어야 작동"), 9.28 의 핵심 문장, 보고서 4.4·4.6 의 같은 문장은 내 해석이었고 점검해 보니 자료가 뒷받침하지 않는다. 거둔다.

- **수리.** 한 거점의 수입은 $f h + p(\\mathrm{med}(h, \\xi, b) - h)$ 다. 약정 $h$ 가 줄 때의 변화는 $\\xi > h$ 이면 $-(p - f)$ (리더 이득), $\\xi < h$ 이면 $+f$ (손해) 이고 용량 $b$ 는 들어가지 않는다. 약정이 줄면 그만큼 용량이 비므로 되찾는 데 원래의 여유 용량이 필요 없다.
- **자료 (결정이 Nominal 과 다른 건, 결정 − Nominal, 가장 나쁜 10 가지 피해 %).** Data-DR (0.1) 50 건: 여유 용량 −88, 수요 쪽 피해 −5.17 ± 0.47, belief 쪽 피해 +0.10 ± 1.13. 같은 거점에서 용량만 줄인 12 건: 여유 용량 −111, belief 쪽 −0.65 ± 0.71. Belief-only (0.1) 70 건: 여유 용량 −6, 수요 쪽 +0.03, belief 쪽 −3.72 ± 0.49. Coupled 82 건: 여유 용량 −43, 약정 −19, 수요 쪽 −2.19 ± 0.48, belief 쪽 −3.31 ± 0.53. 여유 용량 변화와 수요 쪽 피해 변화의 건별 상관 0.73~0.89, belief 쪽과는 −0.02~+0.09.
- **맞는 말.** 두 오류는 수입의 다른 부분을 건드리고 손잡이도 따로다 (여유 용량 ↔ 수요 쪽, 거점·약정 ↔ belief 쪽). 피해는 더해진다. Coupled 는 두 쪽 보호를 더해 얻는다 (−1.2, −1.8, 둘 다 −3.1). "Data-DR 의 결정이 belief 오류에 Nominal 보다 약하다" 도 거둔다. 빠지는 약정은 많지만 (32.7 대 26.5) 피해는 같다 (5.10 대 5.07). 9.25 의 수준 표에서 belief 만 움직일 때 +0.77 인 것은 학습 분포 보험료 (+0.73) 가 그대로 남은 것이다.
- **남는 것.** 9.27 의 앞 세 항목 (평균으로는 92% 되찾고 수요가 약정에 못 미치는 시나리오에서는 24%, 그래서 평균은 비기고 꼬리만 이김) 은 그대로다. 되찾는 몫은 결정 사이에 거의 같다 (바닥 24~38%, 그 밖 112~119%).
- **Data-DR 의 결정만 두 오류가 함께 올 때 피해가 합보다 크다** (10.46 대 3.90 + 5.10 = 9.00, 차 +1.46. Nominal +0.36, Belief-only +0.56, Coupled +0.23). 빠지는 약정이 가장 많고 수요가 낮은 쪽이 진짜이면 그것을 되찾지 못해서로 읽히나 확인하지 않았다.
- **모르는 것.** Coupled 가 결정을 바꿀 때 여유 용량을 Data-DR 의 절반만 줄이는 이유. 전체 평균으로는 비슷하게 줄인다 (83 대 77).
- 같은 거점에서 용량만 줄인 Belief-only·Coupled 의 건 (4~11 건) 에서는 약정이 22~32 줄고 belief 쪽 피해도 준다. 용량을 줄이면 약정 자체가 줄어 잃을 것이 작아지는 경우가 있다 (건수 적음).
"""
p = D + "nz_location_capacity_hypothesis_2026-11-14.md"; t = io.open(p, encoding="utf-8").read(); io.open(p, "w", encoding="utf-8", newline="").write(t.rstrip("\n") + "\n" + H)
MEM = "C:/Users/user/.claude/projects/C--Users-user-robust-bilevel-network-games/memory/"
p = MEM + "nz_location_takeorpay.md"; t = io.open(p, encoding="utf-8").read()
M = "\n- **정정 (2026-11-18 낮, 가설 문서 9.30).** \"여유 용량은 소매상이 덜 약정했을 때 수입을 되찾는 길이라 Coupled 가 덜 줄인다 / Data-DR 은 belief 오류에 더 약해진다\" 는 내 해석이었고 자료가 뒷받침하지 않아 거둠 (사용자가 좋은 인사이트로 꼽았던 문장). 수리: 약정이 줄 때의 수입 변화에 용량 b 가 안 들어감. 자료: Data-DR 이 여유 용량을 88 줄여도 belief 쪽 피해 +0.10 ± 1.13, 같은 거점 용량만 줄인 12 건 −0.65 ± 0.71, 여유 용량 변화와 belief 쪽 피해 변화의 상관 0. 맞는 줄기: 두 오류는 수입의 다른 부분 (추가 주문 / 약정) 을 건드리고 손잡이도 따로 (여유 용량 / 거점), 피해는 더해짐, Coupled 는 두 쪽 보호를 더해 얻음. 해석 문장은 자료로 점검한 뒤에만 보고서에 넣을 것.\n"
io.open(p, "w", encoding="utf-8", newline="").write(t.rstrip("\n") + "\n" + M)
p = MEM + "nz_spot_insurance_tension.md"; t = io.open(p, encoding="utf-8").read()
M = "\n**2026-11-18 낮 정정 (가설 문서 9.30).** 위 재검토의 마지막 문장 (\"Data-DR 이 여유 용량 = 자연 보험의 전제를 줄여 belief 오류에 가장 약해짐\") 은 거둠. 되찾는 것은 줄어든 약정을 넘는 수요이고 여유 용량과 무관 (수리·자료 모두). 평균에서는 되찾고 꼬리에서는 못 되찾는다는 부분은 그대로.\n"
io.open(p, "w", encoding="utf-8", newline="").write(t.rstrip("\n") + "\n" + M); print("recorded")
