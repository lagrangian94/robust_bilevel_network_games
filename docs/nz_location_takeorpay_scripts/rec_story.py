# -*- coding: utf-8 -*-
# report: put the cross-term storyline into the conclusions (row 7) and into section 6 (implications for the manuscript); hypothesis document note; memory
import io, shutil
D = "C:/Users/user/robust_bilevel_network_games/docs/"; p = "report_short_main.md"; s = io.open(p, encoding="utf-8").read()
r6 = "| 6 | **손잡이는 따로이고,"; assert s.count(r6) == 1; k = s.index(r6); k2 = s.index("\n", k) + 1
ROW = "| 7 | **두 불확실성을 따로 헤징하면 놓치는 항이 있다 (논문의 줄기로 제안).** 수요 분포와 소매상의 belief 가 같은 방향으로 틀리면, 피해는 두 피해의 합보다 크다. 진짜 분포가 무게를 더한 저수요 시나리오에서 그 분포를 믿은 소매상의 약정도 줄어 있기 때문이다. Data-DR 은 소매상이 $\hat q$ 를 믿는다고 두고 Belief-only 는 진짜가 $\hat q$ 라고 두므로 둘 다 이 항을 모형에 갖고 있지 않다. 두 분포를 묶는 Coupled 만 본다 (4.6) | 교차항은 함께 올 때 피해의 약 4 분의 1 이다. Nominal 2.6 / 11.1, Belief-only 2.3 / 9.5, Data-DR 2.9 / 10.5, Coupled 2.1 / 7.9. Nominal 대비 Coupled $-0.50 \pm 0.17$, Data-DR $+0.29 \pm 0.16$ |\n"
s = s[:k2] + ROW + s[k2:]
a = "## 6. 원고에 대한 함의\n\n| | 내용 |\n|---|---|\n"; assert s.count(a) == 1
NEW = a + "| 줄기로 제안 | **두 불확실성을 따로 헤징하면 놓치는 항이 있다.** 수요 분포의 오류와 소매상 belief 의 오류가 같은 방향으로 오면 피해가 두 피해의 합보다 크고, 그 차이가 함께 올 때 피해의 약 4 분의 1 이다. 한쪽 분포를 $\hat q$ 에 고정하는 모형 (Data-DR, Belief-only) 은 이 항을 갖지 않는다. 두 분포를 한 집합에 묶는 것이 필요한 이유를 숫자로 보인다 (4.6). 조건: 한 쌍 옮기기 섭동 90 가지, 소매상이 움직인 분포를 그대로 믿는 경우, 예약료 0.75 한 구성 |\n"
s = s.replace(a, NEW)
a = "| 못 씀 | Coupled 가 Belief-only 보다 낫다 (5 절의 3) |"; assert s.count(a) == 1
s = s.replace(a, "| 쓸 수 있음 | 한쪽만 헤징하는 리더는 한 손잡이만 쓴다. Belief-only 는 거점 배치를 바꿔 belief 에 따라 움직이는 약정을 기존 사업자 쪽으로 넘기고, Data-DR 은 여유 용량을 줄인다. Coupled 는 둘을 함께 쓴다 (4.6) |\n| 쓸 수 있음 (조건 명시) | Coupled 는 Data-DR 과 같은 평균 대가로 꼬리를 더 줄인다. Belief-only 보다는 평균을 1 가량 더 내고 꼬리를 2 가량 더 줄이므로 어느 쪽이 낫다고는 못 한다. Coupled 의 결정은 점 탐색으로 고른 것이다 (4.5, 5 절의 3) |")
io.open(p, "w", encoding="utf-8", newline="").write(s); shutil.copyfile(p, D + "nz_location_takeorpay_scripts/report_short_main.md"); shutil.copyfile("rec_story.py", D + "nz_location_takeorpay_scripts/rec_story.py")
q = D + "nz_location_capacity_hypothesis_2026-11-14.md"; t = io.open(q, encoding="utf-8").read()
io.open(q, "w", encoding="utf-8", newline="").write(t.rstrip("\n") + "\n\n- **논문 줄기 (2026-11-18 낮, 사용자 승인으로 보고서 결론 7 번·6 절에 넣음).** 9.31 (나) 의 교차항. 두 불확실성을 따로 헤징하면 놓치는 항이 있고 그 크기가 함께 올 때 피해의 약 4 분의 1 이다. 원고 tex 반영은 아직 안 함.\n")
m = "C:/Users/user/.claude/projects/C--Users-user-robust-bilevel-network-games/memory/nz_location_takeorpay.md"; t = io.open(m, encoding="utf-8").read()
io.open(m, "w", encoding="utf-8", newline="").write(t.rstrip("\n") + "\n- **논문 줄기 후보 (2026-11-18, 사용자가 보고서에 넣으라고 함).** 교차항: 두 불확실성을 따로 헤징하면 놓치는 항이 있고 함께 올 때 피해의 약 4 분의 1 (보고서 결론 7 번, 6 절 첫 줄, 가설 문서 9.31 나). 원고 tex 반영은 허락 필요.\n"); print("ok")
