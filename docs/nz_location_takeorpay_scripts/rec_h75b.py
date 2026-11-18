# -*- coding: utf-8 -*-
# record: r = 1800 result of the unserved-demand screening (H75 supplement), ledger 4.29, hypothesis document 9.29, memory
import io, shutil
D = "C:/Users/user/robust_bilevel_network_games/docs/"; SP = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad/"
V = """
 H75 보충, r = 1800 (2026-11-18 11:20 무렵. 설정 `dwps3r18`, 45 건, 08:01~11:00, 실패 0. 예측은 r = 1400 에만 적었으므로 판정이 아니라 같은 항목의 값만 적는다. 출력 사본 `agg_zsr18_out_2026-11-18.txt`):
  안 채운 수요 6.0% (r = 1400 은 13.4%), 다른 지역 판매 18% (10%, 지금 32%), 점포 하나가 파는 고객 4.9 곳, 총수요와의 상관 0.79, Nominal 이 세 지역에 다 여는 건 31%.
  Data-DR (0.1) 다른 결정 40%, 그중 총용량 줄임 78%. 최악 분포의 쏠림 79%.
  Belief-only (0.1) 아는 조건 q95 −1.25 ± 0.45, 평균 +0.46. Coupled ρ = 1 평균 +0.07 ± 0.29, q95 −4.31 ± 1.13. Data-DR (0.1) +0.76, −2.21.
  읽기: r 을 올리면 지금 모형 쪽으로 돌아간다 (r = 1400 과 지금 모형의 사이). 두 r 모두에서 Data-DR 의 한 방향은 그대로이고 Coupled 의 꼬리 이득은 남는다.
"""
for p in (D + "nz_location_takeorpay_scripts/cf_predictions.txt", SP + "cf_predictions.txt"):
    s = io.open(p, encoding="utf-8").read(); io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + V)
L = """- **r = 1800 선별 결과 (45 건, 2026-11-18 11:00 끝, 실패 0).** 안 채운 수요 6.0%, 다른 지역 판매 18%, 리더의 몫 74%, 총수요와의 상관 0.79, 세 지역에 다 여는 건 31%, 총용량 698. Data-DR (0.1) 다른 결정 40%, 그중 총용량 줄임 78%, 쏠림 79%. 평가: 아는 조건 Belief-only (0.1) +0.46·−1.25 ± 0.45. ρ = 1: Data-DR (0.1) +0.76·−2.21, Coupled +0.07 ± 0.29·−4.31 ± 1.13.
- **상태 (2026-11-18 11:20).** 선별 끝. 두 r 모두 Data-DR 의 한 방향을 못 바꾸고 Coupled 의 값어치는 남음. 본 구성은 지금 모형 (수요 반드시 충족) 그대로 두고 이 모형은 메모로만 남기는 것을 제안함 (사용자 결정 대기). 150 건 확대는 하지 않음.
"""
p = D + "nz_location_config_log.md"; s = io.open(p, encoding="utf-8").read(); io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + L)
H = """
### 9.29 소매상이 주문을 버릴 수 있는 모형: 선별 결과 (2026-11-18 오전, 45 건씩, 실패 0)

대장 4.29. 소매상이 수요를 안 채워도 되고 안 채운 단위마다 판매가 r 을 잃는다. r = 1400·1800, 점포 3 곳 이하, 지리 15 × 시드 1~3, 용량비 0.4. 맨 왼쪽은 같은 45 건의 지금 모형이다. 평가는 30 칸 평균, Nominal 대비 %.

| | 지금 모형 | r = 1800 | r = 1400 |
|---|---|---|---|
| 안 채운 수요 | 0 | 6.0% | 13.4% |
| 리더 판매 중 다른 지역 고객 몫 | 32% | 18% | 10% |
| 리더의 몫 | 73% | 74% | 73% |
| Nominal 이 세 지역에 다 여는 건 | 11% | 31% | 44% |
| 리더 수입과 총수요의 상관 | 0.83 | 0.79 | 0.75 |
| Data-DR (0.1) 다른 결정 중 총용량 줄임 | 78% | 78% | 80% |
| 최악 분포가 총수요 낮은 세 시나리오로 가는 몫 | 91% | 79% | 72% |
| Belief-only (0.1), 아는 조건: 평균 / q95 | −0.22 / −1.65 ± 0.51 | +0.46 / −1.25 ± 0.45 | −0.07 / −3.25 ± 1.32 |
| Data-DR (0.1), ρ = 1: 평균 / q95 | +0.96 / −1.45 | +0.76 / −2.21 | +0.47 / −2.00 |
| Coupled, ρ = 1: 평균 / q95 | +0.65 / −3.29 ± 0.71 | +0.07 / −4.31 ± 1.13 | +0.30 / −3.99 ± 1.40 |

- **시장은 지역 단위로 갈리지만 Data-DR 의 한 방향은 그대로다.** 다른 지역 판매가 32 → 10% 로 줄어도 리더가 세 지역에 나눠 열어서 (11 → 44%) 수입이 여전히 총수요를 따라간다. H75 c 기각.
- **belief 헤징과 Coupled 의 값어치는 남는다.** Coupled 의 꼬리 이득은 세 설정에서 −3.3~−4.3 이고 평균 대가는 오히려 작다. 45 건이라 표준오차가 크다 (0.7~1.4). Belief-only 가 약해질 것이라는 예측 (H75 d) 은 빗나갔다.
- 읽기: 9.23·9.26 (점포 2 곳) 과 함께, 인스턴스를 바꿔 Data-DR 의 대응을 거점별로 갈리게 하려던 두 시도는 모두 안 됐다. 한 방향인 것은 지역 요인 수요에서 리더가 여러 지역에 걸쳐 여는 한 따라오는 성질로 보인다. Coupled 는 그 안에서도 거점별로 올리고 내린다 (다른 결정의 78~88%).
- 150 건 확대와 추가 분석은 하지 않았다. 채택 여부는 사용자 결정.
"""
p = D + "nz_location_capacity_hypothesis_2026-11-14.md"; s = io.open(p, encoding="utf-8").read(); io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + H)
for a, b in (("agg_zsr18_out.txt", "agg_zsr18_out_2026-11-18.txt"), ("rec_h75b.py", "rec_h75b.py")): shutil.copyfile(SP + a, D + "nz_location_takeorpay_scripts/" + b)
p = "C:/Users/user/.claude/projects/C--Users-user-robust-bilevel-network-games/memory/nz_location_takeorpay.md"; s = io.open(p, encoding="utf-8").read()
M = "\n- **주문 버리기 r = 1800 (2026-11-18 11:00, 45 건, 가설 문서 9.29).** r = 1400 과 지금 모형의 사이: 안 채운 수요 6%, Data-DR 다른 결정 중 총용량 줄임 78%, Coupled ρ = 1 +0.07·−4.31 ± 1.13. 점포 2 곳·주문 버리기 두 시도 모두 Data-DR 의 한 방향을 못 바꿈, Coupled 값어치는 세 설정에서 유지. 메모로만 남기자고 제안함 (사용자 결정 대기), 150 건 확대 안 함.\n"
io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + M); print("recorded")
