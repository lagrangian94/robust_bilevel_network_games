# Piece-F Recovery: All-Zero Issue

## 현상
`recover_and_print()` 호출 시 Piece-F obj = 0, λ=0, h=0, 모든 flow=0 반환.
Z₀(Benders) = 4.433 인데 Piece-F = 0.

## 원인 분석
**Piece-F = 0은 정상.** 버그 아님.

ISP-F dual objective:
```
Z^F = -φ̃U Σ x̄ ρ̃¹ - φ̃U Σ(1-x̄) ρ̃³ - λU Σ x̄ ρ⁰¹ - λU Σ(1-x̄) ρ⁰³
```

- 모든 항 ≤ 0 (negative coeff × nonneg var)
- ρ̃² (McCormick (9), RHS=0) 는 objective에 불참
- Dual이 DF-7 (`v·ξ̄·d - ρ̃¹ - ρ̃² + ρ̃³ ≤ 0`) 을 ρ̃² > 0 만으로 만족 가능
- ρ̃¹ = ρ̃³ = ρ⁰¹ = ρ⁰³ = 0 → Z^F = 0

따라서 Z₀ = Z^L + Z^F = Z^L + 0. 전체 값은 Piece-L에서 발생.

## h, λ, ψ⁰ 가 0인 이유
h, λ, ψ⁰는 Piece-F **제약조건**에만 등장하고 **목적함수에 불참**.
→ LP에 obj=0인 최적해가 무한히 존재, solver가 trivial(all-zero) 선택.

## Fix 방안
**2-phase lexicographic recovery:**
1. Phase 1: Piece-F LP 풀어서 최적 obj 확인 (= 0)
2. Phase 2: obj = 0 고정 (제약 추가), max Σ ỹ_ts^s 로 재최적화 → 의미 있는 (h*, λ*, ψ⁰*) 복구
