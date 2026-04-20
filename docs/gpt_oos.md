Finite-Support φ-Divergence DRO with Factor Model (Implementation Guide)
1. Overview

We consider a finite-support DRO setting:

Scenario set: {c¹, ..., cᴺ}
Nominal distribution (used in optimization):
p̂ᵢ = 1/N

Goal:

Keep the support fixed
Construct a non-uniform true distribution p* over the same scenarios
Evaluate out-of-sample performance under p*
2. Factor Model (Data Generation)

Latent structure:

c = F ξ

ξ ∈ ℝ₊² (k = 2 factors)
ξⱼ ~ Exponential(μⱼ), independent
μⱼ ~ Uniform(0,1)

Matrix:

F ∈ ℝ₊^{num_interdictable × 2}
entries ∼ Uniform{1,...,10}

Capacity construction:

Interdictable arcs:
c_intd = F ξ
Non-interdictable arcs:
use a fixed constant (NOT sample max)
Dummy arc:
sum of all regular arcs
3. Required Code Modification

You must store ξ for each scenario.

Example (Julia):

ξ_scenarios = zeros(k, num_scenarios)

for scenario in 1:num_scenarios
  ξ = [rand(Exponential(μ[i])) for i in 1:k]
  ξ_scenarios[:, scenario] = ξ
  capacity_scenarios[intd_idx, scenario] = F * ξ
end

4. Baseline Distribution q(ξ)

Latent data-generating distribution:

q(ξ) = ∏_{j=1}^k (1/μⱼ) exp(-ξⱼ / μⱼ)

Interpretation:

Larger ξ ⇒ lower probability
Large capacity scenarios are rare
5. Stress Distribution g(ξ)

We construct g to emphasize large-capacity scenarios (bad for interdictor).

Scale-up exponential:

μ̃ⱼ = α · μⱼ, where α > 1

g(ξ) = ∏_{j=1}^k (1/μ̃ⱼ) exp(-ξⱼ / μ̃ⱼ)

Recommended:

α = 1.2 (mild)
α = 1.3 (moderate)
α = 1.5 (strong)
6. Importance Weights

For each scenario i:

wᵢ = g(ξᵢ) / q(ξᵢ)

Expanded:

wᵢ = ∏_{j=1}^k (μⱼ / μ̃ⱼ) · exp(-ξᵢⱼ (1/μ̃ⱼ - 1/μⱼ))

Key property:

If μ̃ⱼ > μⱼ:

→ weights increase with ξ
→ large capacity scenarios are overweighted

7. Normalization

Convert weights into probabilities:

pᵢ* = wᵢ / ∑ₜ wₜ

Important:

Sum of wᵢ is NOT automatically 1
Always normalize
8. Mixture (Recommended)

Define:

g = (1 - ε) q + ε g_stress

Then:

wᵢ = (1 - ε) + ε · (g_stress(ξᵢ) / q(ξᵢ))

Recommended:

ε ∈ {0.1, 0.2, 0.3}
9. Julia Implementation

Basic weights:

function compute_weights(ξ_scenarios, μ, μ_tilde)
  k, N = size(ξ_scenarios)
  w = ones(N)

  for i in 1:N
    for j in 1:k
      ξ = ξ_scenarios[j,i]
      w[i] = (μ[j]/μ_tilde[j]) * exp(-ξ(1/μ_tilde[j] - 1/μ[j]))
    end
  end

  w /= sum(w)
  return w
end

Mixture weights (recommended):

function compute_weights_mixture(ξ_scenarios, μ, μ_tilde, ε)
  k, N = size(ξ_scenarios)
  w = ones(N)

  for i in 1:N
    ratio = 1.0
    for j in 1:k
      ξ = ξ_scenarios[j,i]
      ratio = (μ[j]/μ_tilde[j]) * exp(-ξ(1/μ_tilde[j] - 1/μ[j]))
    end
    w[i] = (1-ε) + ε * ratio
  end

  w /= sum(w)
  return w
end

10. Full Pipeline

Step 1: Generate scenarios

Sample ξᵢ ~ q
Compute cᵢ = F ξᵢ

Step 2: Optimization

Use uniform nominal: p̂ᵢ = 1/N

Step 3: Construct true distribution

Compute weights wᵢ
Normalize → p*

Step 4: Evaluation

Compute E_{p*}[L(x, c)]
11. Interpretation (Interdictor)
Large capacity ⇒ harder to disrupt ⇒ worse scenario
g increases probability of large ξ
p* shifts mass toward adversarial scenarios
12. Practical Tips
Avoid weight collapse: start with α ≤ 1.3
Always store ξ
Keep support fixed (no new scenarios)
Avoid sample-dependent scaling (no max over scenarios)
13. Key Takeaways
Nominal distribution: uniform
True distribution: importance-weighted
g: exponential with larger scale
Same support maintained
Stress scenarios emphasized