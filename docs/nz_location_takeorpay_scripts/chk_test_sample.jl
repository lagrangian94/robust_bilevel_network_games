using Random, Distributions, Statistics, Printf
f(z) = cdf(Normal(), 0.9 * z / sqrt(1 + (1 - 0.81)))          # E over idiosyncratic noise of Phi(0.9 z + sqrt(0.19) e)
low(v, a) = (s = sort(v); mean(s[1:round(Int, a * length(s))]))
Zt = randn(MersenneTwister(hash((4242, 7770001))), 3, 500); Zb = randn(MersenneTwister(1), 3, 2_000_000)
for (lab, g) in (("region1", Z -> f.(Z[1, :])), ("region2", Z -> f.(Z[2, :])), ("region3", Z -> f.(Z[3, :])), ("avg of 3", Z -> vec(mean(f.(Z); dims=1))))
    t = 0.5 .+ g(Zt); b = 0.5 .+ g(Zb)
    @printf("%-9s mean test %.4f true %.4f (%+.2f%%) | lowest 60%% avg test %.4f true %.4f (%+.2f%%) | lowest 25%% avg test %.4f true %.4f (%+.2f%%)\n", lab, mean(t), mean(b), 100 * (mean(t) / mean(b) - 1), low(t, 0.6), low(b, 0.6), 100 * (low(t, 0.6) / low(b, 0.6) - 1), low(t, 0.25), low(b, 0.25), 100 * (low(t, 0.25) / low(b, 0.25) - 1))
end
# spread of the same statistic over independent test samples of 500
d = [begin Z = randn(MersenneTwister(1000 + k), 3, 500); t = 0.5 .+ vec(mean(f.(Z); dims=1)); low(t, 0.6) end for k in 1:400]
@printf("lowest-60%% avg of 3-region mean over 400 independent test samples: sd %.2f%% of its mean\n", 100 * std(d) / mean(d))
d1 = [begin Z = randn(MersenneTwister(5000 + k), 3, 500); t = 0.5 .+ f.(Z[1, :]); low(t, 0.6) end for k in 1:400]
@printf("same for a single region: sd %.2f%%\n", 100 * std(d1) / mean(d1))
