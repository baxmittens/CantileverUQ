using Distributions
using ProgressMeter
import .Threads: @threads
using CairoMakie
using Measurements
using Statistics

function cantilever_displacement(E,X,Y,L,w,t)
    # see https://www.sfu.ca/~ssurjano/canti.html
    Dfact1 = 4*(L^3) / (E*w*t)
    Dfact2 = sqrt((Y/(t^2))^2 + (X/(w^2))^2)
    D = Dfact1 * Dfact2
    return D
end

N_E = Normal{Float64}(2.9e7, 1.45e6) # Young's modulus
N_X = Normal{Float64}(500.0, 50.0)	# horizontal load
N_Y = Normal{Float64}(1000.0, 100.0) # vertical load
N_L = Normal{Float64}(100.0, 10.0) # beam length
N_w = Normal{Float64}(4.0, 0.4) # beam width
N_t = Normal{Float64}(2.0, 0.2) # beam width

M_E = measurement(2.9e7, 1.45e6) # Young's modulus
M_X = measurement(500.0, 50.0)	# horizontal load
M_Y = measurement(1000.0, 100.0) # vertical load
M_L = measurement(100.0, 10.0) # beam length
M_w = measurement(4.0, 0.4) # beam width
M_t = measurement(2.0, 0.2) # beam width

samplefunc() = [rand(N_E), rand(N_X), rand(N_Y), rand(N_L), rand(N_w), rand(N_t)]

N = 5_000_000
monte_carlo_resvec = Vector{Float64}(undef, N)

@showprogress @threads for i = 1:N
    monte_carlo_resvec[i] = cantilever_displacement(samplefunc()...)
end

sample_mean = foldl(+, monte_carlo_resvec)/N
sample_var = foldl(+, map(x->(x-sample_mean)^2, monte_carlo_resvec))/(N-1)
sample_sqrt_var = sqrt(sample_var)
quantvals = map(x->cdf(Normal{Float64}(0.0,1.0),x), -3:3)
sample_quantiles = quantile(monte_carlo_resvec, quantvals)

σ_F = cantilever_displacement(M_E,M_X,M_Y,M_L,M_w,M_t)

f = Figure(size=(600,300));
ax = Axis(f[1,1:4])
xlims!(ax, [0,20])
ylims!(ax, [0,.25])

band!(ax, [sample_quantiles[1], sample_quantiles[2]], 0, 5, color=(:red,0.1), label="99.7% quantile")
band!(ax, [sample_quantiles[end-1], sample_quantiles[end]], 0, 5, color=(:red,0.1))
band!(ax, [sample_quantiles[2], sample_quantiles[3]], 0, 5, color=(:yellow,0.2), label="95.4% quantile")
band!(ax, [sample_quantiles[end-2], sample_quantiles[end-1]], 0, 5, color=(:yellow,0.2))
band!(ax, [sample_quantiles[3], sample_quantiles[5]], 0, 5, color=(:green,0.2), label="68.2% quantile")
hist!(ax, monte_carlo_resvec, normalization = :pdf, bins = 500, color=(:red, 0.45), strokewidth=0.1, label="histogram")
lines!(ax, [sample_mean, sample_mean], [0.0,0.25], label="exp. value")
lines!(ax, [sample_mean+sample_sqrt_var, sample_mean+sample_sqrt_var], [0.0,0.25], label="mean+√var")
lines!(ax, [sample_mean-sample_sqrt_var, sample_mean-sample_sqrt_var], [0.0,0.25], label="mean-√var")
lines!(ax, [σ_F.val, σ_F.val], [0.0,0.25], label="σ_F.val",linestyle=:dash)
lines!(ax, [σ_F.val+σ_F.err, σ_F.val+σ_F.err], [0.0,0.25], label="σ_F.val+σ_F.err",linestyle=:dash)
lines!(ax, [σ_F.val-σ_F.err, σ_F.val-σ_F.err], [0.0,0.25], label="σ_F.val-σ_F.err",linestyle=:dash)
Legend(f[1,5], ax, labelsize=8)

f