#testbug_GB18
include("GB19.jl")
using DelimitedFiles
import GLM: fit, predict, LinearModel
using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using CSV, Random, DataFrames, Gadfly


rng = Random.seed!(2019)


gbdt = GB18.GBDT(;
  sampling_rate = 0.75,                # Sampling rate
  learning_rate = 0.1,                 # Learning rate
  num_iterations = 100,                # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB18.GBLearner(gbdt, :regression)

x, y = GB18.test_data(500; p = 10, sigma = sqrt(1))
x_test, y_test = GB18.test_data(500; p = 10, sigma = sqrt(1))
mu = GB18.fried(x)
mu_test = GB18.fried(x_test)

# initialize the hypers
hypers = GB18.Hypers(x, y, 1, 1, 0.8, 2)
hypers.σ = sqrt(1)
hypers.τ = 3 / 10 * var(y) / gbdt.num_iterations
hypers.α = 0.95
hypers.γ = 2.0
hypers.s =  ones(size(x, 2)) / size(x, 2)
hypers

# initialize opt
opts = GB18.Opts()
opts.num_burn = 5

rmse2, Store_s = GB18.TBD_predict_and_rmse!(
    x,
    y,
    x_test,
    y_test,
    mu_test,
    gbdt,
    hypers,
    opts,
    rng)




hypers
sum(hypers.varcount) # 2765(burn 460)
sum(hypers.varcount[1:5]) # 1218(burn 460)
sum(hypers.s[1:5])

hypers.s




for k in 1:500
    GB18.UpdateAlpha!(hypers)
end
hypers
alpha_current = hypers.α
hypers.s .!= 0.0

current_s     = hypers.s[hypers.s .!= 0.0]

function alpha_to_rho(alpha::Float64, scale::Float64)
    return alpha / (alpha + scale)
end

function rho_to_alpha(rho::Float64, scale::Float64)
    return scale * rho / (1 - rho)
end
hypers.λₐ
rho_current = alpha_to_rho(alpha_current, hypers.λₐ)

current_s = hypers.s[hypers.s .> 0.]

psi = mean([log(current_s[i]) for i in 1:length(current_s)])
p = 1.0 * length(current_s)

loglik_current = alpha_current * psi + lgamma(alpha_current) -
    p * lgamma(alpha_current / p) +
    logpdf(Beta(hypers.αₐ, hypers.αᵦ), rho_current)


rho_propose = rand(Beta(hypers.αₐ, hypers.αᵦ))
alpha_propose = rho_to_alpha(rho_propose, hypers.α)

loglik_propose = alpha_propose * psi + lgamma(alpha_propose) -
    p * lgamma(alpha_propose / p) +
        logpdf(Beta(hypers.αₐ, hypers.αᵦ), rho_propose)


log(rand()) < loglik_propose - loglik_current

alpha_current = alpha_propose
rho_current = rho_propose
loglik_current = loglik_propose

hypers.α = alpha_current

hypers
