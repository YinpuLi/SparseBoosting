#testbug_GB100
include("GB100.jl")
using DelimitedFiles
import GLM: fit, predict, LinearModel
using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using CSV, Random, DataFrames, Gadfly
using MLBase

rng = Random.seed!(2019)


gbdt = GB100.GBDT(;
  sampling_rate         = 0.75,                # Sampling rate
  learning_rate         = 0.1,                 # Learning rate
  num_iterations        = 100,                # NuGBer of iterations
  tree_options          = Dict(
      :nsubfeatures     => 0,
      :min_samples_leaf => 1,
      :max_depth        => 5))
gbl                     = GB100.GBLearner(gbdt, :regression)

x, y                    = GB100.test_data(1000; p = 1000, sigma = sqrt(10))
x_test, y_test          = GB100.test_data(500; p = 1000, sigma = sqrt(10))
mu                      = GB100.fried(x)
mu_test                 = GB100.fried(x_test)

# initialize the hypers
hypers = GB100.Hypers(x, y, 1, 1, 0.8)
hypers.σ = sqrt(1)
hypers.τ = 3 / 10 * var(y) / gbdt.num_iterations
hypers.α = 1
hypers.λ = 20 # hypers.λ = 1.0 or hypers.λ = nothing
hypers.s =  ones(size(x, 2)) / size(x, 2)
hypers

# initialize opt
opts = GB100.Opts()
opts.num_burn = 5

rmse2 = GB100.TBD_predict_and_rmse2!(
    x,
    y,
    mu,
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
sum(hypers.s[1:5])
sum(hypers.s)



###### Tuning ̂σ with cross-validation method #####

rng = Random.seed!(2019)


gbdt = GB100.GBDT(;
  sampling_rate         = 0.75,                # Sampling rate
  learning_rate         = 0.1,                 # Learning rate
  num_iterations        = 100,                # NuGBer of iterations
  tree_options          = Dict(
      :nsubfeatures     => 0,
      :min_samples_leaf => 1,
      :max_depth        => 5))
gbl                     = GB100.GBLearner(gbdt, :regression)

x, y                    = GB100.test_data(500; p = 1000, sigma = sqrt(10))
x_test, y_test          = GB100.test_data(500; p = 1000, sigma = sqrt(10))
mu                      = GB100.fried(x)
mu_test                 = GB100.fried(x_test)

# initialize the hypers
hypers = GB100.Hypers(x, y, 1, 1, 0.8)
hypers.σ = sqrt(1)
hypers.τ = 3 / 10 * var(y) / gbdt.num_iterations
hypers.α = 1
hypers.λ = nothing # hypers.λ = 1.0 or hypers.λ = nothing
hypers.s =  ones(size(x, 2)) / size(x, 2)
hypers

# initialize opt
opts = GB100.Opts()
opts.num_burn = 5

param2update = Dict(:σ => [1,3])
# collect(range(1.0; length = 100, stop = 10.0))
BestHypers, BestCVError, ListCVErr = GB100.TunePara_CV(x, y, mu, gbdt, opts, hypers, param2update, rng = rng)
BestHypers
