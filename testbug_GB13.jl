#testbug_GB13
include("GB13.jl")
using DelimitedFiles
import GLM: fit, predict, LinearModel
using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using CSV
using DataFrames

rng = Random.seed!(2019)


gbdt = GB13.GBDT(;
  sampling_rate = 0.75,                # Sampling rate
  learning_rate = 0.1,                 # Learning rate
  num_iterations = 100,                # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB13.GBLearner(gbdt, :regression)

x, y = GB13.test_data(500; p = 10, sigma = sqrt(10))
x_test, y_test = GB13.test_data(500; p = 10, sigma = sqrt(10.0))
mu = GB13.fried(x)
mu_test = GB13.fried(x_test)

# initialize the hypers
hypers = GB13.Hypers(x, y, 1, 1, 0.8)
hypers.σ = sqrt(1)
hypers.τ = 3 / 10 * var(y) / gbdt.num_iterations
hypers.α = 0.95
hypers.s = hypers.α .* ones(size(x, 2)) / size(x, 2)
hypers

# initialize opt
opts = GB13.Opts()
opts.num_burn = 400 # 500 in total now

for l in collect(1 : opts.num_burn)

    hypers_copy = deepcopy(hypers)

    # fit the model
    model = GB13.fit!(gbl, x, y, hypers_copy)

    GB13.UpdateS!(hypers_copy) # returns the updated hypers_copy.s
    if (opts.update_s) # true
        hypers.s = hypers_copy.s
    end
    if (opts.update_varcount) # that is, the var count is going to be set to zero again
        hypers.varcount = hypers_copy.varcount
    end

end


hypers
