#testbug_GB14
include("GB14.jl")
using DelimitedFiles
import GLM: fit, predict, LinearModel
using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using CSV
using DataFrames

rng = Random.seed!(2019)


gbdt = GB14.GBDT(;
  sampling_rate = 0.75,                # Sampling rate
  learning_rate = 0.1,                 # Learning rate
  num_iterations = 100,                # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB14.GBLearner(gbdt, :regression)

x, y = GB14.test_data(500; p = 1000, sigma = sqrt(1))
x_test, y_test = GB14.test_data(500; p = 1000, sigma = sqrt(1))
mu = GB14.fried(x)
mu_test = GB14.fried(x_test)

# initialize the hypers
hypers = GB14.Hypers(x, y, 1, 1, 0.8)
hypers.σ = sqrt(1)
hypers.τ = 3 / 10 * var(y) / gbdt.num_iterations
hypers.α = 0.95
hypers.s =  ones(size(x, 2)) / size(x, 2)
hypers

# initialize opt
opts = GB14.Opts()
opts.num_burn = 4950# 500 in total now

# test the new function and get the rmse2 and check the Store_s
rmse2, Store_s = GB14.TBD_predict_and_rmse!(
    x,
    y,
    x_test,
    y_test,
    mu_test,
    gbdt,
    hypers,
    opts,
    rng)
a
## p = 100, σ² = 1
# num_burn = 50:
    #  rmse2 = 1.717
    #  sum(hypers.s[1:5]) = 0.457
# num_burn = 500
    # rmse2 = 1.773
    # sum(hypers.s[1:5]) = 0.447
# num_burn = 1000
    # rmse2 =
    # sum(hypers.s[1:5]) =

##### below is for testing, they work fine #####
#for l in collect(1 : opts.num_burn)

#    hypers_copy = deepcopy(hypers)

#    # fit the model
#    model = GB14.fit!(gbl, x, y, hypers_copy)

#    GB14.UpdateS!(hypers_copy) # returns the updated hypers_copy.s
#    if (opts.update_s) # true
#        hypers.s = hypers_copy.s
#    end
#    if (opts.update_varcount) # that is, the var count is going to be set to zero again
#        hypers.varcount = hypers_copy.varcount
#    end

#end
######## end of testing


hypers
sum(hypers.varcount) # 2765(burn 460)
sum(hypers.varcount[1:5]) # 1218(burn 460)
sum(hypers.s[1:5])

hypers.s


# num_burn = 50, sum(hypers.s[1:5]) = 0.49
# num_burn = 500, sum(hypers.s[1:5]) = wait..
