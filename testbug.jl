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

hypers = GB13.Hypers(x,y,1,1,0.8)
hypers.σ = sqrt(1)
hypers.τ = 3 / 10 * var(y) / gbdt.num_iterations
hypers.α = 0.95
hypers.s = hypers.α .* ones(size(x, 2)) / size(x, 2)
#hypers.varcount = fill(0, size(hypers.s, 1))

hypers
# test the tree part
mytree    = GB13.build_tree(
    y            ,    # y vector
    x ,    # X matrix
    hypers,
    0,      # n_subfeatures
    5,      # max_depth
    1,      # min_samples_leaf
    2,      # min_samples_split
    0.)

GB13.print_tree(mytree)


root, indX = GB13._fit(
    x,
    y,
    fill(1.0, size(x, 1)),
    0,
    5,
    1,
    2,
    0.0,
    hypers,
    rng) # the results of the tree and the varcount match;
            # the varcount stores the splitting counting number in one tree




mymodel=GB13.fit!(gbl, x, y, hypers)
# get the fitted value
myfit  = GB13.predict!(gbl, x)

# get the residual sum of squares
sqrt(mean(y .- myfit) .^2) # 0.0103

# get prediction
mypred = GB13.predict!(gbl, x_test)

# get mse and root of mean squared error
sqrt(mean((y_test .- mypred ) .^2)) #3.8

sqrt(mean((mypred .- mu_test ) .^2)) #2.0
# 把公式25换了之后，rmse2 for p = 1000 is 2.548, which is lower than 3, better!!!
