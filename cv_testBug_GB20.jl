#### Paprameter Estimation using grid search with Cross-Validation ####
# Code in this part shows how the Algorithm is optimized by Cross-Validation.
include("GB20.jl")
using MLBase
import GLM: fit, predict, LinearModel
using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using CSV, Random
using DataFrames
using SharedArrays
rng = Random.seed!(2019)
################################################################################
##### Step 0: Specify the hyper-parameters to be updated in the Algorithm ######
################################################################################
# algorithm specification

gbdt = GB20.GBDT(;
  sampling_rate = 0.75,                # Sampling rate
  learning_rate = 0.1,                 # Learning rate
  num_iterations = 100,                # Number of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB20.GBLearner(gbdt, :regression)

# data specification
x, y            = GB20.test_data(500; p = 10, sigma = sqrt(1))
x_test, y_test  = GB20.test_data(500; p = 10, sigma = sqrt(1))
mu              = GB20.fried(x)
mu_test         = GB20.fried(x_test)
# initialize opt
opts = GB20.Opts()
opts.num_burn = 5

# initialize the hypers
hypers      = GB20.Hypers(x, y, 1, 1, 0.8, 2)
hypers.σ    = sqrt(1)
hypers.τ    = 3 / 10 * var(y) / gbdt.num_iterations
hypers.α    = 0.95
hypers.γ    = 2.0
hypers.s    =  ones(size(x, 2)) / size(x, 2)

# hyper-params needed to be updated through cross-validation
# param2update    = [hypers_temp.α, hypers_temp.τ, hypers_temp.σ]
hypers


# K-folds:
#n, p = size(x)
#cv_folds = 5
# folds_indx: is the index of the sub-sample in each fold
#folds_indx  = collect(Kfold(n, cv_folds))

#trainindx   = folds_indx[1]
#testindx    = setdiff(collect(1:n), folds_indx[1])
#x_CVtrain = x[folds_indx[1], :]
#y_CVtrain = y[folds_indx[1]]
#length(folds_indx)
#n
################################################################################
########### Step 1: Tuning the hyper-parameters of the Algorithm ###############
################################################################################

# Hyper parameters in this step are parameters that are not directly learned
# by the algorithm, so the hypers.s is not included.

# It is possible and recommended to search the hyper-parameter space(without
# hypers.s for the best cross-validation score.)

# Two generic approaches to sampling search candidates are possible:
### Method 1: for given values, GridSearchCV() exhaustively considers all
#             parameter-combinations.
### Method 2: for given values, RandomizedSearchCV() can sample a given number
#             of candidates from a parameter space with a specified distribution.
# I am using Method 1: GridSearchCV().

################# Exhaustive grid search ##################

param2update  = Dict(   :α => collect(range(0.8; length = 5, stop = 1.0)),
                        :τ => collect(range(0.05; length = 5, stop = 0.1)),
                        :σ => collect(range(1; length = 3, stop = 3)))


@time BestHypers, BestCVError = GB20.TunePara_CV(x, y, mu, gbdt, opts, hypers, 5, param2update; rng = rng)
#161.196413 seconds (329.58 M allocations: 52.879 GiB, 3.60% gc time)
BestHypers
BestCVError
################################################################################
################ Step 2: Evaluate on a new data set ############################
################################################################################

# The performance of the selected hyper-parameters and trained model is then
# measured on a dedicated evaluation set(X_test and Y_test) that was not used
# in the model selection step.

# As we have obtained the best parameter combination from Step 1, we could now
# use these tuned paramters to retrain the model with the train set(x, y, mu)
# and evaluate the model on a totally new test set(x_test, y_test) and get the
# final evaluation.

@time rmse2, Store_s = GB20.TBD_predict_and_rmse!(
    x,
    y,
    x_test,
    y_test,
    mu_test,
    gbdt,
    BestHypers,
    opts,
    rng)
#  0.715254 seconds (1.13 M allocations: 185.964 MiB, 3.49% gc time)
println("And the final evaluation on test data is: rmse = $rmse2.")
