include("GB20.jl")
using MLBase
using DelimitedFiles
import GLM: fit, predict, LinearModel
using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using CSV, Random
using DataFrames
using SharedArrays

rng = Random.seed!(2019)

gbdt = GB20.GBDT(;
  sampling_rate = 0.75,                # Sampling rate
  learning_rate = 0.1,                # Learning rate
  num_iterations = 100,               # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB20.GBLearner(gbdt, :regression)

############## σ² = 1 #################
Pval = []
for i in 1:1000
    if (10 <= i <= 19)
        push!(Pval, i)
    elseif (20 <= i <= 99)
        if (i % 10 == 0)
            push!(Pval, i)
        end
    else
        if (i % 100 == 0)
            push!(Pval, i)
        end

    end
end

Pval
temp_storedim = size(collect(Pval),1)

rmsefitted2 = []
HyperS = Array{Union{Missing, Nothing, Int, Float64}, 2}(nothing, temp_storedim, 1000)
TunedParams = Array{Union{Nothing, Float64}, 2}(nothing, temp_storedim, 3) # 28×3 Array, storing (α, τ, σ)


STORE = []
Pval
Pval
for pval in Pval
    # reproducibility
    rng = Random.seed!(2019)

    ### simulate the training and testing data
    X, Y = GB20.test_data(500; p = pval, sigma = 1.0)
    X_test, Y_test = GB20.test_data(500; p = pval, sigma = 1.0)
    μ = GB20.fried(X)
    μ_test = GB20.fried(X_test)

    hypers = GB20.Hypers(X, Y, 1, 1, 0.95, 0.5)
    hypers.σ = sqrt(1)
    hypers.τ = 3 / 10 * var(Y) / gbdt.num_iterations
    hypers.α = 0.95
    hypers.s = ones(size(X, 2)) / size(X, 2)


    # initialize opt
    opts = GB20.Opts()
    opts.num_burn = 5

    ### Exhaustive grid search for best param-combination ###
    param2update  = Dict(   :α => collect(range(0.8; length = 5, stop = 1.0)),
                            :τ => collect(range(0.05; length = 5, stop = 0.1)),
                            :σ => collect(range(1; length = 3, stop = 3)))
    BestHypers, BestCVError = GB20.TunePara_CV(X, Y, μ, gbdt, opts, hypers, 5, param2update; rng = rng)

    # store the tuned paramters:
    k = findall(x -> x == pval, Pval)
    TunedParams[k, :] = [BestHypers.α, BestHypers.τ, BestHypers.σ]
    #TunedParams[k, 1] = BestHypers.α
    #TunedParams[k, 2] = BestHypers.τ
    #TunedParams[k, 3] = BestHypers.σ

    ### Retrain the model with best params-combination and Evaluate on the Test set ###
    rmse2, Store_s = GB20.TBD_predict_and_rmse!(
        X,
        Y,
        X_test,
        Y_test,
        μ_test,
        gbdt,
        BestHypers,
        opts,
        rng)

    println("The rmse2 b.w. μ_test and pred for p = $(pval)  is $(rmse2).")

    push!(rmsefitted2, rmse2)
    push!(STORE, Store_s)

    # store the hyper.s for the ensembel tree
    j = findall(x -> x == pval, Pval)
    HyperS[j, 1:size(BestHypers.s, 1)] = BestHypers.s


end

rmsefitted2
HyperS
TunedParams



######## σ² = 10 #########
Pval = []
for i in 1:1000
    if (10 <= i <= 19)
        push!(Pval, i)
    elseif (20 <= i <= 99)
        if (i % 10 == 0)
            push!(Pval, i)
        end
    else
        if (i % 100 == 0)
            push!(Pval, i)
        end

    end
end

Pval
temp_storedim = size(collect(Pval),1)
rmsefitted2_sigma10 = []
HyperS_sigma10 = Array{Union{Missing, Nothing, Int, Float64}, 2}(nothing, temp_storedim, 1000)
TunedParams_sigma10 = Array{Union{Nothing, Float64}, 2}(nothing, temp_storedim, 3) # 28×3 Array, storing (α, τ, σ)

STORE_sigma10 = []
Pval

for pval in Pval[1]
    # reproducibility
    rng = Random.seed!(2019)

    ### simulate the training and testing data
    X, Y = GB20.test_data(500; p = pval, sigma = sqrt(10.0))
    X_test, Y_test = GB20.test_data(500; p = pval, sigma = sqrt(10.0))
    μ = GB20.fried(X)
    μ_test = GB20.fried(X_test)

    hypers = GB20.Hypers(X, Y, 1, 1, 0.95, 2.0)
    hypers.σ = sqrt(1)
    hypers.τ = 3 / 10 * var(Y) / gbdt.num_iterations
    hypers.α = 0.95
    hypers.s = ones(size(X, 2)) / size(X, 2)


    # initialize opt
    opts = GB20.Opts()
    opts.num_burn = 5

    ### Exhaustive grid search for best param-combination ###
    param2update  = Dict(   :α => collect(range(0.8; length = 5, stop = 1.0)),
                            :τ => collect(range(0.05; length = 5, stop = 0.1)),
                            :σ => collect(range(1; length = 3, stop = 3)))
    BestHypers, BestCVError = GB20.TunePara_CV(X, Y, μ, gbdt, opts, hypers, 5, param2update; rng = rng)

    # store the tuned paramters:
    k = findall(x -> x == pval, Pval)
    TunedParams_sigma10[k, :] = [BestHypers.α, BestHypers.τ, BestHypers.σ]
    #TunedParams[k, 1] = BestHypers.α
    #TunedParams[k, 2] = BestHypers.τ
    #TunedParams[k, 3] = BestHypers.σ

    # train the model and get the final model as well as rmse2
    rmse2, Store_s = GB20.TBD_predict_and_rmse!(
        X,
        Y,
        X_test,
        Y_test,
        μ_test,
        gbdt,
        BestHypers,
        opts,
        rng)

    println("The rmse2 b.w. μ_test and pred for p = $(pval)  is $(rmse2).")

    push!(rmsefitted2_sigma10, rmse2)
    push!(STORE_sigma10, Store_s)

    # store the hyper.s for the ensembel tree
    j = findall(x -> x == pval, Pval)
    HyperS_sigma10[j, 1:size(BestHypers.s, 1)] = BestHypers.s

    #println("After $(opts.num_burn) times of burning, the sum of the first 5 elements in the prob vec is: $(sum(hypers.s[1:5])).")
end


rmsefitted2_sigma10
HyperS_sigma10
TunedParams_sigma10
