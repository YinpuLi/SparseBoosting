using DelimitedFiles
include("GB13.jl")
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
  learning_rate = 0.1,                # Learning rate
  num_iterations = 100,               # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB13.GBLearner(gbdt, :regression)

######## σ² = 1 #########


Pval = []



for i in 10 : 1000

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


temp_storedim = size(collect(Pval),1)
rmsefitted2 = []
HyperS = Array{Union{Missing, Nothing, Int, Float64}, 2}(nothing, temp_storedim, 1000)
s
for pval in Pval
    # reproducibility
    Random.seed!(2019)
    rng = 2019

    ### simulate the training and testing data
    X, Y = GB13.test_data(500; p = pval, sigma = 1.0)
    X_test, Y_test = GB13.test_data(500; p = pval)
    μ = GB13.fried(X)
    μ_test = GB13.fried(X_test)

    hypers = GB13.Hypers(X, Y, 1, 1, 0.95)
    hypers.σ = sqrt(1)  ### TODO: I am using
    hypers.τ = 3 / 10 * var(Y) / gbdt.num_iterations
    hypers.α = 0.95
    hypers.s =  hypers.α .* ones(size(X, 2)) / size(X, 2)

    # initialize opt
    opts = GB13.Opts()
    opts.num_burn = 500

    # train the model and get the final model as well as rmse2
    rmse2 = GB13.TBD_predict_and_rmse!(
        X,
        Y,
        X_test,
        Y_test,
        μ_test,
        gbdt,
        hypers,
        opts,
        rng)

    println("The rmse2 b.w. μ_test and pred for p = $(pval)  is $(rmse2).")
    println("\nThe first 5 elemnts in hypers.s for p = $(pval) is $(hypers.s[1:5]).\n")
    push!(rmsefitted2, rmse2)

    # store the hyper.s for the ensembel tree
    j = findall(x -> x == pval, Pval)
    HyperS[j, 1:size(hypers.s, 1)] = hypers.s
    println("\nThe sum of the first 5 elemnts in the prob vec is $(sum(hypers.s[1:5])).\n")

end

rmsefitted2
HyperS

HyperS_copy = copy(HyperS)
for i in 1 : size(HyperS, 1)
    for j in 1 : size(HyperS, 2)
        if HyperS_copy[i, j] == nothing
            HyperS_copy[i, j] = 0.0
        end
    end
end
HyperS_copy


rmsefitted2 = DataFrame(rmsefitted2 = rmsefitted2)
CSV.write("_rmse2_sigma1.csv", rmsefitted2)
HyperS = DataFrame(HyperS)
HyperS_copy = DataFrame(HyperS_copy)
CSV.write("_HyperS_copy_sigma1.csv", HyperS_copy)
##### not runned yet
