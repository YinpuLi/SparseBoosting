using DelimitedFiles
include("GB15_2.jl")
import GLM: fit, predict, LinearModel
using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using CSV
using DataFrames

rng = Random.seed!(2019)


gbdt = GB15.GBDT(;
  sampling_rate = 0.75,                # Sampling rate
  learning_rate = 0.1,                # Learning rate
  num_iterations = 100,               # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB15.GBLearner(gbdt, :regression)

######## σ² = 1 #########
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

STORE = []

Pval
for pval in Pval
    # reproducibility
    rng = Random.seed!(2019)

    ### simulate the training and testing data
    X, Y = GB15.test_data(500; p = pval, sigma = 1.0)
    X_test, Y_test = GB15.test_data(500; p = pval, sigma = 1.0)
    μ = GB15.fried(X)
    μ_test = GB15.fried(X_test)

    hypers = GB15.Hypers(X, Y, 1, 1, 0.95)
    hypers.σ = sqrt(1)
    hypers.τ = 3 / 10 * var(Y) / gbdt.num_iterations
    hypers.α = 0.95
    hypers.s = ones(size(X, 2)) / size(X, 2)

    # initialize opt
    opts = GB15.Opts()
    opts.num_burn = 50

    # train the model and get the final model as well as rmse2
    rmse2, Store_s = GB15.TBD_predict_and_rmse!(
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

    push!(rmsefitted2, rmse2)
    push!(STORE, Store_s)
    # store the hyper.s for the ensembel tree
    j = findall(x -> x == pval, Pval)
    HyperS[j, 1:size(hypers.s, 1)] = hypers.s
    println("After $(opts.num_burn) times of burning, the final sum of the first 5 elements of the prob vec is:\n$(sum(hypers.s[1:5])) ")
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
STORE

for i in 1 : size(STORE, 1)
    temp = DataFrame(STORE[i,])
    CSV.write("STORE.csv", temp, append = true)
end
##### not runned yet


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
STORE_sigma10 = []
Pval



for pval in Pval
    # reproducibility
    rng = Random.seed!(2019)
    ### simulate the training and testing data
    X, Y = GB15.test_data(500; p = pval, sigma = sqrt(10.0))
    X_test, Y_test = GB15.test_data(500; p = pval, sigma = sqrt(10.0))
    μ = GB15.fried(X)
    μ_test = GB15.fried(X_test)

    hypers = GB15.Hypers(X, Y, 1, 1, 0.95)
    hypers.σ = sqrt(10)  ### TODO: I am using
    hypers.τ = 3 / 10 * var(Y) / gbdt.num_iterations
    hypers.α = 0.95
    hypers.s = ones(size(X, 2)) / size(X, 2)

    # initialize opt
    opts = GB15.Opts()
    opts.num_burn = 5

    # train the model and get the final model as well as rmse2
    rmse2, Store_s = GB15.TBD_predict_and_rmse!(
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

    push!(rmsefitted2_sigma10, rmse2)
    push!(STORE_sigma10, Store_s)
    # store the hyper.s for the ensembel tree
    j = findall(x -> x == pval, Pval)
    HyperS_sigma10[j, 1:size(hypers.s, 1)] = hypers.s
    println("After $(opts.num_burn) times of burning, the sum of the first 5 elements in the prob vec is: $(sum(hypers.s[1:5])).")
end


rmsefitted2_sigma10
HyperS_sigma10


rmsefitted_sigma10 = DataFrame(rmsefitted_sigma10=rmsefitted_sigma10)
rmsefitted2_sigma10 = DataFrame(rmsefitted2_sigma10 = rmsefitted2_sigma10)
CSV.write("_rmse2_sigma10.csv", rmsefitted2_sigma10)
HyperS_sigma10 = DataFrame(HyperS_sigma10)
HyperS_copy_sigma10 = copy(HyperS_sigma10)

for i in 1 : size(HyperS_sigma10, 1)

    for j in 1 : size(HyperS_sigma10, 2)

        if HyperS_copy_sigma10[i, j] == nothing

            HyperS_copy_sigma10[i, j] = 0.0

        end

    end

end

HyperS_copy_sigma10
CSV.write("_HyperS_sigma10.csv", HyperS_sigma10)

STORE_sigma10

for i in 1 : size(STORE_sigma10, 1)
    temp = DataFrame(STORE_sigma10[i])
    CSV.write("STORE_sigma10.csv", temp, append = true)
end
