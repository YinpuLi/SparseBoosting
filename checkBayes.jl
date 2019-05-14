using DelimitedFiles
include("GB11.jl")
import GLM: fit, predict, LinearModel
using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using CSV
using DataFrames

rng = Random.seed!(2019)


gbdt = GB11.GBDT(;
  sampling_rate = 0.75,                # Sampling rate
  learning_rate = 0.1,                # Learning rate
  num_iterations = 100,               # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB11.GBLearner(gbdt, :regression)

######## σ² = 1 #########
temp_storedim = size(collect(10:1000),1)
fitval = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rssfitted = []
predicted = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rmsefitted = []
Ytestval = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
μ_testval = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rmsefitted2 = []
HyperS = Array{Union{Missing, Nothing, Int, Float64}, 2}(nothing, temp_storedim, 1000)
s
for pval in collect(10:11)
    # reproducibility
    Random.seed!(2019)
    rng = 2019

    ### simulate the training and testing data
    X, Y = GB11.test_data(500; p = pval, sigma = 1.0)
    X_test, Y_test = GB11.test_data(500; p = pval)
    μ = GB11.fried(X)
    μ_test = GB11.fried(X_test)

    println("The last simulated data:
        \nX:$(X[500, pval]) \nY:$(Y[500])
        \nX_test: $(X_test[500, pval]) \nY_test: $(Y_test[500])
        \nμ: $(μ[500]) \nμ_test: $(μ_test[500])\n\n")

    μ_testval[:, pval - 9] = μ_test
    Ytestval[:, pval - 9 ] = Y_test

    hypers = GB11.Hypers(X, Y, 1, 1, 0.95)
    hypers.σ = sqrt(1)  ### TODO: I am using
    hypers.τ = 3 / 10 * var(Y) / gbdt.num_iterations
    hypers.α = 0.95
    hypers.s =  hypers.α .* ones(size(X, 2)) / size(X, 2)

    # train the model
    mymodel=GB11.fit!(gbl, X, Y, hypers)
    # get the fitted value
    myfit  = GB11.predict!(gbl, X)

    # store the hyper.s for the ensembel tree
    HyperS[pval - 9, 1:size(hypers.s, 1)] .= hypers.s

    # store the fitted value
    fitval[:, pval - 9] = myfit

    # get the residual sum of squares
    rssfit = sqrt(mean(Y .- myfit) .^2)
    push!(rssfitted, rssfit)
    # get prediction
    mypred = GB11.predict!(gbl, X_test)

    println("The last pred is $(mypred[500])\n\n ")

    predicted[:, pval - 9] = mypred

    # get mse and root of mean squared error
    rmse = sqrt(mean((Y_test .- mypred ) .^2))

    println("The rmse b.w. Y_test and pred for p = $(pval) is $(rmse) ")

    push!(rmsefitted, rmse)

    # get the rmse(μ_test, mypred)
    rmse2 = sqrt(mean((μ_test .- mypred).^2))

    println("The rmse2 b.w. μ_test and pred for p = $(pval)  is $(rmse2).")

    push!(rmsefitted2, rmse2)
end

rssfitted
rmsefitted
rmsefitted2
Ytestval
μ_testval
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


fitval = DataFrame(fitval)
CSV.write("_fitval_sigma1.csv", fitval)
rssfitted = DataFrame(rssfitted = rssfitted)
CSV.write("_fss_sigma1.csv", rssfitted)
rmsefitted = DataFrame(rmsefitted = rmsefitted)
CSV.write("_rmse_sigma1.csv", rmsefitted)
predicted = DataFrame(predicted)
CSV.write("_pred_sigma1.csv", predicted)
rmsefitted2 = DataFrame(rmsefitted2 = rmsefitted2)
CSV.write("_rmse2_sigma1.csv", rmsefitted2)
Ytestval = DataFrame(Ytestval)
CSV.write("_Ytest_sigma1.csv", Ytestval)
μ_testval = DataFrame(μ_testval)
CSV.write("_mu_test_sigma1.csv", μ_testval)
HyperS = DataFrame(HyperS)
HyperS_copy = DataFrame(HyperS_copy)
CSV.write("_HyperS_copy_sigma1.csv", HyperS_copy)
##### not runned yet


######## σ² = 10 #########
temp_storedim = size(collect(10:1000),1)
fitval_sigma10 = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rssfitted_sigma10 = []
predicted_sigma10 = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rmsefitted_sigma10 = []
Ytestval_sigma10 = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
μ_testval_sigma10 = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rmsefitted2_sigma10 = []
HyperS_sigma10 = Array{Union{Missing, Nothing, Int, Float64}, 2}(nothing, temp_storedim, 1000)
s
for pval in collect(10:1000)
    # reproducibility
    Random.seed!(2019)
    rng = 2019

    ### simulate the training and testing data
    X, Y = GB11.test_data(500; p = pval, sigma = sqrt(10.0))
    X_test, Y_test = GB11.test_data(500; p = pval, sigma = sqrt(10.0))
    μ = GB11.fried(X)
    μ_test = GB11.fried(X_test)

    println("Simulated data:
    \nX: $(X[500, pval])
    \nY: $(Y[500])
    \nX_test: $(X_test[500, pval])
    \nY_test: $(Y_test[500])
    \nμ: $(μ[500])
    \nμ_test: $(μ_test[500])")

    μ_testval_sigma10[:, pval - 9] = μ_test
    Ytestval_sigma10[:, pval - 9] = Y_test

    hypers = GB11.Hypers(X, Y, 1, 1, 0.95)
    hypers.σ = sqrt(1)  # TODO: maybe change to 10 after checking the plot
    hypers.τ = 3 / 10 * var(Y) / gbdt.num_iterations
    hypers.α = 0.95
    hypers.s =  hypers.α .* ones(size(X, 2)) / size(X, 2)

    # train the model
    mymodel=GB11.fit!(gbl, X, Y, hypers)

    # store the hyper.s for the ensembel tree
    HyperS_sigma10[ pval - 9, 1:size(hypers.s, 1)] .= hypers.s


    # get the fitted value
    myfit  = GB11.predict!(gbl, X)
    # store the fitted value
    fitval_sigma10[:, pval - 9] = myfit

    # get the residual sum of squares
    rssfit = sqrt(mean((Y .- myfit) .^2))
    push!(rssfitted_sigma10, rssfit)
    # get prediction
    mypred = GB11.predict!(gbl, X_test)

    println("The last pred is $(mypred[500])")

    predicted_sigma10[:, pval - 9] = mypred

    # get mse and root of mean squared error
    rmse = sqrt(mean((Y_test .- mypred ) .^2))
    println("The p = $(pval), and the rmse is $rmse.")
    push!(rmsefitted_sigma10, rmse)

    # get the rmse(Y_test, μ_test)
    rmse2 = sqrt(mean((mypred .- μ_test) .^2))
    println("The p = $(pval) dim, rmse2 is $rmse2.")
    push!(rmsefitted2_sigma10, rmse2)
end


fitval_sigma10
rssfitted_sigma10
predicted_sigma10
rmsefitted_sigma10
Ytestval_sigma10
μ_testval_sigma10
rmsefitted2_sigma10
HyperS_sigma10

#predicted_sigma10 = DataFrame(predicted_sigma10)
rmsefitted_sigma10 = DataFrame(rmsefitted_sigma10=rmsefitted_sigma10)

#CSV.write("fitval_sigma10.csv", fitval_sigma10)
#CSV.write("rssfitted_sigma10.csv", rssfitted_sigma10)
#CSV.write("predicted_sigma10.csv", predicted_sigma10)
#CSV.write("rmsefitted_sigma10.csv", rmsefitted_sigma10)

fitval_sigma10 = DataFrame(fitval_sigma10)
CSV.write("_fitval_sigma10.csv", fitval_sigma10)
rssfitted_sigma10 = DataFrame(rssfitted_sigma10 = rssfitted_sigma10)
CSV.write("_fss_sigma10.csv", rssfitted_sigma10)

CSV.write("_rmse_sigma10.csv", rmsefitted_sigma10)
predicted_sigma10 = DataFrame(predicted_sigma10)
CSV.write("_pred_sigma10.csv", predicted_sigma10)
rmsefitted2_sigma10 = DataFrame(rmsefitted2_sigma10 = rmsefitted2_sigma10)
CSV.write("_rmse2_sigma10.csv", rmsefitted2_sigma10)
Ytestval_sigma10 = DataFrame(Ytestval_sigma10)
CSV.write("_Ytest_sigma10.csv", Ytestval_sigma10)
μ_testval_sigma10 = DataFrame(μ_testval_sigma10)
CSV.write("_mu_test_sigma10.csv", μ_testval_sigma10)

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
