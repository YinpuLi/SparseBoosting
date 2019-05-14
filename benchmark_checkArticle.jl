 #map(abspath,readdir()) # get the file location and file names
 # create log file
using DelimitedFiles
writedlm("log_output_sigma1.txt",  " ")
writedlm("log_output_sigma10.txt", " ")
include("GB6.jl")
import GLM: fit, predict, LinearModel
using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using CSV
using DataFrames

Random.seed!(2019)
rng = 2019

# 1) Simulate n (= 500 )× p ( = 1~1000) training data and test data, where σ² = 1 and 10 separately
# 2) Plot the broken line chart

gbdt = GB6.GBDT(;
  sampling_rate = 0.75,                # Sampling rate
  learning_rate = 0.1,                # Learning rate
  num_iterations = 100,               # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB6.GBLearner(gbdt, :regression)



######## σ² = 1 #########
temp_storedim = size(collect(10:1000),1)
fitval = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rssfitted = []
predicted = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rmsefitted = []
Ytestval = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
μ_testval = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rmsefitted2 = []
s
for pval in collect(10:1000)
    # reproducibility
    Random.seed!(2019)
    rng = 2019

    ### simulate the training and testing data
    X, Y = GB6.test_data(500; p = pval, sigma = 1.0)
    X_test, Y_test = GB6.test_data(500; p = pval)
    μ = GB6.fried(X)
    μ_test = GB6.fried(X_test)

    println("The last simulated data:
        \nX:$(X[500, pval]) \nY:$(Y[500])
        \nX_test: $(X_test[500, pval]) \nY_test: $(Y_test[500])
        \nμ: $(μ[500]) \nμ_test: $(μ_test[500])\n\n")

    μ_testval[:, pval - 9] = μ_test
    Ytestval[:, pval - 9 ] = Y_test

    # train the model
    mymodel=GB6.fit!(gbl, X, Y)
    # get the fitted value
    myfit  = GB6.predict!(gbl, X)
    # store the fitted value
    fitval[:, pval - 9] = myfit

    # get the residual sum of squares
    rssfit = sqrt(mean(Y .- myfit) .^2)
    push!(rssfitted, rssfit)
    # get prediction
    mypred = GB6.predict!(gbl, X_test)

    println("The last pred is $(mypred[500])\n\n ")

    predicted[:, pval - 9] = mypred

    # get mse and root of mean squared error
    rmse = sqrt(mean((Y_test .- mypred ) .^2))

    println("The rmse b.w. Y_test and pred for p = $(pval) is $(rmse) ")

    push!(rmsefitted, rmse)

    # get the rmse(μ_test, mypred)
    rmse2 = sqrt(mean((μ_test .- mypred).^2))

    println("The rmse b.w. μ_test and pred for p = $(pval)  is $(rmse2).")

    push!(rmsefitted2, rmse2)
end

rssfitted
rmsefitted
rmsefitted2
Ytestval
μ_testval

fitval = DataFrame(fitval)
CSV.write("fitval_sigma1.csv", fitval)
rssfitted = DataFrame(rssfitted = rssfitted)
CSV.write("fss_sigma1.csv", rssfitted)
rmsefitted = DataFrame(rmsefitted = rmsefitted)
CSV.write("rmse_sigma1.csv", rmsefitted)
predicted = DataFrame(predicted)
CSV.write("pred_sigma1.csv", predicted)
rmsefitted2 = DataFrame(rmsefitted2 = rmsefitted2)
CSV.write("rmse2_sigma1.csv", rmsefitted2)
Ytestval = DataFrame(Ytestval)
CSV.write("Ytest_sigma1.csv", Ytestval)
μ_testval = DataFrame(μ_testval)
CSV.write("mu_test_sigma1.csv", μ_testval)
##### not runned yet


######## σ² = 10 #########

fitval_sigma10 = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rssfitted_sigma10 = []
predicted_sigma10 = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rmsefitted_sigma10 = []
Ytestval_sigma10 = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
μ_testval_sigma10 = Array{Union{Missing, Nothing, Int, Float64}}(nothing, 500, temp_storedim)
rmsefitted2_sigma10 = []
s
for pval in collect(10:1000)
    # reproducibility
    Random.seed!(2019)
    rng = 2019

    ### simulate the training and testing data
    X, Y = GB6.test_data(500; p = pval, sigma = sqrt(10.0))
    X_test, Y_test = GB6.test_data(500; p = pval, sigma = sqrt(10.0))
    μ = GB6.fried(X)
    μ_test = GB6.fried(X_test)

    println("Simulated data:
    \nX: $(X[500, pval])
    \nY: $(Y[500])
    \nX_test: $(X_test[500, pval])
    \nY_test: $(Y_test[500])
    \nμ: $(μ[500])
    \nμ_test: $(μ_test[500])")

    μ_testval_sigma10[:, pval - 9] = μ_test
    Ytestval_sigma10[:, pval - 9] = Y_test

    # train the model
    mymodel=GB6.fit!(gbl, X, Y)
    # get the fitted value
    myfit  = GB6.predict!(gbl, X)
    # store the fitted value
    fitval_sigma10[:, pval - 9] = myfit

    # get the residual sum of squares
    rssfit = sqrt(mean((Y .- myfit) .^2))
    push!(rssfitted_sigma10, rssfit)
    # get prediction
    mypred = GB6.predict!(gbl, X_test)

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

#predicted_sigma10 = DataFrame(predicted_sigma10)
rmsefitted_sigma10 = DataFrame(rmsefitted_sigma10=rmsefitted_sigma10)

#CSV.write("fitval_sigma10.csv", fitval_sigma10)
#CSV.write("rssfitted_sigma10.csv", rssfitted_sigma10)
#CSV.write("predicted_sigma10.csv", predicted_sigma10)
#CSV.write("rmsefitted_sigma10.csv", rmsefitted_sigma10)

fitval_sigma10 = DataFrame(fitval_sigma10)
CSV.write("fitval_sigma10.csv", fitval)
rssfitted_sigma10 = DataFrame(rssfitted_sigma10 = rssfitted_sigma10)
CSV.write("fss_sigma10.csv", rssfitted_sigma10)

CSV.write("rmse_sigma10.csv", rmsefitted_sigma10)
predicted_sigma10 = DataFrame(predicted_sigma10)
CSV.write("pred_sigma10.csv", predicted_sigma10)
rmsefitted2_sigma10 = DataFrame(rmsefitted2_sigma10 = rmsefitted2_sigma10)
CSV.write("rmse2_sigma10.csv", rmsefitted2_sigma10)
Ytestval_sigma10 = DataFrame(Ytestval_sigma10)
CSV.write("Ytest_sigma10.csv", Ytestval_sigma10)
μ_testval_sigma10 = DataFrame(μ_testval_sigma10)
CSV.write("mu_test_sigma10.csv", μ_testval_sigma10)

# plot the rmsefitted for sigma = 1
using Gadfly


rmse_sigma1 = CSV.read("rmse_sigma1.csv")
rmse_sigma10 = CSV.read("rmse_sigma10.csv")


rmse2_sigma1 = CSV.read("rmse2_sigma1.csv")
rmse2_sigma10 = CSV.read("rmse2_sigma10.csv")

 # 991
i = collect(1:1:10)
ii = collect(20:10:100)
iii = collect(200:100:900)
ind = vcat(i, ii, iii)
rmse1_forplot = rmse_sigma1[ind,1]
rmse10_forplot = rmse_sigma10[ind,1]


p_1 = layer(x = ind, y = rmse1_forplot,
            Geom.point, Geom.line,
            Theme(default_color="red"))
p_2 = layer(x = ind, y = rmse10_forplot,
            Geom.point, Geom.line, Theme(default_color="blue"))
myplot1 = plot(p_1, p_2, Guide.xlabel("P"),
    Guide.ylabel("rmse"), Guide.title("Root Mean Squared Error"),
    Guide.manual_color_key("", ["σ² = 1", "σ² = 10"], ["red", "blue"]))

using Cairo, Fontconfig
draw(PNG("myplot1.png", 3inch, 3inch), myplot1)

rmse2_1_forplot = rmse2_sigma1[ind,1]
rmse2_10_forplot = rmse2_sigma10[ind,1]

pp_1 = layer(x = ind, y = rmse2_1_forplot,
            Geom.point, Geom.line,
            Theme(default_color="red"))
pp_2 = layer(x = ind, y = rmse2_10_forplot,
            Geom.point, Geom.line, Theme(default_color="blue"))
myplot2 = plot(pp_1, pp_2, Guide.xlabel("P"),
    Guide.ylabel("rmse"), Guide.title("Root Mean Squared Error"),
    Guide.manual_color_key("", ["σ² = 1", "σ² = 10"], ["red", "blue"]));

Gadfly.push_theme(style(line_width=1mm))

ppp_2 = layer(x = ind, y = rmse2_10_forplot,
            style(line_width=10cm),
            Geom.point, Geom.line, Theme(default_color="blue"))
pppp = plot(ppp_2, Guide.xlabel("P"),
    Guide.ylabel("rmse"), Guide.title("Root Mean Squared Error"),
    Guide.manual_color_key("", ["σ² = 1", "σ² = 10"], ["red", "blue"]));
draw(PNG("test.png", 15inch, 15inch), pppp)

Gadfly.pop_theme()

draw(PNG("myplot2.png", 15inch, 15inch), myplot2)



p = 10
x, y = GB6.test_data(500; p = 10, sigma = sqrt(10.0))
x_test, y_test = GB6.test_data(500; p = 10, sigma = sqrt(10.0))
mu = GB6.fried(x)
mu_test = GB6.fried(x_test)
x=DataFrame(x)
y=DataFrame(y=y)
x_test=DataFrame(x_test)
y_test=DataFrame(y_test=y_test)
mu = DataFrame(mu = mu)
mu_test = DataFrame(mu_test = mu_test)

## for comparing the data with bencnmark method, use CSV

CSV.write("_X.csv", x)
CSV.write("_Y.csv", y)
CSV.write("_X_test.csv", x_test)
CSV.write("_Y_test.csv", y_test)
CSV.write("_mu.csv", mu)
CSV.write("_mu_test.csv", mu_test)

for j in 1 : size(x, 2)
    x[:, j] = collect(skipmissing(x[:,j]))
end
x = convert(Matrix, x)
y = collect(skipmissing(y[:,1]))

for j in 1 : size(x_test, 2)
    x_test[:, j] = collect(skipmissing(x_test[:,j]))
end
x_test = convert(Matrix, x_test)
y_test = collect(skipmissing(y_test[:,1]))

mu_test = collect(skipmissing(mu_test[:,1]))

gbdt = GB6.GBDT(;
  sampling_rate = 0.75,                # Sampling rate
  learning_rate = 0.1,                # Learning rate
  num_iterations = 1,               # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 1,
      :max_depth => 5))
gbl = GB6.GBLearner(gbdt, :regression)


hypers = GB6.Hypers()
hypers.σ = sqrt(1)
hypers.τ = 3 / 10 * var(y) / gbdt.num_iterations
mymodel=GB6.fit!(gbl, x, y, hypers)
# get the fitted value
myfit  = GB6.predict!(gbl, x)

# get the residual sum of squares
sqrt(mean(y .- myfit) .^2) # 0.0103

# get prediction
mypred = GB6.predict!(gbl, x_test)

# get mse and root of mean squared error
sqrt(mean((y_test .- mypred ) .^2)) #10.2

sqrt(mean((mypred .- mu_test ) .^2))




#### compare with R
# σ² = 10, σ = √10
xx = CSV.read("xx.csv")[:, 2:end]
yy = CSV.read("yy.csv")[:, 2]
xx_test = CSV.read("xx_test.csv")[:, 2:end]
yy_test = CSV.read("yy_test.csv")[:, 2]
μμ_test = CSV.read("mumu_test.csv")[:, 2]

for j in 1 : size(xx, 2)
    xx[:, j] = collect(skipmissing(xx[:,j]))
end
xx = convert(Matrix, xx)
yy = collect(skipmissing(yy[:,1]))

for j in 1 : size(xx_test, 2)
    xx_test[:, j] = collect(skipmissing(xx_test[:,j]))
end
xx_test = convert(Matrix, xx_test)
yy_test = collect(skipmissing(yy_test[:,1]))

μμ_test = collect(skipmissing(μμ_test[:,1]))

# set the same combination of parameters, though I did not use cross validation here
gbdt_new = GB6.GBDT(;
  sampling_rate = 0.8,                # Sampling rate
  learning_rate = 0.1,                # Learning rate
  num_iterations = 100,               # NuGBer of iterations
  tree_options = Dict(
      :nsubfeatures => 0,
      :min_samples_leaf => 5,
      :max_depth => 6))
gbl_new = GB6.GBLearner(gbdt_new, :regression)



mymodel2=GB6.fit!(gbl_new, xx, yy)
# get the fitted value
myfit2 = GB6.predict!(gbl_new, xx)

# get the residual sum of squares
sqrt(mean(yy .- myfit2) .^2) # 0.02446; 0.0045(σ = √10)

# get prediction
mypred2 = GB6.predict!(gbl_new, xx_test)

# get mse and root of mean squared error
mean((yy_test .- mypred2 ) .^2) # ; 14.1986(σ = √10)
sqrt(mean((yy_test .- mypred2 ) .^2)) #12.02; 3.7678(σ = √10)

sqrt(mean((mypred2 .- μμ_test ) .^2)) # 6.6 ; 1.93(σ = 1.9)
