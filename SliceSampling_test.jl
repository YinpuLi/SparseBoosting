# test the SliceSampling module

include("SliceSampling.jl")
using Distributions
using Gadfly
using Random
#### I : test the Normal Target Distribution #####
function target(x)
    pdf(Normal(10, 2), x)  # density of given x value, destracted from N(10, 2²)
end

points = SliceSampling.sliceSample(10000, target, [0, 20], 0.1)
fromTarget = rand(Normal(10, 2), 10000)
# to plot the results
p1 = layer(x = points, Geom.histogram, Theme(default_color=colorant"orange"))
p2 = layer(x = fromTarget, Geom.histogram, Theme(default_color=colorant"green"))
plot(p1, p2, Theme(grid_color = colorant"white", grid_color_focused = colorant"white"), Guide.manual_color_key("Sampled vs Target",["SliceSampling","Target N(μ = 10, σ = 2)"],["orange","green"]))


# Normal distribution is fine

#### II: test the Beta Target Distribution #####
function target(x)
    pdf(Beta(2,5), x)  # density of given x value, destracted from N(10, 2²)
end

points = SliceSampling.sliceSample(10000, target, [0, 1], 0.001)
fromTarget = rand(Beta(2,5), 10000)
# to plot the results
p1 = layer(x = points, Geom.histogram, Theme(default_color=colorant"orange"))
p2 = layer(x = fromTarget, Geom.histogram, Theme(default_color=colorant"green"))
plot(p1, p2, Theme(grid_color = colorant"white", grid_color_focused = colorant"white"), Guide.manual_color_key("Sampled vs Target",["SliceSampling","Target N(μ = 10, σ = 2)"],["orange","green"]))

#plot(x = fromTarget, Geom.histogram)

# Beta is fine

#### III: test the Cauchy Target Distribution #####
#### TODO: half Cauchy
function target(x)
    pdf(Cauchy(2,0.5), x)  # density of given x value, destracted from N(10, 2²)
end

fromTarget = rand(Cauchy(2,0.5), 10000)
fromTarget = fromTarget[-5 .< fromTarget .< 10]
points = SliceSampling.sliceSample(length(fromTarget), target, [-5, 10], 0.001)

# to plot the results
p1 = layer(x = points, Geom.histogram, Theme(default_color=colorant"orange"))
p2 = layer(x = fromTarget, Geom.density, Theme(default_color=colorant"green"))
plot(p1, p2, Theme(grid_color = colorant"white", grid_color_focused = colorant"white"), Guide.manual_color_key("Sampled vs Target",["SliceSampling","Target N(μ = 10, σ = 2)"],["orange","green"]))

#plot(x = fromTarget, Geom.histogram)


# Cauchy not OK? TODO
# Cauchy not OK? NOT that OK:
    # the slice sampling hist is covering the Target hist

#### III.2: test the Half Cauchy Target Distribution #####
# as Half Cauchy is not available raight now
# HalfCauchy of σ² = Inverse Beta of λ > 0
include("InvBeta.jl")
rng = Random.seed!(2019)
function target(x)
    InvBeta.pdf(InvBeta.InverseBeta(.5,.5), x)  # density of given x value, destracted from N(10, 2²)
end

fromTarget = InvBeta.rand(rng, InvBeta.InverseBeta(.5, .5), 10000)
fromTarget = fromTarget[0 .< fromTarget .< 50]
points = SliceSampling.sliceSample(length(fromTarget), target, [0.000001, 50], 0.1)

# to plot the results
p1 = layer(x = points, Geom.histogram, Theme(default_color=colorant"orange"))
p2 = layer(x = fromTarget, Geom.histogram, Theme(default_color=colorant"green"))
plot(p2, p1, Theme(grid_color = colorant"white", grid_color_focused = colorant"white"), Guide.manual_color_key("Sampled vs Target",["SliceSampling","Target N(μ = 10, σ = 2)"],["orange","green"]))

#plot(x = fromTarget, Geom.histogram)
# HalfCauchy/ InverseBeta is fine!


#### IV: test the Gamma Target Distribution #####
function target(x)
    pdf(Gamma(3, 2), x)  # density of given x value, destracted from N(10, 2²)
end

fromTarget = rand(Cauchy(3,2), 10000)
fromTarget = fromTarget[0 .< fromTarget .<16]
points = SliceSampling.sliceSample(length(fromTarget), target, [0, 16], 0.1)

# to plot the results
p1 = layer(x = points, Geom.histogram, Theme(default_color=colorant"orange"))
p2 = layer(x = fromTarget, Geom.histogram, Theme(default_color=colorant"green"))
plot(p1, p2, Theme(grid_color = colorant"white", grid_color_focused = colorant"white"), Guide.manual_color_key("Sampled vs Target",["SliceSampling","Target"],["orange","green"]))

#plot(x = fromTarget, Geom.histogram)
# Gamma not OK: TODO
    # The sampled data has heavier tails


#### V: test the InverseGamma Target Distribution #####
function target(x)
    pdf(InverseGamma(3,1), x)  # density of given x value, destracted from N(10, 2²)
end

fromTarget = rand(InverseGamma(3,1), 10000)
fromTarget = fromTarget[0.0001 .< fromTarget .< 5]
points = SliceSampling.sliceSample(length(fromTarget), target, [0.0001, 5], 0.01)

# to plot the results
p1 = layer(x = points, Geom.histogram, Theme(default_color=colorant"orange"))
p2 = layer(x = fromTarget, Geom.histogram, Theme(default_color=colorant"green"))
plot(p1, p2, Theme(grid_color = colorant"white", grid_color_focused = colorant"white"), Guide.manual_color_key("Sampled vs Target",["SliceSampling","Target N(μ = 10, σ = 2)"],["orange","green"]))

#plot(x = fromTarget, Geom.histogram)
# InverseGamma is OK !


#### V: test the Exponential Target Distribution #####
function target(x)
    pdf(Exponential(.5), x)  # density of given x value, destracted from N(10, 2²)
end

fromTarget = rand(Exponential(.5), 10000)
fromTarget = fromTarget[0 .< fromTarget .< 5]
points = SliceSampling.sliceSample(length(fromTarget), target, [0, 5], 0.01)

# to plot the results
p1 = layer(x = points, Geom.histogram, Theme(default_color=colorant"orange"))
p2 = layer(x = fromTarget, Geom.histogram, Theme(default_color=colorant"green"))
plot(p2, p1, Theme(grid_color = colorant"white", grid_color_focused = colorant"white"), Guide.manual_color_key("Sampled vs Target",["SliceSampling","Target N(μ = 10, σ = 2)"],["orange","green"]))

#plot(x = fromTarget, Geom.histogram)
# Exponential is OK !







#plot(x = fromTarget, Geom.histogram)
