module ModifiedBoosting

using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase

mutable struct Node

    is_leaf     :: Bool
    is_root     :: Bool
    left        :: Node
    right       :: Node
    parent      :: Node

    # branch parameters
    var         :: Int64
    val         :: Float64

    lower       :: Float64
    upper       :: Float64

    # leaf parameters
    μ           :: Float64

    ## Parameters for computing weights efficiently
    current_weight  :: Float64

    # Node(is_leaf, is_root, left, right, parent, var, val, lower, upper, μ, current_weight)    = new(is_leaf, is_root, left, right, parent, var, val, lower, upper, μ, current_weight)
    Node() = new()

end


function Root()
    x = Node()

    x.is_leaf = true
    x.is_root = true
    x.left = x
    x.right = x
    x.parent = x

    x.var = 0
    x.val = 0.0
    x.lower = 0.0
    x.upper = 1.0
    x.μ = 0.0
    x.current_weight = 1.0

    return x
end

function InitRoot(X::Array{Float64,2})
    x = Root()
    return x
end

# binary split
function AddLeaves!(x::Node)
    x.left = Node()
    x.right = Node()
    x.is_leaf = false

    ## Initialize the leafs when splitting
    x.left.is_leaf = true
    x.left.parent = x
    x.left.var = 0
    x.left.val = 0.0
    x.left.is_root = false
    x.left.lower = 0.0
    x.left.upper = 1.0
    x.left.μ = 0.0
    x.left.current_weight = 0.0
    x.right.is_leaf = true
    x.right.parent = x
    x.right.var = 0
    x.right.val = 0.0
    x.right.is_root = false
    x.right.lower = 0.0
    x.right.upper = 1.0
    x.right.μ = 0.0
    x.right.current_weight = 0.0

    return nothing
end


mutable struct Hypers
    α       :: Float64
    β        :: Float64
    γ       :: Float64
    σ       :: Float64
    σᵤ    :: Float64
    shape       :: Float64
    width       :: Float64
    num_tree    :: Int64
    s           :: Vector{Float64}

    ## Double hypers
    ̂σ   :: Float64
    ̂σᵤ :: Float64
    α_scale :: Float64
    α_shape_1 :: Float64
    α_shape_2 :: Float64

    Hypers()    = new()
end

# hyper-parameters involved
function Hypers(X::Array{Float64, 2},
            Y::Vector{Float64};
            α = 0.95,
            β = 2.0,
            γ = 0.95,
            k = 2.0,
            width = 0.1,
            shape = 1.0,
            num_tree = 50,
            α_scale = 0.0,
            α_shape_1 = 0.5,
            α_shape_2 = 1.0)

            out = Hypers()
            out.α = α
            out.β = β
            out.γ = γ
            out.σ = GetSigma(X,Y)
            out.σᵤ = 0.5 / (k * sqrt(num_tree))
            out.shape = shape
            out.width = width
            out.num_tree = num_tree
            out.s = ones(size(X,2)) / size(X,2)

            out.̂σ = out.σ
            out.̂σᵤ = out.σᵤ

            out.α_scale = α_scale == 0.0 ? size(X,2) : α_scale
            out.α_shape_1 = α_shape_1
            out.α_shape_2 = α_shape_2

            return out;

end


function GetLimits(x::Node)
    y = x
    lower = 0.0
    upper = 1.0
    my_bool = y.is_root ? false : true # my_bool is not defined, so it is initially false
    while my_bool
        is_left = (y.parent.left == y)
        y = y.parent
        my_bool = y.is_root ? false : true # when y is not root
        if y.var == x.var
            my_bool = false
            if is_left
                upper = y.val
                lower = y.lower
            else
                upper = y.upper
                lower = y.val
            end
        end
    end
    return lower, upper
end


function BirthLeaves!(x::Node, hypers::Hypers)
    if !x.is_leaf
        error("Node must be a leaf")
    end
    AddLeaves!(x)
    wv = StatsBase.Weights(hypers.s)
    x.var = sample(wv)
    # x.var = rand(1:P) ## TODO change this to allow non-iid selection
    lower, upper = GetLimits(x)
    x.lower = lower
    x.upper = upper
    x.val = (upper - lower) * rand() + lower
end

function is_left(n::Node)
    if n == n.parent.left
        return true
    else
        return false
    end
end

function DeleteLeaves!(x::Node)
    x.left = x
    x.right = x
    x.is_leaf = true
    return nothing
end

function depth(x::Node)
    x.is_root ? 0 : 1 + depth(x.parent)
end

function leaves!(x::Node, leaves::Vector{Node})
    if x.is_leaf
        push!(leaves, x);
    else
        leaves!(x.left, leaves);
        leaves!(x.right, leaves);
    end
end


function leaves(x::Node)
    leaves = Node[];
    leaves!(x, leaves);
    return leaves;
end











function fried(X::Array{Float64, 2})
    n = size(X,1)
    p = size(X,2)

    Y = zeros(n)
    for i in 1:n
        Y[i] = 10 * sin(2 * pi * X[i,1] * X[i,2]) +
            20 * (X[i,3] - 0.5)^2 + 10 * X[i,4] + 5 * X[i,5]
    end
    return Y
end

function test_data(n::Int64; sigma = 1.0, p = 10)
    X = rand(n,p)
    Y = fried(X) + sigma * randn(n)
    return X, Y
end

function logit(x::Float64)
    return log(x) - log(1 - x)
end

function normalize(y::Vector{Float64})
    a = minimum(y)
    b = maximum(y)
    return (y .- a) ./ (b - a) .- 0.5
end

function expit(x::Float64)
    return 1.0 / (1 + exp(-x))
end

function activation(x::Float64, c::Float64, hypers::Hypers)
    return 1- expit((x - c) / hypers.width)
end

function get_var_counts(forest::Vector{Node}, hypers::Hypers)
    counts = zeros(Int64, size(hypers.s,1))
    num_tree = length(forest)
    for t in 1:num_tree
        get_var_counts!(forest[t], counts)
    end
    return counts
end

function get_var_counts!(node::Node, counts::Vector{Int64})
    if !node.is_leaf
        counts[node.var] += 1
        get_var_counts!(node.left, counts)
        get_var_counts!(node.right, counts)
    end
end




end  # module
