module GB4

# Based on GB3.jl
# update: in decision tree part, change the splitting rule


using Distributions,SpecialFunctions
using LinearAlgebra, MultivariateStats
using StatsBase, Random, Distributed
using DelimitedFiles, DataStructures
using MLBase
import Base: length, show
import MLBase: cross_validate
####################### PART I: Gradiant Boosting Decision Tree  ##########################

# only the terminal/leaf nodes
struct Leaf{T}
    majority    :: T                    # majority vote/ weight of the leaf
    values      :: Vector{T}
end
# only the root as well as the internal nodes, (internal node with children)
struct Node{S, T}
    feature_id  :: Int                          # the feature to split on(to left and right child)
    feature_val :: S                            # the feature value to split at
    left        :: Union{Leaf{T}, Node{S, T}}   # left child
    right       :: Union{Leaf{T}, Node{S, T}}   # right child
end


# defining the global constant type, including all instances of any of LeafOrNode's argument types: Leaf{T}, Node{S, T}
const LeafOrNode{S, T} = Union{Leaf{T}, Node{S, T}}

# ensemble trees
#
# TODO: the ensembled tree should be the summation of base functions,
# not trees

## containing leaves and nodes in all trees TODO: change it, may be useless
struct Ensemble{S, T}
    trees       :: Vector{LeafOrNode{S, T}}
end

# instance-and-node index pointer, two directions
#
# for each instance of 1×p dimention, containing p features' values, locate the terminal leaf node, provided any current node(internal or terminal nodes)

function instance_to_node(tree :: Node, instance)
    # this instance is one of the n instances, as a vector, with all feature values as entries
    features_val = instance         # 1×p dimention
    if tree.feature_val == nothing || features_val[tree.feature_id] < tree.feature_val
        return instance_to_node(tree.left, features_val)
    else
        return instance_to_node(tree.right, features_val)
    end
end

instance_to_node(leaf :: Leaf, instance) = leaf



mutable struct InstanceNodeIndex
    i2n     :: Vector{Leaf}                                     # applying instance_to_node() to return to leafs
    n2i     :: DataStructures.DefaultDict{Leaf, Vector{Int}}    # store the info of Leaf with region(a slice of sample indx of instances)

    function InstanceNodeIndex(tree :: Union{Leaf, Node}, instances)
        num_instances   = size(instances, 1)

        # instance to node and node to instance

        i2n             = Array{Leaf}(undef, num_instances)  # there should be n instances assigned to terminal nodes, return to a n-length vector of leafs that each instance belong to
        n2i             = DataStructures.DefaultDict{Leaf, Vector{Int}}(() -> Int[])  # each leaf node includes a slice of samples: vector of index of sample

        for i = 1 : num_instances
            node        = instance_to_node(tree, instances[i, :])
            i2n[i]      = node   # assign the node::LeafOrNode to the vector
            push!(n2i[node], i)
        end

        new(i2n, n2i)

    end

end

mutable struct NodeMeta{S}

    l           :: NodeMeta{S}      # left child
    r           :: NodeMeta{S}      # right child
    label       :: Float64          # most likely label: y; majority/ weight
    feature     :: Int              # feature_id used for splitting
    threshold   :: S                # threshold value
    is_leaf     :: Bool
    depth       :: Int
    region      :: UnitRange{Int}   # a slice of samples's id used to decide the split of the node
    features    :: Vector{Int}      # a list of features_id not known to be constant
    split_at    :: Int              # index of sample to split at
    impurity    :: Float64

    function NodeMeta{S}(features, region, depth) where {S}

        node            = new{S}()
        node.depth      = depth
        node.region     = region
        node.features   = features
        node.is_leaf    = false

        node

    end
end

mutable struct Tree{S}
    root    :: NodeMeta{S}          # only one root, containing all of the samples, for each tree
    labels  :: Vector{Int}          # index vector of length n
end

# returns respective node type(Leaf or InternalNode) of instance
# labels: a vector of y values
function _convert( node :: NodeMeta{S}, labels :: Array{T}) where {S, T <: Float64}
    if node.is_leaf         # then convert this NodeMeta to a terminal node of type Leaf
        return Leaf{T}(node.label, labels[node.region]) # NodeMeta.label = Leaf.majority; labelsᵢ = yᵢ, the node.region = Vector of indices in this region
    else                    # then convert this NodeMeta to an internal node of type Node, and continuously converting its children
        left    = _convert(node.l, labels)
        right   = _convert(node.r, labels)
        return Node{S, T}(node.feature, node.threshold, left, right)
    end
end



function val_func(leaf :: Leaf)
    #inst_node_index     = InstanceNodeIndex(node, instances)
    #inst_ind            = inst_node_index.n2i[node]
    # now we only consider the case of loss function = LeasetSquares
    # we don't need to change values
    val = leaf.majority

    return val
end

# update region by having updated leaf value encoded
# in a leaf-to-value mapping

function update_regions!(n2v :: Dict{Leaf, T}, node :: Node, val_func :: Function) where T
    update_regions!(n2v, node.left, val_func),
    update_regions!(n2v, node.right, val_func)
end

function update_regions!(n2v :: Dict{Leaf, T}, leaf :: Leaf, val_func :: Function) where T
    n2v[leaf] = val_func(leaf)
end


###  gradient boosting decision tree algorithm
abstract type GBAlgorithm end

mutable struct GBDT <:GBAlgorithm
    cv_folds        :: Int64
    sampling_rate   :: Float64
    learning_rate   :: Float64
    num_iterations  :: Int64
    tree_options    :: Dict
    function GBDT(;
        cv_folds        = 10,
        sampling_rate   = 1.0,
        learning_rate   = 1.0,
        num_iterations  = 10,
        tree_options    = Dict())
        default_option  = Dict(
            :max_depth => 5,
            :nsubfeatures => 0,
            :min_samples_leaf => 1,
            )
        tree_options = merge(default_option, tree_options)
        # the tree_options from the function argument should be set in the end of the merge(), so the input arguments could overwrite the default_option
        new(cv_folds, sampling_rate, learning_rate, num_iterations, tree_options)
    end
end




# gradient boosting base learner

# l2 loss function fits
function fit_best_constant(y, psuedo, psuedo_pred, prev_func_pred)
    1.0 # no refitting needed
end


function check_input(
            X                   :: Matrix{S},
            Y                   :: Vector{T},
            W                   :: Vector{U},
            max_features        :: Int,
            max_depth           :: Int,
            min_samples_leaf    :: Int,
            min_samples_split   :: Int,
            min_purity_increase :: Float64) where {S, T, U}
    n_samples, n_features = size(X)
    if length(Y) != n_samples
        throw("dimension mismatch between X and Y ($(size(X)) vs $(size(Y))")
    elseif length(W) != n_samples
        throw("dimension mismatch between X and W ($(size(X)) vs $(size(Y))")
    elseif max_depth < -1
        throw("unexpected value for max_depth: $(max_depth) (expected:"
            * " max_depth >= 0, or max_depth = -1 for infinite depth)")
    elseif n_features < max_features
        throw("number of features $(n_features) is less than the number "
            * "of max features $(max_features)")
    elseif max_features < 0
        throw("number of features $(max_features) must be >= zero ")
    elseif min_samples_leaf < 1
        throw("min_samples_leaf must be a positive integer "
            * "(given $(min_samples_leaf))")
    elseif min_samples_split < 2
        throw("min_samples_split must be at least 2 "
            * "(given $(min_samples_split))")
    end
end
function _split!(
            X                   :: Matrix{S}, # the feature array
            Y                   :: Vector{Float64}, # the label array
            W                   :: Vector{U},
            node                :: NodeMeta{S}, # the node to split
            max_features        :: Int, # number of features to consider, Int(size(X, 2))
            max_depth           :: Int, # the maximum depth of the resultant tree
            min_samples_leaf    :: Int, # the minimum number of samples each leaf needs to have
            min_samples_split   :: Int, # the minimum number of samples in needed for a split
            min_purity_increase :: Float64, # minimum purity needed for a split
            indX                :: Vector{Int}, # an array of sample indices,
                                                # we split using samples in indX[node.region]
            # the two arrays below are given for optimization purposes
            Xf                  :: Vector{S},
            Yf                  :: Vector{Float64},
            Wf                  :: Vector{U},
            rng                 :: Random.AbstractRNG) where {S, U}

    region = node.region
    n_samples = length(region)
    r_start = region.start - 1

    @inbounds @simd for i in 1:n_samples
        Yf[i] = Y[indX[i + r_start]]
        Wf[i] = W[indX[i + r_start]]
    end

    tssq = zero(U)
    tsum = zero(U)
    wsum = zero(U)
    @inbounds @simd for i in 1:n_samples
        tssq += Wf[i]*Yf[i]*Yf[i]
        tsum += Wf[i]*Yf[i]
        wsum += Wf[i]
    end

    node.label =  tsum / wsum
    if (min_samples_leaf * 2 >  n_samples
     || min_samples_split    >  n_samples
     || max_depth            <= node.depth
      # equivalent to old_purity > -1e-7
     || tsum * node.label    > -1e-7 * wsum + tssq)
        # TODO : Add Wf[1:n_samples] to this thing
        node.is_leaf = true
        return
    end

    features = node.features
    n_features = length(features)
    best_purity = typemin(U)
    best_feature = -1
    threshold_lo = X[1]
    threshold_hi = X[1]

    indf = 1   # index of features to go through; with initial value 1: starting from the first availabel feature
    # the number of new constants found during this split
    n_constant = 0
    # true if every feature is constant
    unsplittable = true
    # the number of non constant features we will see if
    # only sample n_features used features
    # is a hypergeometric random variable
    total_features = size(X, 2)
    total_samples  = size(X, 1)

    # this is the total number of features that we expect to not
    # be one of the known constant features. since we know exactly
    # what the non constant features are, we can sample at 'non_constants_used'
    # non constant features instead of going through every feature randomly.
    non_constants_used = hypergeometric(n_features, total_features-n_features, max_features, rng)  # for my code, it's p, no worries


    @inbounds while (unsplittable || indf <= non_constants_used) && indf <= n_features


        feature = let
                indr = rand(rng, indf:n_features)
                features[indf], features[indr] = features[indr], features[indf]
                features[indf]
            end


        rssq = tssq
        lssq = zero(U)
        rsum = tsum
        lsum = zero(U)

        @simd for i in 1:n_samples
            Xf[i] = X[indX[i + r_start], feature]
        end

        # sort Yf and indX by Xf
        q_bi_sort!(Xf, indX, 1, n_samples, r_start)
        @simd for i in 1:n_samples
            Yf[i] = Y[indX[i + r_start]]
            Wf[i] = W[indX[i + r_start]]
        end
        nl, nr = zero(U), wsum
        # lo and hi are the indices of
        # the least upper bound and the greatest lower bound
        # of the left and right nodes respectively
        hi = 0
        last_f = Xf[1]
        is_constant = true
        while hi < n_samples
            lo = hi + 1
            curr_f = Xf[lo]
            hi = (lo < n_samples && curr_f == Xf[lo+1]
                ? searchsortedlast(Xf, curr_f, lo, n_samples, Base.Order.Forward)
                : lo)

            (lo != 1) && (is_constant = false)
            # honor min_samples_leaf
            if lo-1 >= min_samples_leaf && n_samples - (lo-1) >= min_samples_leaf
                unsplittable = false
                purity = (rsum * rsum / nr) + (lsum * lsum / nl)
                if purity > best_purity
                    # will take average at the end, if possible
                    threshold_lo = last_f
                    threshold_hi = curr_f
                    best_purity  = purity
                    best_feature = feature
                end
            end

            # update, lssq, rssq, lsum, rsum in the direction
            # that would require the smaller number of iterations
            if (hi << 1) < n_samples + lo # i.e., hi - lo < n_samples - hi
                @simd for i in lo:hi
                    nr   -= Wf[i]
                    rsum -= Wf[i]*Yf[i]
                    rssq -= Wf[i]*Yf[i]*Yf[i]
                end
            else
                nr = rsum = rssq = zero(U)
                @simd for i in (hi+1):n_samples
                    nr   += Wf[i]
                    rsum += Wf[i]*Yf[i]
                    rssq += Wf[i]*Yf[i]*Yf[i]
                end
            end
            lsum = tsum - rsum
            lssq = tssq - rssq
            nl   = wsum - nr

            last_f = curr_f
        end

        # keep track of constant features to be used later.
        if is_constant
            n_constant += 1
            features[indf], features[n_constant] = features[n_constant], features[indf]
        end

        indf += 1
    end

    # no splits honor min_samples_leaf
    @inbounds if (unsplittable
            || best_purity - tsum * node.label < min_purity_increase * wsum)
        node.is_leaf = true
        return
    else
        # new_purity - old_purity < stop.min_purity_increase
        bf = Int(best_feature)
        @simd for i in 1:n_samples
            Xf[i] = X[indX[i + r_start], best_feature]
        end

        try
            node.threshold = (threshold_lo + threshold_hi) / 2.0
        catch
            node.threshold = threshold_hi
        end
        # split the samples into two parts: ones that are greater than
        # the threshold and ones that are less than or equal to the threshold
        #                                 ---------------------
        # (so we partition at threshold_lo instead of node.threshold)
        node.split_at = partition!(indX, Xf, threshold_lo, region)
        node.feature = best_feature
        node.features = features[(n_constant+1):n_features]
    end
end

@inline function fork!(node :: NodeMeta{S}) where S
    ind = node.split_at
    region = node.region
    features = node.features
    # no need to copy because we will copy at the end
    node.l = NodeMeta{S}(features, region[    1:ind], node.depth + 1)
    node.r = NodeMeta{S}(features, region[ind+1:end], node.depth + 1)
end

function _fit(
        X                     :: Matrix{S},
        Y                     :: Vector{Float64},
        W                     :: Vector{U},
        max_features          :: Int,
        max_depth             :: Int,
        min_samples_leaf      :: Int,
        min_samples_split     :: Int,
        min_purity_increase   :: Float64,
        rng=Random.GLOBAL_RNG :: Random.AbstractRNG) where {S, U}

    n_samples, n_features = size(X)

    Yf  = Array{Float64}(undef, n_samples)
    Xf  = Array{S}(undef, n_samples)
    Wf  = Array{U}(undef, n_samples)

    indX = collect(Int(1):Int(n_samples))
    root = NodeMeta{S}(collect(1:n_features), 1:n_samples, 0)
    stack = NodeMeta{S}[root]



    @inbounds while length(stack) > 0
        node = pop!(stack)   # remove the last node and return it



        _split!(
            X, Y, W,
            node,
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            indX,
            Xf, Yf, Wf,
            rng)
        if !node.is_leaf



            fork!(node)
            push!(stack, node.r)
            push!(stack, node.l)
        end
    end
    return (root, indX)
end

function fit_tree(;
        X                     :: Matrix{S},
        Y                     :: Vector{Float64},
        W                     :: Union{Nothing, Vector{U}},
        max_features          :: Int,
        max_depth             :: Int,
        min_samples_leaf      :: Int,
        min_samples_split     :: Int,
        min_purity_increase   :: Float64,
        rng=Random.GLOBAL_RNG :: Random.AbstractRNG) where {S, U}

    n_samples, n_features = size(X)
    if W == nothing
        W = fill(1.0, n_samples)
    end
    check_input(
        X,
        Y,
        W,
        max_features,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
    root, indX = _fit(
        X,
        Y,
        W,
        max_features,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase,
        rng)
    return Tree{S}(root, indX)
end


function build_tree(
        labels             :: Vector{T},    # y vector
        features           :: Matrix{S},    # X matrix
        n_subfeatures       = 0,
        max_depth           = -1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Number}

        if max_depth == -1
            max_depth = typemax(Int)
        end
        if n_subfeatures == 0
            n_subfeatures = size(features, 2)
        end

        rng = mk_rng(rng)::Random.AbstractRNG
        t = fit_tree(
            X                   = features,
            Y                   = labels,
            W                   = nothing,
            max_features        = Int(n_subfeatures),
            max_depth           = Int(max_depth),
            min_samples_leaf    = Int(min_samples_leaf),
            min_samples_split   = Int(min_samples_split),
            min_purity_increase = Float64(min_purity_increase),
            rng                 = rng)

            return _convert(t.root, labels[t.labels])
end


# a decision stump consists a one-level decision tree: 1-rules
function build_stump(labels::Vector{T}, features::Matrix{S}; rng = Random.GLOBAL_RNG) where {S, T <: Float64}
    return build_tree(labels, features, 0, 1)
end


# apply_tree will eventually return the Leaf.majority
# features[i, :] = the "feature" is one of the predictors of n instances: n×1 for X.ₚ
apply_tree(leaf :: Leaf{T}, feature :: Vector{S}) where {S, T} = leaf.majority
function apply_tree(tree :: Node{S, T}, features :: Vector{S}) where {S, T}
    if features[tree.feature_id] < tree.feature_val
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end
# alternatively, X matrix is provided, then apply instance-to-instance
function apply_tree(tree :: LeafOrNode{S, T}, features :: Matrix{S}) where {S, T}
    N = size(features, 1)
    predictions = Array{T}(undef, N)
    for i in 1:N
        predictions[i] = apply_tree(tree, features[i, :])
    end
    if T <: Float64
        return Float64.(predictions)
    else
        return predictions
    end
end

# sub sampling
function create_sample_indices(gb :: GBAlgorithm, X, y)
    n       = size(X,1)
    prop    = gb.sampling_rate
    ind     = randperm(n)[1:Int(floor(prop * n))]
end

# cross validation
# exrend the cross_validate from MLBase

function rmse(preds, testval)
    sqrt(mean((preds .- testval) .^2))
end


function build_base_func(
    gb :: GBDT,
    instances,              # X matrix
    labels,                 # y vector
    prev_func_pred,         #
    psuedo;
    rng                 = Random.GLOBAL_RNG
    )

    # model is the decision tree
    model = build_tree(
        psuedo,             # current residuals
        instances,          # X matrix
        gb.tree_options[:nsubfeatures],
        gb.tree_options[:max_depth],
        gb.tree_options[:min_samples_leaf])          # to fit to the current residual

    psuedo_pred = apply_tree(model, instances)


    # then update regions of leaves
    #
    # overwrite leaves by having node-to-val mapping
    inst_node_index = InstanceNodeIndex(model, instances)

    val_type    = eltype(prev_func_pred)  # type of elements
    n2v         = Dict{Leaf, val_type}()
    update_regions!(n2v, model, val_func)

    # prediction function 这里有问题 TODO 输入的参数不够
    @inline function pred(instances)
        num_instances = size(instances, 1)
        predictions   = [n2v[instance_to_node(model, instances[i, :])]
            for i in 1 : num_instances   ]
        return predictions
    end
    # produce function that delegates prediction to model
    #
    # @return Function of form (instances) -> predictions.
    return (features) -> pred(features)
end

# gradient boosted base learner algorithm

mutable struct GBBL <: GBAlgorithm
    sampling_rate   :: Float64
    learning_rate   :: Float64
    num_iterations  :: Int
    learner

    function GBBL(learner; sampling_rate = 1.0, learning_rate = 1.0, num_iterations =100)
        new(sampling_rate, learning_rate, num_iterations, learner)
    end
end


import GLM: fit, predict, LinearModel
# Extend functions
function learner_fit(
  learner::Type{LinearModel}, instances::AbstractMatrix, labels::Vector)
  # From GLM.LinearAlgebra
  model = fit(learner, instances, labels)
end
function learner_predict(
  model::LinearModel, instances::AbstractMatrix)
  predict(model, instances)
end

function learner_fit(learner, instances, labels)
  error("This function must be implemented by $(learner).")
end
function learner_predict(learner, model, instances)
    error("This function must be implemented by $(learner).")
end



function build_base_func(
    gb :: GBBL,
    instances,          # X matrix
    labels,             # y vector
    prev_func_pred,
    psuedo;
    rng = Random.GLOBAL_RNG
    )

    # train learner
    learner = gb.learner

    model   = learner_fit(learner, instances, psuedo)
    psuedo_pred = learner_predict(model, instances)
    model_const = fit_best_constant(labels, psuedo, psuedo_pred, prev_func_pred)

    # produce function that delegates prediction to model
    return (instances) -> model_const .* learner_predict(model, instances)
end


################# End of Gradient Boosting Decision Tree #######################








# dsefining the functions is_leaf for different types: Leaf/Node, resulting in bools
is_leaf(l :: Leaf) = true
is_leaf(n :: Node) = false

# make a randome number generator object
mk_rng(rng :: Random.AbstractRNG) = rng
mk_rng(seed :: T) where T <: Integer = Random.MersenneTwister(seed)

######## methods #######

# extending the Base.length methods for (terminal-)Leaf and (enternal)Node type

length(leaf :: Leaf) = 1
length(tree :: Node) = length(tree.left) + length(tree.right)
## length(e :: Ensemble) measures how many nodes and leaves are contained in the ensembled tree
### ensembel.tree will return to an array containing nodes and leaves
#length(ensemble :: Ensemble) = length(ensemble.trees)

# defining the depths from the current node to the furthest child
depth(leaf :: Leaf) = 0
depth(tree :: Node) = 1 + max(depth(tree.left), depth(tree.right))


#### end of methods ####





################## PART II: Gradient Boosting #################

########## fit gradient boosting algorithem #######


# Gradient boost model

mutable struct GBModel
    learning_rate   :: Float64
    base_funcs      :: Vector{Function}
end
function build_base_func(
  gb::GBAlgorithm,
  instances,
  labels,
  prev_func_pred,
  psuedo)

  error("Error must be overriden!")
end
function stochastic_gradient_boost(gb :: GBAlgorithm,
                                    X,
                                    y)
    # initialize base functions collection
    num_iterations  = gb.num_iterations
    base_funcs      = Array{Function}(undef, num_iterations + 1)

    # create initial base function
    @inline function minimizing_scalar(y) # now, we only consier the case of L2loss
        mean(y)
    end
    initial_val         = minimizing_scalar(y)
    initial_base_func   = (X) -> fill(initial_val, size(X, 1))

    # Add initial base function to ensemble
    base_funcs[1]       = initial_base_func

    # build consecutive base functions
    prev_func_pred      = initial_base_func(X)

    # residuals y-F(x) are the negative gradients wrt F(x) of the l2loss
    @inline function negative_gradient(y, prev_func_pred)
        y .- prev_func_pred
    end

    for iter_ind = 2 : num_iterations + 1
        # obtain the current residuals
        psuedo = negative_gradient(
                y, prev_func_pred)
        # now we only consider the l2loss case

        # sub-sampling of instances
        stage_sample_ind = create_sample_indices(gb, X, y)

        # build current base function
        stage_base_func     = build_base_func(
            gb,
            X[stage_sample_ind, :],
            y[stage_sample_ind],
            prev_func_pred[stage_sample_ind],
            psuedo[stage_sample_ind])

        # update previous function prediction
        prev_func_pred .+= gb.learning_rate .* stage_base_func(X)

        # add optimal base function to ensemble
        base_funcs[iter_ind] = stage_base_func
    end

            # return model
            return GBModel(gb.learning_rate, base_funcs)
end

# fit/predict the data with gb algorithm

function fit_gb(gb :: GBAlgorithm, X, y)
    stochastic_gradient_boost(gb, X, y)
end
function predict_gb(gb_model :: GBModel, X)
    outputs = gb_model.base_funcs[1](X)
    for i = 2 : length(gb_model.base_funcs)
        outputs .+= gb_model.learning_rate .* gb_model.base_funcs[i](X)
    end
    return outputs
end


###### machine learning API ########

mutable struct GBLearner
    algorithm   :: GBAlgorithm
    output      :: Symbol
    model
    function GBLearner(algorithm, output = :regression)
        new(algorithm, output, nothing)
    end
end

import StatsBase: fit!, predict!
function fit!(gbl :: GBLearner,
              instances,
              labels)
              error("Instance type: $(typeof(instances))
              and label type: $(typeof(labels)) together is currently not supported.")
end

function predict!(gbl :: GBLearner,
                  instances)
    error("Instance type: $(eltype(instances)) is currently not supported.")
end
function fit!(gbl           :: GBLearner,
              instances     :: Matrix{Float64},
              labels        :: Vector{Float64})

              gbl.model      = fit_gb(gbl.algorithm, instances, labels)
end



function predict!(gbl       :: GBLearner,
                  instances :: Matrix{Float64})

                  # predict with GB algorithm
                  predictions = predict_gb(gbl.model, instances)
end

function gbm(gbl            :: GBLearner,
             instances      :: Matrix{Float64},
             labels         :: Vector{Float64};
             cv_folds       = nothing)

             if cv_folds    == nothing
                 gbl.model      = fit_gb(gbl.algorithm, instances, labels)
             else
                 scores         = Float64[]
                 n              = size(instances, 1)
                 gen            = Kfold(n, cv_folds)
                 for (i, train_inds) in enumerate(gen)
                     test_inds = setdiff(1:n, train_inds)
                     train_inds = Int.(train_inds)
                     test_inds  = Int.(test_inds)
                     trainX  = instances[train_inds, :]
                     trainY  = labels[train_inds]
                     testX   = instances[test_inds, :]
                     testY   = labels[test_inds]
                     gbl.model   = fit_gb(gbl.algorithm, trainX, trainY)
                     preds   = predict!(gbl, testX)
                     score   = rmse(preds, testY)

                     push!(scores, score)
                     gbl.model      = fit_gb(gbl.algorithm, instances, labels)

                     return mean(scores)

                 end
             end
end

###################### HElPER FUNCTIONS #################
function assign(Y :: Vector{T}, list :: Vector{T}) where T
    dict = Dict{T, Int}()
    @simd for i in 1:length(list)
        @inbounds dict[list[i]] = i
    end

    _Y = Array{Int}(undef, length(Y))
    @simd for i in 1:length(Y)
        @inbounds _Y[i] = dict[Y[i]]
    end

    return list, _Y
end

function assign(Y :: Vector{T}) where T
    set = Set{T}()
    for y in Y
        push!(set, y)
    end
    list = collect(set)
    return assign(Y, list)
end

@inline function zero_one(ns, n)
    return 1.0 - maximum(ns) / n
end

@inline function gini(ns, n)
    s = 0.0
    @simd for k in ns
        s += k * (n - k)
    end
    return s / (n * n)
end

# returns the entropy of ns/n
@inline function entropy(ns, n)
    s = 0.0
    @simd for k in ns
        if k > 0
            s += k * log(k)
        end
    end
    return log(n) - s / n
end

# adapted from the Julia Base.Sort Library
import Base.Sort
@inline function partition!(v, w, pivot, region)
    i, j = 1, length(region)
    r_start = region.start - 1
    @inbounds while true
        while w[i] <= pivot; i += 1; end;
        while w[j]  > pivot; j -= 1; end;
        i >= j && break
        ri = r_start + i
        rj = r_start + j
        v[ri], v[rj] = v[rj], v[ri]
        w[i], w[j] = w[j], w[i]
        i += 1; j -= 1
    end
    return j
end

# adapted from the Julia Base.Sort Library
function insert_sort!(v, w, lo, hi, offset)
    @inbounds for i = lo+1:hi
        j = i
        x = v[i]
        y = w[offset+i]
        while j > lo
            if x < v[j-1]
                v[j] = v[j-1]
                w[offset+j] = w[offset+j-1]
                j -= 1
                continue
            end
            break
        end
        v[j] = x
        w[offset+j] = y
    end
    return v
end

@inline function _selectpivot!(v, w, lo, hi, offset)
    @inbounds begin
        mi = (lo+hi)>>>1

        # sort the values in v[lo], v[mi], v[hi]

        if v[mi] < v[lo]
            v[mi], v[lo] = v[lo], v[mi]
            w[offset+mi], w[offset+lo] = w[offset+lo], w[offset+mi]
        end
        if v[hi] < v[mi]
            if v[hi] < v[lo]
                v[lo], v[mi], v[hi] = v[hi], v[lo], v[mi]
                w[offset+lo], w[offset+mi], w[offset+hi] = w[offset+hi], w[offset+lo], w[offset+mi]
            else
                v[hi], v[mi] = v[mi], v[hi]
                w[offset+hi], w[offset+mi] = w[offset+mi], w[offset+hi]
            end
        end

        # move v[mi] to v[lo] and use it as the pivot
        v[lo], v[mi] = v[mi], v[lo]
        w[offset+lo], w[offset+mi] = w[offset+mi], w[offset+lo]
        v_piv = v[lo]
        w_piv = w[offset+lo]
    end

    # return the pivot
    return v_piv, w_piv
end

# adapted from the Julia Base.Sort Library
@inline function _bi_partition!(v, w, lo, hi, offset)
    pivot, w_piv = _selectpivot!(v, w, lo, hi, offset)
    # pivot == v[lo], v[hi] > pivot
    i, j = lo, hi
    @inbounds while true
        i += 1; j -= 1
        while v[i] < pivot; i += 1; end;
        while pivot < v[j]; j -= 1; end;
        i >= j && break
        v[i], v[j] = v[j], v[i]
        w[offset+i], w[offset+j] = w[offset+j], w[offset+i]
    end
    v[j], v[lo] = pivot, v[j]
    w[offset+j], w[offset+lo] = w_piv, w[offset+j]

    # v[j] == pivot
    # v[k] >= pivot for k > j
    # v[i] <= pivot for i < j
    return j
end


# adapted from the Julia Base.Sort Library
# adapted from the Julia Base.Sort Library
# this sorts v[lo:hi] and w[offset+lo, offset+hi]
# simultaneously by the values in v[lo:hi]
const SMALL_THRESHOLD  = 20
function q_bi_sort!(v, w, lo, hi, offset)
    @inbounds while lo < hi
        hi-lo <= SMALL_THRESHOLD && return insert_sort!(v, w, lo, hi, offset)
        j = _bi_partition!(v, w, lo, hi, offset)
        if j-lo < hi-j
            # recurse on the smaller chunk
            # this is necessary to preserve O(log(n))
            # stack space in the worst case (rather than O(n))
            lo < (j-1) && q_bi_sort!(v, w, lo, j-1, offset)
            lo = j+1
        else
            j+1 < hi && q_bi_sort!(v, w, j+1, hi, offset)
            hi = j-1
        end
    end
    return v
end


# The code function below is a small port from numpy's library
# library which is distributed under the 3-Clause BSD license.
# The rest of DecisionTree.jl is released under the MIT license.

# ported by Poom Chiarawongse <eight1911@gmail.com>

# this is the code for efficient generation
# of hypergeometric random variables ported from numpy.random
function hypergeometric(good, bad, sample, rng)

    @inline function loggam(x)
        x0 = x
        n = 0
        if (x == 1.0 || x == 2.0)
            return 0.0
        elseif x <= 7.0
            n = Int(floor(7 - x))
            x0 = x + n
        end
        x2 = 1.0 / (x0*x0)
        xp = 6.2831853071795864769252867665590 # Tau
        gl0 = -1.39243221690590e+00
        gl0 = gl0 * x2 + 1.796443723688307e-01
        gl0 = gl0 * x2 - 2.955065359477124e-02
        gl0 = gl0 * x2 + 6.410256410256410e-03
        gl0 = gl0 * x2 - 1.917526917526918e-03
        gl0 = gl0 * x2 + 8.417508417508418e-04
        gl0 = gl0 * x2 - 5.952380952380952e-04
        gl0 = gl0 * x2 + 7.936507936507937e-04
        gl0 = gl0 * x2 - 2.777777777777778e-03
        gl0 = gl0 * x2 + 8.333333333333333e-02
        gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0
        if x <= 7.0
            @simd for k in 1:n
                gl -= log(x0 - k)
            end
        end
        return gl
    end

    @inline function hypergeometric_hyp(good, bad, sample)
        d1 = bad + good - sample
        d2 = min(bad, good)

        Y = d2
        K = sample
        while Y > 0
            Y -= floor(UInt, rand(rng) + Y/(d1 + K))
            K -= 1
            if K == 0
                break
            end
        end

        Z = d2 - Y
        return if good > bad
            sample - Z
        else
            Z
        end
    end

    @inline function hypergeometric_hrua(good, bad, sample)
        mingoodbad = min(good, bad)
        maxgoodbad = max(good, bad)
        popsize = good + bad
        m = min(sample, popsize - sample)
        d4 = mingoodbad / popsize
        d5 = 1.0 - d4
        d6 = m*d4 + 0.5
        d7 = sqrt((popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5)
        # d8 = 2*sqrt(2/e) * d7 + (3 - 2*sqrt(3/e))
        d8 = 1.7155277699214135*d7 + 0.8989161620588988
        d9 = floor(UInt, (m + 1) * (mingoodbad + 1) / (popsize + 2))
        d10 = (loggam(d9+1) + loggam(mingoodbad-d9+1) + loggam(m-d9+1) +
               loggam(maxgoodbad-m+d9+1))
        d11 = min(m+1, mingoodbad+1, floor(UInt, d6+16*d7))

        while true
            X = rand(rng)
            Y = rand(rng)
            W = d6 + d8*(Y - 0.5)/X

            (W < 0.0 || W >= d11) && continue
            Z = floor(Int, W)
            T = d10 - (loggam(Z+1) + loggam(mingoodbad-Z+1) + loggam(m-Z+1) +
                       loggam(maxgoodbad-m+Z+1))
            (X*(4.0-X)-3.0) <= T && break
            (X*(X-T) >= 1) && continue
            (2.0*log(X) <= T) && break
        end

        if good > bad
            Z = m - Z
        end

        return if m < sample
            good - Z
        else
            Z
        end
    end

    return if sample > 10
        hypergeometric_hrua(good, bad, sample)
    else
        hypergeometric_hyp(good, bad, sample)
    end
end

######################## print the tree methods ##############################

function print_tree(leaf::Leaf, depth=-1, indent=0)
    matches = findall(leaf.values .== leaf.majority)
    ratio = string(length(matches)) * "/" * string(length(leaf.values))
    println("$(leaf.majority) : $(ratio)")
end

function print_tree(tree::Node, depth=-1, indent=0)
    if depth == indent
        println()
        return
    end
    println("Feature $(tree.feature_id), Threshold $(tree.feature_val)")
    print("    " ^ indent * "L-> ")
    print_tree(tree.left, depth, indent + 1)
    print("    " ^ indent * "R-> ")
    print_tree(tree.right, depth, indent + 1)
end

# extending Base.show
## the leaf/node/ensemble, they will show as called
function show(io::IO, leaf::Leaf)
    println(io, "\nDecision Leaf")
    println(io, "Majority: $(leaf.majority)")
    println(io,   "Samples:  $(length(leaf.values))\n")
end
function show(io::IO, tree::Node)
    println(io, "\nDecision Tree")
    println(io, "Leaves: $(length(tree))")
    println(io, "Feature to split on: $(tree.feature_id)")
    println(io, "feature_val: $(tree.feature_val)")
    println(io,   "Depth:  $(depth(tree))\n")
end
function show(io::IO, ensemble::Ensemble)
    println(io, "\nEnsemble of Decision Trees")
    println(io, "Trees:      $(length(ensemble))")
    println(io, "Avg Leaves: $(mean([length(tree) for tree in ensemble.trees]))")
    println(io,   "Avg Depth:  $(mean([depth(tree) for tree in ensemble.trees]))\n")
end

# extending the Base.hcat

function hcat!(A, B)
    # check dimension
    if (size(A, 1) != size(B, 1))
        throw(error("Dimension of $A and $B does not match!"))
    end

    newrow_num = size(A, 1)
    newcol_num = size(A, 2) + size(B, 2)
    temp = Array{Union{Nothing, Missing,Int, Float64}}(nothing, newrow_num, newcol_num)
    temp[:, 1:size(A, 2)] = A
    temp[:, size(A, 2) + 1 : end] = B

    return temp

end
###########################################



###### Simulation of Fried Example ########
function fried(X::Array{Float64, 2})
    n = size(X,1)
    p = size(X,2)

    Y = zeros(n)
    for i in 1:n
        Y[i] = 10 * sin(pi * X[i,1] * X[i,2]) +
            20 * (X[i,3] - 0.5)^2 + 10 * X[i,4] + 5 * X[i,5]
    end
    return Y
end

# test_data() simulates fried data with noise

function test_data(n::Int64; sigma = 1.0, p = 10)
    X = rand(n,p)
    Y = fried(X) + sigma * randn(n)
    return X, Y
end
############### End of Simulation of Fried Example ##############

############ TODO: Becnmark ########


# Importance of features: 1, 2, 4


#### plot ######



end  # module GB
