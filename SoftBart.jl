module softbart

using Distributions
using MultivariateStats
using SpecialFunctions
using StatsBase
using LinearAlgebra

mutable struct Node

    is_leaf::Bool
    is_root::Bool
    left::Node
    right::Node
    parent::Node

    ## branch parameters
    var::Int64
    val::Float64

    lower::Float64
    upper::Float64

    ## Leaf parameters
    mu::Float64

    ## Parameters for computing weights efficiently
    current_weight::Float64

    Node() = new()

end

mutable struct Hypers
    alpha::Float64
    beta::Float64
    gamma::Float64
    sigma::Float64
    sigma_mu::Float64
    shape::Float64
    width::Float64
    num_tree::Int64
    s::Vector{Float64}

    ## Double hypers
    sigma_hat::Float64
    sigma_mu_hat::Float64
    alpha_scale::Float64
    alpha_shape_1::Float64
    alpha_shape_2::Float64


    Hypers() = new()
end

mutable struct Opts
    num_burn::Int64
    num_thin::Int64
    num_save::Int64
    num_print::Int64

    update_sigma_mu::Bool
    update_s::Bool
    update_alpha::Bool
    update_beta::Bool
    update_gamma::Bool
end

mutable struct SoftbartOut
    y_hat_train::Array{Float64,2}
    y_hat_test::Array{Float64,2}
    y_hat_train_mean::Vector{Float64}
    y_hat_test_mean::Vector{Float64}
    sigma::Vector{Float64}
    sigma_mu::Vector{Float64}
    s::Array{Float64,2}
    alpha::Vector{Float64}
    beta::Vector{Float64}
    gamma::Vector{Float64}
end

function SoftbartOut(opts::Opts, n_train::Int64, n_test::Int64, hypers::Hypers)

    num_save = opts.num_save

    y_hat_train = zeros(num_save, n_train)
    y_hat_test = zeros(num_save, n_test)
    y_hat_train_mean = zeros(n_train)
    y_hat_test_mean = zeros(n_test)


    sigma = zeros(num_save)

    sigma_mu = opts.update_sigma_mu ? zeros(num_save) : hypers.sigma_mu * ones(1)
    s = opts.update_s ?
        zeros(num_save, length(hypers.s)) :
        reshape(hypers.s, 1, length(hypers.s))
    alpha = opts.update_alpha ? zeros(num_save) : hypers.alpha * ones(1)
    beta = opts.update_beta ? zeros(num_save) : hypers.beta * ones(1)
    gamma = opts.update_gamma ? zeros(num_save) : hypers.gamma * ones(1)


    return SoftbartOut(y_hat_train, y_hat_test, y_hat_train_mean,
                       y_hat_test_mean, sigma, sigma_mu, s, alpha, beta, gamma)
end

function Opts(;num_burn = 1000,
              num_thin = 1,
              num_save = 1000,
              num_print = 100,
              update_sigma_mu = true,
              update_s = true,
              update_alpha = true,
              update_beta = true,
              update_gamma = true)
    return Opts(num_burn, num_thin, num_save, num_print, update_sigma_mu,
                update_s, update_alpha, update_beta, update_gamma)
end

function Hypers(X::Array{Float64, 2},
                Y::Vector{Float64};
                alpha = 0.95,
                beta = 2.0,
                gamma = 0.95,
                k = 2.0,
                width = 0.1,
                shape = 1.0,
                num_tree = 50,
                alpha_scale = 0.0,
                alpha_shape_1 = 0.5,
                alpha_shape_2 = 1.0)

    out = Hypers()
    out.alpha = alpha
    out.beta = beta
    out.gamma = gamma
    out.sigma = GetSigma(X,Y)
    out.sigma_mu = 0.5 / (k * sqrt(num_tree))
    out.shape = shape
    out.width = width
    out.num_tree = num_tree
    out.s = ones(size(X,2)) / size(X,2)

    out.sigma_hat = out.sigma
    out.sigma_mu_hat = out.sigma_mu

    out.alpha_scale = alpha_scale == 0.0 ? size(X,2) : alpha_scale
    out.alpha_shape_1 = alpha_shape_1
    out.alpha_shape_2 = alpha_shape_2

    return out;

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
    x.mu = 0.0
    x.current_weight = 1.0

    return x
end

function InitRoot(X::Array{Float64,2})
    x = Root()
    return x
end

function GetSigma(X::Array{Float64, 2}, Y::Vector{Float64})
    if size(X,2) < size(X,1) / 5.0
        beta = llsq(X,Y)        # linear least square
        p = size(X,2)
        Yhat = X * beta[1:p] .+ beta[p+1]
        return sqrt(mean((Y - Yhat).^2))
    else
        return sqrt(var(Y))
    end
end

function AddLeaves!(x::Node)
    x.left = Node()
    x.right = Node()
    x.is_leaf = false

    ## Initialize the leafs
    x.left.is_leaf = true
    x.left.parent = x
    x.left.var = 0
    x.left.val = 0.0
    x.left.is_root = false
    x.left.lower = 0.0
    x.left.upper = 1.0
    x.left.mu = 0.0
    x.left.current_weight = 0.0
    x.right.is_leaf = true
    x.right.parent = x
    x.right.var = 0
    x.right.val = 0.0
    x.right.is_root = false
    x.right.lower = 0.0
    x.right.upper = 1.0
    x.right.mu = 0.0
    x.right.current_weight = 0.0

    return nothing
end

function BirthLeaves!(x::Node, hypers::Hypers)
    if !x.is_leaf
        error("Node must be a leaf")
    end
    AddLeaves!(x)
    wv = ProbabilityWeights(hypers.s)
    x.var = sample(wv)
    # x.var = rand(1:P) ## TODO change this to allow non-iid selection
    lower, upper = GetLimits(x)
    x.lower = lower
    x.upper = upper
    x.val = (upper - lower) * rand() + lower
end

function GenTree(hypers::Hypers, P::Int64)
    x = Root()
    GenTree!(x, hypers, P)
    return x
end

function GenTree!(x::Node, hypers::Hypers, P::Int64)
    grow_prob = SplitProb(x, hypers)
    u = rand()
    if u < grow_prob
        BirthLeaves!(x, hypers)
        GenTree!(x.left, hypers, P)
        GenTree!(x.right, hypers, P)
    end
end

function SplitProb(n::Node, hypers::Hypers)
    d = depth(n)
    grow_prob = hypers.gamma / (1 + d)^(hypers.beta)
    return grow_prob
end

function GetLimits(x::Node)
    y = x
    lower = 0.0
    upper = 1.0
    my_bool = y.is_root ? false : true
    while my_bool
        is_left = (y.parent.left == y)
        y = y.parent
        my_bool = y.is_root ? false : true
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

function GetSuffStats(n::Node,
                      y::Vector{Float64},
                      x::Array{Float64,2},
                      hypers::Hypers)
    leafs = leaves(n)
    num_leaves = length(leafs)
    w_i = zeros(num_leaves)
    mu_hat = zeros(num_leaves)
    Lambda = zeros(num_leaves, num_leaves)
    for i in 1:size(x,1)
        GetW!(n, x, i, hypers)
        for j in 1:num_leaves
            w_i[j] = leafs[j].current_weight
        end
        mu_hat += y[i] * w_i
        Lambda += w_i * transpose(w_i)
    end

    Lambda /= hypers.sigma^2
    mu_hat /= hypers.sigma^2


    Omega_inv = Lambda + LinearAlgebra.I  / hypers.sigma_mu^2

    mu_hat = Omega_inv\mu_hat  # TODO: what does this symbol "\" represent???
    # \(x, y) Left division operator: multiplication of y by the inverse of x on the left.
    return mu_hat, Omega_inv

end

function LogLT(n::Node, Y::Vector{Float64}, X::Array{Float64,2}, hypers::Hypers)

    leafs = leaves(n)
    num_leaves = length(leafs)

    ## Get sufficient statistics
    mu_hat, Omega_inv = GetSuffStats(n, Y, X, hypers)

    N = size(Y,1)

    out = -0.5 * N * log(2.0 * pi * hypers.sigma^2)
    out -= 0.5 * num_leaves * log(2.0 * pi * hypers.sigma_mu^2)
    out -= 0.5 * logdet(Omega_inv / (2.0 * pi))
    out -= 0.5 * sum(Y.^2) / hypers.sigma^2
    out += 0.5 * dot(mu_hat, Omega_inv * mu_hat)

    return out


end

function cauchy_jacobian(tau::Float64, sigma_hat::Float64)
    sigma = tau^(-0.5)
    out = logpdf(Cauchy(0,sigma_hat), sigma)

    out = out + log(0.5) - 3.0 / 2.0 * log(tau)

    return out

end

function update_sigma(r::Vector{Float64}, sigma_hat::Float64, sigma_old::Float64)
    SSE = sum(r .* r)
    n = length(r)               # Gamma(shape, scale)
    sigma_prop = rand(Gamma(0.5 * n + 1.0, 2.0 / SSE))^(-0.5) 
    tau_prop = sigma_prop^(-2)
    tau_old = sigma_old^(-2)

    loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) -
        cauchy_jacobian(tau_old, sigma_hat)

    return log(rand()) < loglik_rat ? sigma_prop : sigma_old

end

function UpdateSigma!(hypers::Hypers, r::Vector{Float64})
    hypers.sigma = update_sigma(r, hypers.sigma_hat, hypers.sigma)
end

function UpdateSigmaMu!(hypers::Hypers, means::Vector{Float64})
    hypers.sigma_mu = update_sigma(means, hypers.sigma_mu_hat, hypers.sigma_mu)
end

function UpdateMu!(n::Node,
                   Y::Vector{Float64},
                   X::Array{Float64,2},
                   hypers::Hypers)

    leafs = leaves(n)
    num_leaves = length(leafs)

    ## Get mean and covariance
    mu_hat, Omega_inv = GetSuffStats(n,Y,X,hypers)

    ## Update
    mu = mu_hat .+ rand(MvNormalCanon(Omega_inv),1) # sample 1 vector from the distribution, each column is a sample

    # for Multinomial distributin, the canonical parameters:
    # h = Σ^-1 × μ
    # J = Σ^-1
    # MvNormalCanon(J):  Construct a multivariate normal distribution with zero mean (thus zero potential vector) and precision matrix represented by J.
    for i in 1:num_leaves
        leafs[i].mu = mu[i]
    end

end

function predict(forest::Vector{Node},
                 X::Array{Float64,2},
                 hypers::Hypers)

    out = zeros(size(X,1))
    T = length(forest)
    for t in 1:T
        out += predict(forest[t], X, hypers)
    end

    return out

end

function predict(n::Node,
                 X::Array{Float64,2},
                 hypers::Hypers)

    leafs = leaves(n)
    num_leaves = length(leafs)
    N = size(X,1)
    out = zeros(N)

    for i in 1:N
        GetW!(n, X, i, hypers)
        for j in 1:num_leaves
            out[i] += leafs[j].current_weight * leafs[j].mu
        end
    end

    return out

end

function GetW!(n::Node, x::Array{Float64, 2}, i::Int64, hypers::Hypers)

    if !n.is_leaf


        weight = activation(x[i, n.var], n.val, hypers)
        n.left.current_weight = weight * n.current_weight
        n.right.current_weight = (1 - weight) * n.current_weight

        GetW!(n.left, x, i, hypers)
        GetW!(n.right, x, i, hypers)

    end

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

function PrintTree(x::Node)
    d = depth(x)
    for i in 0:d
        print("    ")
    end
    print(x.var)
    print("\n")
    if !x.is_leaf
        PrintTree(x.right)
        PrintTree(x.left)
    end
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

function get_means(forest::Vector{Node})
    means = Float64[]
    for tree in forest
        get_means!(tree, means)
    end
    return means
end

function get_means!(node::Node, means::Vector{Float64})

    if node.is_leaf
        push!(means, node.mu)
    else
        get_means!(node.left, means)
        get_means!(node.right, means)
    end
end

function test_tree()
    x = Root()
    AddLeaves!(x)
    AddLeaves!(x.left)
    AddLeaves!(x.left.left)
    AddLeaves!(x.left.right)

    x.var = 1
    x.left.var = 2
    x.right.var = 0
    x.left.left.var = 3
    x.left.right.var = 4
    x.left.left.left.var = 0
    x.left.left.right.var = 0
    x.left.right.left.var = 0
    x.left.right.right.var = 0
    return x
end

function test_tree_2(P::Int64, hypers::Hypers)
    x = Root()
    BirthLeaves!(x, hypers)
    BirthLeaves!(x.left, hypers)
    BirthLeaves!(x.left.left, hypers)
    BirthLeaves!(x.left.right, hypers)


    return x
end


function init_forest(X::Array{Float64, 2}, Y::Vector{Float64}, hypers)
    forest = Node[]
    for t in 1:hypers.num_tree
        n = InitRoot(X)
        push!(forest, n)
    end
    return forest
end

function soft_bart(
                        X::Array{Float64, 2},
                        Y::Vector{Float64},
                        X_test::Array{Float64,2},
                        hypers::Hypers,
                        opts::Opts)

    a = minimum(Y)
    b = maximum(Y)
    Y_u = (Y .- a) ./ (b - a) .- 0.5
    hypers_u = deepcopy(hypers)
    hypers_u.sigma /= (b-a)
    hypers_u.sigma_hat /= (b-a)

    out = soft_bart_norm(X,Y_u,X_test,hypers_u,opts)

    ## Unnormalize
    out.y_hat_train = (b - a) * (out.y_hat_train .+ 0.5) .+ a
    out.y_hat_test = (b - a) * (out.y_hat_test .+ 0.5) .+ a
    out.sigma = (b - a) * out.sigma
    out.y_hat_train_mean = (b - a) * (out.y_hat_train_mean .+ 0.5) .+ a
    out.y_hat_test_mean = (b - a) * (out.y_hat_test_mean .+ 0.5) .+ a

    # return (b-a) * (y_hat + 0.5) + a, (b-a) * sigma, sigma_mu, forest, hypers_u

    return out

end

function soft_bart_norm(
                   X::Array{Float64,2},
                   Y::Vector{Float64},
                   X_test::Array{Float64,2},
                   hypers::Hypers,
                   opts::Opts)

    MH_BD = 0.7

    forest = init_forest(X,Y,hypers)

    Y_hat = zeros(size(X,1))

    ## Output
    # Y_hat_test = zeros(opts.num_save, size(X_test, 1))
    # sigma = zeros(opts.num_save)
    # sigma_mu = zeros(opts.num_save)
    out = SoftbartOut(opts, size(X,1), size(X_test,1), hypers)

    for i in 1:opts.num_burn
        for t in 1:hypers.num_tree
            Y_star = Y_hat - predict(forest[t], X, hypers)
            res = Y - Y_star

            if forest[t].is_leaf
                birth_death!(forest[t], X, res, hypers)
            elseif rand() < MH_BD
                birth_death!(forest[t], X, res, hypers)
            else
                change_decision_rule!(forest[t], X, res, hypers)
            end

            UpdateMu!(forest[t],res,X,hypers)
            Y_hat = Y_star .+ predict(forest[t], X, hypers)
        end

        res = Y - Y_hat

        ## Update sigma
        UpdateSigma!(hypers, res)

        ## Update sigma_mu
        if opts.update_sigma_mu
            means = get_means(forest)
            UpdateSigmaMu!(hypers, means)
        end

        ## update s and alpha
        if (i > opts.num_burn / 2.0 && opts.update_s)
            UpdateS!(forest, hypers)
            if(opts.update_alpha)
                UpdateAlpha!(hypers)
            end
        end

        ## Update beta and gamma
        if opts.update_beta
            UpdateBeta!(forest, hypers)
        end
        if opts.update_gamma
            UpdateGamma!(forest, hypers)
        end

        if(i % opts.num_print == 0)
            print("Finishing burn", i, "\n")
        end
    end

    for i in 1:opts.num_save
        for b in 1:opts.num_thin
            for t in 1:hypers.num_tree
                Y_star = Y_hat - predict(forest[t], X, hypers)
                res = Y - Y_star

                if forest[t].is_leaf
                    birth_death!(forest[t], X, res, hypers)
                elseif rand() < MH_BD
                    birth_death!(forest[t], X, res, hypers)
                else
                    change_decision_rule!(forest[t], X, res, hypers)
                end
                UpdateMu!(forest[t],res,X,hypers)
                Y_hat = Y_star .+ predict(forest[t], X, hypers)
            end

            res = Y - Y_hat
            UpdateSigma!(hypers, res)

            ## Update sigma
            UpdateSigma!(hypers, res)

            ## Update sigma_mu
            if opts.update_sigma_mu
                means = get_means(forest)
                UpdateSigmaMu!(hypers, means)
            end

            ## update s and alpha
            if (opts.update_s)
                UpdateS!(forest, hypers)
                if(opts.update_alpha)
                    UpdateAlpha!(hypers)
                end
            end


            ## Update beta and gamma
            if opts.update_beta
                UpdateBeta!(forest, hypers)
            end
            if opts.update_gamma
                UpdateGamma!(forest, hypers)
            end

        end

        ## Record predictions
        out.y_hat_test[i,:] = predict(forest,X_test,hypers)
        out.y_hat_train[i,:] = Y_hat

        ## Record sigma
        out.sigma[i] = hypers.sigma

        ## Record hypers
        if opts.update_sigma_mu
            out.sigma_mu[i] = hypers.sigma_mu
        end
        if opts.update_s
            out.s[i,:] = hypers.s
        end
        if opts.update_alpha
            out.alpha[i] = hypers.alpha
        end
        if opts.update_beta
            out.beta[i] = hypers.beta
        end
        if opts.update_gamma
            out.gamma[i] = hypers.gamma
        end

        out.sigma_mu[i] = hypers.sigma_mu
        out.sigma[i] = hypers.sigma
        if (i % opts.num_print == 0)
            print("Finishing save", i, "\n")
        end
    end

    out.y_hat_train_mean = vec(mean(out.y_hat_train, dims = (1)))
    out.y_hat_test_mean = vec(mean(out.y_hat_test, dims = (1)))

    return out
end

function fried(X::Array{Float64, 2})
    n = size(X,1)
    p = size(X,2)

    Y = zeros(n)
    for i in 1:n
        Y[i] = 10 * sin(2 * pi * X[i,1] .* X[i,2]) .+
            20 * (X[i,3] - 0.5).^2 .+ 10 * X[i,4] .+ 5 * X[i,5]
    end
    return Y
end

function logit(x::Float64)
    return log(x) - log(1 - x)
end

function normalize(y::Vector{Float64})
    a = minimum(y)
    b = maximum(y)
    return (y - a) / (b - a) - 0.5
end

function expit(x::Float64)
    return 1.0 / (1 + exp(-x))
end

function activation(x::Float64, c::Float64, hypers::Hypers)
    return 1- expit((x - c) / hypers.width)
end



function birth_death!(tree::Node,
                     X::Array{Float64, 2},
                     Y::Vector{Float64},
                     hypers::Hypers)

    u = rand()
    p_birth = probability_node_birth(tree)
    if u < p_birth
        node_birth!(tree::Node,
                    X::Array{Float64, 2},
                    Y::Vector{Float64},
                    hypers::Hypers)
    else
        node_death!(tree::Node,
                    X::Array{Float64,2},
                    Y::Vector{Float64},
                    hypers::Hypers)
    end
end

function node_birth!(tree::Node,
                     X::Array{Float64,2},
                     Y::Vector{Float64},
                     hypers::Hypers)

    leaf, leaf_probability = birth_node(tree)

    leaf_depth = depth(leaf)
    leaf_prior = growth_prior(leaf, leaf_depth, hypers)

    ## Get likelihood before
    ll_before = LogLT(tree, Y, X, hypers)
    ll_before += log(1.0 - leaf_prior)

    ## Get transition probability
    p_forward = log(probability_node_birth(tree) * leaf_probability)

    ## Birth new leaves
    BirthLeaves!(leaf, hypers) ## TODO Change selection of feature

    ## Get likelihood after
    ll_after = LogLT(tree, Y, X, hypers)
    ll_after += log(leaf_prior) +
        log(1.0 - growth_prior(leaf.left, leaf_depth + 1, hypers)) +
        log(1.0 - growth_prior(leaf.right, leaf_depth + 1, hypers))

    ## Get probability of reverse transition: TODO not sure if I need to account
    ## for the choice of splitting value; I don't think I need to, but not sure
    num_not_grand_branches = length(not_grand_branches(tree))
    p_not_grand = 1.0/num_not_grand_branches
    p_backward = log((1 - probability_node_birth(tree)) * p_not_grand)

    transition_prob = ll_after + p_backward - ll_before - p_forward

    if log(rand()) > transition_prob
        DeleteLeaves!(leaf)
        leaf.var = 0
    end

end

function node_death!(tree::Node,
                     X::Array{Float64,2},
                     Y::Vector{Float64},
                     hypers::Hypers)

    ## Select branch to kill children
    branch, p_not_grand = death_node(tree)

    ## Compute before likelihood
    leaf_depth = depth(branch.left)
    leaf_prob = growth_prior(branch, leaf_depth - 1, hypers)
    left_prior = growth_prior(branch.left, leaf_depth, hypers)
    right_prior = left_prior
    ll_before = LogLT(tree, Y, X, hypers) +
        log(1.0 - left_prior) + log(1.0 - right_prior) + log(leaf_prob)

    ## Compute forward transition prob
    p_forward = log(p_not_grand * (1.0 - probability_node_birth(tree)))

    ## Save old leafs
    left = branch.left
    right = branch.right

    ## Compute after likelihood
    DeleteLeaves!(branch)

    ll_after = LogLT(tree, Y, X, hypers) + log(1.0 - leaf_prob)

    ## Compute backwards transition
    p_backwards = log(1.0 / length(leaves(tree)) * probability_node_birth(tree))

    ## Do MH
    transition_prob = ll_after + p_backwards - ll_before - p_forward
    if log(rand()) > transition_prob
        branch.left = left
        branch.right = right
        branch.is_leaf = false
    end

end

function change_decision_rule!(tree::Node,
                               X::Array{Float64, 2},
                               Y::Vector{Float64},
                               hypers::Hypers)

    ## Select a node to change
    ngb = not_grand_branches(tree)
    branch = rand(ngb)

    ## Calculate likelihood before proposal
    ll_before = LogLT(tree, Y, X, hypers)
    # ll_before = ll_before

    ## save old split
    old_feature = branch.var
    old_value   = branch.val
    old_lower = branch.lower
    old_upper = branch.upper

    # branch.var = rand(1:size(X,2)) ## TODO change for sparsity
    branch.var = sample(ProbabilityWeights(hypers.s))
    lower, upper = GetLimits(branch)
    branch.lower = lower
    branch.upper = upper
    branch.val = (upper - lower) * rand() + lower

    ll_after = LogLT(tree, Y, X, hypers)

    log_trans_prob = ll_after - ll_before

    if log(rand()) > log_trans_prob
        branch.var = old_feature
        branch.val = old_value
        branch.lower = old_lower
        branch.upper = old_upper
    end

end

function growth_prior(leaf::Node, leaf_depth::Int64, hypers::Hypers)
    return hypers.gamma * (1.0 + leaf_depth)^(-hypers.beta)
end

function birth_node(tree::Node)
    leafs = leaves(tree)
    leaf = rand(leafs)
    num_leafs = length(leafs)
    leaf_node_probability = 1.0 / num_leafs

    return leaf, leaf_node_probability
end

function test_data(n::Int64; sigma = 1.0, p = 10)
    X = rand(n,p)
    Y = fried(X) .+ sigma * randn(n)
    return X, Y
end

function probability_node_birth(tree::Node)
    out = tree.is_leaf ? 1.0 : 0.5
end

function death_node(tree::Node)
    ngb = not_grand_branches(tree)
    branch = rand(ngb)
    p_not_grand = 1.0 / length(ngb)
    return branch, p_not_grand
end

function not_grand_branches(tree::Node)
    ngb = Node[]
    not_grand_branches!(ngb, tree)
    return ngb
end

function not_grand_branches!(ngb::Vector{Node}, node::Node)

    if !node.is_leaf
        left_is_leaf = node.left.is_leaf
        right_is_leaf = node.right.is_leaf
        if left_is_leaf && right_is_leaf
            push!(ngb, node)
        else
            not_grand_branches!(ngb, node.left)
            not_grand_branches!(ngb, node.right)
        end
    end

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

function UpdateS!(forest::Vector{Node}, hypers::Hypers)
    shape_up = get_var_counts(forest, hypers) .+ hypers.alpha / length(hypers.s)
    hypers.s = rand(Dirichlet(shape_up))
end

function alpha_to_rho(alpha::Float64, scale::Float64)
    return alpha / (alpha + scale)
end

function rho_to_alpha(rho::Float64, scale::Float64)
    return scale * rho / (1 - rho)
end

function UpdateAlpha!(hypers::Hypers)
    alpha_current = hypers.alpha

    rho_current = alpha_to_rho(alpha_current, hypers.alpha_scale)
    psi = mean([log(hypers.s[i]) for i in 1:length(hypers.s)])    # psi = mean(log.(s))
    p = 1.0 * length(hypers.s)

    # SpecialFunctions.lgamma(x) = |log(gamma(x::Real))|
    # log_lik_current = log(π(ρ | -)) in S.1
    loglik_current = alpha_current * psi + lgamma(alpha_current) -
        p * lgamma(alpha_current / p) +
        logpdf(Beta(hypers.alpha_shape_1, hypers.alpha_shape_2), rho_current)

    for i in 1:50
        rho_propose = rand(Beta(hypers.alpha_shape_1, hypers.alpha_shape_2))
        alpha_propose = rho_to_alpha(rho_propose, hypers.alpha_scale)

        loglik_propose = alpha_propose * psi + lgamma(alpha_propose) -
            p * lgamma(alpha_propose / p) +
            logpdf(Beta(hypers.alpha_shape_1, hypers.alpha_shape_2), rho_propose)

        if log(rand()) < loglik_propose - loglik_current
            alpha_current = alpha_propose
            rho_current = rho_propose
            loglik_current = loglik_propose
        end
    end

    hypers.alpha = alpha_current

end

function growth_prior(node_depth::Int64, gamma::Float64, beta::Float64)
    return gamma * (1.0 + node_depth)^(-beta)
end

function forest_loglik(forest::Vector{Node}, gamma::Float64, beta::Float64)
    out = 0.0
    for t in 1:length(forest)
        out += tree_loglik(forest[t], 0, gamma, beta)
    end
    return out
end

function tree_loglik(node::Node, node_depth::Int64, gamma::Float64, beta::Float64)
    out = 0.0
    if node.is_leaf
        out += log(1.0 - growth_prior(node_depth, gamma, beta))
    else
        out += log(growth_prior(node_depth, gamma, beta))
        out += tree_loglik(node.left, node_depth + 1, gamma, beta)
        out += tree_loglik(node.right, node_depth + 1, gamma, beta)
    end
    return out
end

function UpdateGamma!(forest::Vector{Node}, hypers::Hypers)
    gamma_current = hypers.gamma
    loglik_current = forest_loglik(forest, gamma_current, hypers.beta)

    for i in 1:10
        gamma_prop = 0.5 * rand() + 0.5
        loglik_prop = forest_loglik(forest, gamma_prop, hypers.beta)
        if log(rand()) < loglik_prop - loglik_current
            gamma_current = gamma_prop
            loglik_current = loglik_prop
        end
    end

    hypers.gamma = gamma_current

end

function UpdateBeta!(forest::Vector{Node}, hypers::Hypers)
    beta_current = hypers.beta
    loglik_current = forest_loglik(forest, hypers.gamma, beta_current)

    for i in 1:10
        beta_prop = 2 * abs(randn())
        loglik_prop = forest_loglik(forest, hypers.gamma, beta_prop)
        if log(rand()) < loglik_prop - loglik_current
            beta_current = beta_prop
            loglik_current = loglik_prop
        end
    end

    hypers.beta = beta_current

end

end
import Base.print
print(io::IO, x::softbart.Node) = print(io, softbart.PrintTree(x));
import Base.show
show(io::IO, x::softbart.Node) = print(io, x);


####### Yinpu added #####
import Base.show
function show(io::IO, fit :: SoftbartOut)
    println(io, "\nFit:")
    println(io, "Updated α: $(fit.alpha[end])")
    println(io, "Updated β: $(fit.beta[end])")
    println(io, "Updated γ: $(fit.gamma[end])")
    println(io, "Updated σ: $(fit,.igma[end])")
    println(io, "Updated σᵤ: $(fit.sigma_mu[end])")
    println(io, "Updated s: $(fit.s[end,:])\n")
end

##### end of Yinpu added ######
