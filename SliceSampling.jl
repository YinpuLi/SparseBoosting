module SliceSampling

using Distributions, Roots, DataFrames


########## ALgorithm ##########
# 1. Choose a random starting value of x, let's call it x₀
# 2. Uniformly sample on interval [0, f(x₀)], let's call it a
# 3. Here's the tricky part - imagine a horizontal line at y = a. Figure out all the line segments under the curve.
# 4. From all the line segments, draw a value of x uniformly.
# 5. Repeat from step 2 until you have as many draws as you want.
######### end of Alg #########


# this implements slice sampling of a normal distribution with mean 10
    # and standard deviatio  2.
    # I used an x interval of 0 to 20,
    # with 0.1 root finding step size


# sliceSample function:
    # @ n =  number of points wanted
    # @ f = target distribution
    # @ w = root accuracy
    # @ interval = (lb, ub) of x values possible

function sliceSample(n, f , interval, w)
    pts = fill(0.0, n) # this vector will hold our points

    L = interval[1]
    R = interval[2]

    x   = rand(1) * (R - L)
    x   = x[1]  # random starting value x₀

    # we want to sample n points:
    for i in 1:n
        pts[i] = x
        y      = rand(1) * (f(x)[1] - 0.0)
        y      = y[1]
        # imagine a horizontal line across the dstribution
        # find intersections across that line
        function fshift(x)
            f(x) - y
        end

        roots = []
        seq   = collect(range(L, step = w, stop = R))

        for j in seq
            if (fshift(j) < 0) != (f(j + w)<y)
                # signs don't macth, so we have a root
                root = find_zero(z -> f(z) - y, j)
                if j< root < j + w
                    push!(roots, root)
                end
            end
        end

        # include the endpoints of the interval
        pushfirst!(roots, L)
        push!(roots, R)

        # devide the intersections into line segments
        segments = Array{Float64, 2}(undef, 1, 2)
        segments = DataFrame(segments)
        for j in 1 : (length(roots) - 1)
            midpoint = (roots[j + 1] + roots[j]) / 2.0
            if (f(midpoint) > y[1])
                # since this segment is under the curve, add it to the segments
                intv = [roots[j] roots[j+1]]
                append!(segments, intv)
            end
        end

        # uniformly sample next x from segements
        # assign each segment a probability, then unif based on those probabilities
        # note that the first row of segments is some undef number
        segments = segments[2:end, :] # drop the first row

        total = sum(segments[:, 2] .- segments[:, 1])

        prob_vec = (segments[:, 2] .- segments[:, 1]) ./ total

        # assign probabilities to each line segement based on how long it is
        # select a line segment by index(named seg)

        p = rand(1)

        @inline function selectSegment(x, i)
            if p[1] < x
                return i
            else
                return (selectSegment(x + prob_vec[i + 1], i + 1))
            end
        end

        seg = selectSegment(prob_vec[1], 1) # start from the first segment

        # uniformly sample new x value
        x = rand(Uniform(segments[seg, 1], segments[seg, 2]), 1)
        x = x[1]
    end

    return pts

end





end  # module SliceSamplinmg
