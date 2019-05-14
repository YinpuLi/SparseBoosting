hi = 2
nr = 0
nl = 500
if (isa(hi, Int64) || nr != 0 || nl != 0)
    print(1)
else
    print(0)
end




function myfunc(nr, nl, hi)
    if (nr < nl && isa(hi, Float64))
        print(2)
        result = 1
    elseif (nr<nl && isa(hi, Int64))
        print(3)
        return nothing
    end
    return result
end

myfunc(nr, nl, hi)


function UpdateSigma!(hypers    :: Hypers,
                    testX       :: Matrix{Float64},
                    testY       :: Vector{Float64},
                    testμ       :: Vector{Float64},
                        gbl     :: GBLearner)

        y_hat = predict!(gbl, testX)
        σ_hat1 = rmse(y_hat, testY)
        σ_hat2 = rmse(y_hat, testμ)

        hypers.σ = σ_hat1
end
