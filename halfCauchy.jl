using Distributions
using SpecialFunctions
import Base.convert
# this is a file defining the half-Cauchy Distribution

struct HalfCauchy{T <: Real} <: Distributions.ContinuousDistribution
    a :: T # Beta(a, b) shape₁
    b :: T # Beta(a, b) shape₂

    function HalfCauchy{T}(a :: T, b :: T) where T
        Distributions.@check_args(HalfCauchy, a > zero(a), b > zero(b))
        new{T}(a, b)
    end
end

HalfCauchy(a :: T, b :: T) where{T <: Real} = HalfCauchy{T}(a, b)
HalfCauchy(a :: Real, b :: Real) =  HalfCauchy(promote(a, b)...)
HalfCauchy(a :: Integer, b :: Integer) = HalfCauchy(Float64(a), Float64(b))
HalfCauchy() = HalfCauchy(0.5, 0.5)

Distributions.@distr_support HalfCauchy 0 Inf

#### Conversions

function convert(:: Type{HalfCauchy{T}}, a :: Real, b :: Real) where T<: Real
    HalfCauchy(T(a), T(b))
end
function convert(:: Type{HalfCauchy{T}}, d :: HalfCauchy{S}) where {T <: Real, S <: Real}
    HalfCauchy(T(d.a), T(d.b))
end

#### Parameters
params(d :: HalfCauchy) = (d.a, d.b)

#### Statistics
# I skipped this part

#### Functions
pdf(d :: HalfCauchy, x :: Real) = (x ^(b - 1) /(1 + x)^(d.a + d.b)) / beta(d.a, d.b)
