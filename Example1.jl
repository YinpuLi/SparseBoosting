module Example1

include("Example.jl")
# import Example: MyType, foo, +

export  MyType, foo, mysqure
mysqure(x :: Example.MyType) = Example.foo(x)^2

end
