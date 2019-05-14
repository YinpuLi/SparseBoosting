module Example

import Base.sho
import Base.+

export MyType, foo, +

struct MyType
    x
end

bar(x) = 2x
foo(a :: MyType) = bar(a.x) + 1
+(a :: MyType, b :: MyType) = a.x + b.x



show(io :: IO, a :: MyType)  = println(io, "MyType $(a.x)")



end
