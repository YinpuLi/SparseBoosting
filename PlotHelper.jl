using DataFrames, Gadfly
ind = collect(1:28)
mydata = DataFrame(hcat(ind, rmsefitted2, rmsefitted2_sigma10))
stack(mydata, [:x2, :x3])

myplot = plot(stack(mydata, [:x2, :x3]),
    x = :x1,
    y = :value,
    color = :variable,
    Geom.point,
    Geom.line)
draw(PNG("myplot.png", 3inch, 3inch), myplot)
