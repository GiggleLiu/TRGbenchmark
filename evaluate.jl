using TensorNetworkAD
using TensorNetworkAD: isingtensor
using DataFrames, DataFramesMeta, CSV
using Plots, StatsPlots
using Statistics
using LsqFit, LaTeXStrings

timeitensor(β, niter, χ) = @elapsed read(`./itensor/trg $β $niter $χ`, String)
timepython(β, niter, χ)  = @elapsed read(`python3 pytorch/trg.py -T $β -iters $niter -cut $χ`, String)
timetnad(β, niter, χ)    = @elapsed trg(isingtensor(β), χ, niter)

timeitensor(1.,10,10)
timepython(1.,10,10)
timetnad(1.,10,10)

df = DataFrame(β = Float64[], niter = Int[], χ = Int[], label = String[], time = Float64[])
for β = 1.0, niter in 5:5:60, χ = 20, _ in 1:5
    push!(df, (β = β, niter = niter, χ = χ, label = "itensor", time = timeitensor(β, niter, χ)))
    push!(df, (β = β, niter = niter, χ = χ, label = "torch", time = timepython(β, niter, χ)))
    push!(df, (β = β, niter = niter, χ = χ, label = "julia", time = timetnad(β, niter, χ)))
end
for β = 1.0, niter in 20, χ = 10:5:40, _ in 1:5
    push!(df, (β = β, niter = niter, χ = χ, label = "itensor", time = timeitensor(β, niter, χ)))
    push!(df, (β = β, niter = niter, χ = χ, label = "torch", time = timepython(β, niter, χ)))
    push!(df, (β = β, niter = niter, χ = χ, label = "julia", time = timetnad(β, niter, χ)))
end

CSV.write("times.csv", df)


#=
    Scaling with iterations
=#

dfniter = by(@where(df, :χ .== 20), [:label,:niter], time = :time => median, sort=true)
@. model(x, (a,b)) = a * x + b
dffit = by(dfniter, :label, sort=true) do df
    label = df.label
    fit = curve_fit(model, df.niter, df.time, [1.,0])
    a,b = fit.param
    fun = x -> a * x + b
    (label = label[1], a = a, b = b, f = fun)
end


@df dfniter scatter(
    :niter,
    :time,
    ylabel = "s",
    xlabel = "niter",
    title = L"\chi = 20",
    legend = :topleft,
    group = :label)

@df dffit plot!(:f, 0, 60,
    label = "fit " .* :label .* ": \t"  .* string.(round.(:a, digits=4)) .* "s/niter",
    lw = 0.2,
    ls = :solid)

savefig("plots/iterscaling.png")

#=
    Scaling with χ
=#

dfχ = by(@where(df, :niter .== 20), [:label,:χ], time = :time => median, sort=true)
@df dfχ plot(
    :χ,
    :time,
    ylabel = "s",
    xlabel = L"\chi",
    title = "niter = 20",
    yticks = [10^-1,10,100],
    ylims = (0.1,100),
    yscale = :log10,
    legend = :topleft,
    group = :label)

savefig("plots/chiscaling.png")
