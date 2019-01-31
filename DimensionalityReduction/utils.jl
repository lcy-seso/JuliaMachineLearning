# Generate random input data.
using Random
using Distributions
using Statistics
using Plots; gr()

function plotData(data, class_num, sample_per_class, title)
    sp = 1
    ep = sample_per_class
    sliced = view(data[sp : ep, :], :, :)
    scatter(sliced[:, 1], sliced[:, 2], sliced[:, 3])
    
    for _ = 2 : class_num
        sp = ep + 1
        ep += sample_per_class
        
        sliced = view(data[sp : ep, :], :, :)     
        s = scatter!(sliced[:, 1], sliced[:, 2], sliced[:, 3])
    end
    title!(title)
end

function genData(sample_per_class, μs, σs)
    xs::Array{Array{Float64},1} = []
    for (μ, σ) in zip(μs, σs)
        push!(xs, rand!(Normal(μ, σ), zeros(sample_per_class, 3))) 
    end
    return vcat(xs...)
end

function centered(data)
    m = mean(data, dims=1)
    centered = similar(data)
    
    start = 1
    for pos in sample_per_class : sample_per_class : sample_per_class * class_num
        centered[start:pos, :] = data[start:pos, :] .- m
        start = pos + 1
    end
    return (centered, m)
end