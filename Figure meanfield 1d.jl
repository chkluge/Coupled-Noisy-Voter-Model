if !isdefined(Main, :CurveApproximation)
    include("CurveApproximation.jl")
end
if !isdefined(Main, :SimpleSolvers)
    include("SimpleSolvers.jl")
end
using .CurveApproximation
using .SimpleSolvers: newton

using Plots
using ProgressMeter
using Plots: plot, plot!
using LinearAlgebra

function constructRHS_1d(α::Float64,ε::Float64,δ::Float64)
    function f(b)
        return ε*(1-2*b) + b*(1.0-b)*(1.0-α+δ*b)
    end
    return f
end

function constructJacobian(α::Float64,ε::Float64,δ::Float64)
    function J(b)
        return -2*ε + (1.0-2.0*b)*(1.0-α) + b*δ*(2.0-3.0*b)
    end
    return J
end

function get_equilibria(α,δ,ε)
    F = constructRHS_1d(α,ε,δ)
    J = constructJacobian(α,ε,δ)

    starting_points = range(0,1,length=16)

    # find equilibria
    roots = zeros(length(starting_points))
    for (i,p) in enumerate(starting_points)
        roots[i] = newton(F,J,p, 16)
    end
    for i in 1:length(starting_points)
        if abs(F(roots[i])) > 1e-9
            roots[i] = -1.0
        end
    end

    # remove duplicates and points outside [0,1]²
    filtered_equilibria = Float64[]
    for i in 1:length(starting_points)
        candidate = roots[i]
        if candidate != -1.0 && candidate ∉ filtered_equilibria
            if candidate<0 || candidate>1
                continue
            end
            accept = true
            for eq in filtered_equilibria
                if sum(abs2.(eq-candidate)) < 1e-9
                    accept = false
                    break
                end
            end
            if accept
                push!(filtered_equilibria, candidate)
            end
        end
    end

    # determine stability
    stable_equilibria = Float64[]
    unstable_equilibria = Float64[]
    for eq in filtered_equilibria
        val = J(eq)
        if val < 0
            push!(stable_equilibria, eq)
        else
            push!(unstable_equilibria, eq)
        end
    end

    return stable_equilibria, unstable_equilibria
end

function bifurcation_alpha(δ,ε)
    ALPHAS = 0.0:.025:2.0

    p = plot(legend=:none, xlabel="κ₁", ylabel="b")
    for a in ALPHAS
        seq,ueq = get_equilibria(a,δ,ε)
        scatter!(p, fill(1-a, length(seq)), seq, color = :black)
        scatter!(p, fill(1-a, length(ueq)), ueq, color = :red)
    end
    display(p)
end

function bifurcation_noise(δ,ε;hide_ticks=false)
    ALPHAS = 0.0:.001:2.0
    stable_low = Tuple{Float32,Float32}[]
    stable_high = Tuple{Float32,Float32}[]
    unstable = Tuple{Float32,Float32}[]

    leq1 = Inf
    leq2 = Inf
    leq3 = Inf

    for a in ALPHAS
        seq,ueq = get_equilibria(a,δ,ε)
        for eq in seq
            if eq ≤ 0.5
                push!(stable_low, (1-a,eq))
                leq1 = eq
            end
            if eq ≥ 0.5
                push!(stable_high, (1-a,eq))
                leq2 = eq
            end
        end
        for eq in ueq
            push!(unstable, (1-a,eq))
            leq3 = eq
        end
    end

    if !isempty(unstable)
        append!(unstable, [stable_high[end]])
        prepend!(unstable, [stable_low[1]])
    end

    if length(stable_high) > 16
        stable_high = approximate_curve(stable_high, .001, 16)
    end
    if length(stable_low) > 16
        stable_low = approximate_curve(stable_low, .001, 16)
    end
    if length(unstable) > 16
        unstable = approximate_curve(unstable, .001, 16)
    end

    tick_color = hide_ticks ? :transparent : :match

    p = plot(legend=:none, xlabel=hide_ticks ? " " : "κ₁", ylabel=hide_ticks ? " " : "b", ylims=(-.025,1.025), size=(256,256), tickfontcolor=tick_color)
    plot!(stable_low, color = :black)
    plot!(stable_high, color = :black)
    plot!(unstable, color = :red, ls=:dash)
    display(p)
    return p
end

function bifurcation_nonoise(δ;hide_ticks=false)
    # we know this diagram on paper
    
    unstable0 = [(0.0,0.0),(1.0,0.0)]
    stable0 = [(-1.0,0.0),(0.0,0.0)]

    stable1 = [(-δ,1.0),(1.0,1.0)]
    unstable1 = [(-1.0,1.0),(-δ,1.0)]

    vertical = [(0.0,0.0),(-δ,1.0)]

    tick_color = hide_ticks ? :transparent : :match

    p = plot(legend=:none, xlabel=hide_ticks ? " " : "κ₁", ylabel=hide_ticks ? " " : "b", ylims=(-.025,1.025), size=(256,256), tickfontcolor=tick_color)
    plot!(stable0, color = :black)
    plot!(stable1, color = :black)
    plot!(unstable0, color = :red, ls=:dash)
    plot!(unstable1, color = :red, ls=:dash)
    if δ > 0
        plot!(vertical, color = :red, ls=:dash)
    else
        plot!(vertical, color = :black)
    end

    display(p)
    return p
end

function bifurcation_diagram(δ,ε;hide_ticks=false)
    if ε > 0.0
        return bifurcation_noise(δ,ε;hide_ticks=hide_ticks)
    else
        return bifurcation_nonoise(δ;hide_ticks=hide_ticks)
    end
end

function save_bif_figures()
    savefig( bifurcation_diagram(.5,.000;hide_ticks=true) , "1d-bif-diag-nonoise-1-noticks.pdf" )
    savefig( bifurcation_diagram(-.5,.000;hide_ticks=true) , "1d-bif-diag-nonoise-2-noticks.pdf" )
    savefig( bifurcation_diagram(.5,.005;hide_ticks=true) , "1d-bif-diag-noise-1-noticks.pdf" )
    savefig( bifurcation_diagram(-.5,.005;hide_ticks=true) , "1d-bif-diag-noise-2-noticks.pdf" )
end

function phase_diagram(ε)
    ALPHAS = 0.0:.005:2.0
    DELTAS = -1.0:.005:1.0

    res = zeros((length(ALPHAS),length(DELTAS)))
    for (i,a) in enumerate(ALPHAS)
        for (j,d) in enumerate(DELTAS)
            seq,ueq = get_equilibria(a,d,ε)
            res[i,j] = min(2,length(seq))
        end
    end
    p = heatmap(ALPHAS, DELTAS, res, clims=(0,2), aspect_ratio = :equal, size = (512,512))
    display(p)
end