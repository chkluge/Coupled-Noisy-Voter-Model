if !isdefined(Main, :SimpleSolvers)
    include("SimpleSolvers.jl")
end
if !isdefined(Main, :CNVM_MeanField_2d)
    include("CNVM_MeanField_2d.jl")
end
using .SimpleSolvers
using .CNVM_MeanField_2d

using Plots
using ProgressMeter
using Plots: plot, plot!
using LinearAlgebra

# some utilities for plotting
function curve_from_raw_trajectory(A::Matrix{Float16}, tolerance = .1)
    last_pt = [Inf16,Inf16]
    result = Tuple{Float16,Float16}[]
    for i in 1:(size(A)[1])
        pt = A[i,:]
        if norm(pt - last_pt) ≥ tolerance
            push!(result, (pt[1],pt[2]))
            last_pt = pt
        end
    end
    endpt = (A[end,:][1],A[end,:][2])
    if result[end] != endpt
        push!(result, endpt)
    end
    return result
end

function curve_from_raw_trajectory_arclength(A::Matrix{Float16}, tolerance = .1)
    last_pt = [Inf16,Inf16]
    result = Tuple{Float16,Float16}[]
    accumulated_distance = 0.0
    for i in 1:(size(A)[1])
        pt = A[i,:]
        accumulated_distance += norm(pt - last_pt)
        last_pt = pt
        if accumulated_distance ≥ tolerance
            push!(result, (pt[1],pt[2]))
            accumulated_distance = 0.0
        end
    end
    endpt = (A[end,:][1],A[end,:][2])
    if result[end] != endpt
        push!(result, endpt)
    end
    return result
end

function split_curve(X::Array{Tuple{Float16,Float16}})
    # assumes that we're roughly arc-length sampled
    cut = length(X)÷2 +1
    return X[1:cut],X[cut:end]
end

function dist(x::Tuple{Float16,Float16},y::Tuple{Float16,Float16})::Float16
    return norm( (x[1]-y[1], x[2]-y[2]) )
end

# and all the plots!
function flow(F::Function, h::Function)
    steps = range(0,1,length=16)
    steps2 = range(0,1,length=64)
    gridpoints = [[x,y] for x in steps, y in steps]
    gridpoints2 = [[x,y] for x in steps2, y in steps2]
    f(x::Float64,y::Float64) = .1*F([x,y])
    p = heatmap(steps2,steps2,h.(gridpoints2)')
    quiver!(p,getindex.(gridpoints[:],1), getindex.(gridpoints[:],2), quiver=f, c="turquoise");
    return p
end

function compare(b0, α,β,ε,δ, Tmax)
    α,β,ε,δ,Tmax = float.([α,β,ε,δ,Tmax])

    F = constructRHS_2d(α,β,ε,δ)
    p = plot(xlims=(0,Tmax), ylims=(0,1), legend=:none)

    X = RK4(b0, F, .05, Tmax, 2)
    plot!(p, 0:.1:Tmax, X[:,1], c=:red, alpha = .25)
    plot!(p, 0:.1:Tmax, X[:,2], c=:blue, alpha = .25)

    plot!(p, 0:.1:Tmax, .5*X[:,1]+.5*X[:,2], c=:gray)

    b_avg = .5*(b0[1] + b0[2])*[1,1]
    Y = RK4(b_avg, F, .05, Tmax, 2)
    plot!(p, 0:.1:Tmax, Y[:,1], c=:black, ls=:dash)

    plot!(p, 0:.1:Tmax, abs.(.5*X[:,1]+.5*X[:,2]-Y[:,1]), c=:purple)

    display(p)
end

function test_backward(b0, α,β,ε,δ, Tmax)
    α,β,ε,δ,Tmax = float.([α,β,ε,δ,Tmax])

    F = constructRHS_2d(α,β,ε,δ)
    p = plot(xlims=(0,1), ylims=(0,1), legend=:none, aspect_ratio=:equal)
    plot!(p, [0,1],[0,1], c=:gray, ls=:dash)
    for r = -.02:.005:.02
        X = RK4(b0 + [-r,r], x->-F(x), .05, Tmax, 2)
        plot!(p, X[:,1],X[:,2], c=:red, alpha = .5)
    end
    display(p)
end

function phase_diagram_full(β,ε, Tmax)
    β,ε,Tmax = float.([β,ε,Tmax])

    ALPHAS = 0.0:.025:2.0
    DELTAS = -1.0:.025:1.0

    p1 = plot(xlims=(-1,1),ylims=(0,2), title = "b₁, β=$β, ε=$ε", xlabel="δ", ylabel="α", aspect_ratio=:equal)
    p2 = plot(xlims=(-1,1),ylims=(0,2), title = "b₂, β=$β, ε=$ε", xlabel="δ", ylabel="α", aspect_ratio=:equal)

    res1 = zeros(Float64, (length(ALPHAS),length(DELTAS)))
    res2 = zeros(Float64, (length(ALPHAS),length(DELTAS)))

    @showprogress for (i,a) in enumerate(ALPHAS)
        for (j,d) in enumerate(DELTAS)
            F = constructRHS_2d(a,β,ε,d)

            # high
            x0 = [.999,.9999]
            X = RK4(x0, F, .1, Tmax, 1)
            from_high1 = X[end,1]
            from_high2 = X[end,2]

            # low
            x0 = [0.0001,0.001]
            X = RK4(x0, F, .1, Tmax, 1)
            from_low1 = X[end,1]
            from_low2 = X[end,2]

            if abs(from_high1-from_low1) < .05
                res1[i,j] = .5*from_high1 + .5*from_low1
            else
                res1[i,j] = NaN64
            end
            if abs(from_high2-from_low2) < .05
                res2[i,j] = .5*from_high2 + .5*from_low2
            else
                res2[i,j] = NaN64
            end
        end
    end

    heatmap!(p1, DELTAS,ALPHAS,res1, aspect_ratio=:equal)
    heatmap!(p2, DELTAS,ALPHAS,res2, aspect_ratio=:equal)

    l = @layout [a b]
    plot(p1,p2, layout = l, size = (1024,512))
end

function portrait(α,δ,ε; draw_perturbations::Bool=false, hide_ticks::Bool=false)
    α,δ,ε = float.([α,δ,ε])
    β = 1.0

    stable_equilibria, unstable_equilibria = get_equilibria(α,β,δ,ε)

    tick_color = hide_ticks ? :transparent : :match
    
    p = plot(xlims=(-.1,1.1), ylims=(-.1,1.1), aspect_ratio=:equal, legend = :none, size = (512,512),
            #title = "κ₁=$(round(1-α,digits=2)), κ₂=$(round(-1+α-δ,digits=2)), ε=$ε",
            ticks=[0,.5,1],tickdirection = :in, tickfontcolor=tick_color
            )
    
    # let's try to pick some nice trajectories
    F = constructRHS_2d(α,β,ε,δ)
    J = constructJacobian(α,β,ε,δ)
    interesting_points = Vector{Float64}[]
    interesting_points_bw = Vector{Float64}[]
    for eq in unstable_equilibria
        eq = [eq[1],eq[2]] #convert back to vector
        vals,vecs = eigen(J(eq))
        for i in 1:2
            if vals[i] > 0.0
                pt = eq + 1e-2*vecs[:,i]
                if 0≤pt[1]≤1 && 0≤pt[2]≤1
                    push!(interesting_points, pt)
                end
                pt = eq - 1e-2*vecs[:,i]
                if 0≤pt[1]≤1 && 0≤pt[2]≤1
                    push!(interesting_points, pt)
                end
            else
                pt = eq + 1e-2*vecs[:,i]
                if 0≤pt[1]≤1 && 0≤pt[2]≤1
                    push!(interesting_points_bw, pt)
                end
                pt = eq - 1e-2*vecs[:,i]
                if 0≤pt[1]≤1 && 0≤pt[2]≤1
                    push!(interesting_points_bw, pt)
                end
            end
        end
    end
    interesting_curves = []
    for x0 in interesting_points
        #println(x0)
        X = RK4(x0,F,.1,1000.0)
        X = curve_from_raw_trajectory_arclength(X, .05)
        push!(interesting_curves, X)
    end
    for x0 in interesting_points_bw
        X = RK4(x0, x->-F(x) ,.1,1000.0)
        X = curve_from_raw_trajectory_arclength(X, .05)
        reverse!(X)
        push!(interesting_curves, X)
    end

    # make sure we draw the diagonal
    diagonal_covered = false
    for pt in interesting_points
        if abs(pt[1] - pt[2]) < 1e-3
            diagonal_covered = true
            break
        end
    end
    for pt in interesting_points_bw
        if abs(pt[1] - pt[2]) < 1e-3
            diagonal_covered = true
            break
        end
    end
    if !diagonal_covered
        x0 = [.5,.5]
        # forward
        X1 = RK4(x0,F,.1,1000.0)
        X1 = curve_from_raw_trajectory_arclength(X1, .05)
        # backward
        X2 = RK4(x0, x->-F(x) ,.1,1000.0)
        X2 = curve_from_raw_trajectory_arclength(X2, .05)
        reverse!(X2)
        # glue
        append!(X2,X1[2:end])
        push!(interesting_curves, X2)
    end

    # filter out duplicate trajectories
    interesting_curves_filtered = []
    for X in interesting_curves
        unique = true
        first,mid,last = X[1], X[Int(round(length(X)/2))], X[end]
        for X2 in interesting_curves_filtered
            if max( minimum(x->dist(x, first), X2), minimum(x->dist(x, mid), X2), minimum(x->dist(x, last), X2) ) < .1
                unique = false
            end
        end
        if unique
            push!(interesting_curves_filtered, X)
        end
    end

    for X in interesting_curves_filtered
        x1,x2 = split_curve(X)
        plot!(p, x1, color=:gray, alpha=1, arrow=:closed)
        plot!(p, x2, color=:gray, alpha=1, arrow=:none)
    end

    if draw_perturbations
        x,y,u,v = perturbations(α,δ,.025)
        quiver!(p, x,y, quiver=(u,v), c=:orange)
    end

    scatter!(p, stable_equilibria, color=:black)
    scatter!(p, unstable_equilibria, color=:red)
    #display(p)
    return p
end

function portrait_figure_test(;draw_perturbations::Bool=false, hide_ticks::Bool=false)
    p11 = portrait(1.5,+.75,0.0;draw_perturbations=draw_perturbations,hide_ticks=hide_ticks)
    p21 = portrait(1.5,+.25,0.0;draw_perturbations=draw_perturbations,hide_ticks=hide_ticks)
    p12 = portrait(0.5,-.25,0.0;draw_perturbations=draw_perturbations,hide_ticks=hide_ticks)
    p22 = portrait(0.5,-.75,0.0;draw_perturbations=draw_perturbations,hide_ticks=hide_ticks)

    savefig(p11, "phase_portrait_11.pdf")
    savefig(p21, "phase_portrait_21.pdf")
    savefig(p12, "phase_portrait_12.pdf")
    savefig(p22, "phase_portrait_22.pdf")
    
    L = @layout [a b ; c d]
    p=plot(p11,p12,p21,p22, layout = L, size = (512,512))
    return p
end