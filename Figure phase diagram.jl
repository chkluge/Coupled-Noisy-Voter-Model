if !isdefined(Main, :MultiplexParticleSystem)
    include("MultiplexParticleSystem.jl")
end
if !isdefined(Main, :CNVM_Simulation)
    include("CNVM_Simulation.jl")
end
if !isdefined(Main, :SimpleSolvers)
    include("SimpleSolvers.jl")
end
if !isdefined(Main, :CNVM_MeanField_2d)
    include("CNVM_MeanField_2d.jl")
end
if !isdefined(Main, :CurveApproximation)
    include("CurveApproximation.jl")
end
using .CurveApproximation
using BifurcationKit

using .MultiplexParticleSystem
using .CNVM_Simulation
using .SimpleSolvers
using .CNVM_MeanField_2d
using Plots
using ProgressMeter
using Graphs
using StatsBase: mean
using JLD2

struct SimPhaseDiagResult
    N::Int64
    mean_deg::Float64
    Tmax::Float64
    ε::Float64

    alphas::AbstractArray
    deltas::AbstractArray
    values_lo::Matrix{Tuple{Float64,Float64}} # b_1, b_2 for all delta,alpha from low initial condition
    values_hi::Matrix{Tuple{Float64,Float64}} # b_1, b_2 for all delta,alpha from high initial condition
end

function sim_phase_diagram(ε, Tmax, mean_deg::Real, N::Integer; a_lims = (0.0, 2.0), d_lims = (-1.0, 1.0))::SimPhaseDiagResult
    ε,Tmax = float.([ε,Tmax])
    β = 1.0

    alphasteps = 32
    deltasteps = 32

    ALPHAS = range(a_lims[1],a_lims[2], alphasteps)
    DELTAS = range(d_lims[1],d_lims[2], deltasteps)

    res_lo = fill((NaN,NaN), (length(ALPHAS),length(DELTAS)))
    res_hi = fill((NaN,NaN), (length(ALPHAS),length(DELTAS)))

    @showprogress for (i,a) in enumerate(ALPHAS)
        #for (j,d) in enumerate(DELTAS)
        Threads.@threads for j in eachindex(DELTAS)
            d = DELTAS[j]
            init_weights_high = [[.01,.99], [.01,.99]]
            init_weights_low = [[.99,.01], [.99,.01]]
            M = Coupled_Noisy_Voter_Model(a,β,ε,d)
            n_edges = Int(round(.5*mean_deg*N))
            G = [erdos_renyi(N, n_edges), erdos_renyi(N, n_edges)]

            _,X1,X2 = simulate_marginals_only!(initSim(M,G, init_weights_high), Tmax)
            res_hi[i,j] = (X1[end]/N, X2[end]/N)

            _,X1,X2 = simulate_marginals_only!(initSim(M,G, init_weights_low), Tmax)
            res_lo[i,j] = (X1[end]/N, X2[end]/N)
        end
    end

    return SimPhaseDiagResult(Int(N),Float64(mean_deg),Float64(Tmax),Float64(ε), ALPHAS,DELTAS, res_lo,res_hi)
end

function num_phase_diagram(ε)
    ε = float(ε)
    β = 1.0


    ALPHAS = 0.0:.02:2.0
    DELTAS = -1.0:.02:1.0

    p = plot(xlims=(-1,1),ylims=(0,2), clims=(0,1), title = "̄b, β=$β, ε=$ε", xlabel="δ", ylabel="α", aspect_ratio=:equal)

    res = zeros(Float64, (length(ALPHAS),length(DELTAS)))

    @showprogress for (i,a) in enumerate(ALPHAS)
        for (j,d) in enumerate(DELTAS)
            stable_equilibria, _ = get_equilibria(a,β,d,ε)

            values = [.5*(e[1]+e[2]) for e in stable_equilibria]
            value2 = [(e[1]-e[2]) for e in stable_equilibria]
            if !isempty(values) && maximum(values)-minimum(values) < .05 && maximum(value2)<.01
                res[i,j] = mean(values)
            else
                res[i,j] = NaN64
            end
        end
    end

    heatmap!(p, DELTAS,ALPHAS,res, aspect_ratio=:equal)
end

function RHS_2d(b, p)
    α,δ,ε = p
    return [ε*(1.0-2*b[1]) + b[1]*(1.0-b[1])*(1.0 -α +δ*b[2]),
            ε*(1.0-2*b[2]) + b[2]*(1.0-b[2])*(1.0 -α +δ*b[1])]
end

function bifCurves(ε::Real; a_lims = (0.0, 2.0), d_lims = (-1.0, 1.0), resolution = 64)
    a_min, a_max = a_lims
    d_min, d_max = d_lims

    fold_low  = Tuple{Float64,Float64}[]
    fold_high = Tuple{Float64,Float64}[]
    pitchfork = Tuple{Float64,Float64}[]

    for d in range(d_min, d_max, 2*resolution)
        prob = BifurcationProblem(RHS_2d, [.95,.95], [a_min, d, ε], 1;
                                record_from_solution=(x,p;k...)->x[1])
        copa = ContinuationPar(p_min = a_min, p_max = a_max, ds=.0025, dsmax=.005)
        br = continuation(prob, PALC(), copa)

        for sp in br.specialpoint
            if sp.type == :bp && sp.status==:converged && 0≤sp.x[1]≤1
                if d > 0.0
                    # cusp & folds
                    if sp.x[1] ≥ 0.5
                        push!(fold_high, (d, sp.param))
                    end
                    if sp.x[1] ≤ 0.5
                        push!(fold_low,  (d, sp.param))
                    end
                else # pitchfork
                    push!(pitchfork, (d, sp.param))
                end
            end
        end
    end
    for a in range(a_min, 1.0, resolution)
        prob = BifurcationProblem(RHS_2d, [.5,.5], [a, d_min, ε], 2;
                                record_from_solution=(x,p;k...)->x[1])
        copa = ContinuationPar(p_min = d_min, p_max = 0.0, ds=.0025, dsmax=.005)
        br = continuation(prob, PALC(), copa)
        
        for sp in br.specialpoint
            if sp.type == :bp && sp.status==:converged && 0≤sp.x[1]≤1
                push!(pitchfork, (sp.param, a))
            end
        end
    end

    # sort the pitchfork points for plotting
    sort!(pitchfork, by=p->p[2])
    i = findmax(p->p[1], pitchfork)[end]
    pitchfork[(i+1):end] .= sort(pitchfork[(i+1):end], by=p->-p[1])

    # subsample
    fold_high = approximate_curve(fold_high, 1e-3, 16)
    fold_low =  approximate_curve(fold_low,  1e-3, 16)
    pitchfork = approximate_curve(pitchfork, 1e-3, 16)

    return fold_high, fold_low, pitchfork
end

function cuspPoint(ε::Real)
    alpha = 1.0+4*ε
    delta = 8.0*ε
    return (delta,alpha)
end

function PhaseDiagram(simRes::SimPhaseDiagResult)
    a_lims = (minimum(simRes.alphas), maximum(simRes.alphas))
    d_lims = (minimum(simRes.deltas), maximum(simRes.deltas))
    ε = simRes.ε
    ε_meanfield = ε / simRes.mean_deg

    # prepare simulation results for plotting
    sim_values = zeros(length(simRes.alphas),length(simRes.deltas))
    for i in eachindex(simRes.values_hi)
        v = simRes.values_hi[i]
        vh=.5*(v[1]+v[2])
        v = simRes.values_lo[i]
        vl=.5*(v[1]+v[2])
        if abs(vh - vl) > .1
            sim_values[i] = NaN64
        else
            sim_values[i] = .5*(vh+vl)
        end
    end

    # mean-field bifurcation branches
    fold_high, fold_low, pitchfork = bifCurves(ε_meanfield, d_lims=d_lims,a_lims=a_lims)

    # plot
    p = plot(title = "ε=$ε, N=$(simRes.N), mean_deg=$(simRes.mean_deg)", aspect_ratio=:equal,
            xlabel="δ",xlims=d_lims, ylabel="α",ylims=a_lims)

    heatmap!(p, simRes.deltas,simRes.alphas,sim_values, clims=(0,1))

    plot!(p, fold_high, label="Fold")
    plot!(p, fold_low,  label="Fold")
    plot!(p, pitchfork, label="Pitchfork")
    scatter!(p, [cuspPoint(ε_meanfield)], label="Cusp")
    return p
end