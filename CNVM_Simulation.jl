if !isdefined(Main, :MultiplexParticleSystem)
    include("MultiplexParticleSystem.jl")
end

module CNVM_Simulation
export Coupled_Noisy_Voter_Model, count_states, simulate!, simulate_marginals_only!

# Implements the Coupled Noisy Voter Model and functions for running simulations.

using Main.MultiplexParticleSystem

function Coupled_Noisy_Voter_Model(α,β,ε,δ)::Model

    α,β,ε,δ :: Float64

    function tr(from_state::AbstractArray{Int}, to_state::Int, layer::Int)::Float64
        if from_state[layer] != to_state
            return ε
        else
            return 0.0
        end
    end
    function tr(from_state::AbstractArray{Int}, to_state::Int, layer::Int, nstate::AbstractArray{Int})::Float64
        if nstate[layer] == to_state && to_state != from_state[layer]
            if to_state == 1
                return α
            else
                return ( (from_state[3-layer]==2) ? (1.0+δ) : 1.0 ) * β
            end
        else
            return 0.0
        end
    end
    return Model(2, [2,2], tr)
end

"""
Count AA,AB,BA,BB vertices in `sim.states`.
"""
function count_states(states::Array{Int64, 2})::Tuple{Int,Int,Int,Int}
    AA,AB,BA,BB = 0,0,0,0
    for v in 1:size(states)[1]
        s1 = states[v,1]
        s2 = states[v,2]
        if s1 == 1
            if s2 == 1
                AA += 1
            else
                AB += 1
            end
        else
            if s2 == 1
                BA += 1
            else
                BB += 1
            end
        end
    end
    return AA,AB,BA,BB
end

"""
Outputs a time series for the prevalence of each state, i.e. AA,AB,BA,BB.
"""
function simulate!(sim::SimulationState, TMAX::Float64)
    T = [sim.T]
    aa::Int64,ab::Int64,ba::Int64,bb::Int64 = count_states(sim.states)
    AA = [aa]
    AB = [ab]
    BA = [ba]
    BB = [bb]
    while sim.T < TMAX
        step!(sim)
        push!(T, sim.T)
        aa,ab,ba,bb = count_states(sim.states)
        push!(AA, aa)
        push!(AB, ab)
        push!(BA, ba)
        push!(BB, bb)
    end
    return T,AA,AB,BA,BB
end

"""
Outputs time series only for the prevalence of B on each layer, i.e. b1,b2.
"""
function simulate_marginals_only!(sim::SimulationState, TMAX::Float64)
    T = [sim.T]
    observable1::Int64 = sum(s -> (s==2) ? 1 : 0, view(sim.states,:,1))
    observable2::Int64 = sum(s -> (s==2) ? 1 : 0, view(sim.states,:,2))
    X1 = [observable1]
    X2 = [observable2]
    while sim.T < TMAX
        step!(sim)
        push!(T, sim.T)
        observable1 = sum(s -> (s==2) ? 1 : 0, view(sim.states,:,1))
        observable2 = sum(s -> (s==2) ? 1 : 0, view(sim.states,:,2))
        push!(X1, observable1)
        push!(X2, observable2)
    end
    return T,X1,X2
end

end #END MODULE