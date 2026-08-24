module MultiplexParticleSystem
export Model, SimulationState, initSim, step!

# Implements Gillespie-style simulation for certain interacting particle systems on multiplex networks.
# Each vertex has a state on each layer.
# Transition rates must be a sum of internal dynamics and single-edge interactions.

using Graphs
using Random

# DATA STRUCTURES
struct Model{T <: Function}
    N_layers::Int64          # Number of layers of the multiplex network.
    N_states::Array{Int64, 1}# Number of discrete states on each layer.
    transitionRate::T        # from_state::Array{Int}, to_state::Int, layer::Int, [nstate::Array{Int}] ↦ Float64
end

mutable struct SimulationState{T <: Function}
    T::Float64
    const M::Model{T}
    const G::Array{SimpleGraph{Int64}, 1}
    const N_vertices::Int64
    const states::Array{Int64, 2}           # vertex , layer
    const rates::Array{Array{Float64, 2},1} # [layer][state, vertex]
    const total_rates::Array{Float64, 1}    # vertex_layer_id
end

# HELPER FUNCTIONS
function recomputeRates!(sim::SimulationState, vert::Int64, layer::Int64)::Nothing
    vstate = view(sim.states, vert, :)
    for s in 1:sim.M.N_states[layer]
        sim.rates[layer][s, vert] = sim.M.transitionRate(vstate, s, layer)::Float64
    end
    for n in neighbors(sim.G[layer], vert)
        nstate = view(sim.states, n, :)
        for s in 1:sim.M.N_states[layer]
            sim.rates[layer][s, vert] += sim.M.transitionRate(vstate, s, layer, nstate)::Float64
        end
    end
    @views sim.total_rates[vertexLayerToID(vert, layer, sim.N_vertices)] = sum(sim.rates[layer][:, vert])
    return nothing;
end

function pickWeighted(vals, weights, totalweight=sum(weights))::Int64
    # I compared this to sample from StatsBase and it has equal speed, less memory use
    x = rand() * totalweight
    for v in vals
        w = weights[v]
        if w > x
            return v
        end
        x -= w
    end
    @assert(false, "No return in pickWeighted !")
end

function vertexLayerToID(vertex::Int64, layer::Int64, N_vertices::Int64)::Int64
    return (layer-1)*N_vertices + vertex
end

function vertexLayerFromID(id::Int64, N_vertices::Int64)::Tuple{Int64,Int64}
    layer = id ÷ N_vertices + 1
    vert = id % N_vertices
    if vert == 0
        layer -= 1
        vert = N_vertices
    end
    return (vert, layer)
end

# INITIALIZATION
function initSim(M::Model, G::Array{SimpleGraph{Int64}, 1}, state_weights::Array{Array{Float64,1},1})::SimulationState
    # better to catch these here than to have weird behaviour later
    @assert(length(G) == M.N_layers, string(length(G), " layers supplied, expected ", M.N_layers," !"))
    @assert(all(nv.(G) .== nv(G[1])), "Layers do not have the same number of vertices!")
    @assert(length(state_weights) == M.N_layers && all(length.(state_weights) .== M.N_states),"Number of state_weights does not match the number of states!")

    N_vertices = nv(G[1])

    # sample initial states i.i.d. according to state_weights[LAYER][STATE]
    vert_states = zeros(Int64, (N_vertices, M.N_layers))
    for l in 1:M.N_layers
        for v in 1:N_vertices
            vert_states[v,l] = pickWeighted(1:M.N_states[l], state_weights[l])
        end
    end

    # compute all transition rates
    rates::Array{Array{Float64, 2},1} = [zeros(Float64, M.N_states[L], N_vertices) for L=1:M.N_layers]
    total_rates = zeros(Float64, N_vertices * M.N_layers)
    sim = SimulationState(0.0, M, G, N_vertices, vert_states, rates, total_rates)
    for l in 1:M.N_layers
        for v in 1:N_vertices
            recomputeRates!(sim, v, l)
        end
    end
    return sim
end

# STEP
function step!(sim::SimulationState)::Nothing
    total_rate::Float64 = sum(sim.total_rates)

    # time to next event
    sim.T += randexp()/total_rate

    # in an absorbing state? terminate. T is Inf now, but that's fine. I hope.
    if total_rate == 0.0
        return nothing
    end

    # sample next vertex
    vl_id = pickWeighted(1:length(sim.total_rates), sim.total_rates, total_rate)
    vert, layer = vertexLayerFromID(vl_id, sim.N_vertices)

    # sample new state for the vertex
    @views s = pickWeighted(1:sim.M.N_states[layer], sim.rates[layer][:, vert])
    
    # apply changes
    sim.states[vert, layer] = s
    for l in 1:sim.M.N_layers
        recomputeRates!(sim, vert, l)
        for n in neighbors(sim.G[l], vert)
            recomputeRates!(sim, n, l)
        end
    end
    return nothing
end

end #END MODULE