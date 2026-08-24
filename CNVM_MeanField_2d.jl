if !isdefined(Main, :SimpleSolvers)
    include("SimpleSolvers.jl")
end

module CNVM_MeanField_2d
export constructRHS_2d, constructJacobian, get_equilibria, perturbations

using Main.SimpleSolvers
using LinearAlgebra

"""
The 2d mean-field.
"""
function constructRHS_2d(α::Float64,β::Float64,ε::Float64,δ::Float64)::Function
    function f(b)
        return [
            ε*(1-2*b[1]) + b[1]*(1-b[1])*(β-α+β*δ*b[2]),
            ε*(1-2*b[2]) + b[2]*(1-b[2])*(β-α+β*δ*b[1])
        ]
    end
    return f
end

"""
Jacobian of the 2d mean-field.
"""
function constructJacobian(α::Float64,β::Float64,ε::Float64,δ::Float64)::Function
    function J(b)
        return [
            -2*ε + (1-2*b[1])*(β-α+β*δ*b[2])     b[1]*(1-b[1])*(β*δ);
            b[2]*(1-b[2])*(β*δ)     -2*ε + (1-2*b[2])*(β-α+β*δ*b[1])
        ]
    end
    return J
end

"""
Find unique stable, unstable equilibria of the 2d mean-field.
"""
function get_equilibria(α,β,δ,ε)
    F = constructRHS_2d(α,β,ε,δ)
    J = constructJacobian(α,β,ε,δ)

    steps = range(0,1,length=16)
    starting_points = [[x,y] for x in steps, y in steps]

    # find equilibria
    roots = zeros((length(starting_points),2))
    for (i,p) in enumerate(starting_points)
        roots[i,:] = newton(F,J,p, 16)
    end
    for i in 1:length(starting_points)
        if sum(F(roots[i,:]).^2) > 1e-8
            roots[i,:] .= -1.0
        end
    end

    # remove duplicates and points outside [0,1]²
    filtered_equilibria = Vector{Float64}[]
    for i in 1:length(starting_points)
        candidate = roots[i,:]
        if candidate != [-1.0,-1.0] && candidate ∉ filtered_equilibria
            if candidate[1]<0 || candidate[2]<0 || candidate[1]>1 || candidate[2]>1
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
    stable_equilibria = Tuple{Float64,Float64}[]
    unstable_equilibria = Tuple{Float64,Float64}[]
    for eq in filtered_equilibria
        vals = eigen(J(eq)).values
        if vals[1] < 0 && vals[2] < 0
            push!(stable_equilibria, (eq[1],eq[2]))
        else
            push!(unstable_equilibria, (eq[1],eq[2]))
        end
    end

    return stable_equilibria, unstable_equilibria
end

"""
Outputs location and direction of the arrows from our perturbation calculation.
"""
function perturbations(α,δ,ε)
    b = (α-1)/δ
    l1 = 1-α
    l2 = -(1-α+δ)
    x = [0,1,1,0,b]
    y = [0,1,0,1,b]
    u = zeros(5)
    v = zeros(5)
    # 0 0
    u[1]=v[1] = -1/l1
    # 1 1
    u[2]=v[2] = 1/l2
    # 1 0
    u[3]= -1/l1
    v[3]= 1/l2
    # 0 1
    u[4]= 1/l2
    v[4]= -1/l1
    # * *
    u[5]=v[5] = (2*b-1)/(δ*b*(1-b))

    u *= ε
    v *= ε
    return x,y,u,v
end

end #END MODULE