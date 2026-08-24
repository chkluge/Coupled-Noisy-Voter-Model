module SimpleSolvers
export RK4, newton

using LinearAlgebra

function RK4(x0, F::T, dt::Real, t_max::Real, out_every::Int=1) where T<:Function
    n_steps = Int(ceil(t_max/dt))
    X = zeros(Float16,(n_steps÷out_every +1,length(x0)))
    x = x0
    X[1,:] = x
    for i=1:n_steps
        k1 = F(x)
        k2 = F(x + .5*dt*k1)
        k3 = F(x + .5*dt*k2)
        k4 = F(x + dt*k3)
        x += dt*(k1/6 + k2/3 + k3/3 + k4/6)
        if(i%out_every==0)
            X[(i÷out_every+1),:] = Float16.(x)
        end
    end
    return X
end

function newton(F,J,x0::AbstractVector,iters::Integer)
    x = copy(x0)
    for _ in 1:iters
        if det(J(x)) == 0.0
            break
        end
        x̂ = x + J(x)\(-F(x))
        x = x̂
    end
    return x
end

function newton(F,J,x0::Real,iters::Integer)
    x = copy(x0)
    for _ in 1:iters
        if J(x) == 0.0
            break
        end
        x̂ = x + (-F(x))/J(x)
        x = x̂
    end
    return x
end

end #END MODULE