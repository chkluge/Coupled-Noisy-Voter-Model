module CurveApproximation
export approximate_curve

using LinearAlgebra

# the algorithm is simple:
# draw a line segment from the start to the end of the curve
# find the point on the curve that has the greatest distance to the line segment
#       if all points are close to the line segment, congratulations! the line segment is your approximation.
# split the curve at that point and approximate the two parts

# distance to line segment
function sdSegment2d(p::Vector, a::Vector, b::Vector)
    pa = p-a
    ba = b-a
    h = clamp.(dot(pa,ba)/dot(ba,ba), 0.0, 1.0)
    return norm(pa-ba*h)
end

# heavy lifting
function approximate_curve_recursive(curve::AbstractArray, tolerance::Float64, depth::Integer)::AbstractArray
    a = curve[1]
    b = curve[end]
    if depth == 0 || length(curve) ≤ 2
        return [a,b]
    end
    maxdist = 0.0
    pivot = -1
    for (i,p) in enumerate(curve)
        d = sdSegment2d(p,a,b)
        if d > maxdist + tolerance
            maxdist = d
            pivot = i
        end
    end
    if pivot == -1
        return [a,b]
    end
    seg1 = approximate_curve_recursive(curve[1:pivot], tolerance, depth-1)
    seg2 = approximate_curve_recursive(curve[pivot:end], tolerance, depth-1)
    append!(seg1, seg2[2:end])
    return seg1
end

# compatible with Arrays of Tuples, which are convenient for plotting
function approximate_curve(curve::AbstractArray, tolerance::Real, depth::Integer)::Array
    c = [[p[1],p[2]] for p in curve]
    res = approximate_curve_recursive(c, Float64(tolerance), depth)
    return [(p[1],p[2]) for p in res]
end

end # END MODULE