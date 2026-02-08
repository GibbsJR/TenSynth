# Robust SVD with fallback chain, polar decomposition, and random unitaries
# Adapted from MPS2Circuit/src/svd.jl

"""
    robust_svd(M::Matrix{ComplexF64})

Compute SVD with robust fallback handling.

Attempts multiple SVD algorithms in sequence:
1. Standard Julia SVD
2. QR iteration algorithm
3. QR iteration with small noise perturbations (10^-16 to 10^-12)
"""
function robust_svd(M::Matrix{ComplexF64})
    if any(isnan, M)
        throw(ErrorException("SVD failed — input contains NaN values"))
    end

    # Try standard SVD
    try
        return svd(M)
    catch
    end

    # Try QR iteration
    try
        return svd(M, alg=LinearAlgebra.QRIteration())
    catch
    end

    # Try with small noise perturbations
    for i in -16.0:-12.0
        try
            noise = 10^i * randn(ComplexF64, size(M, 1), size(M, 2))
            return svd(M + noise, alg=LinearAlgebra.QRIteration())
        catch
        end
    end

    throw(ErrorException("SVD failed — all fallback methods exhausted"))
end

"""
    polar_unitary(M::Matrix{ComplexF64})

Compute the unitary factor U*V† from the polar decomposition of M.
For square matrices, the result is unitary.
"""
function polar_unitary(M::Matrix{ComplexF64})::Matrix{ComplexF64}
    F = robust_svd(M)
    return F.U * F.Vt
end

"""
    polar_unitary(M::Matrix{ComplexF64}, M_prev::Matrix{ComplexF64}, beta::Float64)

Compute regularized polar decomposition with geodesic interpolation.
"""
function polar_unitary(M::Matrix{ComplexF64}, M_prev::Matrix{ComplexF64}, beta::Float64)::Matrix{ComplexF64}
    F = robust_svd(M_prev * (inv(M_prev) * M)^(1 - beta))
    return F.U * F.Vt
end

"""
    randU(beta::Float64, dim::Int=2)

Generate a random unitary matrix near the identity.
Matrix size is 2^dim × 2^dim. Smaller beta → closer to identity.
"""
function randU(beta::Float64, dim::Int=2)::Matrix{ComplexF64}
    n = 2^dim
    M = rand(ComplexF64, n, n)
    return exp(im * beta * (M + M'))
end

"""
    randU_sym(beta::Float64)

Generate a random symmetric two-qubit unitary near the identity.
"""
function randU_sym(beta::Float64)::Matrix{ComplexF64}
    M = rand(ComplexF64, 2, 2)
    M2 = Matrix{ComplexF64}(I, 4, 4)
    M2[2:3, 2:3] = M
    M2[1, 1] = rand()
    M2[4, 4] = rand()
    return exp(im * beta * (M2 + M2'))
end
