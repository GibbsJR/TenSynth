# iMPS type extensions — convenience constructors and utility methods
# The iMPSType{T} type itself is defined in Core/types.jl
# This file adds constructors and basic operations.

using LinearAlgebra

"""
    iMPSType{T}(unit_cell::Int; physical_dim::Int=2)

Construct an iMPS initialized to the |0⟩^⊗∞ product state.
"""
function iMPSType{T}(unit_cell::Int; physical_dim::Int=2) where T
    gamma = [reshape(T[1; zeros(T, physical_dim-1)], 1, physical_dim, 1) for _ in 1:unit_cell]
    lambda = [reshape(T[1], 1, 1) for _ in 1:unit_cell]
    iMPSType{T}(gamma, lambda, unit_cell, physical_dim, true, nothing)
end

iMPSType(unit_cell::Int; physical_dim::Int=2) = iMPSType{ComplexF64}(unit_cell; physical_dim=physical_dim)

"""
    bond_dimensions(psi::iMPSType) -> Vector{Int}
"""
function bond_dimensions(psi::iMPSType)
    return [size(λ, 1) for λ in psi.lambda]
end

"""
    max_bond_dimension(psi::iMPSType) -> Int
"""
function max_bond_dimension(psi::iMPSType)
    return maximum(bond_dimensions(psi))
end

function Base.copy(psi::iMPSType{T}) where T
    iMPSType{T}(psi.gamma, psi.lambda, psi.unit_cell, psi.physical_dim,
             psi.normalized, psi._gamma_absorbed)
end

function Base.deepcopy(psi::iMPSType{T}) where T
    iMPSType{T}(
        [copy(g) for g in psi.gamma],
        [copy(l) for l in psi.lambda],
        psi.unit_cell, psi.physical_dim, psi.normalized,
        isnothing(psi._gamma_absorbed) ? nothing : [copy(g) for g in psi._gamma_absorbed]
    )
end

Base.length(psi::iMPSType) = psi.unit_cell
Base.eltype(::iMPSType{T}) where T = T

"""
    invalidate_cache!(psi::iMPSType)
"""
function invalidate_cache!(psi::iMPSType)
    psi._gamma_absorbed = nothing
    psi.normalized = false
    return psi
end

"""
    get_site_tensor(psi::iMPSType{T}, i::Int) -> Array{T,3}

Get full site tensor A[i] = lambda[i-1] * gamma[i] * lambda[i].
"""
function get_site_tensor(psi::iMPSType{T}, i::Int) where T
    n = psi.unit_cell
    i_prev = mod1(i - 1, n)
    i_curr = mod1(i, n)

    gamma_i = psi.gamma[i_curr]
    lambda_left = psi.lambda[i_prev]
    lambda_right = psi.lambda[i_curr]

    χ_L, d, χ_R = size(gamma_i)

    result = similar(gamma_i)
    @inbounds for r in 1:χ_R, p in 1:d, l in 1:χ_L
        val = zero(T)
        for m in 1:size(lambda_left, 2)
            val += lambda_left[l, m] * gamma_i[m, p, r]
        end
        result[l, p, r] = val
    end

    result2 = similar(result)
    @inbounds for r in 1:χ_R, p in 1:d, l in 1:χ_L
        val = zero(T)
        for m in 1:size(lambda_right, 1)
            val += result[l, p, m] * lambda_right[m, r]
        end
        result2[l, p, r] = val
    end

    return result2
end

function Base.show(io::IO, psi::iMPSType{T}) where T
    χs = bond_dimensions(psi)
    print(io, "iMPSType{$T}(unit_cell=$(psi.unit_cell), physical_dim=$(psi.physical_dim), bond_dims=$χs)")
end
