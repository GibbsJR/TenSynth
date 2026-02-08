# iMPO — Infinite Matrix Product Operator for long-range gate application
# Adapted from iMPS2Circuit/src/Operators/iMPO.jl

using LinearAlgebra
using TensorOperations

"""
    iMPO{T<:Number}

Infinite Matrix Product Operator with a unit cell structure.

# Fields
- `W::Vector{Array{T,4}}`: MPO tensors with shape [left_bond, physical_out, physical_in, right_bond]
- `unit_cell::Int`: Number of sites in the unit cell
- `physical_dim::Int`: Physical dimension per site
"""
mutable struct iMPO{T<:Number}
    W::Vector{Array{T,4}}
    unit_cell::Int
    physical_dim::Int

    function iMPO{T}(W::Vector{Array{T,4}}, unit_cell::Int, physical_dim::Int) where T
        length(W) == unit_cell || throw(ArgumentError("W length must equal unit_cell"))
        unit_cell > 0 || throw(ArgumentError("unit_cell must be positive"))
        physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
        new{T}(W, unit_cell, physical_dim)
    end
end

"""
    iMPO{T}(unit_cell::Int; physical_dim::Int=2) where T

Create an identity iMPO.
"""
function iMPO{T}(unit_cell::Int; physical_dim::Int=2) where T
    W = Array{T,4}[]
    for _ in 1:unit_cell
        w = zeros(T, 1, physical_dim, physical_dim, 1)
        for p in 1:physical_dim
            w[1, p, p, 1] = one(T)
        end
        push!(W, w)
    end
    iMPO{T}(W, unit_cell, physical_dim)
end

iMPO(unit_cell::Int; physical_dim::Int=2) = iMPO{ComplexF64}(unit_cell; physical_dim=physical_dim)

function bond_dimensions(O::iMPO)
    return [size(w, 4) for w in O.W]
end

function max_bond_dimension(O::iMPO)
    return maximum(bond_dimensions(O))
end

function Base.show(io::IO, O::iMPO{T}) where T
    print(io, "iMPO{$T}(unit_cell=$(O.unit_cell), physical_dim=$(O.physical_dim), bond_dims=$(bond_dimensions(O)))")
end

# ============================================================================
# Gate to iMPO conversion
# ============================================================================

"""
    gate_to_impo(U::Matrix{ComplexF64}, sites::Tuple{Int, Int}, unit_cell::Int) -> iMPO{ComplexF64}

Convert a two-site unitary gate to an iMPO representation.
"""
function gate_to_impo(U::Matrix{ComplexF64}, sites::Tuple{Int, Int}, unit_cell::Int)::iMPO{ComplexF64}
    size(U) == (4, 4) || throw(ArgumentError("Gate must be 4x4 matrix"))

    i, j = sites
    1 <= i <= unit_cell || throw(ArgumentError("Site $i out of range"))
    1 <= j <= unit_cell || throw(ArgumentError("Site $j out of range"))
    i != j || throw(ArgumentError("Sites must be different for two-qubit gate"))

    if i > j
        i, j = j, i
        U = reverse_qubits(U)
    end

    d = 2
    U_tensor = reshape(U, d, d, d, d)

    W = Array{ComplexF64,4}[]

    for site in 1:unit_cell
        if site == i
            if j == i + 1
                w = zeros(ComplexF64, 1, d, d, d)
                for p_out in 1:d, p_in in 1:d, r in 1:d
                    w[1, p_out, p_in, r] = U_tensor[p_out, r, p_in, r]
                end
            else
                w = zeros(ComplexF64, 1, d, d, d)
                for p_out in 1:d, p_in in 1:d, p2 in 1:d
                    w[1, p_out, p_in, p2] = sum(U_tensor[p_out, pp, p_in, p2] for pp in 1:d)
                end
            end
            push!(W, w)

        elseif site == j
            w = zeros(ComplexF64, d, d, d, 1)
            for p2_in in 1:d, p2_out in 1:d, p1 in 1:d
                w[p1, p2_out, p2_in, 1] = U_tensor[p1, p2_out, p1, p2_in]
            end
            push!(W, w)

        elseif i < site < j
            w = zeros(ComplexF64, d, d, d, d)
            for p in 1:d, bond in 1:d
                w[bond, p, p, bond] = one(ComplexF64)
            end
            push!(W, w)

        else
            w = zeros(ComplexF64, 1, d, d, 1)
            for p in 1:d
                w[1, p, p, 1] = one(ComplexF64)
            end
            push!(W, w)
        end
    end

    return iMPO{ComplexF64}(W, unit_cell, d)
end

"""
    apply_impo!(psi::iMPSType{T}, O::iMPO{T}, config::BondConfig) where T

Apply an iMPO to an iMPS, modifying the iMPS in place.
"""
function apply_impo!(psi::iMPSType{T}, O::iMPO{T}, config::BondConfig) where T
    psi.unit_cell == O.unit_cell || throw(DimensionMismatch(
        "iMPS and iMPO must have same unit cell size"
    ))
    psi.physical_dim == O.physical_dim || throw(DimensionMismatch(
        "iMPS and iMPO must have same physical dimension"
    ))

    n = psi.unit_cell

    for site in 1:n
        gamma = psi.gamma[site]
        Wt = O.W[site]

        χ_L, d, χ_R = size(gamma)
        χ_L_O, d_out, d_in, χ_R_O = size(Wt)

        @tensor new_gamma[-1, -2, -3, -4, -5, -6] := Wt[-1, -2, 1, -4] * gamma[-3, 1, -6]

        new_gamma = reshape(new_gamma, χ_L_O * χ_L, d_out, χ_R_O * χ_R)
        psi.gamma[site] = new_gamma
    end

    for site in 1:n
        λ = psi.lambda[site]
        χ_O = size(O.W[site], 4)

        new_λ = kron(Matrix{T}(I, χ_O, χ_O), λ)
        psi.lambda[site] = new_λ
    end

    invalidate_cache!(psi)

    truncate!(psi, config.max_chi; max_trunc_err=config.max_trunc_err)

    return psi
end

"""
    apply_gate_impo!(psi::iMPSType{T}, U::Matrix{ComplexF64}, sites::Tuple{Int, Int}, config::BondConfig) where T

Apply a two-qubit gate using the iMPO representation.
"""
function apply_gate_impo!(
    psi::iMPSType{T},
    U::Matrix{ComplexF64},
    sites::Tuple{Int, Int},
    config::BondConfig
) where T
    O = gate_to_impo(U, sites, psi.unit_cell)
    return apply_impo!(psi, O, config)
end

function Base.copy(O::iMPO{T}) where T
    iMPO{T}(O.W, O.unit_cell, O.physical_dim)
end

function Base.deepcopy(O::iMPO{T}) where T
    iMPO{T}([copy(w) for w in O.W], O.unit_cell, O.physical_dim)
end
