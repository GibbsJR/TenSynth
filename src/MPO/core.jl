# Core MPO/MPS operations
# Adapted from MPO2Circuit/src/mpo/core.jl
# Index convention: [left_bond, phys_out, phys_in, right_bond]

using LinearAlgebra

"""
    identity_mpo(n::Int) -> FiniteMPO{ComplexF64}

Create an identity MPO for `n` sites.
Each tensor has bond dimension 1 and represents the 2×2 identity matrix.
Convention: [left_bond, phys_out, phys_in, right_bond]
"""
function identity_mpo(n::Int)::FiniteMPO{ComplexF64}
    n > 0 || throw(ArgumentError("Number of sites must be positive, got $n"))
    # Identity: phys_out[i] == phys_in[j] → δ_{ij}
    # reshape I(2) as [1, 2, 2, 1] with convention [left, phys_out, phys_in, right]
    tensors = [reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1) for _ in 1:n]
    return FiniteMPO{ComplexF64}(tensors)
end

"""
    n_sites(mpo::FiniteMPO) -> Int
"""
n_sites(mpo::FiniteMPO)::Int = length(mpo.tensors)

"""
    n_sites(mps::FiniteMPS) -> Int
"""
n_sites(mps::FiniteMPS)::Int = length(mps.tensors)

"""
    bond_dimensions(mpo::FiniteMPO) -> Vector{Int}

Return the bond dimensions between sites.
"""
function bond_dimensions(mpo::FiniteMPO)::Vector{Int}
    n = n_sites(mpo)
    n > 1 || return Int[]
    return [size(mpo.tensors[i], 4) for i in 1:n-1]
end

"""
    bond_dimensions(mps::FiniteMPS) -> Vector{Int}
"""
function bond_dimensions(mps::FiniteMPS)::Vector{Int}
    n = n_sites(mps)
    n > 1 || return Int[]
    return [size(mps.tensors[i], 3) for i in 1:n-1]
end

"""
    trace_mpo(mpo::FiniteMPO) -> ComplexF64

Compute the trace of an MPO.
Convention: [left, phys_out, phys_in, right] → trace over phys_out == phys_in
"""
function trace_mpo(mpo::FiniteMPO)::ComplexF64
    T = ones(ComplexF64, 1)

    for tensor in mpo.tensors
        # tensor: [left, phys_out, phys_in, right]
        # Trace: sum over phys_out == phys_in (positions 2 and 3)
        @views Tnew = zeros(ComplexF64, size(tensor, 4))
        for r in axes(tensor, 4)
            for l in axes(tensor, 1)
                for p in axes(tensor, 2)  # phys_out == phys_in for trace
                    Tnew[r] += T[l] * tensor[l, p, p, r]
                end
            end
        end
        T = Tnew
    end

    return T[1]
end

Base.copy(mpo::FiniteMPO{T}) where T = FiniteMPO{T}(copy.(mpo.tensors))
Base.copy(mps::FiniteMPS{T}) where T = FiniteMPS{T}(copy.(mps.tensors))
