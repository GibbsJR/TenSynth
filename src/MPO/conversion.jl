# Conversion between MPO and MPS representations
# Adapted from MPO2Circuit/src/mpo/conversion.jl
# INDEX SWAP: Original [left, phys_in, phys_out, right] → TenSynth [left, phys_out, phys_in, right]
# The combined physical dimension is 4 = 2×2 regardless of ordering.
# When reshaping MPO→MPS, we combine (phys_out, phys_in) → phys_combined=4.
# When reshaping MPS→MPO, we split phys_combined=4 → (phys_out, phys_in).

using LinearAlgebra
using TensorOperations

"""
    mpo_to_mps(mpo::FiniteMPO{ComplexF64}) -> Tuple{FiniteMPS{ComplexF64}, Vector{Float64}}

Convert an MPO to an MPS by combining physical indices.
The MPS is left-canonicalized. Returns (MPS, normalization_factors).
"""
function mpo_to_mps(mpo::FiniteMPO{ComplexF64})::Tuple{FiniteMPS{ComplexF64}, Vector{Float64}}
    n = n_sites(mpo)
    s_norms = zeros(Float64, n)

    mps_tensors = Vector{Array{ComplexF64, 3}}(undef, n)
    for i in 1:n
        tensor = mpo.tensors[i]
        # tensor: [left, phys_out, phys_in, right]
        # combine to: [left, phys_combined=4, right]
        mps_tensors[i] = reshape(tensor, size(tensor, 1), 4, size(tensor, 4))
    end

    # Left-canonicalize and collect norms
    for i in 1:n-1
        T = mps_tensors[i]
        T_r = reshape(T, size(T, 1) * size(T, 2), size(T, 3))

        F = robust_svd(T_r)
        s_norms[i] = sqrt(sum(F.S .^ 2))

        mps_tensors[i] = reshape(F.U, size(T, 1), size(T, 2), length(F.S))

        T1 = diagm(F.S) / s_norms[i]
        T2 = F.Vt
        T3 = mps_tensors[i+1]
        @tensor Tnew[i, l, m] := T1[i, j] * T2[j, k] * T3[k, l, m]
        mps_tensors[i+1] = Tnew
    end

    # Normalize last tensor
    F = robust_svd(reshape(mps_tensors[end], size(mps_tensors[end], 1) * size(mps_tensors[end], 2), size(mps_tensors[end], 3)))
    s_norms[end] = sqrt(sum(F.S .^ 2))
    mps_tensors[end] = mps_tensors[end] ./ s_norms[end]

    return FiniteMPS{ComplexF64}(mps_tensors), s_norms
end

"""
    mps_to_mpo(mps::FiniteMPS{ComplexF64}, norms::Vector{Float64}) -> FiniteMPO{ComplexF64}

Convert an MPS back to an MPO using stored normalization factors.
"""
function mps_to_mpo(mps::FiniteMPS{ComplexF64}, norms::Vector{Float64})::FiniteMPO{ComplexF64}
    n = n_sites(mps)
    length(norms) == n || throw(ArgumentError("Norms vector length must match MPS length"))

    mpo_tensors = Vector{Array{ComplexF64, 4}}(undef, n)
    for i in 1:n
        tensor = mps.tensors[i]
        # [left, phys_combined=4, right] → [left, phys_out=2, phys_in=2, right]
        mpo_tensors[i] = norms[i] .* reshape(tensor, size(tensor, 1), 2, 2, size(tensor, 3))
    end

    return FiniteMPO{ComplexF64}(mpo_tensors)
end

"""
    mpo_to_mps_direct(mpo::FiniteMPO{ComplexF64}) -> FiniteMPS{ComplexF64}

Convert an MPO to an MPS without tracking normalization.
"""
function mpo_to_mps_direct(mpo::FiniteMPO{ComplexF64})::FiniteMPS{ComplexF64}
    mps_tensors = Vector{Array{ComplexF64, 3}}(undef, n_sites(mpo))
    for i in 1:n_sites(mpo)
        tensor = mpo.tensors[i]
        mps_tensors[i] = reshape(tensor, size(tensor, 1), 4, size(tensor, 4))
    end
    return FiniteMPS{ComplexF64}(mps_tensors)
end

"""
    mps_to_mpo_direct(mps::FiniteMPS{ComplexF64}) -> FiniteMPO{ComplexF64}

Convert an MPS to an MPO without applying normalization.
"""
function mps_to_mpo_direct(mps::FiniteMPS{ComplexF64})::FiniteMPO{ComplexF64}
    mpo_tensors = Vector{Array{ComplexF64, 4}}(undef, n_sites(mps))
    for i in 1:n_sites(mps)
        tensor = mps.tensors[i]
        mpo_tensors[i] = reshape(tensor, size(tensor, 1), 2, 2, size(tensor, 3))
    end
    return FiniteMPO{ComplexF64}(mpo_tensors)
end
