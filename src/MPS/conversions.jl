# MPS/MPO conversion functions
# Adapted from MPS2Circuit/src/mps/conversions.jl
# Dropped: NPZ I/O (NPZData2MPS), ITensor conversion (moved to separate file)
# MPO convention: [left_bond, phys_out, phys_in, right_bond]

using LinearAlgebra
using TensorOperations

"""
    mps2mpo(mps::FiniteMPS, s_norms::Vector{Float64}) -> FiniteMPO

Convert an MPS with physical dimension 4 to an MPO with physical dimension 2×2.
The s_norms scaling factors are applied to each tensor.
Convention: [left_bond, phys_out, phys_in, right_bond]
"""
function mps2mpo(mps::FiniteMPS{ComplexF64}, s_norms::Vector{Float64})
    tensors = Vector{Array{ComplexF64, 4}}(undef, length(mps.tensors))

    for i in 1:length(mps.tensors)
        T = reshape(mps.tensors[i], size(mps.tensors[i], 1), 2, 2, size(mps.tensors[i], 3))
        tensors[i] = s_norms[i] .* T
    end

    return FiniteMPO{ComplexF64}(tensors)
end

"""
    mps2mpo(mps::FiniteMPS) -> FiniteMPO

Single-argument version that uses unit scaling factors.
"""
function mps2mpo(mps::FiniteMPS{ComplexF64})
    s_norms = ones(Float64, length(mps.tensors))
    return mps2mpo(mps, s_norms)
end

"""
    mps2mpo_JustR(mps::FiniteMPS) -> FiniteMPO

Convert an MPS with physical dimension 4 to an MPO with physical dimension 2×2.
Simple reshaping without normalization factors.
"""
function mps2mpo_JustR(mps::FiniteMPS{ComplexF64})::FiniteMPO{ComplexF64}
    tensors = Vector{Array{ComplexF64, 4}}(undef, length(mps.tensors))

    for i in 1:length(mps.tensors)
        tensors[i] = reshape(mps.tensors[i], size(mps.tensors[i], 1), 2, 2, size(mps.tensors[i], 3))
    end

    return FiniteMPO{ComplexF64}(tensors)
end

"""
    mpo2mps_JustR(mpo::FiniteMPO) -> FiniteMPS

Convert an MPO with physical dimension 2×2 to an MPS with physical dimension 4.
Simple reshaping without computing normalization factors.
"""
function mpo2mps_JustR(mpo::FiniteMPO{ComplexF64})::FiniteMPS{ComplexF64}
    tensors = Vector{Array{ComplexF64, 3}}(undef, length(mpo.tensors))

    for (i, m) in enumerate(mpo.tensors)
        tensors[i] = reshape(m, size(m, 1), 4, size(m, 4))
    end

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    mpo2mps(mpo::FiniteMPO) -> Tuple{FiniteMPS, Vector{Float64}}

Convert an MPO to an MPS with extracted normalization factors.
"""
function mpo2mps(mpo::FiniteMPO{ComplexF64})::Tuple{FiniteMPS{ComplexF64}, Vector{Float64}}
    N = length(mpo.tensors)
    s_norms = zeros(Float64, N)

    tensors = Vector{Array{ComplexF64, 3}}(undef, N)
    for (i, m) in enumerate(mpo.tensors)
        tensors[i] = reshape(m, size(m, 1), 4, size(m, 4))
    end

    for i in 1:N-1
        F = robust_svd(reshape(tensors[i], size(tensors[i], 1) * size(tensors[i], 2), size(tensors[i], 3)))
        s_norms[i] = sqrt(sum(F.S .^ 2))
        tensors[i] = reshape(F.U, size(tensors[i], 1), size(tensors[i], 2), size(F.U, 2))
        T1 = diagm(F.S) / s_norms[i]
        T2 = F.Vt
        T3 = tensors[i+1]
        @tensoropt Tnew[i, l, m] := T1[i, j] * T2[j, k] * T3[k, l, m]
        tensors[i+1] = Tnew
    end

    F = robust_svd(reshape(tensors[end], size(tensors[end], 1) * size(tensors[end], 2), size(tensors[end], 3)))
    s_norms[end] = sqrt(sum(F.S .^ 2))
    tensors[end] = tensors[end] ./ s_norms[end]

    return FiniteMPS{ComplexF64}(tensors), s_norms
end

# ============================================================================
# ITensor conversion functions
# ============================================================================

"""
    from_itensors(psi) -> FiniteMPS{ComplexF64}

Convert an ITensors MPS to a TenSynth FiniteMPS.
Input MPS will be orthogonalized to the last site.
"""
function from_itensors(psi)::FiniteMPS{ComplexF64}
    N = length(psi)
    ITensorMPS.orthogonalize!(psi, N)

    tensors = Vector{Array{ComplexF64, 3}}(undef, N)
    for i in 1:N
        T = Array(psi[i].tensor)  # Convert NDTensors.DenseTensor to plain Array
        if i == 1
            tensors[i] = reshape(T, 1, 2, :)
        elseif i == N
            tensors[i] = permutedims(reshape(T, 2, :, 1), (2, 1, 3))
        else
            tensors[i] = permutedims(T, (2, 1, 3))
        end
    end

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    to_itensors(mps::FiniteMPS, sites) -> ITensors.MPS

Convert a TenSynth FiniteMPS to an ITensors MPS.
"""
function to_itensors(mps::FiniteMPS{ComplexF64}, sites)
    N = length(mps.tensors)
    psi = ITensorMPS.MPS(sites)

    BInds = [ITensors.Index(size(mps.tensors[i], 3), "Link,l=$(i)") for i in 1:N-1]

    for i in 1:N
        M = mps.tensors[i]
        if i == 1
            M_2d = reshape(M, 2, size(M, 3))
            psi[i] = ITensors.ITensor(M_2d, sites[i], BInds[i])
        elseif i == N
            M_2d = permutedims(reshape(M, size(M, 1), size(M, 2)), (2, 1))
            psi[i] = ITensors.ITensor(M_2d, sites[i], BInds[i-1])
        else
            M_perm = permutedims(M, (2, 1, 3))
            psi[i] = ITensors.ITensor(M_perm, sites[i], BInds[i-1], BInds[i])
        end
    end

    return psi
end

"""
    transposeMPO(mpo::FiniteMPO) -> FiniteMPO

Transpose MPO by swapping physical indices (phys_out ↔ phys_in).
Convention: [left, phys_out, phys_in, right] → [left, phys_in, phys_out, right]
"""
function transposeMPO(mpo::FiniteMPO{ComplexF64})::FiniteMPO{ComplexF64}
    tensors = [permutedims(t, (1, 3, 2, 4)) for t in mpo.tensors]
    return FiniteMPO{ComplexF64}(tensors)
end
