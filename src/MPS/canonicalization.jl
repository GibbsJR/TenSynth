# MPS canonicalization functions
# Adapted from MPS2Circuit/src/mps/canonicalization.jl

using LinearAlgebra
using TensorOperations

"""
    canonicalise(mps_old::FiniteMPS, site::Int) -> FiniteMPS

Put an MPS into mixed-canonical form with the orthogonality center at the specified site.
"""
function canonicalise(mps_old::FiniteMPS{ComplexF64}, site::Int)
    tensors = copy(mps_old.tensors)

    # Sweep left-to-right up to site-1
    for ii in 1:site-1
        T = tensors[ii]
        T_r = reshape(T, size(T, 1) * size(T, 2), size(T, 3))

        F = robust_svd(T_r)

        tensors[ii] = reshape(F.U, size(T, 1), size(T, 2), length(F.S))

        T1 = diagm(F.S) * F.Vt
        T2 = tensors[ii+1]
        @tensor Tnew[i, k, l] := T1[i, j] * T2[j, k, l]
        tensors[ii+1] = copy(Tnew)
    end

    # Sweep right-to-left down to site+1
    for ii in length(tensors):-1:site+1
        T = tensors[ii]
        T_r = reshape(T, size(T, 1), size(T, 2) * size(T, 3))

        F = robust_svd(T_r)

        tensors[ii] = reshape(F.Vt, length(F.S), size(T, 2), size(T, 3))

        T2 = tensors[ii-1]
        T1 = F.U * diagm(F.S)
        @tensor Tnew[i, j, l] := T2[i, j, k] * T1[k, l]

        tensors[ii-1] = copy(Tnew)
    end

    # Normalize the center tensor
    tensors[site] = tensors[site] ./ norm(tensors[site])

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    canonicalise(mpo_in::FiniteMPO, site::Int) -> FiniteMPO

Put an MPO into mixed-canonical form with the orthogonality center at the specified site.
Converts to MPS form, canonicalizes, and converts back.
"""
function canonicalise(mpo_in::FiniteMPO{ComplexF64}, site::Int)
    mps = mpo2mps_JustR(mpo_in)
    mps_canon = canonicalise(mps, site)
    mpo = mps2mpo_JustR(mps_canon)
    return mpo
end

"""
    canonicalise_FromTo(mps_old::FiniteMPS, sites_From::Tuple{Int,Int}, sites_To::Tuple{Int,Int}) -> FiniteMPS

Move the orthogonality center from one pair of sites to another.
More efficient than full canonicalization when only a partial sweep is needed.
"""
function canonicalise_FromTo(mps_old::FiniteMPS{ComplexF64}, sites_From::Tuple{Int, Int}, sites_To::Tuple{Int, Int})

    if sites_From[1] < sites_To[2]
        site1 = sites_From[1]
        site2 = sites_To[1]
    elseif sites_From[2] > sites_To[1]
        site1 = sites_From[2]
        site2 = sites_To[2]
    else
        if sites_From == sites_To
            return mps_old
        end
        throw(ArgumentError("canonicalise_FromTo: invalid site combination $(sites_From) → $(sites_To)"))
    end

    tensors = copy(mps_old.tensors)

    # Sweep forward (left to right)
    for ii in site1:site2-1
        T = tensors[ii]
        T_r = reshape(T, size(T, 1) * size(T, 2), size(T, 3))

        F = robust_svd(T_r)

        tensors[ii] = reshape(F.U, size(T, 1), size(T, 2), length(F.S))

        T1 = diagm(F.S) * F.Vt
        T2 = tensors[ii+1]
        @tensor Tnew[i, k, l] := T1[i, j] * T2[j, k, l]
        tensors[ii+1] = Tnew
    end

    # Sweep backward (right to left)
    for ii in site1:-1:site2+1
        T = tensors[ii]
        T_r = reshape(T, size(T, 1), size(T, 2) * size(T, 3))

        F = robust_svd(T_r)

        tensors[ii] = reshape(F.Vt, length(F.S), size(T, 2), size(T, 3))

        T1 = tensors[ii-1]
        T2 = F.U * diagm(F.S)
        @tensor Tnew[i, j, l] := T1[i, j, k] * T2[k, l]

        tensors[ii-1] = copy(Tnew)
    end

    tensors[site2] = tensors[site2] ./ norm(tensors[site2])

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    canonicalise_FromTo(mpo_old::FiniteMPO, sites_From::Tuple{Int,Int}, sites_To::Tuple{Int,Int}) -> FiniteMPO

Move the orthogonality center of an MPO from one pair of sites to another.
"""
function canonicalise_FromTo(mpo_old::FiniteMPO{ComplexF64}, sites_From::Tuple{Int, Int}, sites_To::Tuple{Int, Int})
    mps = mpo2mps_JustR(mpo_old)
    mps = canonicalise_FromTo(mps, sites_From, sites_To)
    return mps2mpo_JustR(mps)
end
