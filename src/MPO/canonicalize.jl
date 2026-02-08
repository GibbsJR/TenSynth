# Canonicalization routines for MPS and MPO
# Adapted from MPO2Circuit/src/mpo/canonicalize.jl
# No index swap needed in MPS canonicalization (MPS is [left, phys, right]).
# MPO canonicalization delegates to MPS via conversion.

using LinearAlgebra
using TensorOperations

"""
    canonicalise(mps::FiniteMPS{ComplexF64}, site::Int) -> FiniteMPS{ComplexF64}

Put an MPS in mixed canonical form with orthogonality center at `site`.
"""
function canonicalise(mps::FiniteMPS{ComplexF64}, site::Int)::FiniteMPS{ComplexF64}
    n = n_sites(mps)
    (1 <= site <= n) || throw(ArgumentError("Site must be between 1 and $n, got $site"))

    mps_out = FiniteMPS{ComplexF64}(copy.(mps.tensors))

    # Left-canonicalize sites 1 to site-1
    for i in 1:site-1
        T = mps_out.tensors[i]
        T_r = reshape(T, size(T, 1) * size(T, 2), size(T, 3))

        F = robust_svd(T_r)

        mps_out.tensors[i] = reshape(F.U, size(T, 1), size(T, 2), length(F.S))

        SVt = diagm(F.S) * F.Vt
        T_next = mps_out.tensors[i+1]
        @tensor Tnew[i, k, l] := SVt[i, j] * T_next[j, k, l]
        mps_out.tensors[i+1] = Tnew
    end

    # Right-canonicalize sites n down to site+1
    for i in n:-1:site+1
        T = mps_out.tensors[i]
        T_r = reshape(T, size(T, 1), size(T, 2) * size(T, 3))

        F = robust_svd(T_r)

        mps_out.tensors[i] = reshape(F.Vt, length(F.S), size(T, 2), size(T, 3))

        US = F.U * diagm(F.S)
        T_prev = mps_out.tensors[i-1]
        @tensor Tnew[i, j, l] := T_prev[i, j, k] * US[k, l]
        mps_out.tensors[i-1] = Tnew
    end

    # Normalize the center tensor
    mps_out.tensors[site] = mps_out.tensors[site] ./ norm(mps_out.tensors[site])

    return mps_out
end

"""
    canonicalise(mpo::FiniteMPO{ComplexF64}, site::Int) -> FiniteMPO{ComplexF64}

Put an MPO in mixed canonical form via MPS conversion.
"""
function canonicalise(mpo::FiniteMPO{ComplexF64}, site::Int)::FiniteMPO{ComplexF64}
    mps = mpo_to_mps_direct(mpo)
    mps_canon = canonicalise(mps, site)
    return mps_to_mpo_direct(mps_canon)
end

"""
    canonicalise_from_to(mps::FiniteMPS{ComplexF64}, from::Tuple{Int,Int}, to::Tuple{Int,Int}) -> FiniteMPS{ComplexF64}

Efficiently reposition canonical center from `from` to `to`.
"""
function canonicalise_from_to(mps::FiniteMPS{ComplexF64}, from::Tuple{Int,Int}, to::Tuple{Int,Int})::FiniteMPS{ComplexF64}
    if from == to
        return FiniteMPS{ComplexF64}(copy.(mps.tensors))
    end

    if from[1] < to[2]
        site1 = from[1]
        site2 = to[1]
    elseif from[2] > to[1]
        site1 = from[2]
        site2 = to[2]
    else
        throw(ArgumentError("Invalid from/to sites: $from -> $to"))
    end

    mps_out = FiniteMPS{ComplexF64}(copy.(mps.tensors))

    # Left sweep
    for i in site1:site2-1
        T = mps_out.tensors[i]
        T_r = reshape(T, size(T, 1) * size(T, 2), size(T, 3))

        F = robust_svd(T_r)

        mps_out.tensors[i] = reshape(F.U, size(T, 1), size(T, 2), length(F.S))

        SVt = diagm(F.S) * F.Vt
        T_next = mps_out.tensors[i+1]
        @tensor Tnew[i, k, l] := SVt[i, j] * T_next[j, k, l]
        mps_out.tensors[i+1] = Tnew / norm(Tnew)
    end

    # Right sweep
    for i in site1:-1:site2+1
        T = mps_out.tensors[i]
        T_r = reshape(T, size(T, 1), size(T, 2) * size(T, 3))

        F = robust_svd(T_r)

        mps_out.tensors[i] = reshape(F.Vt, length(F.S), size(T, 2), size(T, 3))

        US = F.U * diagm(F.S)
        T_prev = mps_out.tensors[i-1]
        @tensor Tnew[i, j, l] := T_prev[i, j, k] * US[k, l]
        mps_out.tensors[i-1] = Tnew / norm(Tnew)
    end

    return mps_out
end

"""
    canonicalise_from_to(mpo::FiniteMPO{ComplexF64}, from::Tuple{Int,Int}, to::Tuple{Int,Int}) -> FiniteMPO{ComplexF64}
"""
function canonicalise_from_to(mpo::FiniteMPO{ComplexF64}, from::Tuple{Int,Int}, to::Tuple{Int,Int})::FiniteMPO{ComplexF64}
    mps = mpo_to_mps_direct(mpo)
    mps_canon = canonicalise_from_to(mps, from, to)
    return mps_to_mpo_direct(mps_canon)
end
