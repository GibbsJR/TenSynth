# SVD-based truncation/compression for MPS and MPO
# Adapted from MPO2Circuit/src/mpo/compression.jl
# No index swap needed — operates via MPS conversion.

using LinearAlgebra
using TensorOperations

"""
    mpo_svd_truncate(mps::FiniteMPS{ComplexF64}, max_chi::Int; normalize::Bool=true) -> FiniteMPS{ComplexF64}

Truncate an MPS to maximum bond dimension `max_chi` using SVD.
"""
function mpo_svd_truncate(mps::FiniteMPS{ComplexF64}, max_chi::Int; normalize::Bool=true)::FiniteMPS{ComplexF64}
    n = n_sites(mps)
    max_chi > 0 || throw(ArgumentError("max_chi must be positive, got $max_chi"))

    mps_out = FiniteMPS{ComplexF64}(copy.(mps.tensors))

    for i in 1:n-1
        T = mps_out.tensors[i]
        T_r = reshape(T, size(T, 1) * size(T, 2), size(T, 3))

        F = robust_svd(T_r)
        chi = min(length(F.S), max_chi)

        U_trunc = F.U[:, 1:chi]
        S_trunc = F.S[1:chi]
        Vt_trunc = F.Vt[1:chi, :]

        mps_out.tensors[i] = reshape(U_trunc, size(T, 1), size(T, 2), chi)

        SVt = diagm(S_trunc) * Vt_trunc
        T_next = mps_out.tensors[i+1]
        @tensor Tnew[i, k, l] := SVt[i, j] * T_next[j, k, l]
        mps_out.tensors[i+1] = Tnew
    end

    if normalize
        mps_out.tensors[end] = mps_out.tensors[end] ./ norm(mps_out.tensors[end])
    end

    return mps_out
end

"""
    mpo_svd_truncate(mpo::FiniteMPO{ComplexF64}, max_chi::Int; normalize::Bool=true) -> FiniteMPO{ComplexF64}

Truncate an MPO to maximum bond dimension via MPS conversion.
"""
function mpo_svd_truncate(mpo::FiniteMPO{ComplexF64}, max_chi::Int; normalize::Bool=true)::FiniteMPO{ComplexF64}
    mpo_canon = canonicalise(mpo, 1)
    mps = mpo_to_mps_direct(mpo_canon)
    mps_trunc = mpo_svd_truncate(mps, max_chi; normalize=normalize)
    return mps_to_mpo_direct(mps_trunc)
end
