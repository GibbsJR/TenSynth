# MPS truncation functions
# Adapted from MPS2Circuit/src/mps/truncation.jl

using LinearAlgebra
using TensorOperations

"""
    truncate_mps(mps::FiniteMPS, chi_trunc::Int) -> FiniteMPS

Truncate an MPS by directly cutting bond dimensions.
For proper SVD-based truncation, use `SVD_truncate` instead.
"""
function truncate_mps(mps::FiniteMPS{ComplexF64}, chi_trunc::Int)
    t = mps.tensors
    tensors_out = Vector{Array{ComplexF64, 3}}(undef, length(t))

    # First tensor: only truncate right bond
    tensors_out[1] = t[1][:, :, 1:min(chi_trunc, size(t[1], 3))]

    # Middle tensors: truncate both bonds
    for i in 2:length(t)-1
        tensors_out[i] = t[i][1:min(chi_trunc, size(t[i], 1)), :, 1:min(chi_trunc, size(t[i], 3))]
    end

    # Last tensor: only truncate left bond
    tensors_out[end] = t[end][1:min(chi_trunc, size(t[end], 1)), :, :]

    return FiniteMPS{ComplexF64}(tensors_out)
end

"""
    SVD_truncate(mps_in::FiniteMPS, chi_trunc::Int, max_SV::Float64, inds::Tuple{Int,Int}, normalize::Bool=true) -> FiniteMPS

Truncate an MPS using SVD to optimally preserve the state.
Input MPS must be in canonical form with center at `inds[1]`.
"""
function SVD_truncate(mps_in::FiniteMPS{ComplexF64}, chi_trunc::Int, max_SV::Float64, inds::Tuple{Int, Int}, normalize::Bool=true)::FiniteMPS{ComplexF64}

    tensors = copy(mps_in.tensors)
    if abs(norm(tensors[inds[1]]) - 1) > 1e-14
        throw(ArgumentError("SVD_truncate: input MPS not canonical at specified site"))
    end

    for ii in inds[1]:inds[2]-1
        T = tensors[ii]
        T_r = reshape(T, size(T, 1) * size(T, 2), size(T, 3))

        F = robust_svd(T_r)
        SVs = F.S

        SVs_temp = SVs[SVs.>max_SV]

        chi = min(length(SVs_temp), chi_trunc)

        tensors[ii] = reshape(F.U, size(T, 1), size(T, 2), length(F.S))[:, :, 1:chi]

        T1 = (diagm(F.S) * F.Vt)[1:chi, :]
        T2 = tensors[ii+1]
        @tensor Tnew[i, k, l] := T1[i, j] * T2[j, k, l]
        tensors[ii+1] = Tnew
    end

    if normalize
        tensors[inds[2]] = tensors[inds[2]] ./ norm(tensors[inds[2]])
    end

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    pad_zeros(mps_init::FiniteMPS, n_zeros::Int) -> FiniteMPS

Pad an MPS with zeros to increase bond dimension.
"""
function pad_zeros(mps_init::FiniteMPS{ComplexF64}, n_zeros::Int)::FiniteMPS{ComplexF64}
    tensors = copy(mps_init.tensors)

    for i in 1:length(tensors)
        if i < length(tensors)
            tensors[i] = cat(tensors[i], zeros(ComplexF64, size(tensors[i], 1), size(tensors[i], 2), n_zeros); dims=3)::Array{ComplexF64, 3}
        end
        if i > 1
            tensors[i] = cat(tensors[i], zeros(ComplexF64, n_zeros, size(tensors[i], 2), size(tensors[i], 3)); dims=1)::Array{ComplexF64, 3}
        end
    end

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    pad_zeros(mpo_init::FiniteMPO, n_zeros::Int) -> FiniteMPO

Pad an MPO with zeros to increase bond dimension.
"""
function pad_zeros(mpo_init::FiniteMPO{ComplexF64}, n_zeros::Int)::FiniteMPO{ComplexF64}
    tensors = copy(mpo_init.tensors)

    for i in 1:length(tensors)
        if i < length(tensors)
            tensors[i] = cat(tensors[i], zeros(ComplexF64, size(tensors[i], 1), size(tensors[i], 2), size(tensors[i], 3), n_zeros); dims=4)
        end
        if i > 1
            tensors[i] = cat(tensors[i], zeros(ComplexF64, n_zeros, size(tensors[i], 2), size(tensors[i], 3), size(tensors[i], 4)); dims=1)
        end
    end

    return FiniteMPO{ComplexF64}(tensors)
end
