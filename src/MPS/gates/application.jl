# Gate application functions
# Adapted from MPS2Circuit/src/gates/application.jl
# Key changes: SVD() → robust_svd(), type wrapping with FiniteMPS/FiniteMPO

using LinearAlgebra
using TensorOperations

"""
    apply_2q_gate(gate, TL_in, TR_in, max_chi, max_trunc_err) -> (TL_out, TR_out, trunc_err)

Apply a two-qubit gate to adjacent MPS tensors (rank-3).
Operates on raw tensors, not wrapped FiniteMPS objects.
"""
function apply_2q_gate(gate::Matrix{ComplexF64}, TL_in::Array{ComplexF64, 3}, TR_in::Array{ComplexF64, 3},
                       max_chi::Int, max_trunc_err::Float64)
    gate_r = reshape(gate, 2, 2, 2, 2)

    @tensoropt T3[m, i, j, o] := gate_r[i, j, k, l] * TL_in[m, k, n] * TR_in[n, l, o]
    T3_r = reshape(T3, size(T3, 1) * size(T3, 2), size(T3, 3) * size(T3, 4))

    F = robust_svd(T3_r)

    SVs::Vector{Float64} = F.S / norm(F.S)
    SVs_temp = SVs[SVs .> max_trunc_err]
    r_delta = length(SVs_temp)
    chi = min(max_chi, r_delta)

    TL_out = reshape(F.U * diagm(sqrt.(SVs)), size(TL_in, 1), size(TL_in, 2), length(SVs))[:, :, 1:chi]
    TR_out = reshape(diagm(sqrt.(SVs)) * F.Vt, length(SVs), size(TR_in, 2), size(TR_in, 3))[1:chi, :, :]

    return TL_out, TR_out, sqrt(sum(SVs[chi+1:end] .^ 2))
end

"""
    apply_2q_gate(gate, TL_in, TR_in, max_chi, max_trunc_err) -> (TL_out, TR_out, trunc_err)

Apply a two-qubit gate to adjacent MPO tensors (rank-4).
Convention: [left_bond, phys_out, phys_in, right_bond]
"""
function apply_2q_gate(gate::Matrix{ComplexF64}, TL_in::Array{ComplexF64, 4}, TR_in::Array{ComplexF64, 4},
                       max_chi::Int, max_trunc_err::Float64)
    gate_r = reshape(gate, 2, 2, 2, 2)

    @tensoropt T3[m, i, a, j, b, o] := gate_r[i, j, k, l] * TL_in[m, k, a, n] * TR_in[n, l, b, o]
    T3_r = reshape(T3, size(T3, 1) * size(T3, 2) * size(T3, 3), size(T3, 4) * size(T3, 5) * size(T3, 6))

    F = robust_svd(T3_r)

    SVs::Vector{Float64} = F.S / norm(F.S)
    SVs_temp = SVs[SVs .> max_trunc_err]
    r_delta = length(SVs_temp)
    chi = min(max_chi, r_delta)

    TL_out = reshape(F.U * diagm(sqrt.(SVs)), size(T3, 1), size(TL_in, 2), size(TL_in, 3), length(SVs))[:, :, :, 1:chi]
    TR_out = reshape(diagm(sqrt.(SVs)) * F.Vt, length(SVs), size(TR_in, 2), size(TR_in, 3), size(TR_in, 4))[1:chi, :, :, :]

    return TL_out, TR_out, sqrt(sum(SVs[chi+1:end] .^ 2))
end

"""
    apply1Q(mps::FiniteMPS, ind::Int, gate::Matrix{ComplexF64}) -> FiniteMPS

Apply a single-qubit gate to an MPS at a specified site.
"""
function apply1Q(mps::FiniteMPS{ComplexF64}, ind::Int, gate::Matrix{ComplexF64})
    if size(gate, 1) != 2
        throw(ArgumentError("apply1Q: gate must be a 2x2 matrix"))
    end

    tensors = copy(mps.tensors)
    @tensor T[-1, -2, -3] := (tensors[ind])[-1, 1, -3] * gate[-2, 1]
    tensors[ind] = T

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    apply1Q!(mps::FiniteMPS, ind::Int, gate::Matrix{ComplexF64}) -> FiniteMPS

Apply a single-qubit gate in-place.
"""
function apply1Q!(mps::FiniteMPS{ComplexF64}, ind::Int, gate::Matrix{ComplexF64})
    if size(gate, 1) != 2
        throw(ArgumentError("apply1Q!: gate must be a 2x2 matrix"))
    end

    @tensor T[-1, -2, -3] := (mps.tensors[ind])[-1, 1, -3] * gate[-2, 1]
    mps.tensors[ind] = T

    return mps
end

# ============================================================================
# Direct MPO construction for long-range gates (no SWAP decomposition)
# ============================================================================

"""
    gate_to_mpo_direct(gate, inds) -> Vector{Array{ComplexF64,4}}

Convert a two-qubit gate to an MPO without SWAP decomposition.
MPO bond dimension is at most 4, independent of gate range.
Returns raw tensor vector (not wrapped FiniteMPO).
"""
function gate_to_mpo_direct(gate::Matrix{ComplexF64}, inds::Tuple{Int,Int})::Vector{Array{ComplexF64,4}}
    i, j = inds
    @assert j > i "Second index must be greater than first index"

    L = j - i + 1
    gate_r = reshape(gate, 2, 2, 2, 2)
    mpo = Vector{Array{ComplexF64,4}}(undef, L)

    # Permute to [out_q1, in_q1, out_q2, in_q2] = indices (2, 4, 1, 3) of gate_r
    gate_perm = permutedims(gate_r, (2, 4, 1, 3))
    gate_matrix = reshape(gate_perm, 4, 4)
    F = svd(gate_matrix)

    tol = 1e-14
    chi = max(sum(F.S .> tol * F.S[1]), 1)

    # Site 1: shape (1, 2, 2, chi)
    U_scaled = F.U[:, 1:chi] * Diagonal(sqrt.(F.S[1:chi]))
    mpo[1] = reshape(U_scaled, 1, 2, 2, chi)

    if L == 2
        Vt_scaled = Diagonal(sqrt.(F.S[1:chi])) * F.Vt[1:chi, :]
        mpo[2] = reshape(Vt_scaled, chi, 2, 2, 1)
    else
        # Middle sites: identity on physical, pass bond
        for k in 2:L-1
            mpo[k] = zeros(ComplexF64, chi, 2, 2, chi)
            for alpha in 1:chi, p in 1:2
                mpo[k][alpha, p, p, alpha] = 1.0
            end
        end

        # Site L: shape (chi, 2, 2, 1)
        Vt_scaled = Diagonal(sqrt.(F.S[1:chi])) * F.Vt[1:chi, :]
        mpo[L] = reshape(Vt_scaled, chi, 2, 2, 1)
    end

    return mpo
end

"""
    apply_mpo_to_mps_local(mps_tensors, mpo, inds) -> Vector{Array{ComplexF64,3}}

Apply an MPO to MPS tensors over a local region [i, j] without truncation.
Operates on raw tensor vectors.
"""
function apply_mpo_to_mps_local(mps_tensors::Vector{Array{ComplexF64,3}},
                                 mpo::Vector{Array{ComplexF64,4}},
                                 inds::Tuple{Int,Int})::Vector{Array{ComplexF64,3}}
    i, j = inds
    N = length(mps_tensors)
    L = j - i + 1

    @assert length(mpo) == L "MPO length must match site range"
    @assert 1 <= i <= j <= N "Invalid site indices"

    mps_out = deepcopy(mps_tensors)

    for k in i:j
        mpo_idx = k - i + 1
        mpo_k = mpo[mpo_idx]
        mps_k = mps_out[k]

        dim_left_mps = size(mps_k, 1)
        dim_phys_out = size(mpo_k, 2)
        dim_left_mpo = size(mpo_k, 1)
        dim_right_mpo = size(mpo_k, 4)
        dim_right_mps = size(mps_k, 3)

        @tensoropt new_tensor[lm, lo, p_out, ro, rm] := mpo_k[lo, p_out, p_in, ro] * mps_k[lm, p_in, rm]

        mps_out[k] = reshape(new_tensor, dim_left_mps * dim_left_mpo, dim_phys_out,
                             dim_right_mpo * dim_right_mps)
    end

    return mps_out
end

"""
    truncate_local_region(mps_tensors, inds, max_chi, max_trunc_err) -> (tensors, trunc_err)

Truncate MPS tensors via SVD sweep over a local region [i, j].
Operates on raw tensor vectors.
"""
function truncate_local_region(mps_tensors::Vector{Array{ComplexF64,3}},
                               inds::Tuple{Int,Int},
                               max_chi::Int,
                               max_trunc_err::Float64)::Tuple{Vector{Array{ComplexF64,3}}, Float64}
    i, j = inds
    mps_out = deepcopy(mps_tensors)
    total_trunc_err_sq = 0.0

    for k in i:j-1
        T = mps_out[k]
        dim_left = size(T, 1)
        dim_phys = size(T, 2)
        dim_right = size(T, 3)

        T_r = reshape(T, dim_left * dim_phys, dim_right)
        F = robust_svd(T_r)
        SVs = F.S

        sv_norm = norm(SVs)
        SVs_normalized = sv_norm > 0 ? SVs ./ sv_norm : SVs

        # Determine truncation point
        chi = length(SVs)
        for idx in length(SVs):-1:1
            if SVs_normalized[idx] > max_trunc_err
                chi = idx
                break
            end
            chi = idx - 1
        end
        chi = max(chi, 1)
        chi = min(chi, max_chi)

        if chi < length(SVs)
            total_trunc_err_sq += sum(SVs[chi+1:end].^2)
        end

        U_trunc = F.U[:, 1:chi]
        mps_out[k] = reshape(U_trunc, dim_left, dim_phys, chi)

        SV_matrix = Diagonal(SVs[1:chi]) * F.Vt[1:chi, :]
        T_next = mps_out[k+1]
        dim_phys_next = size(T_next, 2)
        dim_right_next = size(T_next, 3)

        T_next_r = reshape(T_next, size(T_next, 1), dim_phys_next * dim_right_next)
        new_next = SV_matrix * T_next_r
        mps_out[k+1] = reshape(new_next, chi, dim_phys_next, dim_right_next)
    end

    mps_out[j] = mps_out[j] ./ norm(mps_out[j])

    return mps_out, sqrt(total_trunc_err_sq)
end

"""
    apply_gate_efficient(mps::FiniteMPS, gate, inds, max_chi, max_trunc_err; orth_centre=0)
        -> (FiniteMPS, Float64, Int)

Apply a two-qubit gate efficiently, handling both adjacent and long-range cases.
"""
function apply_gate_efficient(mps::FiniteMPS{ComplexF64},
                              gate::Matrix{ComplexF64},
                              inds::Tuple{Int,Int},
                              max_chi::Int,
                              max_trunc_err::Float64;
                              orth_centre::Int=0)::Tuple{FiniteMPS{ComplexF64}, Float64, Int}
    i, j = inds
    @assert i < j "First index must be less than second index"
    @assert 1 <= i && j <= length(mps.tensors) "Indices out of bounds"

    if j == i + 1
        # Adjacent gate: standard two-site SVD method
        mps_out = orth_centre > 0 && orth_centre != i ?
            canonicalise_FromTo(mps, (orth_centre, orth_centre), (i, i)) :
            canonicalise(mps, i)

        tensors = copy(mps_out.tensors)
        tensors[i], tensors[j], err = apply_2q_gate(gate, tensors[i], tensors[j], max_chi, max_trunc_err)
        return FiniteMPS{ComplexF64}(tensors), err, j
    else
        # Long-range gate: direct MPO method
        mps_out = orth_centre > 0 && orth_centre != i ?
            canonicalise_FromTo(mps, (orth_centre, orth_centre), (i, i)) :
            canonicalise(mps, i)

        mpo = gate_to_mpo_direct(gate, (1, j - i + 1))
        tensors_out = apply_mpo_to_mps_local(mps_out.tensors, mpo, inds)
        tensors_out, err = truncate_local_region(tensors_out, inds, max_chi, max_trunc_err)

        return FiniteMPS{ComplexF64}(tensors_out), err, j
    end
end
