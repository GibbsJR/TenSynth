# Gate application to MPO tensors
# Adapted from MPO2Circuit/src/gates/application.jl
# INDEX SWAP: TenSynth convention [left, phys_out, phys_in, right]
# Original MPO2Circuit convention: [left, phys_in, phys_out, right]
#
# Gate applies to phys_in indices (the "input" side of the operator).
# In original code: gate contracts with positions 2 (phys_in) of TL, TR.
# In TenSynth: phys_in is at position 3, so gate contracts with position 3.

using LinearAlgebra
using TensorOperations

"""
    apply_gate_to_tensors(TL::Array{ComplexF64,4}, TR::Array{ComplexF64,4},
                          gate::Matrix{ComplexF64}, max_chi::Int, max_trunc_err::Float64)

Apply a two-qubit gate to adjacent MPO tensors and split using SVD.
Convention: [left, phys_out, phys_in, right]
Gate acts on phys_in indices.
"""
function apply_gate_to_tensors(TL::Array{ComplexF64,4}, TR::Array{ComplexF64,4},
                                gate::Matrix{ComplexF64}, max_chi::Int,
                                max_trunc_err::Float64)::Tuple{Array{ComplexF64,4}, Array{ComplexF64,4}, Float64}
    # Reshape gate to 4-tensor: [out1, out2, in1, in2]
    gate_r = reshape(gate, 2, 2, 2, 2)

    # TL: [left, phys_out_L, phys_in_L, bond]
    # TR: [bond, phys_out_R, phys_in_R, right]
    # gate_r: [out_L, out_R, in_L, in_R]
    #
    # Contract gate's input indices with MPO's phys_in indices (position 3):
    # TL[m=left, a=phys_out_L, k=phys_in_L, n=bond]
    # TR[n=bond, b=phys_out_R, l=phys_in_R, o=right]
    # gate_r[i=out_L, j=out_R, k=in_L, l=in_R]
    # Result: [left, phys_out_L, new_phys_in_L, phys_out_R, new_phys_in_R, right]
    # SVD splits into [left,phys_out,phys_in,chi] and [chi,phys_out,phys_in,right]
    @tensoropt T3[m, a, i, b, j, o] := gate_r[i, j, k, l] * TL[m, a, k, n] * TR[n, b, l, o]

    # Reshape for SVD: [left * phys_out_L * new_phys_in_L, phys_out_R * new_phys_in_R * right]
    d1, d2, d3, d4, d5, d6 = size(T3)
    T3_r = reshape(T3, d1 * d2 * d3, d4 * d5 * d6)

    F = robust_svd(T3_r)

    SVs = F.S
    SVs_norm = SVs ./ norm(SVs)

    r_delta = sum(SVs_norm .> max_trunc_err)
    r_delta = max(r_delta, 1)

    chi = min(max_chi, r_delta, length(SVs))

    sqrt_S = sqrt.(SVs)

    # Reshape back: TL_out[left, phys_out, phys_in, bond], TR_out[bond, phys_out, phys_in, right]
    TL_out = reshape(F.U * diagm(sqrt_S), d1, d2, d3, length(SVs))[:, :, :, 1:chi]
    TR_out = reshape(diagm(sqrt_S) * F.Vt, length(SVs), d4, d5, d6)[1:chi, :, :, :]

    trunc_err = chi < length(SVs) ? sqrt(sum(SVs[chi+1:end].^2)) : 0.0

    return TL_out, TR_out, trunc_err
end

"""
    apply_gate!(mpo::FiniteMPO{ComplexF64}, gate::GateMatrix, site1::Int, site2::Int;
                max_chi::Int=128, max_trunc_err::Float64=1e-14) -> Float64

Apply a two-qubit gate to adjacent sites. Returns truncation error.
"""
function apply_gate!(mpo::FiniteMPO{ComplexF64}, gate::GateMatrix, site1::Int, site2::Int;
                     max_chi::Int=128, max_trunc_err::Float64=1e-14)::Float64
    n = n_sites(mpo)

    (1 <= site1 <= n && 1 <= site2 <= n) ||
        throw(ArgumentError("Sites must be between 1 and $n, got $site1 and $site2"))
    site2 == site1 + 1 ||
        throw(ArgumentError("Sites must be adjacent, got $site1 and $site2"))

    TL_new, TR_new, err = apply_gate_to_tensors(
        mpo.tensors[site1], mpo.tensors[site2],
        gate.matrix, max_chi, max_trunc_err
    )

    mpo.tensors[site1] = TL_new
    mpo.tensors[site2] = TR_new

    return err
end

"""
    gate_to_mpo(gate::GateMatrix, site1::Int, site2::Int, n::Int;
                max_chi::Int=128, max_trunc_err::Float64=1e-14) -> FiniteMPO{ComplexF64}

Convert a gate to MPO representation spanning n sites.
"""
function gate_to_mpo(gate::GateMatrix, site1::Int, site2::Int, n::Int;
                     max_chi::Int=128, max_trunc_err::Float64=1e-14)::FiniteMPO{ComplexF64}
    (1 <= site1 < site2 <= n) ||
        throw(ArgumentError("Sites must satisfy 1 <= site1 < site2 <= n"))

    mpo = identity_mpo(n)

    if site2 == site1 + 1
        mpo.tensors[site1], mpo.tensors[site2], _ = apply_gate_to_tensors(
            mpo.tensors[site1], mpo.tensors[site2],
            gate.matrix, max_chi, max_trunc_err
        )
    else
        local_len = site2 - site1 + 1
        local_mpo = identity_mpo(local_len)

        swap_mat = SWAP

        for k in 1:(local_len - 2)
            local_mpo.tensors[k], local_mpo.tensors[k+1], _ = apply_gate_to_tensors(
                local_mpo.tensors[k], local_mpo.tensors[k+1],
                swap_mat, max_chi, max_trunc_err
            )
        end

        local_mpo.tensors[local_len-1], local_mpo.tensors[local_len], _ = apply_gate_to_tensors(
            local_mpo.tensors[local_len-1], local_mpo.tensors[local_len],
            gate.matrix, max_chi, max_trunc_err
        )

        for k in (local_len - 2):-1:1
            local_mpo.tensors[k], local_mpo.tensors[k+1], _ = apply_gate_to_tensors(
                local_mpo.tensors[k], local_mpo.tensors[k+1],
                swap_mat, max_chi, max_trunc_err
            )
        end

        for k in 1:local_len
            mpo.tensors[site1 + k - 1] = local_mpo.tensors[k]
        end
    end

    return mpo
end

"""
    mpo_mpo_contract!(mpo_bottom::FiniteMPO{ComplexF64}, mpo_top::FiniteMPO{ComplexF64},
                      site1::Int, site2::Int; max_chi::Int=128, max_trunc_err::Float64=1e-14) -> Float64

Contract mpo_top into mpo_bottom using zipup SVD compression.
Convention: [left, phys_out, phys_in, right]
mpo_top's phys_in connects to mpo_bottom's phys_out.
"""
function mpo_mpo_contract!(mpo_bottom::FiniteMPO{ComplexF64}, mpo_top::FiniteMPO{ComplexF64},
                           site1::Int, site2::Int;
                           max_chi::Int=128, max_trunc_err::Float64=1e-14)::Float64
    n = n_sites(mpo_bottom)
    n_top = n_sites(mpo_top)

    (1 <= site1 <= site2 <= n) ||
        throw(ArgumentError("Invalid site range: $site1:$site2 for MPO with $n sites"))
    (site2 - site1 + 1 == n_top) ||
        throw(ArgumentError("Top MPO size ($n_top) doesn't match site range"))

    total_err = 0.0

    # T_bottom: [left_b, phys_out_b, phys_in_b, right_b]
    # T_top:    [left_t, phys_out_t, phys_in_t, right_t]
    # phys_in_t connects to phys_out_b
    T_b = mpo_bottom.tensors[site1]
    T_t = mpo_top.tensors[1]

    # T_b[i, j=phys_out_b, k=phys_in_b, l=right_b]
    # T_t[m, a=phys_out_t, j'=phys_in_t, n=right_t]
    # Contract: T_t's phys_in (j') with T_b's phys_out (j)
    # Result: [left_b, left_t, phys_out_t, phys_in_b, right_b, right_t]
    #       = [left_b, left_t, phys_out, phys_in, right_b, right_t]  (TenSynth convention)
    @tensoropt tensor[i, m, a, k, l, n] := T_b[i, j, k, l] * T_t[m, a, j, n]

    d1, d2, d3, d4, d5, d6 = size(tensor)
    tensor = reshape(tensor, d1 * d2, d3, d4, d5, d6)

    for ii in 1:(site2 - site1)
        T_b = mpo_bottom.tensors[site1 + ii]
        T_t = mpo_top.tensors[1 + ii]

        # tensor: [combined_left, phys_out, phys_in, right_b, right_t]
        # T_b: [right_b, phys_out_b, phys_in_b, right_b_new]
        # T_t: [right_t, phys_out_t, phys_in_t, right_t_new]
        # Contract: T_t's phys_in (o) with T_b's phys_out (o)
        # Result: [combined_left, phys_out_prev, phys_in_prev, phys_out_t_new, phys_in_b_new, right_b_new, right_t_new]
        @tensoropt tensor_new[i, j, k, q, n, p, r] := tensor[i, j, k, l, m] * T_b[l, o, n, p] * T_t[m, q, o, r]

        s = size(tensor_new)
        tensor_r = reshape(tensor_new, s[1] * s[2] * s[3], s[4] * s[5] * s[6] * s[7])

        F = robust_svd(tensor_r)

        SVs_norm = F.S ./ norm(F.S)
        chi_keep = sum(SVs_norm .> max_trunc_err)
        chi_keep = max(chi_keep, 1)
        chi = min(max_chi, chi_keep, length(F.S))

        if chi < length(F.S)
            total_err += sqrt(sum(F.S[chi+1:end].^2))
        end

        T_left = F.U[:, 1:chi]
        S_Vt = diagm(F.S[1:chi]) * F.Vt[1:chi, :]

        mpo_bottom.tensors[site1 + ii - 1] = reshape(T_left, s[1], s[2], s[3], chi)
        tensor = reshape(S_Vt, chi, s[4], s[5], s[6], s[7])
    end

    s = size(tensor)
    mpo_bottom.tensors[site2] = reshape(tensor, s[1], s[2], s[3], s[4] * s[5])
    mpo_bottom.tensors[site2] ./= norm(mpo_bottom.tensors[site2])

    return total_err
end

"""
    apply_gate_long_range!(mpo::FiniteMPO{ComplexF64}, gate::GateMatrix, site1::Int, site2::Int;
                           max_chi::Int=128, max_trunc_err::Float64=1e-14,
                           canonicalize::Bool=true) -> Float64

Apply a gate to non-adjacent sites using MPO contraction.
"""
function apply_gate_long_range!(mpo::FiniteMPO{ComplexF64}, gate::GateMatrix, site1::Int, site2::Int;
                                max_chi::Int=128, max_trunc_err::Float64=1e-14,
                                canonicalize::Bool=true)::Float64
    n = n_sites(mpo)

    (1 <= site1 < site2 <= n) ||
        throw(ArgumentError("Sites must satisfy 1 <= site1 < site2 <= n"))

    if site2 == site1 + 1
        return apply_gate!(mpo, gate, site1, site2; max_chi=max_chi, max_trunc_err=max_trunc_err)
    end

    if canonicalize
        mpo_canon = canonicalise(mpo, site1)
        for i in 1:n
            mpo.tensors[i] = mpo_canon.tensors[i]
        end
    end

    local_len = site2 - site1 + 1
    gate_mpo = gate_to_mpo(gate, 1, local_len, local_len; max_chi=max_chi, max_trunc_err=max_trunc_err)
    gate_mpo_canon = canonicalise(gate_mpo, 1)

    err = mpo_mpo_contract!(mpo, gate_mpo_canon, site1, site2; max_chi=max_chi, max_trunc_err=max_trunc_err)

    return err
end

"""
    random_unitary(n::Int) -> Matrix{ComplexF64}

Generate a random n×n unitary matrix via QR decomposition.
"""
function random_unitary(n::Int)::Matrix{ComplexF64}
    A = randn(ComplexF64, n, n)
    Q, R = qr(A)
    d = diag(R)
    d = d ./ abs.(d)
    return Matrix(Q) * diagm(d)
end
