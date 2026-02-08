# Gate application to iMPS states

using LinearAlgebra

"""
    apply_gate_single!(psi::iMPSType, U::Matrix, site::Int)

Apply a 2×2 single-qubit gate to a specific site.
"""
function apply_gate_single!(psi::iMPSType{T}, U::Matrix{ComplexF64}, site::Int) where T
    i = mod1(site, psi.unit_cell)
    γ = psi.gamma[i]
    χ_L, d, χ_R = size(γ)

    new_γ = zeros(T, χ_L, d, χ_R)
    @inbounds for r in 1:χ_R, p_out in 1:d, l in 1:χ_L
        val = zero(T)
        for p_in in 1:d
            val += U[p_out, p_in] * γ[l, p_in, r]
        end
        new_γ[l, p_out, r] = val
    end

    psi.gamma[i] = new_γ
    psi._gamma_absorbed = nothing
    return psi
end

"""
    apply_gate_nn!(psi::iMPSType, U::Matrix, sites::Tuple{Int,Int}, config::BondConfig)

Apply a 4×4 two-qubit gate to nearest-neighbor sites.
Uses SVD to restore canonical form.
"""
function apply_gate_nn!(psi::iMPSType{T}, U_orig::Matrix{ComplexF64},
                         sites::Tuple{Int,Int}, config::BondConfig) where T
    n = psi.unit_cell
    i1 = mod1(sites[1], n)
    i2 = mod1(sites[2], n)

    d = psi.physical_dim

    # Handle wrap-around: ensure left-to-right ordering
    U = copy(U_orig)
    if mod1(i1 + 1, n) == i2
        # Forward bond: i1 -> i2
    elseif mod1(i2 + 1, n) == i1
        # Wrap-around: swap sites and reverse qubit order in gate
        i1, i2 = i2, i1
        U = reverse_qubits(U)
    end

    i_prev = mod1(i1 - 1, n)

    # Build two-site tensor: θ = λ[i1-1] * γ[i1] * λ[i1] * γ[i2] * λ[i2]
    λ_left = psi.lambda[i_prev]
    γ1 = psi.gamma[i1]
    λ_mid = psi.lambda[i1]
    γ2 = psi.gamma[i2]
    λ_right = psi.lambda[i2]

    χ_L = size(γ1, 1)
    χ_M = size(γ1, 3)
    χ_R = size(γ2, 3)

    # θ[l, p1, p2, r] = Σ_m λ_left[l,l'] * γ1[l',p1,m] * λ_mid[m,m'] * γ2[m',p2,r'] * λ_right[r',r]
    # Step 1: A1 = λ_left * γ1
    A1 = zeros(T, size(λ_left, 1), d, χ_M)
    @inbounds for m in 1:χ_M, p in 1:d, l in 1:size(λ_left, 1)
        val = zero(T)
        for k in 1:χ_L
            val += λ_left[l, k] * γ1[k, p, m]
        end
        A1[l, p, m] = val
    end

    # Step 2: A1_λ = A1 * λ_mid
    A1_λ = zeros(T, size(A1, 1), d, size(λ_mid, 2))
    @inbounds for m2 in 1:size(λ_mid, 2), p in 1:d, l in 1:size(A1, 1)
        val = zero(T)
        for m in 1:χ_M
            val += A1[l, p, m] * λ_mid[m, m2]
        end
        A1_λ[l, p, m2] = val
    end

    # Step 3: A2_λ = γ2 * λ_right
    χ_M2 = size(γ2, 1)
    A2_λ = zeros(T, χ_M2, d, size(λ_right, 2))
    @inbounds for r in 1:size(λ_right, 2), p in 1:d, m in 1:χ_M2
        val = zero(T)
        for k in 1:χ_R
            val += γ2[m, p, k] * λ_right[k, r]
        end
        A2_λ[m, p, r] = val
    end

    # Step 4: θ[l, p1, p2, r] = Σ_m A1_λ[l, p1, m] * A2_λ[m, p2, r]
    χ_L_eff = size(A1_λ, 1)
    χ_R_eff = size(A2_λ, 3)
    χ_M_eff = size(A1_λ, 3)

    θ = zeros(T, χ_L_eff, d, d, χ_R_eff)
    @inbounds for r in 1:χ_R_eff, p2 in 1:d, p1 in 1:d, l in 1:χ_L_eff
        val = zero(T)
        for m in 1:χ_M_eff
            val += A1_λ[l, p1, m] * A2_λ[m, p2, r]
        end
        θ[l, p1, p2, r] = val
    end

    # Apply gate: reverse qubit ordering to match column-major tensor convention
    U_reversed = reverse_qubits(U)
    U_4d = reshape(U_reversed, d, d, d, d)
    θ_new = zeros(T, χ_L_eff, d, d, χ_R_eff)
    @inbounds for r in 1:χ_R_eff, p2o in 1:d, p1o in 1:d, l in 1:χ_L_eff
        val = zero(T)
        for p2i in 1:d, p1i in 1:d
            val += U_4d[p1o, p2o, p1i, p2i] * θ[l, p1i, p2i, r]
        end
        θ_new[l, p1o, p2o, r] = val
    end

    # SVD to split back into two tensors
    θ_mat = reshape(θ_new, χ_L_eff * d, d * χ_R_eff)
    F = _imps_robust_svd(θ_mat)

    # Truncate
    χ_max = config.max_chi
    χ_keep = min(χ_max, length(F.S))

    # Determine truncation by error threshold
    total_weight = sum(F.S.^2)
    cumulative = 0.0
    for k in 1:length(F.S)
        cumulative += F.S[k]^2
        if (total_weight - cumulative) / total_weight < config.max_trunc_err^2
            χ_keep = min(χ_keep, k)
            break
        end
    end

    S_trunc = F.S[1:χ_keep]
    # Normalize singular values
    s_norm = norm(S_trunc)
    if s_norm > eps(Float64)
        S_trunc ./= s_norm
    end

    # New tensors
    # γ1_new: need to remove λ_left from left
    # γ2_new: need to remove λ_right from right
    # New lambda: S_trunc

    # U_trunc * S_trunc^(1/2) gives left part, S_trunc^(1/2) * Vt_trunc gives right part
    # But in Vidal form: A1 = λ_left^{-1} * U * sqrt(S), A2 = sqrt(S) * Vt * λ_right^{-1}
    # Actually for iTEBD: new_γ1 = λ_left^{-1} * reshape(U_trunc, ...), new_λ = S_trunc, new_γ2 = reshape(Vt_trunc, ...) * λ_right^{-1}

    U_trunc = F.U[:, 1:χ_keep]
    Vt_trunc = F.Vt[1:χ_keep, :]

    # Compute λ_left^{-1}
    λ_left_inv = _safe_inverse_diag(λ_left)
    λ_right_inv = _safe_inverse_diag(λ_right)

    # New gamma1: λ_left^{-1} * reshape(U_trunc, χ_L_eff, d, χ_keep)
    U_tens = reshape(U_trunc, χ_L_eff, d, χ_keep)
    new_γ1 = zeros(T, size(λ_left_inv, 1), d, χ_keep)
    @inbounds for r in 1:χ_keep, p in 1:d, l in 1:size(λ_left_inv, 1)
        val = zero(T)
        for k in 1:χ_L_eff
            val += λ_left_inv[l, k] * U_tens[k, p, r]
        end
        new_γ1[l, p, r] = val
    end

    # New gamma2: reshape(Vt_trunc, χ_keep, d, χ_R_eff) * λ_right^{-1}
    Vt_tens = reshape(Vt_trunc, χ_keep, d, χ_R_eff)
    new_γ2 = zeros(T, χ_keep, d, size(λ_right_inv, 2))
    @inbounds for r in 1:size(λ_right_inv, 2), p in 1:d, l in 1:χ_keep
        val = zero(T)
        for k in 1:χ_R_eff
            val += Vt_tens[l, p, k] * λ_right_inv[k, r]
        end
        new_γ2[l, p, r] = val
    end

    # Update state
    psi.gamma[i1] = new_γ1
    psi.lambda[i1] = diagm(T.(S_trunc))
    psi.gamma[i2] = new_γ2

    psi.normalized = false
    psi._gamma_absorbed = nothing

    return psi
end

"""
    apply_gate!(psi::iMPSType, U::Matrix, sites::Tuple{Int,Int}, config::BondConfig)

Apply a gate to arbitrary sites (dispatches to NN or SWAP-based).
"""
function apply_gate!(psi::iMPSType{T}, U::Matrix{ComplexF64},
                      sites::Tuple{Int,Int}, config::BondConfig) where T
    if size(U) == (2, 2)
        apply_gate_single!(psi, U, sites[1])
        return psi
    end

    n = psi.unit_cell
    i1 = mod1(sites[1], n)
    i2 = mod1(sites[2], n)

    # Check adjacency (with periodic wrapping)
    if i2 == mod1(i1 + 1, n) || i1 == mod1(i2 + 1, n)
        apply_gate_nn!(psi, U, sites, config)
    else
        # Use SWAP network for non-adjacent sites
        apply_gate_with_swaps!(psi, U, sites, config)
    end

    return psi
end

"""
    apply_gate!(psi::iMPSType, gate::ParameterizedGate, config::BondConfig)

Apply a parameterized gate.
"""
function apply_gate!(psi::iMPSType{T}, gate::ParameterizedGate, config::BondConfig) where T
    U = to_matrix(gate)
    apply_gate!(psi, U, gate.qubits, config)
    return psi
end

"""
    apply_gates!(psi::iMPSType, gates::Vector, sites_list::Vector, config::BondConfig)
"""
function apply_gates!(psi::iMPSType{T}, gates::Vector{Matrix{ComplexF64}},
                       sites_list::Vector{Tuple{Int,Int}}, config::BondConfig) where T
    for (U, sites) in zip(gates, sites_list)
        apply_gate!(psi, U, sites, config)
    end
    return psi
end

# --- Internal helpers ---

function _imps_robust_svd(M::AbstractMatrix)
    try
        return svd(M)
    catch
        try
            return svd(M; alg=LinearAlgebra.DivideAndConquer())
        catch
            return svd(M .+ eps(Float64) * randn(ComplexF64, size(M)))
        end
    end
end

function _safe_inverse_diag(λ::Matrix{T}) where T
    n = size(λ, 1)
    inv_λ = zeros(T, n, n)
    for i in 1:n
        val = λ[i, i]
        if abs(val) > eps(Float64) * 100
            inv_λ[i, i] = one(T) / val
        end
    end
    return inv_λ
end
