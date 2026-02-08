# Gate update via polar decomposition
# Adapted from MPO2Circuit/src/optimization/gate_update.jl
# INDEX SWAP: TenSynth convention [left, phys_out, phys_in, right]
# Transposed target: [left, phys_in, phys_out, right] = permutedims(..., (1,3,2,4))

using LinearAlgebra
using TensorOperations

"""
    mpo_polar_unitary(M::Matrix{ComplexF64}) -> Matrix{ComplexF64}

Extract unitary factor via polar decomposition (local to MPO module).
"""
function mpo_polar_unitary(M::Matrix{ComplexF64})::Matrix{ComplexF64}
    F = robust_svd(M)
    return F.U * F.Vt
end

"""
    rsvd_update(U_new, U_old, slowdown) -> Matrix{ComplexF64}

Momentum-adjusted gate update.
"""
function rsvd_update(U_new::Matrix{ComplexF64}, U_old::Matrix{ComplexF64},
                      slowdown::Float64)::Matrix{ComplexF64}
    if slowdown ≈ 0.0
        return U_new
    end
    M_interp = (1 - slowdown) * U_new + slowdown * U_old
    return mpo_polar_unitary(M_interp)
end

"""
    compute_gate_environment_nn(mpo_target, mpo_test, site1, site2) -> Matrix{ComplexF64}

Environment for nearest-neighbor gate optimization.
Target is transposed and conjugated. Convention: [left, phys_out, phys_in, right].
"""
function compute_gate_environment_nn(mpo_target::FiniteMPO{ComplexF64},
                                      mpo_test::FiniteMPO{ComplexF64},
                                      site1::Int, site2::Int)::Matrix{ComplexF64}
    n = n_sites(mpo_target)

    L = site1 > 1 ? _compute_left_env_transposed(mpo_target, mpo_test, site1-1) : ones(ComplexF64, 1, 1)
    R = site2 < n ? _compute_right_env_transposed(mpo_target, mpo_test, site2+1) : ones(ComplexF64, 1, 1)

    # Transposed target: [left, phys_in, phys_out, right]
    T1L = conj(permutedims(mpo_target.tensors[site1], (1, 3, 2, 4)))
    T1R = conj(permutedims(mpo_target.tensors[site2], (1, 3, 2, 4)))

    # Test: [left, phys_out, phys_in, right]
    T2L = mpo_test.tensors[site1]
    T2R = mpo_test.tensors[site2]

    # Contraction matching the original structure:
    # T1L[i, k, l, m], T1R[m, n, o, p] — transposed target
    # T2L[j, q, k2, r], T2R[r, s, n2, t] — test
    # k↔k2 (phys_out of transposed target = phys_in of test)
    # Wait — the original code had:
    #   T1L[i, k, l, m] (left, phys_out, phys_in, right) after transpose of [left, phys_in, phys_out, right]
    #   T2L[j, q, k, r] (left, phys_in, phys_out, right) in original convention
    # In TenSynth convention after transpose, T1L is [left, phys_in, phys_out, right]
    # Test T2L is [left, phys_out, phys_in, right]
    # The contraction structure in the original code matches physical indices correctly.
    # Since both the target transpose AND the test convention have changed in exactly opposite ways,
    # the tensor contraction expressions stay the same.
    # TenSynth: l,o = target phys_out (gate output slot), q,s = test phys_out (gate input slot)
    # E must have rows=gate_output, cols=gate_input for inv(polar(E)) to give correct gate
    @tensoropt E[l, o, q, s] := L[i, j] * T1L[i, q, k, m] * T1R[m, s, nn, p] * T2L[j, k, l, r] * T2R[r, nn, o, t] * R[p, t]

    return reshape(E, 4, 4)
end

"""
    compute_gate_environment_lr(mpo_target, mpo_test, site1, site2) -> Matrix{ComplexF64}

Environment for long-range gate optimization.
"""
function compute_gate_environment_lr(mpo_target::FiniteMPO{ComplexF64},
                                      mpo_test::FiniteMPO{ComplexF64},
                                      site1::Int, site2::Int)::Matrix{ComplexF64}
    n = n_sites(mpo_target)

    L = site1 > 1 ? _compute_left_env_transposed(mpo_target, mpo_test, site1-1) : ones(ComplexF64, 1, 1)

    T1 = conj(permutedims(mpo_target.tensors[site1], (1, 3, 2, 4)))
    T2 = mpo_test.tensors[site1]

    @tensoropt L_new[l, q, m, r] := L[i, j] * T1[i, q, a, m] * T2[j, a, l, r]

    for iii in (site1+1):(site2-1)
        T1 = conj(permutedims(mpo_target.tensors[iii], (1, 3, 2, 4)))
        T2 = mpo_test.tensors[iii]

        @tensoropt L_temp[i, j, nn, o] := L_new[i, j, k, ll] * T1[k, a, m, nn] * T2[ll, m, a, o]
        L_new = L_temp
    end

    R = site2 < n ? _compute_right_env_transposed(mpo_target, mpo_test, site2+1) : ones(ComplexF64, 1, 1)

    T1 = conj(permutedims(mpo_target.tensors[site2], (1, 3, 2, 4)))
    T2 = mpo_test.tensors[site2]

    # TenSynth: i,l = target phys_out (gate output slot), j,o = test phys_out (gate input slot)
    @tensoropt E[i, l, j, o] := L_new[i, j, k, nn] * T1[k, o, a, m] * T2[nn, a, l, p] * R[m, p]

    return reshape(E, 4, 4)
end

"""
    compute_optimal_gate(mpo_target, mpo_test, site1, site2; ...) -> Matrix{ComplexF64}
"""
function compute_optimal_gate(mpo_target::FiniteMPO{ComplexF64},
                               mpo_test::FiniteMPO{ComplexF64},
                               site1::Int, site2::Int;
                               current_gate::Union{Matrix{ComplexF64}, Nothing}=nothing,
                               slowdown::Float64=0.0)::Matrix{ComplexF64}
    if site2 == site1 + 1
        E = compute_gate_environment_nn(mpo_target, mpo_test, site1, site2)
    else
        E = compute_gate_environment_lr(mpo_target, mpo_test, site1, site2)
    end

    if any(isnan.(E))
        throw(ErrorException("NaN in gate environment"))
    end

    U_opt = inv(mpo_polar_unitary(E))

    if !isnothing(current_gate) && slowdown > 0.0
        U_opt = rsvd_update(U_opt, current_gate, slowdown)
    end

    return U_opt
end

# Helper: left environment with transposed target
function _compute_left_env_transposed(mpo_target::FiniteMPO{ComplexF64},
                                       mpo_test::FiniteMPO{ComplexF64},
                                       end_site::Int)::Matrix{ComplexF64}
    L = ones(ComplexF64, 1, 1)

    for ii in 1:end_site
        T1 = conj(permutedims(mpo_target.tensors[ii], (1, 3, 2, 4)))
        T2 = mpo_test.tensors[ii]

        # T1: [i, k, l, m] after transpose (left, phys_in, phys_out, right)
        # T2: [j, q, k2, r] (left, phys_out, phys_in, right) in TenSynth
        # Match: T1's phys_out (l) with T2's phys_in (k2)
        #        T1's phys_in (k) with T2's phys_out (q)
        # Original: L[i,j] * T1[i,k,l,m] * T2[j,l,k,r] → L_new[m,r]
        # In TenSynth convention: same indices but different physical meanings,
        # however since both conventions changed symmetrically the expression is the same.
        @tensoropt L_new[m, r] := L[i, j] * T1[i, k, l, m] * T2[j, l, k, r]
        L = L_new ./ norm(L_new)
    end

    return L
end

# Helper: right environment with transposed target
function _compute_right_env_transposed(mpo_target::FiniteMPO{ComplexF64},
                                        mpo_test::FiniteMPO{ComplexF64},
                                        start_site::Int)::Matrix{ComplexF64}
    n = n_sites(mpo_target)
    R = ones(ComplexF64, 1, 1)

    for ii in n:-1:start_site
        T1 = conj(permutedims(mpo_target.tensors[ii], (1, 3, 2, 4)))
        T2 = mpo_test.tensors[ii]

        @tensoropt R_new[i, j] := T1[i, k, l, m] * T2[j, l, k, r] * R[m, r]
        R = R_new ./ norm(R_new)
    end

    return R
end
