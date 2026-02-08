# Layer sweep optimization
# Adapted from MPO2Circuit/src/optimization/layer_sweep.jl
# INDEX SWAP: TenSynth convention [left, phys_out, phys_in, right]
# The transposed target and contraction patterns follow the same logic as environments.jl.

using LinearAlgebra
using TensorOperations

"""
    layer_to_mpo(layer::GateLayer, n_qubits::Int; max_chi::Int=128,
                  max_trunc_err::Float64=1e-14) -> FiniteMPO{ComplexF64}

Convert a layer of gates to an MPO representation.
"""
function layer_to_mpo(layer::GateLayer, n_qubits::Int;
                       max_chi::Int=128, max_trunc_err::Float64=1e-14)::FiniteMPO{ComplexF64}
    mpo = identity_mpo(n_qubits)

    for (gate, (i, j)) in zip(layer.gates, layer.indices)
        if j == i + 1
            apply_gate!(mpo, gate, i, j; max_chi=max_chi, max_trunc_err=max_trunc_err)
        else
            apply_gate_long_range!(mpo, gate, i, j; max_chi=max_chi, max_trunc_err=max_trunc_err)
        end
    end

    return mpo
end

"""
    _transpose_mpo(mpo::FiniteMPO{ComplexF64}) -> FiniteMPO{ComplexF64}

Transpose MPO tensors: swap phys_out and phys_in.
[left, phys_out, phys_in, right] → [left, phys_in, phys_out, right]
"""
function _transpose_mpo(mpo::FiniteMPO{ComplexF64})::FiniteMPO{ComplexF64}
    return FiniteMPO{ComplexF64}([permutedims(t, (1, 3, 2, 4)) for t in mpo.tensors])
end

"""
    layer_sweep!(mpo_target, mpo_test, layer; ...) -> GateLayer

Perform bidirectional sweeps over a layer to optimize all gates.
"""
function layer_sweep!(mpo_target::FiniteMPO{ComplexF64}, mpo_test::FiniteMPO{ComplexF64},
                       layer::GateLayer;
                       n_sweeps::Int=2, slowdown::Float64=0.0,
                       max_chi::Int=128, max_trunc_err::Float64=1e-14)::GateLayer
    n = n_sites(mpo_target)
    n_gates = length(layer.gates)

    if n_gates == 0
        return layer
    end

    # Transpose target
    mpo_targ = _transpose_mpo(mpo_target)

    gate_matrices = [copy(g.matrix) for g in layer.gates]
    gate_indices = layer.indices

    # Create gate MPO
    gate_mpo = layer_to_mpo(layer, n; max_chi=max_chi, max_trunc_err=max_trunc_err)

    # Initialize environments
    layer_Lenvs = [ones(ComplexF64, 1, 1, 1) for _ in 1:n]
    layer_Renvs = [ones(ComplexF64, 1, 1, 1) for _ in 1:n]

    # Right environments
    R = ones(ComplexF64, 1, 1, 1)
    for ii in n:-1:1
        T1 = conj(mpo_targ.tensors[ii])
        T2 = gate_mpo.tensors[ii]
        T3 = mpo_test.tensors[ii]

        @tensoropt R_new[i, m, p] := T1[i, a, b, l] * T2[m, c, a, o] * T3[p, b, c, q] * R[l, o, q]

        layer_Renvs[ii] = R_new ./ norm(R_new)
        R = layer_Renvs[ii]
    end

    # Left environments up to first gate
    L = ones(ComplexF64, 1, 1, 1)
    first_gate_start = gate_indices[1][1]
    for ii in 1:(first_gate_start - 1)
        T1 = conj(mpo_targ.tensors[ii])
        T2 = gate_mpo.tensors[ii]
        T3 = mpo_test.tensors[ii]

        @tensoropt L_new[l, o, q] := L[i, m, p] * T1[i, a, b, l] * T2[m, c, a, o] * T3[p, b, c, q]
        layer_Lenvs[ii] = L_new ./ norm(L_new)
        L = layer_Lenvs[ii]
    end

    # Perform sweeps
    for sweep in 1:n_sweeps
        # Right sweep: gates 1 to n_gates-1
        for ii in 1:(n_gates - 1)
            site1, site2 = gate_indices[ii]

            if ii > 1
                prev_start = gate_indices[ii-1][1]
                L = prev_start > 1 ? copy(layer_Lenvs[prev_start-1]) : ones(ComplexF64, 1, 1, 1)
                for jj in prev_start:(site1 - 1)
                    T1 = conj(mpo_targ.tensors[jj])
                    T2 = gate_mpo.tensors[jj]
                    T3 = mpo_test.tensors[jj]
                    @tensoropt L_new[l, o, q] := L[i, m, p] * T1[i, a, b, l] * T2[m, c, a, o] * T3[p, b, c, q]
                    layer_Lenvs[jj] = L_new ./ norm(L_new)
                    L = layer_Lenvs[jj]
                end
            end

            new_gate = _compute_gate_env_and_update(
                mpo_targ, mpo_test, gate_mpo, layer_Lenvs, layer_Renvs,
                site1, site2, n, gate_matrices[ii], slowdown
            )

            gate_matrices[ii] = new_gate
            _update_gate_mpo!(gate_mpo, new_gate, site1, site2, max_chi, max_trunc_err)
        end

        # Update left environments for last gate
        if n_gates > 1
            prev_start = gate_indices[n_gates-1][1]
            curr_start = gate_indices[n_gates][1]
            L = prev_start > 1 ? copy(layer_Lenvs[prev_start-1]) : ones(ComplexF64, 1, 1, 1)
            for jj in prev_start:(curr_start - 1)
                T1 = conj(mpo_targ.tensors[jj])
                T2 = gate_mpo.tensors[jj]
                T3 = mpo_test.tensors[jj]
                @tensoropt L_new[l, o, q] := L[i, m, p] * T1[i, a, b, l] * T2[m, c, a, o] * T3[p, b, c, q]
                layer_Lenvs[jj] = L_new ./ norm(L_new)
                L = layer_Lenvs[jj]
            end
        end

        # Left sweep: gates n_gates down to 1
        for ii in n_gates:-1:1
            site1, site2 = gate_indices[ii]

            if ii < n_gates
                next_end = gate_indices[ii+1][2]
                R = next_end < n ? copy(layer_Renvs[next_end+1]) : ones(ComplexF64, 1, 1, 1)
                for jj in next_end:-1:(site2 + 1)
                    T1 = conj(mpo_targ.tensors[jj])
                    T2 = gate_mpo.tensors[jj]
                    T3 = mpo_test.tensors[jj]
                    @tensoropt R_new[i, m, p] := T1[i, a, b, l] * T2[m, c, a, o] * T3[p, b, c, q] * R[l, o, q]
                    layer_Renvs[jj] = R_new ./ norm(R_new)
                    R = layer_Renvs[jj]
                end
            end

            new_gate = _compute_gate_env_and_update(
                mpo_targ, mpo_test, gate_mpo, layer_Lenvs, layer_Renvs,
                site1, site2, n, gate_matrices[ii], slowdown
            )

            gate_matrices[ii] = new_gate
            _update_gate_mpo!(gate_mpo, new_gate, site1, site2, max_chi, max_trunc_err)
        end
    end

    new_gates = [GateMatrix(gate_matrices[i], layer.gates[i].name,
                             layer.gates[i].params) for i in 1:n_gates]

    return GateLayer(new_gates, copy(gate_indices))
end

"""
    _compute_gate_env_and_update(...)

Compute the environment for a gate and return the optimal update.
mpo_targ is ALREADY transposed.
"""
function _compute_gate_env_and_update(mpo_targ::FiniteMPO{ComplexF64},
                                       mpo_test::FiniteMPO{ComplexF64},
                                       gate_mpo::FiniteMPO{ComplexF64},
                                       layer_Lenvs::Vector{Array{ComplexF64, 3}},
                                       layer_Renvs::Vector{Array{ComplexF64, 3}},
                                       site1::Int, site2::Int, n::Int,
                                       current_gate::Matrix{ComplexF64},
                                       slowdown::Float64)::Matrix{ComplexF64}

    if site2 == site1 + 1
        L_full = site1 > 1 ? layer_Lenvs[site1-1] : ones(ComplexF64, 1, 1, 1)
        R_full = site2 < n ? layer_Renvs[site2+1] : ones(ComplexF64, 1, 1, 1)

        L = L_full[:, 1, :]
        R = R_full[:, 1, :]

        T1L = conj(mpo_targ.tensors[site1])
        T1R = conj(mpo_targ.tensors[site2])
        T2L = mpo_test.tensors[site1]
        T2R = mpo_test.tensors[site2]

        # TenSynth convention: l,o = target phys_out (gate output slot),
        # q,s = test phys_out (gate input slot).
        # E must have rows=gate_output, cols=gate_input for inv(polar(E)) to give correct gate.
        @tensoropt ans[l, o, q, s] := L[i, j] * T1L[i, q, k, m] * T1R[m, s, nn, p] * T2L[j, k, l, r] * T2R[r, nn, o, t] * R[p, t]

    else
        L_full = site1 > 1 ? layer_Lenvs[site1-1] : ones(ComplexF64, 1, 1, 1)
        L = L_full[:, 1, :]

        T1 = conj(mpo_targ.tensors[site1])
        T2 = mpo_test.tensors[site1]

        @tensoropt L_new[l, q, m, r] := L[i, j] * T1[i, q, a, m] * T2[j, a, l, r]

        for iii in (site1+1):(site2-1)
            T1 = conj(mpo_targ.tensors[iii])
            T2 = mpo_test.tensors[iii]
            @tensoropt L_temp[i, j, nn, o] := L_new[i, j, k, ll] * T1[k, a, m, nn] * T2[ll, m, a, o]
            L_new = L_temp
        end

        R_full = site2 < n ? layer_Renvs[site2+1] : ones(ComplexF64, 1, 1, 1)
        R = R_full[:, 1, :]

        T1 = conj(mpo_targ.tensors[site2])
        T2 = mpo_test.tensors[site2]

        # TenSynth: i,l = target phys_out (gate output slot), j,o = test phys_out (gate input slot)
        # E must have rows=gate_output, cols=gate_input
        @tensoropt ans[i, l, j, o] := L_new[i, j, k, nn] * T1[k, o, a, m] * T2[nn, a, l, p] * R[m, p]
    end

    if any(isnan.(ans))
        throw(ErrorException("NaN in gate environment"))
    end

    F = robust_svd(reshape(ans, 4, 4))
    U_opt = inv(F.U * F.Vt)

    if slowdown > 0.0
        U_opt = rsvd_update(U_opt, current_gate, slowdown)
    end

    return U_opt
end

"""
    _update_gate_mpo!(gate_mpo, gate, site1, site2, max_chi, max_trunc_err)

Update gate MPO tensors after optimization.
Convention: [left, phys_out, phys_in, right]
"""
function _update_gate_mpo!(gate_mpo::FiniteMPO{ComplexF64}, gate::Matrix{ComplexF64},
                            site1::Int, site2::Int,
                            max_chi::Int, max_trunc_err::Float64)
    if site2 == site1 + 1
        gate_r = reshape(gate, 2, 2, 2, 2)
        # Original: permutedims(gate_r, (3, 1, 4, 2)) for [left,phys_in,phys_out,right] convention
        # TenSynth [left,phys_out,phys_in,right]: we need [phys_out_L, phys_in_L, phys_out_R, phys_in_R]
        # gate_r indices: [out_L, out_R, in_L, in_R]
        # For SVD splitting into [1, phys_out_L, phys_in_L, chi] and [chi, phys_out_R, phys_in_R, 1]:
        # Permute to: [in_L, out_L, in_R, out_R] = (3, 1, 4, 2)
        # Then reshape to [in_L*out_L, in_R*out_R] = [phys_out*phys_in, phys_out*phys_in]
        # Wait, we need to think about this carefully for our convention.
        # Output tensors should be [1, phys_out, phys_in, chi] and [chi, phys_out, phys_in, 1]
        # gate_r[out_L, out_R, in_L, in_R]
        # Permute to group left site and right site:
        # [out_L, in_L, out_R, in_R] = permutedims(gate_r, (1, 3, 2, 4))
        # Reshape: [out_L*in_L, out_R*in_R] for SVD
        gate_p = permutedims(gate_r, (1, 3, 2, 4))
        gate_mat = reshape(gate_p, 4, 4)

        F = robust_svd(gate_mat)
        sqrt_S = sqrt.(F.S)
        chi = min(max_chi, length(F.S))

        # Left tensor: [1, phys_out, phys_in, chi]
        gate_mpo.tensors[site1] = reshape(F.U[:, 1:chi] * diagm(sqrt_S[1:chi]), 1, 2, 2, chi)
        # Right tensor: [chi, phys_out, phys_in, 1]
        gate_mpo.tensors[site2] = reshape(diagm(sqrt_S[1:chi]) * F.Vt[1:chi, :], chi, 2, 2, 1)
    else
        local_len = site2 - site1 + 1
        local_mpo = identity_mpo(local_len)

        for k in 1:(local_len - 2)
            local_mpo.tensors[k], local_mpo.tensors[k+1], _ = apply_gate_to_tensors(
                local_mpo.tensors[k], local_mpo.tensors[k+1],
                SWAP, max_chi, max_trunc_err
            )
        end

        local_mpo.tensors[local_len-1], local_mpo.tensors[local_len], _ = apply_gate_to_tensors(
            local_mpo.tensors[local_len-1], local_mpo.tensors[local_len],
            gate, max_chi, max_trunc_err
        )

        for k in (local_len - 2):-1:1
            local_mpo.tensors[k], local_mpo.tensors[k+1], _ = apply_gate_to_tensors(
                local_mpo.tensors[k], local_mpo.tensors[k+1],
                SWAP, max_chi, max_trunc_err
            )
        end

        for k in 1:local_len
            gate_mpo.tensors[site1 + k - 1] = local_mpo.tensors[k]
        end
    end
end
