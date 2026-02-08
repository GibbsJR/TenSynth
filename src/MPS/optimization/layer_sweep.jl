# Layer sweep optimization functions
# Adapted from MPS2Circuit/src/optimization/layer_sweep.jl
# Key changes: SVD() → robust_svd(), RSVD() → polar_unitary(),
#              Vector{Array} → FiniteMPS/FiniteMPO

using LinearAlgebra
using TensorOperations

# ============================================================================
# Gate MPO construction
# ============================================================================

"""
    LRgate2mpo(gate, inds, max_chi=8, max_trunc_err=1e-8) -> Vector{Array{ComplexF64,4}}

Convert a two-qubit gate matrix to an MPO spanning the gate's site range.
Returns raw MPO tensors covering sites inds[1] to inds[2].
"""
function LRgate2mpo(gate::Matrix{ComplexF64}, inds::Tuple{Int,Int},
                    max_chi::Int=8, max_trunc_err::Float64=1e-8)::Vector{Array{ComplexF64,4}}
    L = inds[2] - inds[1] + 1
    return gate_to_mpo_direct(gate, (1, L))
end

"""
    CreateGateMPO(N, layer_ansatz, layer_ansatz_inds) -> Vector{Array{ComplexF64,4}}

Create an MPO representation of a layer of gates by applying them to an identity MPO.
Returns raw MPO tensors.
"""
function CreateGateMPO(N::Int, layer_ansatz::Vector{Matrix{ComplexF64}},
                       layer_ansatz_inds::Vector{Tuple{Int,Int}})::Vector{Array{ComplexF64,4}}
    mpo_in = IdentityMPO(N)
    mpo_out = ApplyGateLayers(mpo_in, layer_ansatz, layer_ansatz_inds, 16, 1e-16, true)
    return mpo_out.tensors
end

# ============================================================================
# LayerSweep — MPO version
# ============================================================================

"""
    LayerSweep(mpo_targ::FiniteMPO, mpo_test::FiniteMPO, layer_ansatz, layer_ansatz_inds,
               n_sweeps, slowdown) -> Vector{Matrix{ComplexF64}}

Optimize a layer of two-qubit gates for MPO-to-MPO matching via alternating sweeps.
Only supports adjacent gates (indices differ by 1).
"""
function LayerSweep(mpo_targ_in::FiniteMPO{ComplexF64}, mpo_test_in::FiniteMPO{ComplexF64},
                    layer_ansatz_in::Vector{Matrix{ComplexF64}}, layer_ansatz_inds::Vector{Tuple{Int,Int}},
                    n_sweeps::Int, slowdown::Float64)::Vector{Matrix{ComplexF64}}

    mpo_targ = copy(transposeMPO(mpo_targ_in).tensors)
    mpo_test = mpo_test_in.tensors
    layer_ansatz = copy(layer_ansatz_in)
    N = length(mpo_targ)

    layer_Lenvs = Vector{Array{ComplexF64,3}}([ones(ComplexF64, 1, 1, 1) for _ in 1:N])
    layer_Renvs = Vector{Array{ComplexF64,3}}([ones(ComplexF64, 1, 1, 1) for _ in 1:N])

    # Build gate layer MPO piecewise: identity tensors + gate tensors
    gate_mpo = Vector{Array{ComplexF64,4}}()
    for i in 1:length(layer_ansatz)
        # Add identity tensors before this gate
        start_site = i == 1 ? 1 : layer_ansatz_inds[i-1][2] + 1
        for kk in start_site:layer_ansatz_inds[i][1]-1
            push!(gate_mpo, reshape(Matrix(ComplexF64(1.0) * I, 2, 2), 1, 2, 2, 1))
        end
        append!(gate_mpo, LRgate2mpo(layer_ansatz[i], layer_ansatz_inds[i], 8, 1e-8))
    end
    # Add trailing identity tensors
    for kk in layer_ansatz_inds[end][2]+1:N
        push!(gate_mpo, reshape(Matrix(ComplexF64(1.0) * I, 2, 2), 1, 2, 2, 1))
    end

    # Initialize Renvs
    R = ones(ComplexF64, 1, 1, 1)
    for ii in N:-1:1
        T1 = conj(mpo_targ[ii])
        T2 = gate_mpo[ii]
        T3 = mpo_test[ii]
        @tensoropt R[i, m, p] := T1[i, j, k, l] * T2[m, k, n, o] * T3[p, n, j, q] * R[l, o, q]
        layer_Renvs[ii] = R ./ norm(R)
    end

    # Initialize Lenvs up to first gate
    L = ones(ComplexF64, 1, 1, 1)
    for ii in 1:layer_ansatz_inds[1][1]-1
        T1 = conj(mpo_targ[ii])
        T2 = gate_mpo[ii]
        T3 = mpo_test[ii]
        @tensoropt L[l, o, q] := L[i, m, p] * T1[i, j, k, l] * T2[m, k, n, o] * T3[p, n, j, q]
        layer_Lenvs[ii] = L ./ norm(L)
    end

    # Sweeping
    for kk in 1:n_sweeps
        # Right sweep
        for ii in 1:length(layer_ansatz)-1
            if ii != 1
                L = layer_ansatz_inds[ii-1][1] - 1 >= 1 ? copy(layer_Lenvs[layer_ansatz_inds[ii-1][1]-1]) : ones(ComplexF64, 1, 1, 1)
                for jj in layer_ansatz_inds[ii-1][1]:layer_ansatz_inds[ii][1]-1
                    T1 = conj(mpo_targ[jj])
                    T2 = gate_mpo[jj]
                    T3 = mpo_test[jj]
                    @tensoropt L[l, o, q] := L[i, m, p] * T1[i, j, k, l] * T2[m, k, n, o] * T3[p, n, j, q]
                    layer_Lenvs[jj] = L ./ norm(L)
                end
            end

            if layer_ansatz_inds[ii][1] == layer_ansatz_inds[ii][2] - 1
                L_env = layer_ansatz_inds[ii][1] > 1 ? copy(layer_Lenvs[layer_ansatz_inds[ii][1]-1][:, 1, :]) : ones(ComplexF64, 1, 1)
                R_env = layer_ansatz_inds[ii][2] < N ? copy(layer_Renvs[layer_ansatz_inds[ii][2]+1][:, 1, :]) : ones(ComplexF64, 1, 1)

                T1L = conj(mpo_targ[layer_ansatz_inds[ii][1]])
                T1R = conj(mpo_targ[layer_ansatz_inds[ii][2]])
                T2L = mpo_test[layer_ansatz_inds[ii][1]]
                T2R = mpo_test[layer_ansatz_inds[ii][2]]

                @tensoropt ans[s, q, o, l] := L_env[i, j] * T1L[i, k, l, m] * T1R[m, n, o, p] * T2L[j, q, k, r] * T2R[r, s, n, t] * R_env[p, t]
            else
                continue
            end

            F = robust_svd(reshape(ans, 4, 4))
            layer_ansatz[ii] = F.Vt' * F.U'

            gate_mpo[layer_ansatz_inds[ii][1]:layer_ansatz_inds[ii][2]] = LRgate2mpo(layer_ansatz[ii], layer_ansatz_inds[ii], 8, 1e-8)
        end

        # Update Lenvs before left sweep
        ii = length(layer_ansatz)
        if ii > 1
            L = layer_ansatz_inds[ii-1][1] - 1 >= 1 ? copy(layer_Lenvs[layer_ansatz_inds[ii-1][1]-1]) : ones(ComplexF64, 1, 1, 1)
            for jj in layer_ansatz_inds[ii-1][1]:layer_ansatz_inds[ii][1]-1
                T1 = conj(mpo_targ[jj])
                T2 = gate_mpo[jj]
                T3 = mpo_test[jj]
                @tensoropt L[l, o, q] := L[i, m, p] * T1[i, j, k, l] * T2[m, k, n, o] * T3[p, n, j, q]
                layer_Lenvs[jj] = L ./ norm(L)
            end
        end

        # Left sweep
        for ii in length(layer_ansatz):-1:1
            if ii != length(layer_ansatz)
                R = layer_ansatz_inds[ii+1][2] + 1 <= N ? copy(layer_Renvs[layer_ansatz_inds[ii+1][2]+1]) : ones(ComplexF64, 1, 1, 1)
                for jj in layer_ansatz_inds[ii+1][2]:-1:layer_ansatz_inds[ii][2]+1
                    T1 = conj(mpo_targ[jj])
                    T2 = gate_mpo[jj]
                    T3 = mpo_test[jj]
                    @tensoropt R[i, m, p] := T1[i, j, k, l] * T2[m, k, n, o] * T3[p, n, j, q] * R[l, o, q]
                    layer_Renvs[jj] = R ./ norm(R)
                end
            end

            if layer_ansatz_inds[ii][1] == layer_ansatz_inds[ii][2] - 1
                L_env = layer_ansatz_inds[ii][1] > 1 ? copy(layer_Lenvs[layer_ansatz_inds[ii][1]-1][:, 1, :]) : ones(ComplexF64, 1, 1)
                R_env = layer_ansatz_inds[ii][2] < N ? copy(layer_Renvs[layer_ansatz_inds[ii][2]+1][:, 1, :]) : ones(ComplexF64, 1, 1)

                T1L = conj(mpo_targ[layer_ansatz_inds[ii][1]])
                T1R = conj(mpo_targ[layer_ansatz_inds[ii][2]])
                T2L = mpo_test[layer_ansatz_inds[ii][1]]
                T2R = mpo_test[layer_ansatz_inds[ii][2]]

                @tensoropt ans[s, q, o, l] := L_env[i, j] * T1L[i, k, l, m] * T1R[m, n, o, p] * T2L[j, q, k, r] * T2R[r, s, n, t] * R_env[p, t]
            else
                continue
            end

            F = robust_svd(reshape(ans, 4, 4))
            layer_ansatz[ii] = F.Vt' * F.U'

            gate_mpo[layer_ansatz_inds[ii][1]:layer_ansatz_inds[ii][2]] = LRgate2mpo(layer_ansatz[ii], layer_ansatz_inds[ii], 8, 1e-8)
        end
    end

    return [polar_unitary(layer_ansatz[i], layer_ansatz_in[i], slowdown) for i in 1:length(layer_ansatz)]
end

# ============================================================================
# LayerSweep_general — MPS version (supports long-range gates)
# ============================================================================

"""
    LayerSweep_general(mps_targ::FiniteMPS, mps_test::FiniteMPS, layer_ansatz, layer_ansatz_inds,
                       n_sweeps, slowdown) -> Vector{Matrix{ComplexF64}}

Optimize a layer of two-qubit gates for MPS-to-MPS matching.
Supports both adjacent and long-range gates.
"""
function LayerSweep_general(mps_targ_in::FiniteMPS{ComplexF64}, mps_test_in::FiniteMPS{ComplexF64},
                            layer_ansatz_in::Vector{Matrix{ComplexF64}}, layer_ansatz_inds::Vector{Tuple{Int,Int}},
                            n_sweeps::Int, slowdown::Float64)::Vector{Matrix{ComplexF64}}

    mps_targ = [conj(t) for t in mps_targ_in.tensors]
    mps_test = mps_test_in.tensors
    layer_ansatz = copy(layer_ansatz_in)
    N = length(mps_targ)

    layer_Lenvs = Vector{Array{ComplexF64,3}}([ones(ComplexF64, 1, 1, 1) for _ in 1:N])
    layer_Renvs = Vector{Array{ComplexF64,3}}([ones(ComplexF64, 1, 1, 1) for _ in 1:N])

    gate_mpo = CreateGateMPO(N, layer_ansatz, layer_ansatz_inds)

    # Initialize Renvs
    R = ones(ComplexF64, 1, 1, 1)
    for ii in N:-1:1
        T1 = mps_targ[ii]
        T2 = gate_mpo[ii]
        T3 = mps_test[ii]
        @tensoropt R[i, m, p] := T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q] * R[l, o, q]
        layer_Renvs[ii] = R ./ norm(R)
    end

    # Initialize Lenvs up to first gate
    L = ones(ComplexF64, 1, 1, 1)
    for ii in 1:layer_ansatz_inds[1][1]-1
        T1 = mps_targ[ii]
        T2 = gate_mpo[ii]
        T3 = mps_test[ii]
        @tensoropt L[l, o, q] := L[i, m, p] * T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q]
        layer_Lenvs[ii] = L ./ norm(L)
    end

    # Sweeping
    for kk in 1:n_sweeps
        # Right sweep
        for ii in 1:length(layer_ansatz)-1
            if ii != 1
                L = layer_ansatz_inds[ii-1][1] - 1 >= 1 ? copy(layer_Lenvs[layer_ansatz_inds[ii-1][1]-1]) : ones(ComplexF64, 1, 1, 1)
                for jj in layer_ansatz_inds[ii-1][1]:layer_ansatz_inds[ii][1]-1
                    T1 = mps_targ[jj]
                    T2 = gate_mpo[jj]
                    T3 = mps_test[jj]
                    @tensoropt L[l, o, q] := L[i, m, p] * T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q]
                    layer_Lenvs[jj] = L ./ norm(L)
                end
            end

            ans = _compute_gate_env_mps(mps_targ, mps_test, layer_ansatz_inds[ii],
                                        layer_Lenvs, layer_Renvs, N)

            F = robust_svd(reshape(ans, 4, 4))
            layer_ansatz[ii] = F.Vt' * F.U'

            smol_N = layer_ansatz_inds[ii][2] - layer_ansatz_inds[ii][1] + 1
            gate_mpo[layer_ansatz_inds[ii][1]:layer_ansatz_inds[ii][2]] = CreateGateMPO(
                smol_N, [layer_ansatz[ii]], [layer_ansatz_inds[ii] .- (layer_ansatz_inds[ii][1] - 1)])
        end

        # Update Lenvs before left sweep
        ii = length(layer_ansatz)
        if ii > 1
            L = layer_ansatz_inds[ii-1][1] - 1 >= 1 ? copy(layer_Lenvs[layer_ansatz_inds[ii-1][1]-1]) : ones(ComplexF64, 1, 1, 1)
            for jj in layer_ansatz_inds[ii-1][1]:layer_ansatz_inds[ii][1]-1
                T1 = mps_targ[jj]
                T2 = gate_mpo[jj]
                T3 = mps_test[jj]
                @tensoropt L[l, o, q] := L[i, m, p] * T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q]
                layer_Lenvs[jj] = L ./ norm(L)
            end
        end

        # Left sweep
        for ii in length(layer_ansatz):-1:1
            if ii != length(layer_ansatz)
                R = layer_ansatz_inds[ii+1][2] + 1 <= N ? copy(layer_Renvs[layer_ansatz_inds[ii+1][2]+1]) : ones(ComplexF64, 1, 1, 1)
                for jj in layer_ansatz_inds[ii+1][2]:-1:layer_ansatz_inds[ii][2]+1
                    T1 = mps_targ[jj]
                    T2 = gate_mpo[jj]
                    T3 = mps_test[jj]
                    @tensoropt R[i, m, p] := T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q] * R[l, o, q]
                    layer_Renvs[jj] = R ./ norm(R)
                end
            end

            ans = _compute_gate_env_mps(mps_targ, mps_test, layer_ansatz_inds[ii],
                                        layer_Lenvs, layer_Renvs, N)

            F = robust_svd(reshape(ans, 4, 4))
            layer_ansatz[ii] = F.Vt' * F.U'

            smol_N = layer_ansatz_inds[ii][2] - layer_ansatz_inds[ii][1] + 1
            gate_mpo[layer_ansatz_inds[ii][1]:layer_ansatz_inds[ii][2]] = CreateGateMPO(
                smol_N, [layer_ansatz[ii]], [layer_ansatz_inds[ii] .- (layer_ansatz_inds[ii][1] - 1)])
        end
    end

    return [polar_unitary(layer_ansatz[i], layer_ansatz_in[i], slowdown) for i in 1:length(layer_ansatz)]
end

# ============================================================================
# Internal: compute gate environment for MPS (adjacent + long-range)
# ============================================================================

"""
    _compute_gate_env_mps(mps_targ, mps_test, inds, layer_Lenvs, layer_Renvs, N)

Compute the 4×4 gate environment tensor for MPS-to-MPS matching.
mps_targ should already be conjugated. Returns raw tensor.
"""
function _compute_gate_env_mps(mps_targ::Vector{Array{ComplexF64,3}},
                               mps_test::Vector{Array{ComplexF64,3}},
                               inds::Tuple{Int,Int},
                               layer_Lenvs::Vector{Array{ComplexF64,3}},
                               layer_Renvs::Vector{Array{ComplexF64,3}},
                               N::Int)
    if inds[1] == inds[2] - 1
        # Adjacent gate case
        L_env = inds[1] > 1 ? copy(layer_Lenvs[inds[1]-1][:, 1, :]) : ones(ComplexF64, 1, 1)
        R_env = inds[2] < N ? copy(layer_Renvs[inds[2]+1][:, 1, :]) : ones(ComplexF64, 1, 1)

        T1L = mps_targ[inds[1]]
        T1R = mps_targ[inds[2]]
        T2L = mps_test[inds[1]]
        T2R = mps_test[inds[2]]

        @tensoropt ans[q, s, l, o] := L_env[i, j] * T1L[i, l, m] * T1R[m, o, p] * T2L[j, q, r] * T2R[r, s, t] * R_env[p, t]
        return ans
    else
        # Long-range gate case
        L_env = inds[1] > 1 ? copy(layer_Lenvs[inds[1]-1][:, 1, :]) : ones(ComplexF64, 1, 1)

        T1 = mps_targ[inds[1]]
        T2 = mps_test[inds[1]]
        @tensoropt L_contract[l, q, m, r] := L_env[i, j] * T1[i, l, m] * T2[j, q, r]

        # Contract middle sites
        for iii in inds[1]+1:inds[2]-1
            T1 = mps_targ[iii]
            T2 = mps_test[iii]
            @tensoropt L_contract[i, j, n, o] := L_contract[i, j, k, l] * T1[k, m, n] * T2[l, m, o]
            L_contract = L_contract ./ norm(L_contract)
        end

        # Final contraction
        R_env = inds[2] < N ? copy(layer_Renvs[inds[2]+1][:, 1, :]) : ones(ComplexF64, 1, 1)
        T1 = mps_targ[inds[2]]
        T2 = mps_test[inds[2]]
        @tensoropt ans[j, o, i, l] := L_contract[i, j, k, n] * T1[k, l, m] * T2[n, o, p] * R_env[m, p]
        return ans
    end
end
