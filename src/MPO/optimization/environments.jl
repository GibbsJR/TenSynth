# Environment tensor computations for optimization
# Adapted from MPO2Circuit/src/optimization/environments.jl
# INDEX SWAP: TenSynth convention [left, phys_out, phys_in, right]
#
# Three-layer structure: target (conjugated) / gate layer / test
# The target is transposed (phys_out ↔ phys_in) for proper contraction.
# In TenSynth convention, transpose means: [left, phys_out, phys_in, right] → [left, phys_in, phys_out, right]
# which is permutedims(..., (1, 3, 2, 4)).

using LinearAlgebra
using TensorOperations

mutable struct LayerEnvironments
    left_envs::Vector{Array{ComplexF64, 3}}
    right_envs::Vector{Array{ComplexF64, 3}}
    n_sites::Int
end

"""
    init_layer_environments(mpo_target, gate_mpo, mpo_test) -> LayerEnvironments

Initialize environments for layer optimization.
Target is transposed for proper contraction.

After transposing the target: [left, phys_in, phys_out, right]
Gate MPO: [left, phys_out, phys_in, right]
Test MPO: [left, phys_out, phys_in, right]

Contraction pattern:
  T1 (transposed target, conjugated): [i, j, k, l] = [left, phys_in, phys_out, right]
  T2 (gate): [m, k_g, n_g, o] = [left, phys_out, phys_in, right]
  T3 (test): [p, n_t, j_t, q] = [left, phys_out, phys_in, right]

For the 3-layer contraction, we match:
  T1's phys_out (k) with T2's phys_out (k_g) — same physical space
  T2's phys_in (n_g) with T3's phys_out (n_t) — gate output feeds test
  T1's phys_in (j) with T3's phys_in (j_t) — traced over

Original MPO2Circuit indices (with [left, phys_in, phys_out, right] after transpose):
  T1[i, j, k, l], T2[m, k, n, o], T3[p, n, j, q]
These SAME contractions apply in our convention because the transpose swaps the meaning.
"""
function init_layer_environments(mpo_target::FiniteMPO{ComplexF64},
                                  gate_mpo::FiniteMPO{ComplexF64},
                                  mpo_test::FiniteMPO{ComplexF64})::LayerEnvironments
    n = n_sites(mpo_target)

    left_envs = Vector{Array{ComplexF64, 3}}(undef, n)
    right_envs = Vector{Array{ComplexF64, 3}}(undef, n)

    # Right environments (from right to left)
    R = ones(ComplexF64, 1, 1, 1)
    for ii in n:-1:1
        # Transpose target: swap phys_out(2) and phys_in(3)
        T1 = conj(permutedims(mpo_target.tensors[ii], (1, 3, 2, 4)))
        T2 = gate_mpo.tensors[ii]
        T3 = mpo_test.tensors[ii]

        # T1: [i, j, k, l] (transposed target)
        # T2: [m, k2, n, o] (gate)
        # T3: [p, n2, j2, q] (test)
        # Contract: k↔k2 (phys_out match), n↔n2 (gate phys_in ↔ test phys_out), j↔j2 (phys_in match)
        @tensoropt R_new[i, m, p] := T1[i, j, k, l] * T2[m, k, n, o] * T3[p, n, j, q] * R[l, o, q]

        R_new ./= norm(R_new)
        right_envs[ii] = R_new
        R = R_new
    end

    # Left environments (from left to right)
    L = ones(ComplexF64, 1, 1, 1)
    for ii in 1:n
        T1 = conj(permutedims(mpo_target.tensors[ii], (1, 3, 2, 4)))
        T2 = gate_mpo.tensors[ii]
        T3 = mpo_test.tensors[ii]

        @tensoropt L_new[l, o, q] := L[i, m, p] * T1[i, j, k, l] * T2[m, k, n, o] * T3[p, n, j, q]

        L_new ./= norm(L_new)
        left_envs[ii] = L_new
        L = L_new
    end

    return LayerEnvironments(left_envs, right_envs, n)
end

function update_left_environments!(envs::LayerEnvironments, mpo_target::FiniteMPO{ComplexF64},
                                    gate_mpo::FiniteMPO{ComplexF64}, mpo_test::FiniteMPO{ComplexF64},
                                    start_site::Int, end_site::Int)
    for ii in start_site:end_site
        L = ii > 1 ? envs.left_envs[ii-1] : ones(ComplexF64, 1, 1, 1)

        T1 = conj(permutedims(mpo_target.tensors[ii], (1, 3, 2, 4)))
        T2 = gate_mpo.tensors[ii]
        T3 = mpo_test.tensors[ii]

        @tensoropt L_new[l, o, q] := L[i, m, p] * T1[i, j, k, l] * T2[m, k, n, o] * T3[p, n, j, q]

        envs.left_envs[ii] = L_new ./ norm(L_new)
    end
end

function update_right_environments!(envs::LayerEnvironments, mpo_target::FiniteMPO{ComplexF64},
                                     gate_mpo::FiniteMPO{ComplexF64}, mpo_test::FiniteMPO{ComplexF64},
                                     start_site::Int, end_site::Int)
    n = envs.n_sites

    for ii in end_site:-1:start_site
        R = ii < n ? envs.right_envs[ii+1] : ones(ComplexF64, 1, 1, 1)

        T1 = conj(permutedims(mpo_target.tensors[ii], (1, 3, 2, 4)))
        T2 = gate_mpo.tensors[ii]
        T3 = mpo_test.tensors[ii]

        @tensoropt R_new[i, m, p] := T1[i, j, k, l] * T2[m, k, n, o] * T3[p, n, j, q] * R[l, o, q]

        envs.right_envs[ii] = R_new ./ norm(R_new)
    end
end

# Two-layer environments (target-test, no gate layer)

mutable struct TwoLayerEnvironments
    left_envs::Vector{Array{ComplexF64, 2}}
    right_envs::Vector{Array{ComplexF64, 2}}
    n_sites::Int
end

"""
    init_two_layer_environments(mpo1, mpo2) -> TwoLayerEnvironments

Initialize environments for <mpo1|mpo2>. First MPO is conjugated.
Convention: [left, phys_out, phys_in, right]
"""
function init_two_layer_environments(mpo1::FiniteMPO{ComplexF64},
                                      mpo2::FiniteMPO{ComplexF64})::TwoLayerEnvironments
    n = n_sites(mpo1)

    left_envs = Vector{Array{ComplexF64, 2}}(undef, n)
    right_envs = Vector{Array{ComplexF64, 2}}(undef, n)

    # Right environments
    # Original: T1[i, a, j, k] with [left, phys_in, phys_out, right]
    # TenSynth: T1[i, j, a, k] with [left, phys_out, phys_in, right]
    # Original contraction: T1[i,a,j,k] * T2[l,j,a,m] * R[k,m] → R_new[i,l]
    # TenSynth: T1[i,j,a,k] * T2[l,a,j,m] * R[k,m] → R_new[i,l]
    R = ones(ComplexF64, 1, 1)
    for ii in n:-1:1
        T1 = conj(mpo1.tensors[ii])
        T2 = mpo2.tensors[ii]

        @tensoropt R_new[i, l] := T1[i, j, a, k] * T2[l, a, j, m] * R[k, m]
        R_new ./= norm(R_new)
        right_envs[ii] = R_new
        R = R_new
    end

    # Left environments
    # Original: L[i,n] * T1[i,a,j,k] * T2[n,j,a,p] → L_new[k,p]
    # TenSynth: L[i,n] * T1[i,j,a,k] * T2[n,a,j,p] → L_new[k,p]
    L = ones(ComplexF64, 1, 1)
    for ii in 1:n
        T1 = conj(mpo1.tensors[ii])
        T2 = mpo2.tensors[ii]

        @tensoropt L_new[k, p] := L[i, n] * T1[i, j, a, k] * T2[n, a, j, p]
        L_new ./= norm(L_new)
        left_envs[ii] = L_new
        L = L_new
    end

    return TwoLayerEnvironments(left_envs, right_envs, n)
end

function update_two_layer_left!(envs::TwoLayerEnvironments, mpo1::FiniteMPO{ComplexF64},
                                 mpo2::FiniteMPO{ComplexF64}, start_site::Int, end_site::Int)
    for ii in start_site:end_site
        L = ii > 1 ? envs.left_envs[ii-1] : ones(ComplexF64, 1, 1)
        T1 = conj(mpo1.tensors[ii])
        T2 = mpo2.tensors[ii]

        @tensoropt L_new[k, p] := L[i, n] * T1[i, j, a, k] * T2[n, a, j, p]
        envs.left_envs[ii] = L_new ./ norm(L_new)
    end
end

function update_two_layer_right!(envs::TwoLayerEnvironments, mpo1::FiniteMPO{ComplexF64},
                                  mpo2::FiniteMPO{ComplexF64}, start_site::Int, end_site::Int)
    n = envs.n_sites

    for ii in end_site:-1:start_site
        R = ii < n ? envs.right_envs[ii+1] : ones(ComplexF64, 1, 1)
        T1 = conj(mpo1.tensors[ii])
        T2 = mpo2.tensors[ii]

        @tensoropt R_new[i, l] := T1[i, j, a, k] * T2[l, a, j, m] * R[k, m]
        envs.right_envs[ii] = R_new ./ norm(R_new)
    end
end
