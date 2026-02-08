# Environment contraction functions for MPS optimization
# Adapted from MPS2Circuit/src/environments.jl
# Key changes: Vector{Array{...}} → FiniteMPS/FiniteMPO, types accessed via .tensors
# MPO convention: [left_bond, phys_out, phys_in, right_bond] — same as MPS2Circuit

using LinearAlgebra
using TensorOperations

# ============================================================================
# MPO-MPO environments
# ============================================================================

"""
    init_Lenvs(mpo1::FiniteMPO, mpo2::FiniteMPO) -> Vector{Array{ComplexF64,2}}

Initialize left environments for two MPOs (contraction from left to right).
"""
function init_Lenvs(mpo1::FiniteMPO{ComplexF64}, mpo2::FiniteMPO{ComplexF64})::Vector{Array{ComplexF64,2}}
    L = ones(ComplexF64, 1, 1)
    Lenvs = Vector{Array{ComplexF64,2}}(undef, 0)

    for ii in 1:length(mpo1.tensors)
        T1 = conj(mpo1.tensors[ii])
        T2 = mpo2.tensors[ii]
        @tensoropt temp[k, m] := L[i, l] * T1[i, a, j, k] * T2[l, j, a, m]
        temp = temp ./ norm(temp)
        push!(Lenvs, temp)
        L = copy(temp)
    end

    return Lenvs
end

"""
    init_Renvs(mpo1::FiniteMPO, mpo2::FiniteMPO) -> Vector{Array{ComplexF64,2}}

Initialize right environments for two MPOs (contraction from right to left).
"""
function init_Renvs(mpo1::FiniteMPO{ComplexF64}, mpo2::FiniteMPO{ComplexF64})::Vector{Array{ComplexF64,2}}
    R = ones(ComplexF64, 1, 1)
    Renvs = Vector{Array{ComplexF64,2}}(undef, 0)

    for ii in length(mpo1.tensors):-1:1
        T1 = conj(mpo1.tensors[ii])
        T2 = mpo2.tensors[ii]
        @tensoropt temp[i, l] := T1[i, a, j, k] * T2[l, j, a, m] * R[k, m]
        temp = temp ./ norm(temp)
        prepend!(Renvs, [temp])
        R = copy(temp)
    end

    return Renvs
end

# ============================================================================
# MPS-MPS environments
# ============================================================================

"""
    init_Lenvs(mps1::FiniteMPS, mps2::FiniteMPS) -> Vector{Array{ComplexF64,2}}

Initialize left environments for two MPS.
"""
function init_Lenvs(mps1::FiniteMPS{ComplexF64}, mps2::FiniteMPS{ComplexF64})::Vector{Array{ComplexF64,2}}
    L = ones(ComplexF64, 1, 1)
    Lenvs = Vector{Array{ComplexF64,2}}(undef, 0)

    for ii in 1:length(mps1.tensors)
        T1 = conj(mps1.tensors[ii])
        T2 = mps2.tensors[ii]
        @tensoropt temp[k, m] := L[i, l] * T1[i, j, k] * T2[l, j, m]
        temp = temp ./ norm(temp)
        push!(Lenvs, temp)
        L = copy(temp)
    end

    return Lenvs
end

"""
    init_Renvs(mps1::FiniteMPS, mps2::FiniteMPS) -> Vector{Array{ComplexF64,2}}

Initialize right environments for two MPS.
"""
function init_Renvs(mps1::FiniteMPS{ComplexF64}, mps2::FiniteMPS{ComplexF64})::Vector{Array{ComplexF64,2}}
    R = ones(ComplexF64, 1, 1)
    Renvs = Vector{Array{ComplexF64,2}}(undef, 0)

    for ii in length(mps1.tensors):-1:1
        T1 = conj(mps1.tensors[ii])
        T2 = mps2.tensors[ii]
        @tensoropt temp[i, l] := T1[i, j, k] * T2[l, j, m] * R[k, m]
        temp = temp ./ norm(temp)
        prepend!(Renvs, [temp])
        R = copy(temp)
    end

    return Renvs
end

# ============================================================================
# Environment updates (MPO-MPO)
# ============================================================================

"""
    update_Lenvs(inds, mpo1, mpo2, Lenvs, Renvs) -> updated Lenvs

Re-compute left environments from inds[1] to inds[2].
"""
function update_Lenvs(inds::Tuple{Int,Int}, mpo1::FiniteMPO{ComplexF64}, mpo2::FiniteMPO{ComplexF64},
                      Lenvs::Vector{Array{ComplexF64,2}}, Renvs::Vector{Array{ComplexF64,2}})::Vector{Array{ComplexF64,2}}
    for i_L in inds[1]:inds[2]
        L = i_L != 1 ? Lenvs[i_L-1] : ones(ComplexF64, 1, 1)
        T1 = conj(mpo1.tensors[i_L])
        T2 = mpo2.tensors[i_L]
        @tensoropt temp[k, p] := L[i, n] * T1[i, a, j, k] * T2[n, j, a, p]
        Lenvs[i_L] = temp ./ norm(temp)
    end
    return Lenvs
end

"""
    update_Renvs(inds, mpo1, mpo2, Lenvs, Renvs) -> updated Renvs

Re-compute right environments from inds[2] down to inds[1].
"""
function update_Renvs(inds::Tuple{Int,Int}, mpo1::FiniteMPO{ComplexF64}, mpo2::FiniteMPO{ComplexF64},
                      Lenvs::Vector{Array{ComplexF64,2}}, Renvs::Vector{Array{ComplexF64,2}})::Vector{Array{ComplexF64,2}}
    N = length(mpo1.tensors)
    for i_R in inds[2]:-1:inds[1]
        R = i_R != N ? Renvs[i_R+1] : ones(ComplexF64, 1, 1)
        T1 = conj(mpo1.tensors[i_R])
        T2 = mpo2.tensors[i_R]
        @tensoropt temp[i, l] := T1[i, a, j, k] * T2[l, j, a, m] * R[k, m]
        Renvs[i_R] = temp ./ norm(temp)
    end
    return Renvs
end

# ============================================================================
# Environment updates (MPS-MPS)
# ============================================================================

function update_Lenvs(inds::Tuple{Int,Int}, mps1::FiniteMPS{ComplexF64}, mps2::FiniteMPS{ComplexF64},
                      Lenvs::Vector{Array{ComplexF64,2}}, Renvs::Vector{Array{ComplexF64,2}})::Vector{Array{ComplexF64,2}}
    for i_L in inds[1]:inds[2]
        L = i_L != 1 ? Lenvs[i_L-1] : ones(ComplexF64, 1, 1)
        T1 = conj(mps1.tensors[i_L])
        T2 = mps2.tensors[i_L]
        @tensoropt temp[k, p] := L[i, n] * T1[i, j, k] * T2[n, j, p]
        Lenvs[i_L] = temp ./ norm(temp)
    end
    return Lenvs
end

function update_Renvs(inds::Tuple{Int,Int}, mps1::FiniteMPS{ComplexF64}, mps2::FiniteMPS{ComplexF64},
                      Lenvs::Vector{Array{ComplexF64,2}}, Renvs::Vector{Array{ComplexF64,2}})::Vector{Array{ComplexF64,2}}
    N = length(mps1.tensors)
    for i_R in inds[2]:-1:inds[1]
        R = i_R != N ? Renvs[i_R+1] : ones(ComplexF64, 1, 1)
        T1 = conj(mps1.tensors[i_R])
        T2 = mps2.tensors[i_R]
        @tensoropt temp[i, l] := T1[i, j, k] * T2[l, j, m] * R[k, m]
        Renvs[i_R] = temp ./ norm(temp)
    end
    return Renvs
end

# ============================================================================
# Three-tensor (MPS-MPO-MPS) layer environments
# ============================================================================

"""
    init_layer_envs(mps_targ, gate_mpo, mps_test) -> (Lenvs, Renvs)

Initialize environments for layer optimization: ⟨mps_targ | gate_mpo | mps_test⟩.
Takes raw tensor vectors (not wrapped types) for internal use.
"""
function init_layer_envs(mps_targ::Vector{Array{ComplexF64,3}}, gate_mpo::Vector{Array{ComplexF64,4}},
                         mps_test::Vector{Array{ComplexF64,3}})
    N = length(mps_targ)

    # Left environments
    L = ones(ComplexF64, 1, 1, 1)
    layer_Lenvs = Vector{Array{ComplexF64,3}}([ones(ComplexF64, 1, 1, 1) for _ in 1:N])

    for ii in 1:N
        T1 = conj(mps_targ[ii])
        T2 = gate_mpo[ii]
        T3 = mps_test[ii]
        @tensoropt L[l, o, q] := L[i, m, p] * T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q]
        layer_Lenvs[ii] = L ./ norm(L)
    end

    # Right environments
    R = ones(ComplexF64, 1, 1, 1)
    layer_Renvs = Vector{Array{ComplexF64,3}}([ones(ComplexF64, 1, 1, 1) for _ in 1:N])

    for ii in N:-1:1
        T1 = conj(mps_targ[ii])
        T2 = gate_mpo[ii]
        T3 = mps_test[ii]
        @tensoropt R[i, m, p] := T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q] * R[l, o, q]
        layer_Renvs[ii] = R ./ norm(R)
    end

    return layer_Lenvs, layer_Renvs
end
