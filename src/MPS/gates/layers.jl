# Gate layer application functions
# Adapted from MPS2Circuit/src/gates/layers.jl
# Key changes: Vector{Array{...}} → FiniteMPS/FiniteMPO, SVD() → robust_svd()
# Note: LRgate2mpo moved to optimization/layer_sweep.jl (depends on AdaptVarCompress)

using LinearAlgebra
using TensorOperations

"""
    ApplyGateLayers(mps_in::FiniteMPS, ansatz_layer, ansatz_layer_inds, max_chi, max_trunc_err, prev_inds=(1,1))
        -> (FiniteMPS, Tuple{Int,Int})

Apply a layer of two-qubit gates to an MPS using SWAP networks for long-range gates.
"""
function ApplyGateLayers(mps_in::FiniteMPS{ComplexF64}, ansatz_layer::Vector{Matrix{ComplexF64}},
                         ansatz_layer_inds::Vector{Tuple{Int, Int}}, max_chi::Int, max_trunc_err::Float64,
                         prev_inds::Tuple{Int, Int}=(1, 1))

    if prev_inds == (1, 1)
        mps = canonicalise(mps_in, ansatz_layer_inds[1][1])
    else
        mps = FiniteMPS{ComplexF64}(copy(mps_in.tensors))
    end
    tensors = mps.tensors

    for j in 1:length(ansatz_layer)
        # SWAP qubits together
        for k in ansatz_layer_inds[j][1]:ansatz_layer_inds[j][2]-2
            if k != 1
                mps_temp = canonicalise_FromTo(FiniteMPS{ComplexF64}(tensors), prev_inds, (k, k + 1))
                tensors = mps_temp.tensors
            end
            tensors[k], tensors[k+1], _ = apply_2q_gate(SWAP, tensors[k], tensors[k+1], max_chi, max_trunc_err)
            prev_inds = (k, k + 1)
        end

        # Apply the actual gate
        if prev_inds != (1, 1)
            mps_temp = canonicalise_FromTo(FiniteMPS{ComplexF64}(tensors), prev_inds,
                                           (ansatz_layer_inds[j][2] - 1, ansatz_layer_inds[j][2]))
            tensors = mps_temp.tensors
        end

        tensors[ansatz_layer_inds[j][2]-1], tensors[ansatz_layer_inds[j][2]], _ =
            apply_2q_gate(ansatz_layer[j], tensors[ansatz_layer_inds[j][2]-1],
                         tensors[ansatz_layer_inds[j][2]], max_chi, max_trunc_err)
        prev_inds = (ansatz_layer_inds[j][2] - 1, ansatz_layer_inds[j][2])

        # SWAP qubits back
        for k in ansatz_layer_inds[j][2]-2:-1:ansatz_layer_inds[j][1]
            mps_temp = canonicalise_FromTo(FiniteMPS{ComplexF64}(tensors), prev_inds, (k, k + 1))
            tensors = mps_temp.tensors
            tensors[k], tensors[k+1], _ = apply_2q_gate(SWAP, tensors[k], tensors[k+1], max_chi, max_trunc_err)
            prev_inds = (k, k + 1)
        end
    end

    result = canonicalise(FiniteMPS{ComplexF64}(tensors), prev_inds[1])
    return result, prev_inds
end

"""
    ApplyGateLayers(mpo_in::FiniteMPO, ansatz_layer, ansatz_layer_inds, max_chi, max_trunc_err, Ifcanonicalise=true)
        -> FiniteMPO

Apply a layer of two-qubit gates to an MPO using SWAP networks.
"""
function ApplyGateLayers(mpo_in::FiniteMPO{ComplexF64}, ansatz_layer::Vector{Matrix{ComplexF64}},
                         ansatz_layer_inds::Vector{Tuple{Int, Int}}, max_chi::Int, max_trunc_err::Float64,
                         Ifcanonicalise::Bool=true)

    if Ifcanonicalise
        mpo = canonicalise(mpo_in, 1)
    else
        mpo = FiniteMPO{ComplexF64}(copy(mpo_in.tensors))
    end
    tensors = mpo.tensors

    prev_inds = (1, 1)

    for j in 1:length(ansatz_layer)
        # SWAP qubits together
        for k in ansatz_layer_inds[j][1]:ansatz_layer_inds[j][2]-2
            if k != 1 && Ifcanonicalise
                mpo_temp = canonicalise_FromTo(FiniteMPO{ComplexF64}(tensors), prev_inds, (k, k + 1))
                tensors = mpo_temp.tensors
            end
            tensors[k], tensors[k+1], _ = apply_2q_gate(SWAP, tensors[k], tensors[k+1], max_chi, max_trunc_err)
            prev_inds = (k, k + 1)
        end

        # Apply the actual gate
        if prev_inds != (1, 1) && Ifcanonicalise
            mpo_temp = canonicalise_FromTo(FiniteMPO{ComplexF64}(tensors), prev_inds,
                                           (ansatz_layer_inds[j][2] - 1, ansatz_layer_inds[j][2]))
            tensors = mpo_temp.tensors
        end

        tensors[ansatz_layer_inds[j][2]-1], tensors[ansatz_layer_inds[j][2]], _ =
            apply_2q_gate(ansatz_layer[j], tensors[ansatz_layer_inds[j][2]-1],
                         tensors[ansatz_layer_inds[j][2]], max_chi, max_trunc_err)
        prev_inds = (ansatz_layer_inds[j][2] - 1, ansatz_layer_inds[j][2])

        # SWAP qubits back
        for k in ansatz_layer_inds[j][2]-2:-1:ansatz_layer_inds[j][1]
            if Ifcanonicalise
                mpo_temp = canonicalise_FromTo(FiniteMPO{ComplexF64}(tensors), prev_inds, (k, k + 1))
                tensors = mpo_temp.tensors
            end
            tensors[k], tensors[k+1], _ = apply_2q_gate(SWAP, tensors[k], tensors[k+1], max_chi, max_trunc_err)
            prev_inds = (k, k + 1)
        end
    end

    return FiniteMPO{ComplexF64}(tensors)
end

"""
    Ansatz2MPS(N, ansatz, ansatz_inds, max_chi, max_trunc_err) -> FiniteMPS

Apply multiple layers of gates to the |00...0> state.
"""
function Ansatz2MPS(N::Int, ansatz::Vector{Vector{Matrix{ComplexF64}}},
                    ansatz_inds::Vector{Vector{Tuple{Int, Int}}}, max_chi::Int, max_trunc_err::Float64)

    mps_test = canonicalise(zeroMPS(N), 1)
    prev_inds = (1, 1)

    for i in 1:length(ansatz)
        mps_test, prev_inds = ApplyGateLayers(mps_test, ansatz[i], ansatz_inds[i], max_chi, max_trunc_err, prev_inds)
    end

    return canonicalise(mps_test, 1)
end

"""
    Ansatz2MPS(mps_in::FiniteMPS, ansatz, ansatz_inds, max_chi, max_trunc_err) -> FiniteMPS

Apply multiple layers of gates to an existing MPS.
"""
function Ansatz2MPS(mps_in::FiniteMPS{ComplexF64}, ansatz::Vector{Vector{Matrix{ComplexF64}}},
                    ansatz_inds::Vector{Vector{Tuple{Int, Int}}}, max_chi::Int, max_trunc_err::Float64)

    mps_test = canonicalise(FiniteMPS{ComplexF64}(deepcopy(mps_in.tensors)), 1)
    prev_inds = (1, 1)

    for i in 1:length(ansatz)
        mps_test, prev_inds = ApplyGateLayers(mps_test, ansatz[i], ansatz_inds[i], max_chi, max_trunc_err, prev_inds)
    end

    return canonicalise(mps_test, 1)
end

"""
    mpo_mps_zipup(mpo_in::FiniteMPO, mps_in::FiniteMPS, inds, max_chi, max_trunc_err) -> FiniteMPS

Apply an MPO to an MPS using zip-up contraction.
"""
function mpo_mps_zipup(mpo_in::FiniteMPO{ComplexF64}, mps_in::FiniteMPS{ComplexF64},
                       inds::Tuple{Int, Int}, max_chi::Int, max_trunc_err::Float64)

    tensors = deepcopy(mps_in.tensors)
    mpo_tensors = deepcopy(mpo_in.tensors)

    T1 = mpo_tensors[1]
    T2 = tensors[inds[1]]

    tensor = ones(ComplexF64, 1, 1, 1, 1, 1)
    @tensoropt tensor[i, m, j, l, n] := T1[i, j, k, l] * T2[m, k, n]
    tensor = reshape(tensor, size(tensor, 1) * size(tensor, 2), size(tensor, 3), size(tensor, 4), size(tensor, 5))

    chi_temp = 1

    for ii in 1:inds[2]-inds[1]
        T1 = mpo_tensors[1+ii]
        T2 = tensors[inds[1]+ii]

        @tensoropt tensor5[i, j, m, o, p] := tensor[i, j, k, l] * T1[k, m, n, o] * T2[l, n, p]
        tensor_r = reshape(tensor5, size(tensor5, 1) * size(tensor5, 2), size(tensor5, 3) * size(tensor5, 4) * size(tensor5, 5))

        F = robust_svd(tensor_r)

        SVs = F.S / norm(F.S)
        SVs = SVs[SVs .> max_trunc_err]
        chi_temp = min(max_chi, length(SVs))

        T1_new = F.U[:, 1:chi_temp]
        T2_new = (diagm(F.S) * F.Vt)[1:chi_temp, :]

        tensors[inds[1]+ii-1] = reshape(T1_new, size(tensor5, 1), size(tensor5, 2), chi_temp)
        tensor = reshape(T2_new, chi_temp, size(tensor5, 3), size(tensor5, 4), size(tensor5, 5))
    end

    t_shape = size(tensor)
    tensors[inds[2]] = reshape(tensor, chi_temp, t_shape[2], t_shape[3] * t_shape[4])
    tensors[inds[2]] = tensors[inds[2]] ./ norm(tensors[inds[2]])

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    ApplyGatesLR(mps_in::FiniteMPS, gates, gates_inds, max_chi, max_trunc_err, orth_centre=0)
        -> (FiniteMPS, Int)

Apply a sequence of possibly long-range two-qubit gates to an MPS
using direct MPO construction (no SWAP decomposition).
"""
function ApplyGatesLR(mps_in::FiniteMPS{ComplexF64}, gates::Vector{Matrix{ComplexF64}},
                      gates_inds::Vector{Tuple{Int, Int}}, max_chi::Int, max_trunc_err::Float64,
                      orth_centre::Int=0)

    mps = FiniteMPS{ComplexF64}(deepcopy(mps_in.tensors))

    if orth_centre == 0
        mps = canonicalise(mps, 1)
        orth_centre = 1
    end

    for (gate, gate_inds) in zip(gates, gates_inds)
        mps = canonicalise_FromTo(mps, (orth_centre, orth_centre), (gate_inds[1], gate_inds[1]))
        gate_mpo = gate_to_mpo_direct(gate, gate_inds)
        gate_mpo_wrapped = FiniteMPO{ComplexF64}(gate_mpo)
        mps = mpo_mps_zipup(canonicalise(gate_mpo_wrapped, 1), mps, gate_inds, max_chi, max_trunc_err)
        orth_centre = gate_inds[2]
    end

    return mps, orth_centre
end

"""
    ApplyGateLayersEfficient(mps_in::FiniteMPS, ansatz_layer, ansatz_layer_inds,
                             max_chi, max_trunc_err, orth_centre=0) -> (FiniteMPS, Int)

Apply a layer of gates efficiently using direct MPO for long-range gates.
"""
function ApplyGateLayersEfficient(mps_in::FiniteMPS{ComplexF64},
                                   ansatz_layer::Vector{Matrix{ComplexF64}},
                                   ansatz_layer_inds::Vector{Tuple{Int,Int}},
                                   max_chi::Int, max_trunc_err::Float64,
                                   orth_centre::Int=0)::Tuple{FiniteMPS{ComplexF64}, Int}
    mps = FiniteMPS{ComplexF64}(deepcopy(mps_in.tensors))

    if orth_centre == 0
        mps = canonicalise(mps, 1)
        orth_centre = 1
    end

    for (gate, gate_inds) in zip(ansatz_layer, ansatz_layer_inds)
        mps, _, orth_centre = apply_gate_efficient(mps, gate, gate_inds, max_chi, max_trunc_err;
                                                    orth_centre=orth_centre)
    end

    return mps, orth_centre
end
