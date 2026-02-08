# Analytical MPS to Circuit Decomposition (Ran2020 / Rudolph et al.)
# Adapted from MPS2Circuit/src/protocols/analytical.jl
# Key changes: Vector{Array{ComplexF64,3}} → FiniteMPS{ComplexF64}, SVD() → robust_svd()

using LinearAlgebra
using TensorOperations

# ==============================================================================
# Helper Functions: Completing Unitaries
# ==============================================================================

"""
    _complete_unitary_from_vector_2x2(v) -> Matrix{ComplexF64}

Complete a normalized 2-element vector to a 2×2 unitary matrix (v as first column).
"""
function _complete_unitary_from_vector_2x2(v::Vector{ComplexF64})::Matrix{ComplexF64}
    v = v / norm(v)
    w = ComplexF64[-conj(v[2]), conj(v[1])]
    return hcat(v, w)
end

"""
    _complete_unitary_from_column(v) -> Matrix{ComplexF64}

Complete a normalized 4-element vector to a 4×4 unitary via Gram-Schmidt.
"""
function _complete_unitary_from_column(v::Vector{ComplexF64})::Matrix{ComplexF64}
    n = length(v)
    U = zeros(ComplexF64, n, n)
    U[:, 1] = v

    for i in 2:n
        ei = zeros(ComplexF64, n)
        ei[i] = 1.0
        w = ei
        for j in 1:i-1
            w = w - (U[:, j]' * w) * U[:, j]
        end

        if norm(w) < 1e-10
            for k in 1:n
                k == i && continue
                ek = zeros(ComplexF64, n)
                ek[k] = 1.0
                w = ek
                for j in 1:i-1
                    w = w - (U[:, j]' * w) * U[:, j]
                end
                norm(w) >= 1e-10 && break
            end
        end

        U[:, i] = norm(w) >= 1e-14 ? w / norm(w) : ei
    end

    F = robust_svd(U)
    return F.U * F.Vt
end

"""
    _extend_isometry_to_unitary(Q) -> Matrix{ComplexF64}

Extend an isometry Q (m×n with m≥n, Q'Q ≈ I) to a full m×m unitary.
"""
function _extend_isometry_to_unitary(Q::Matrix{ComplexF64})::Matrix{ComplexF64}
    m, n = size(Q)
    if m == n
        F = robust_svd(Q)
        return F.U * F.Vt
    end

    U = zeros(ComplexF64, m, m)
    U[:, 1:n] = Q

    for i in (n+1):m
        ei = zeros(ComplexF64, m)
        ei[i] = 1.0
        w = ei
        for j in 1:(i-1)
            w = w - (U[:, j]' * w) * U[:, j]
        end

        if norm(w) < 1e-10
            for k in 1:m
                k == i && continue
                ek = zeros(ComplexF64, m)
                ek[k] = 1.0
                w = ek
                for j in 1:(i-1)
                    w = w - (U[:, j]' * w) * U[:, j]
                end
                norm(w) >= 1e-10 && break
            end
        end

        U[:, i] = norm(w) >= 1e-14 ? w / norm(w) : ei
    end

    F = robust_svd(U)
    return F.U * F.Vt
end

# ==============================================================================
# Gate Extraction from MPS Tensors (Ran2020)
# ==============================================================================

"""
    extract_gate_ran2020(A::Array{ComplexF64,3}, site::Symbol) -> Matrix{ComplexF64}

Extract a 2-qubit unitary gate from an MPS tensor following Ran2020.
`site` is :first, :middle, or :last indicating position in MPS.
"""
function extract_gate_ran2020(A::Array{ComplexF64,3}, site::Symbol)::Matrix{ComplexF64}
    χL, d, χR = size(A)

    if site == :last
        if χL == 1
            v = A[1, :, 1]
            v = v / norm(v)
            U_1q = _complete_unitary_from_vector_2x2(v)
            return kron(Matrix{ComplexF64}(I, 2, 2), U_1q)
        elseif χL == 2
            U_1q = A[:, :, 1]
            F = svd(U_1q)
            U_1q = F.U * F.Vt
            return kron(Matrix{ComplexF64}(I, 2, 2), U_1q)
        else
            error("Last site should have χL ≤ 2, got χL=$χL")
        end

    elseif site == :first
        if χR == 1
            v = reshape(A[1, :, 1], 2)
            v = v / norm(v)
            U_1q = _complete_unitary_from_vector_2x2(v)
            return kron(U_1q, Matrix{ComplexF64}(I, 2, 2))
        else  # χR == 2
            state_vec = reshape(A[1, :, :], 4)
            nrm = norm(state_vec)
            if nrm > 1e-14
                state_vec = state_vec / nrm
            else
                return Matrix{ComplexF64}(I, 4, 4)
            end
            return _complete_unitary_from_column(state_vec)
        end

    elseif site == :middle
        if χL == 1 && χR == 1
            v = reshape(A[1, :, 1], 2)
            v = v / norm(v)
            U_1q = _complete_unitary_from_vector_2x2(v)
            return kron(U_1q, Matrix{ComplexF64}(I, 2, 2))
        elseif χL == 1 && χR == 2
            state_vec = reshape(A[1, :, :], 4)
            nrm = norm(state_vec)
            return nrm > 1e-14 ? _complete_unitary_from_column(state_vec / nrm) : Matrix{ComplexF64}(I, 4, 4)
        elseif χL == 2 && χR == 1
            state_vec = reshape(A[:, :, 1], 4)
            nrm = norm(state_vec)
            return nrm > 1e-14 ? _complete_unitary_from_column(state_vec / nrm) : Matrix{ComplexF64}(I, 4, 4)
        else  # χL == 2 && χR == 2
            A_mat = reshape(A, χL, d * χR)
            return _extend_isometry_to_unitary(Matrix(A_mat'))
        end
    else
        error("site must be :first, :middle, or :last, got $site")
    end
end

"""
    extract_staircase_gates(mps::FiniteMPS) -> (gates, indices)

Extract a full staircase layer of 2-qubit gates from a χ≤2 MPS using Ran2020 algorithm.
"""
function extract_staircase_gates(mps::FiniteMPS{ComplexF64})::Tuple{Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}}}
    N = length(mps.tensors)
    mps_lc = canonicalise(FiniteMPS{ComplexF64}(deepcopy(mps.tensors)), N)

    gates = Matrix{ComplexF64}[]
    inds = Tuple{Int,Int}[]

    for i in 1:N-1
        A = mps_lc.tensors[i]
        χL, _d, χR = size(A)

        site_type = if χL == 1
            :first
        elseif χR == 1
            :last
        else
            :middle
        end

        gate = extract_gate_ran2020(A, site_type)
        push!(gates, gate)
        push!(inds, (i, i+1))
    end

    return gates, inds
end

"""
    extract_gate_from_bond(TL, TR) -> Matrix{ComplexF64}

Extract a 2-qubit unitary from two adjacent MPS tensors.
"""
function extract_gate_from_bond(TL::Array{ComplexF64,3}, TR::Array{ComplexF64,3})::Matrix{ComplexF64}
    χL, d1, χM = size(TL)
    χM2, d2, χR = size(TR)

    @tensor T2[a, s1, s2, b] := TL[a, s1, c] * TR[c, s2, b]

    if χL == 1 && χR == 1
        state_vec = reshape(T2[1, :, :, 1], 4)
        nrm = norm(state_vec)
        return nrm > 1e-14 ? _complete_unitary_from_column(state_vec / nrm) : randU(0.5, 2)
    elseif χL == 1
        state_mat = zeros(ComplexF64, d1, d2)
        for r in 1:χR; state_mat .+= T2[1, :, :, r]; end
        state_vec = reshape(state_mat, 4)
        nrm = norm(state_vec)
        return nrm > 1e-14 ? _complete_unitary_from_column(state_vec / nrm) : randU(0.5, 2)
    elseif χR == 1
        state_mat = zeros(ComplexF64, d1, d2)
        for l in 1:χL; state_mat .+= T2[l, :, :, 1]; end
        state_vec = reshape(state_mat, 4)
        nrm = norm(state_vec)
        return nrm > 1e-14 ? _complete_unitary_from_column(state_vec / nrm) : randU(0.5, 2)
    else
        T2_mat = reshape(T2, χL, d1 * d2 * χR)
        F = svd(T2_mat)
        v_dominant = F.Vt[1, :]
        T2_reduced = reshape(v_dominant, d1, d2, χR)
        state_mat = zeros(ComplexF64, d1, d2)
        for r in 1:χR; state_mat .+= T2_reduced[:, :, r]; end
        state_vec = reshape(state_mat, 4)
        nrm = norm(state_vec)
        return nrm > 1e-14 ? _complete_unitary_from_column(state_vec / nrm) : randU(0.5, 2)
    end
end

# ==============================================================================
# Layer Extraction
# ==============================================================================

"""
    truncate_to_chi2(mps::FiniteMPS) -> (truncated_mps, fidelity)

Truncate an MPS to maximum bond dimension 2 via SVD.
"""
function truncate_to_chi2(mps::FiniteMPS{ComplexF64})::Tuple{FiniteMPS{ComplexF64}, Float64}
    N = length(mps.tensors)
    mps_work = canonicalise(FiniteMPS{ComplexF64}(deepcopy(mps.tensors)), 1)

    if max_bond_dim(mps_work) <= 2
        fid = fidelity(mps, mps_work)
        return mps_work, fid
    end

    mps_trunc = SVD_truncate(mps_work, 2, 1e-16, (1, N), true)
    fid = fidelity(mps, mps_trunc)
    return mps_trunc, fid
end

"""
    mps_to_staircase_layer(mps::FiniteMPS) -> (gates, indices, truncation_fidelity)

Extract a staircase layer (N-1 gates) from an MPS. Includes truncation to χ=2.
"""
function mps_to_staircase_layer(mps::FiniteMPS{ComplexF64})::Tuple{Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}}, Float64}
    mps_chi2, trunc_fid = truncate_to_chi2(mps)
    gates, inds = extract_staircase_gates(mps_chi2)
    return gates, inds, trunc_fid
end

"""
    mps_to_layer(mps::FiniteMPS, parity::Int) -> (gates, indices)

Extract a brick-wall layer of 2-qubit gates from an MPS.
"""
function mps_to_layer(mps::FiniteMPS{ComplexF64}, parity::Int)::Tuple{Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}}}
    N = length(mps.tensors)
    gates = Matrix{ComplexF64}[]
    inds = Tuple{Int,Int}[]
    start = parity == 0 ? 1 : 2
    mps_work = FiniteMPS{ComplexF64}(deepcopy(mps.tensors))

    for i in start:2:N-1
        mps_work = canonicalise(mps_work, i)
        gate = extract_gate_from_bond(mps_work.tensors[i], mps_work.tensors[i+1])
        push!(gates, gate)
        push!(inds, (i, i+1))
    end

    return gates, inds
end

# ==============================================================================
# Disentangling
# ==============================================================================

"""
    disentangle_staircase_layer(mps, gates, inds, max_chi, max_trunc_err) -> FiniteMPS

Apply inverse of a staircase gate layer to disentangle an MPS.
"""
function disentangle_staircase_layer(mps::FiniteMPS{ComplexF64},
                                     gates::Vector{Matrix{ComplexF64}},
                                     inds::Vector{Tuple{Int,Int}},
                                     max_chi::Int,
                                     max_trunc_err::Float64)::FiniteMPS{ComplexF64}
    inv_gates = [Matrix{ComplexF64}(gate') for gate in gates]
    mps_result = FiniteMPS{ComplexF64}(deepcopy(mps.tensors))

    for (gate_inv, (i, j)) in zip(inv_gates, inds)
        mps_result = canonicalise(mps_result, i)
        mps_result.tensors[i], mps_result.tensors[j], _ = apply_2q_gate(gate_inv, mps_result.tensors[i], mps_result.tensors[j], max_chi, max_trunc_err)
    end

    mps_result = canonicalise(mps_result, 1)
    nrm = norm(mps_result.tensors[1])
    if nrm > 1e-14
        mps_result.tensors[1] = mps_result.tensors[1] / nrm
    end
    return mps_result
end

"""
    disentangle_layer(mps, gates, inds, max_chi, max_trunc_err) -> FiniteMPS

Apply inverse of a gate layer to disentangle an MPS (general version).
"""
function disentangle_layer(mps::FiniteMPS{ComplexF64},
                           gates::Vector{Matrix{ComplexF64}},
                           inds::Vector{Tuple{Int,Int}},
                           max_chi::Int,
                           max_trunc_err::Float64)::FiniteMPS{ComplexF64}
    inv_gates = [Matrix{ComplexF64}(gate') for gate in gates]
    mps_result = canonicalise(FiniteMPS{ComplexF64}(deepcopy(mps.tensors)), 1)

    for (gate_inv, (i, j)) in zip(inv_gates, inds)
        mps_result = canonicalise(mps_result, i)
        mps_result.tensors[i], mps_result.tensors[j], _ = apply_2q_gate(gate_inv, mps_result.tensors[i], mps_result.tensors[j], max_chi, max_trunc_err)
    end

    mps_result = canonicalise(mps_result, 1)
    mps_result.tensors[1] = mps_result.tensors[1] / norm(mps_result.tensors[1])
    return mps_result
end

# ==============================================================================
# Analytical Decomposition
# ==============================================================================

"""
    analytical_decomposition(mps::FiniteMPS; max_layers=20, max_chi=64,
                            max_trunc_err=1e-10, verbose=false)
        -> (circuit, circuit_inds, history)

Perform analytical (SVD-based) MPS to circuit decomposition using brick-wall topology.
"""
function analytical_decomposition(mps::FiniteMPS{ComplexF64};
                                  max_layers::Int=20,
                                  max_chi::Int=64,
                                  max_trunc_err::Float64=1e-10,
                                  verbose::Bool=false)
    N = length(mps.tensors)
    mps_target = FiniteMPS{ComplexF64}(deepcopy(mps.tensors))
    normalize_mps!(mps_target)

    circuit = Vector{Vector{Matrix{ComplexF64}}}()
    circuit_inds = Vector{Vector{Tuple{Int,Int}}}()
    history = Float64[]

    mps_current = FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors))

    if verbose
        println("Analytical decomposition: $N qubits, max $max_layers layers")
    end

    for layer_num in 1:max_layers
        parity = (layer_num - 1) % 2
        mps_chi2, _ = truncate_to_chi2(mps_current)
        layer_gates, layer_inds = mps_to_layer(mps_chi2, parity)

        isempty(layer_gates) && continue

        push!(circuit, layer_gates)
        push!(circuit_inds, layer_inds)

        mps_circuit = Ansatz2MPS(N, circuit, circuit_inds, max_chi, max_trunc_err)
        current_fid = fidelity(mps_target, mps_circuit)
        push!(history, current_fid)

        if verbose
            println("  Layer $layer_num: fidelity=$(round(current_fid, digits=6)), χ_max=$(max_bond_dim(mps_current))")
        end

        mps_current = disentangle_layer(mps_current, layer_gates, layer_inds, max_chi, max_trunc_err)

        max_bond_dim(mps_current) <= 1 && break
        !isempty(history) && history[end] > 0.9999 && break
    end

    return circuit, circuit_inds, history
end

"""
    analytical_decomposition_staircase(mps::FiniteMPS; max_layers=20, max_chi=64,
                                       max_trunc_err=1e-10, verbose=false)
        -> (circuit, circuit_inds, history)

Analytical decomposition using staircase layers (Ran2020). Each layer has N-1 gates.
"""
function analytical_decomposition_staircase(mps::FiniteMPS{ComplexF64};
                                            max_layers::Int=20,
                                            max_chi::Int=64,
                                            max_trunc_err::Float64=1e-10,
                                            verbose::Bool=false)
    N = length(mps.tensors)
    mps_target = FiniteMPS{ComplexF64}(deepcopy(mps.tensors))
    normalize_mps!(mps_target)

    circuit = Vector{Vector{Matrix{ComplexF64}}}()
    circuit_inds = Vector{Vector{Tuple{Int,Int}}}()
    history = Float64[]

    mps_current = FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors))

    if verbose
        println("Analytical staircase decomposition: $N qubits, max $max_layers layers")
        println("  Initial χ_max = $(max_bond_dim(mps_current))")
    end

    for layer_num in 1:max_layers
        layer_gates, layer_inds, trunc_fid = mps_to_staircase_layer(mps_current)
        isempty(layer_gates) && break

        push!(circuit, layer_gates)
        push!(circuit_inds, layer_inds)

        mps_circuit = Ansatz2MPS(N, circuit, circuit_inds, max_chi, max_trunc_err)
        current_fid = fidelity(mps_target, mps_circuit)
        push!(history, current_fid)

        if verbose
            println("  Layer $layer_num: fidelity=$(round(current_fid, digits=6)), trunc_fid=$(round(trunc_fid, digits=6))")
        end

        mps_current = disentangle_staircase_layer(mps_current, layer_gates, layer_inds, max_chi, max_trunc_err)

        max_bond_dim(mps_current) <= 1 && break
        current_fid > 0.9999 && break
        trunc_fid > 0.9999 && layer_num > 1 && break
    end

    return circuit, circuit_inds, history
end
