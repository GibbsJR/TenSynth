# Circuit redundancy removal
# Adapted from MPS2Circuit/src/synthesis/redundancy.jl
# Key changes: Id → PAULI_I, Hadamard → H_GATE, S → S_GATE, CNOT → CNOT
# Uses TenSynth Core constants and MPS gates functions

using LinearAlgebra

# ==============================================================================
# Basic Rotation Merging Functions
# ==============================================================================

"""
    merge_single_qubit_gates(thetas1, thetas2) -> Vector{Float64}

Combine two consecutive single-qubit gates into one via XZXXZX_2_XZX.
"""
function merge_single_qubit_gates(thetas1::Vector{Float64}, thetas2::Vector{Float64})::Vector{Float64}
    return XZXXZX_2_XZX(thetas1, thetas2)
end

"""
    merge_rz_gates(theta1, theta2) -> Float64

Merge two consecutive RZ rotations: RZ(θ1) RZ(θ2) = RZ(θ1 + θ2).
Result normalized to [-π, π].
"""
function merge_rz_gates(theta1::Float64, theta2::Float64)::Float64
    combined = theta1 + theta2
    while combined > π
        combined -= 2π
    end
    while combined < -π
        combined += 2π
    end
    return combined
end

"""
    is_identity_rotation(theta; tol=1e-10) -> Bool

Check if an RZ rotation angle is effectively identity (2πn for integer n).
"""
function is_identity_rotation(theta::Float64; tol::Float64=1e-10)::Bool
    normalized = mod(theta, 2π)
    return normalized < tol || abs(normalized - 2π) < tol
end

"""
    remove_identity_rotations(thetas; tol=1e-10) -> Vector{Float64}

Replace identity rotations with exactly 0.
"""
function remove_identity_rotations(thetas::Vector{Float64}; tol::Float64=1e-10)::Vector{Float64}
    return [is_identity_rotation(θ; tol=tol) ? 0.0 : θ for θ in thetas]
end

"""
    count_non_trivial_rotations(thetas; tol=1e-10) -> Int

Count non-identity rotations in a vector.
"""
function count_non_trivial_rotations(thetas::Vector{Float64}; tol::Float64=1e-10)::Int
    return count(!is_identity_rotation(θ; tol=tol) for θ in thetas)
end

# ==============================================================================
# Gate Decomposition Functions
# ==============================================================================

"""
    decompose_gates_to_rotations(ansatz, ansatz_inds)

Decompose flat list of 2-qubit gates into individual rotation components.
Each SU(4) gate decomposes into 5 parts (15 params) in application order:
trailing_i, trailing_j, interaction(i,j), leading_i, leading_j.
"""
function decompose_gates_to_rotations(ansatz::Vector{Matrix{ComplexF64}},
                                       ansatz_inds::Vector{Tuple{Int,Int}})
    thetas_decomp = Vector{Vector{Float64}}()
    inds_decomp = Vector{Union{Int, Tuple{Int,Int}}}()

    for (gate_idx, gate) in enumerate(ansatz)
        (i, j) = ansatz_inds[gate_idx]

        local thetas
        try
            thetas = SU42Thetas(gate; max_iters=1000, tol=1e-6)
        catch
            @warn "SU42Thetas failed for gate $gate_idx, using identity rotations"
            thetas = zeros(15)
        end

        # Application order: trailing → interaction → leading
        push!(thetas_decomp, thetas[13:15]); push!(inds_decomp, i)
        push!(thetas_decomp, thetas[10:12]); push!(inds_decomp, j)
        push!(thetas_decomp, thetas[7:9]);   push!(inds_decomp, (i, j))
        push!(thetas_decomp, thetas[4:6]);   push!(inds_decomp, i)
        push!(thetas_decomp, thetas[1:3]);   push!(inds_decomp, j)
    end

    return thetas_decomp, inds_decomp
end

"""
    decompose_gates_to_rotations_from_thetas(ansatz_thetas, ansatz_inds)

Decompose gates given as 15-parameter theta vectors into rotation components.
"""
function decompose_gates_to_rotations_from_thetas(ansatz_thetas::Vector{Vector{Float64}},
                                                   ansatz_inds::Vector{Tuple{Int,Int}})
    thetas_decomp = Vector{Vector{Float64}}()
    inds_decomp = Vector{Union{Int, Tuple{Int,Int}}}()

    for (gate_idx, gate_thetas) in enumerate(ansatz_thetas)
        (i, j) = ansatz_inds[gate_idx]

        push!(thetas_decomp, gate_thetas[13:15]); push!(inds_decomp, i)
        push!(thetas_decomp, gate_thetas[10:12]); push!(inds_decomp, j)
        push!(thetas_decomp, gate_thetas[7:9]);   push!(inds_decomp, (i, j))
        push!(thetas_decomp, gate_thetas[4:6]);   push!(inds_decomp, i)
        push!(thetas_decomp, gate_thetas[1:3]);   push!(inds_decomp, j)
    end

    return thetas_decomp, inds_decomp
end

# ==============================================================================
# Mergeable Pair Detection
# ==============================================================================

"""
    find_mergeable_pairs(thetas_decomp, inds_decomp)

Find pairs of single-qubit gates on the same qubit that can be merged
(not blocked by an intervening 2-qubit gate).
"""
function find_mergeable_pairs(thetas_decomp::Vector{Vector{Float64}},
                              inds_decomp::Vector{Union{Int, Tuple{Int,Int}}})
    updates = Int[]
    ignores = Int[]
    already_ignored = Set{Int}()

    for i in 1:length(thetas_decomp)
        if i in already_ignored
            continue
        end

        inds_curr = inds_decomp[i]
        if inds_curr isa Int
            for j in (i+1):length(thetas_decomp)
                if j in already_ignored
                    continue
                end
                inds_next = inds_decomp[j]
                if inds_next == inds_curr
                    push!(updates, i)
                    push!(ignores, j)
                    push!(already_ignored, j)
                    break
                end
                if inds_next isa Tuple && inds_curr in inds_next
                    break
                end
            end
        end
    end

    return updates, ignores
end

"""
    apply_rotation_merges(thetas_decomp, updates, ignores)

Apply the merges identified by find_mergeable_pairs.
"""
function apply_rotation_merges(thetas_decomp::Vector{Vector{Float64}},
                               updates::Vector{Int},
                               ignores::Vector{Int})
    thetas_new = deepcopy(thetas_decomp)
    for (idx1, idx2) in zip(updates, ignores)
        thetas_new[idx1] = XZXXZX_2_XZX(thetas_decomp[idx1], thetas_decomp[idx2])
    end
    for idx in sort(ignores, rev=true)
        deleteat!(thetas_new, idx)
    end
    return thetas_new
end

"""
    remove_merged_indices(inds_decomp, ignores)

Remove indices corresponding to merged gates.
"""
function remove_merged_indices(inds_decomp::Vector{Union{Int, Tuple{Int,Int}}},
                               ignores::Vector{Int})
    inds_new = deepcopy(inds_decomp)
    for idx in sort(ignores, rev=true)
        deleteat!(inds_new, idx)
    end
    return inds_new
end

# ==============================================================================
# Reconstruction
# ==============================================================================

"""
    reconstruct_circuit_simple(circuit, circuit_inds)

Simple reconstruction: re-decompose each gate and remove identity rotations.
"""
function reconstruct_circuit_simple(circuit::Vector{Vector{Matrix{ComplexF64}}},
                                    circuit_inds::Vector{Vector{Tuple{Int,Int}}})
    circuit_opt = Vector{Vector{Matrix{ComplexF64}}}()

    for (_layer_idx, layer) in enumerate(circuit)
        layer_gates = Matrix{ComplexF64}[]
        for gate in layer
            try
                thetas = SU42Thetas(gate; max_iters=500, tol=1e-6)
                thetas_clean = remove_identity_rotations(thetas; tol=1e-8)
                gate_opt = Thetas2SU4(thetas_clean)
                push!(layer_gates, gate_opt)
            catch
                push!(layer_gates, gate)
            end
        end
        push!(circuit_opt, layer_gates)
    end

    return circuit_opt
end

# ==============================================================================
# Main Redundancy Removal
# ==============================================================================

"""
    remove_redundant_rotations(circuit, circuit_inds; verbose=false)
        -> (optimized_circuit, stats_dict)

Remove gauge freedom in consecutive single-qubit rotations to reduce Rz count.
"""
function remove_redundant_rotations(circuit::Vector{Vector{Matrix{ComplexF64}}},
                                    circuit_inds::Vector{Vector{Tuple{Int,Int}}};
                                    verbose::Bool=false)::Tuple{Vector{Vector{Matrix{ComplexF64}}}, Dict{Symbol,Any}}
    if isempty(circuit)
        return circuit, Dict{Symbol,Any}(:original_rotations => 0, :final_rotations => 0, :saved => 0)
    end

    # Flatten circuit
    flat_gates = Matrix{ComplexF64}[]
    flat_inds = Tuple{Int,Int}[]
    for (layer_idx, layer) in enumerate(circuit)
        for (gate_idx, gate) in enumerate(layer)
            push!(flat_gates, gate)
            push!(flat_inds, circuit_inds[layer_idx][gate_idx])
        end
    end

    if isempty(flat_gates)
        return circuit, Dict{Symbol,Any}(:original_rotations => 0, :final_rotations => 0, :saved => 0)
    end

    # Decompose to rotations
    thetas_decomp, inds_decomp = decompose_gates_to_rotations(flat_gates, flat_inds)

    # Count original non-trivial rotations
    original_count = 0
    for (thetas, _inds) in zip(thetas_decomp, inds_decomp)
        original_count += count_non_trivial_rotations(thetas; tol=1e-8)
    end

    if verbose
        println("Original rotation components: $(length(thetas_decomp))")
        println("Original non-trivial rotations: $original_count")
    end

    # Find and apply merges
    updates, ignores = find_mergeable_pairs(thetas_decomp, inds_decomp)

    if verbose
        println("Found $(length(updates)) mergeable pairs")
    end

    thetas_merged = apply_rotation_merges(thetas_decomp, updates, ignores)
    _inds_merged = remove_merged_indices(inds_decomp, ignores)

    # Count final rotations
    final_count = 0
    for (thetas, _inds) in zip(thetas_merged, _inds_merged)
        final_count += count_non_trivial_rotations(thetas; tol=1e-8)
    end

    saved = length(ignores) * 3

    if verbose
        println("Final rotation components: $(length(thetas_merged))")
        println("Final non-trivial rotations: $final_count")
        println("Merged $(length(ignores)) single-qubit gate pairs")
        println("Estimated Rz savings: $saved")
    end

    # Reconstruct
    circuit_opt = reconstruct_circuit_simple(circuit, circuit_inds)

    stats = Dict{Symbol,Any}(
        :original_rotations => original_count,
        :final_rotations => final_count,
        :original_components => length(thetas_decomp),
        :final_components => length(thetas_merged),
        :merged_pairs => length(ignores),
        :saved => saved,
        :reduction_percent => original_count > 0 ? 100 * saved / original_count : 0.0
    )

    return circuit_opt, stats
end

"""
    GetRMGaugedCircuit(circuit, circuit_inds; verbose=false) -> (optimized_circuit, saved_count)

Remove gauge freedom from layered circuit using RM gauge algorithm.
"""
function GetRMGaugedCircuit(circuit::Vector{Vector{Matrix{ComplexF64}}},
                            circuit_inds::Vector{Vector{Tuple{Int,Int}}};
                            verbose::Bool=false)::Tuple{Vector{Vector{Matrix{ComplexF64}}}, Int}
    circuit_opt, stats = remove_redundant_rotations(circuit, circuit_inds; verbose=verbose)
    return circuit_opt, stats[:saved]
end

"""
    GetRMGaugedCircuit(ansatz, ansatz_inds; verbose=false)

Flat circuit version - returns decomposed rotation representation.
"""
function GetRMGaugedCircuit(ansatz::Vector{Matrix{ComplexF64}},
                            ansatz_inds::Vector{Tuple{Int,Int}};
                            verbose::Bool=false)
    if isempty(ansatz)
        return Vector{Vector{Float64}}(), Vector{Union{Int,Tuple{Int,Int}}}(), 0
    end

    thetas_decomp, inds_decomp = decompose_gates_to_rotations(ansatz, ansatz_inds)
    original_count = length(thetas_decomp)

    if verbose
        println("Original rotation components: $original_count")
    end

    updates, ignores = find_mergeable_pairs(thetas_decomp, inds_decomp)
    thetas_new = apply_rotation_merges(thetas_decomp, updates, ignores)
    inds_new = remove_merged_indices(inds_decomp, ignores)

    n_saved = length(ignores) * 3

    if verbose
        println("Merged $(length(ignores)) single-qubit gate pairs")
        println("Final rotation components: $(length(thetas_new))")
        println("Estimated Rz savings: $n_saved")
    end

    return thetas_new, inds_new, n_saved
end
