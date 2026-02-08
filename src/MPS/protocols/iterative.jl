# Iterative MPS to Circuit Decomposition (Iter[DiOall] Protocol)
# Adapted from MPS2Circuit/src/protocols/iterative.jl
# Key changes: Vector{Array{ComplexF64,3}} → FiniteMPS{ComplexF64},
#              optimize(hp, nothing, nothing, ...) → optimize_layered(hp, ...)

using LinearAlgebra
using TensorOperations

# ==============================================================================
# Helper Functions
# ==============================================================================

"""
    _apply_inverse_circuit(mps, circuit, circuit_inds, max_chi, max_trunc_err)
        -> FiniteMPS{ComplexF64}

Apply the inverse of a circuit to an MPS.
Disentangles the MPS by applying U† for each gate in reverse order.
"""
function _apply_inverse_circuit(mps::FiniteMPS{ComplexF64},
                                circuit::Vector{Vector{Matrix{ComplexF64}}},
                                circuit_inds::Vector{Vector{Tuple{Int,Int}}},
                                max_chi::Int,
                                max_trunc_err::Float64)::FiniteMPS{ComplexF64}
    mps_result = FiniteMPS{ComplexF64}(deepcopy(mps.tensors))
    orth_centre = 0

    # Apply layers in reverse order
    for layer_idx in length(circuit):-1:1
        layer_gates = circuit[layer_idx]
        layer_inds = circuit_inds[layer_idx]

        # Apply inverse gates in reverse order within layer
        for gate_idx in length(layer_gates):-1:1
            gate = layer_gates[gate_idx]
            (i, j) = layer_inds[gate_idx]

            gate_inv = Matrix{ComplexF64}(gate')  # U† for unitary

            mps_result, _, orth_centre = apply_gate_efficient(mps_result, gate_inv, (i, j),
                                                               max_chi, max_trunc_err;
                                                               orth_centre=orth_centre)
        end
    end

    # Normalize
    mps_result = canonicalise(mps_result, 1)
    mps_result.tensors[1] = mps_result.tensors[1] / norm(mps_result.tensors[1])

    return mps_result
end

"""
    _get_analytical_layer_init(mps_disentangled, parity)
        -> (Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}})

Get initial gates for a new layer using analytical extraction (brick-wall topology).
"""
function _get_analytical_layer_init(mps_disentangled::FiniteMPS{ComplexF64},
                                    parity::Int)::Tuple{Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}}}
    # First truncate to χ=2 for gate extraction
    mps_chi2, _ = truncate_to_chi2(mps_disentangled)

    # Extract gates using analytical method (brick-wall)
    gates, inds = mps_to_layer(mps_chi2, parity)

    return gates, inds
end

"""
    _get_staircase_layer_init(mps_disentangled)
        -> (Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}})

Get initial gates for a new layer using staircase extraction (Ran2020 algorithm).
"""
function _get_staircase_layer_init(mps_disentangled::FiniteMPS{ComplexF64})::Tuple{Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}}}
    # Use staircase layer extraction (includes truncation to χ=2)
    gates, inds, _ = mps_to_staircase_layer(mps_disentangled)

    return gates, inds
end

"""
    _compute_fidelity_fast(mps_target, circuit, circuit_inds, N, max_chi, max_trunc_err) -> Float64

Compute fidelity between target MPS and circuit applied to |0...0⟩.
"""
function _compute_fidelity_fast(mps_target::FiniteMPS{ComplexF64},
                                circuit::Vector{Vector{Matrix{ComplexF64}}},
                                circuit_inds::Vector{Vector{Tuple{Int,Int}}},
                                N::Int, max_chi::Int, max_trunc_err::Float64)::Float64
    if isempty(circuit)
        return 0.0
    end

    # Apply circuit to |0...0⟩
    mps_circuit = Ansatz2MPS(N, circuit, circuit_inds, max_chi, max_trunc_err)

    # Compute fidelity
    return fidelity(mps_target, mps_circuit)
end

# ==============================================================================
# Main Iterative Decomposition
# ==============================================================================

"""
    iterative_decomposition(mps; kwargs...)
        -> (circuit, circuit_inds, history)

Perform iterative (Iter[DiOall]) MPS to circuit decomposition.

Implements the best protocol from Rudolph et al.:
1. Add layers one at a time
2. Initialize new layers using analytical decomposition
3. Optimize ALL layers jointly after each addition
4. Continue until target fidelity or max layers reached

# Arguments
- `mps::FiniteMPS{ComplexF64}`: Input MPS to decompose
- `target_fidelity::Float64=0.95`: Target fidelity
- `max_layers::Int=10`: Maximum number of circuit layers
- `sweeps_per_layer::Int=50`: Optimization sweeps after each layer
- `max_chi::Int=64`: Maximum bond dimension
- `max_trunc_err::Float64=1e-10`: Truncation error threshold
- `use_analytical_init::Bool=true`: Use analytical initialization
- `verbose::Bool=false`: Print progress
- `topology::Symbol=:staircase`: Circuit topology (:staircase or :brickwall)
- `circuit_inds`: Optional user-specified circuit structure
"""
function iterative_decomposition(mps::FiniteMPS{ComplexF64};
                                 target_fidelity::Float64=0.95,
                                 max_layers::Int=10,
                                 sweeps_per_layer::Int=50,
                                 max_chi::Int=64,
                                 max_trunc_err::Float64=1e-10,
                                 use_analytical_init::Bool=true,
                                 verbose::Bool=false,
                                 topology::Symbol=:staircase,
                                 circuit_inds::Union{Nothing, Vector{Vector{Tuple{Int,Int}}}}=nothing)
    N = length(mps.tensors)

    @assert topology in (:staircase, :brickwall) "topology must be :staircase or :brickwall, got $topology"

    # Normalize input
    mps_target = FiniteMPS{ComplexF64}(deepcopy(mps.tensors))
    normalize_mps!(mps_target)

    # Determine number of layers and whether using custom indices
    using_custom_inds = circuit_inds !== nothing
    actual_max_layers = using_custom_inds ? length(circuit_inds) : max_layers

    # Check if custom indices contain long-range gates
    has_long_range = using_custom_inds && any(layer -> any(((i, j),) -> j > i + 1, layer), circuit_inds)

    if verbose
        println("Iterative (Iter[DiOall]) decomposition:")
        println("  Qubits: $N, Target fidelity: $target_fidelity")
        println("  Max layers: $actual_max_layers, Sweeps per layer: $sweeps_per_layer")
        if using_custom_inds
            println("  Circuit topology: user-specified $(has_long_range ? "(with long-range gates)" : "(nearest-neighbor)")")
        else
            println("  Circuit topology: $topology")
        end
        println("  Analytical init: $use_analytical_init")
    end

    # Initialize circuit
    result_circuit = Vector{Vector{Matrix{ComplexF64}}}()
    result_circuit_inds = Vector{Vector{Tuple{Int,Int}}}()
    history = Float64[]

    current_fid = 0.0
    prev_fid = 0.0

    for layer_num in 1:actual_max_layers
        # Get layer indices: custom, staircase, or brick-wall
        local layer_inds::Vector{Tuple{Int,Int}}

        if using_custom_inds
            layer_inds = circuit_inds[layer_num]
        elseif topology == :staircase
            layer_inds = [(i, i+1) for i in 1:N-1]
        else  # :brickwall
            parity = (layer_num - 1) % 2
            layer_inds = Tuple{Int,Int}[]
            start = parity == 0 ? 1 : 2
            for i in start:2:N-1
                push!(layer_inds, (i, i+1))
            end
        end

        if isempty(layer_inds)
            continue
        end

        # Check if this layer has only adjacent gates
        layer_is_adjacent_only = all(((i, j),) -> j == i + 1, layer_inds)

        # Initialize new layer gates
        local new_layer_gates::Vector{Matrix{ComplexF64}}

        can_use_analytical = use_analytical_init && layer_is_adjacent_only && !using_custom_inds

        if can_use_analytical && layer_num > 1
            mps_disentangled = _apply_inverse_circuit(mps_target, result_circuit, result_circuit_inds, max_chi, max_trunc_err)

            if topology == :staircase
                analytical_gates, _ = _get_staircase_layer_init(mps_disentangled)
            else
                parity = (layer_num - 1) % 2
                analytical_gates, _ = _get_analytical_layer_init(mps_disentangled, parity)
            end

            if length(analytical_gates) == length(layer_inds)
                new_layer_gates = analytical_gates
            else
                new_layer_gates = [randU(0.1, 2) for _ in layer_inds]
            end
        elseif can_use_analytical && layer_num == 1
            if topology == :staircase
                analytical_gates, _ = _get_staircase_layer_init(mps_target)
            else
                analytical_gates, _ = _get_analytical_layer_init(mps_target, 0)
            end

            if length(analytical_gates) == length(layer_inds)
                new_layer_gates = analytical_gates
            else
                new_layer_gates = [randU(0.1, 2) for _ in layer_inds]
            end
        else
            new_layer_gates = [randU(0.1, 2) for _ in layer_inds]
        end

        # Append new layer to circuit
        push!(result_circuit, new_layer_gates)
        push!(result_circuit_inds, layer_inds)

        # Optimize ALL layers jointly
        hp = HyperParams(N=N, max_chi=max_chi, max_trunc_err=max_trunc_err,
                         n_sweeps=sweeps_per_layer, n_layer_sweeps=5, slowdown=0.1,
                         converge_threshold=1e-8, print_every=0)

        result_circuit, result_circuit_inds = optimize_layered(hp, mps_target, result_circuit, result_circuit_inds; print_info=false)

        # Compute fidelity after optimization
        current_fid = _compute_fidelity_fast(mps_target, result_circuit, result_circuit_inds, N, max_chi, max_trunc_err)
        push!(history, current_fid)

        if verbose
            improvement = layer_num > 1 ? current_fid - prev_fid : current_fid
            println("  Layer $layer_num: fidelity=$(round(current_fid, digits=6)), delta=$(round(improvement, digits=6))")
        end

        # Check if target reached
        if current_fid >= target_fidelity && current_fid >= 0.9999
            if verbose
                println("  Reached excellent fidelity at layer $layer_num")
            end
            break
        end

        # Early stopping: check for convergence
        if length(history) >= 4
            recent_improvement = history[end] - history[end-3]
            if recent_improvement < 1e-8 && layer_num >= 4 && current_fid >= target_fidelity
                if verbose
                    println("  Stopping early: minimal improvement and target reached")
                end
                break
            end
        end

        prev_fid = current_fid
    end

    if verbose
        n_gates = sum(length.(result_circuit); init=0)
        println("  Final: fidelity=$(round(current_fid, digits=6)), depth=$(length(result_circuit)), gates=$n_gates")
    end

    return result_circuit, result_circuit_inds, history
end

"""
    optimize_existing_circuit(mps_target, circuit, circuit_inds; kwargs...)
        -> (circuit, circuit_inds, fidelity)

Optimize an existing circuit to better approximate a target MPS.
"""
function optimize_existing_circuit(mps_target::FiniteMPS{ComplexF64},
                                   circuit::Vector{Vector{Matrix{ComplexF64}}},
                                   circuit_inds::Vector{Vector{Tuple{Int,Int}}};
                                   n_sweeps::Int=100,
                                   max_chi::Int=64,
                                   max_trunc_err::Float64=1e-10,
                                   verbose::Bool=false)
    N = length(mps_target.tensors)

    # Normalize target
    mps_norm = FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors))
    normalize_mps!(mps_norm)

    # Initial fidelity
    init_fid = _compute_fidelity_fast(mps_norm, circuit, circuit_inds, N, max_chi, max_trunc_err)

    if verbose
        println("Optimizing existing circuit:")
        println("  Initial fidelity: $(round(init_fid, digits=6))")
    end

    # Set up optimization
    hp = HyperParams(N=N, max_chi=max_chi, max_trunc_err=max_trunc_err,
                     n_sweeps=n_sweeps, n_layer_sweeps=5, slowdown=0.1,
                     converge_threshold=1e-8, print_every=verbose ? 20 : 0)

    # Optimize
    circuit_opt, circuit_inds_opt = optimize_layered(hp, mps_norm, deepcopy(circuit), deepcopy(circuit_inds); print_info=verbose)

    # Final fidelity
    final_fid = _compute_fidelity_fast(mps_norm, circuit_opt, circuit_inds_opt, N, max_chi, max_trunc_err)

    if verbose
        println("  Final fidelity: $(round(final_fid, digits=6))")
        println("  Improvement: $(round(final_fid - init_fid, digits=6))")
    end

    return circuit_opt, circuit_inds_opt, final_fid
end

"""
    add_layer_and_optimize(mps_target, circuit, circuit_inds; kwargs...)
        -> (new_circuit, new_circuit_inds, fidelity)

Add a new layer to an existing circuit and optimize all layers jointly.
Single step of the iterative decomposition for manual control.
"""
function add_layer_and_optimize(mps_target::FiniteMPS{ComplexF64},
                                circuit::Vector{Vector{Matrix{ComplexF64}}},
                                circuit_inds::Vector{Vector{Tuple{Int,Int}}};
                                sweeps::Int=50,
                                max_chi::Int=64,
                                max_trunc_err::Float64=1e-10,
                                use_analytical_init::Bool=true,
                                topology::Symbol=:staircase,
                                layer_inds::Union{Nothing, Vector{Tuple{Int,Int}}}=nothing)
    N = length(mps_target.tensors)

    @assert topology in (:staircase, :brickwall) "topology must be :staircase or :brickwall, got $topology"

    # Normalize target
    mps_norm = FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors))
    normalize_mps!(mps_norm)

    # Determine layer indices
    local actual_layer_inds::Vector{Tuple{Int,Int}}

    if layer_inds !== nothing
        actual_layer_inds = layer_inds
    elseif topology == :staircase
        actual_layer_inds = [(i, i+1) for i in 1:N-1]
    else  # :brickwall
        layer_num = length(circuit) + 1
        parity = (layer_num - 1) % 2
        actual_layer_inds = Tuple{Int,Int}[]
        start = parity == 0 ? 1 : 2
        for i in start:2:N-1
            push!(actual_layer_inds, (i, i+1))
        end
    end

    if isempty(actual_layer_inds)
        fid = isempty(circuit) ? 0.0 : _compute_fidelity_fast(mps_norm, circuit, circuit_inds, N, max_chi, max_trunc_err)
        return circuit, circuit_inds, fid
    end

    # Check if this layer has only adjacent gates
    layer_is_adjacent_only = all(((i, j),) -> j == i + 1, actual_layer_inds)

    # Initialize new layer gates
    local new_layer_gates::Vector{Matrix{ComplexF64}}

    can_use_analytical = use_analytical_init && layer_is_adjacent_only && layer_inds === nothing

    if can_use_analytical && !isempty(circuit)
        mps_disentangled = _apply_inverse_circuit(mps_norm, circuit, circuit_inds, max_chi, max_trunc_err)

        if topology == :staircase
            analytical_gates, _ = _get_staircase_layer_init(mps_disentangled)
        else
            parity = length(circuit) % 2
            analytical_gates, _ = _get_analytical_layer_init(mps_disentangled, parity)
        end

        if length(analytical_gates) == length(actual_layer_inds)
            new_layer_gates = analytical_gates
        else
            new_layer_gates = [randU(0.1, 2) for _ in actual_layer_inds]
        end
    elseif can_use_analytical
        # First layer
        if topology == :staircase
            analytical_gates, _ = _get_staircase_layer_init(mps_norm)
        else
            analytical_gates, _ = _get_analytical_layer_init(mps_norm, 0)
        end

        if length(analytical_gates) == length(actual_layer_inds)
            new_layer_gates = analytical_gates
        else
            new_layer_gates = [randU(0.1, 2) for _ in actual_layer_inds]
        end
    else
        new_layer_gates = [randU(0.1, 2) for _ in actual_layer_inds]
    end

    # Copy and extend circuit
    new_circuit = deepcopy(circuit)
    new_circuit_inds = deepcopy(circuit_inds)
    push!(new_circuit, new_layer_gates)
    push!(new_circuit_inds, actual_layer_inds)

    # Optimize all layers
    hp = HyperParams(N=N, max_chi=max_chi, max_trunc_err=max_trunc_err,
                     n_sweeps=sweeps, n_layer_sweeps=5, slowdown=0.1,
                     converge_threshold=1e-8, print_every=0)

    new_circuit, new_circuit_inds = optimize_layered(hp, mps_norm, new_circuit, new_circuit_inds; print_info=false)

    # Compute fidelity
    fid = _compute_fidelity_fast(mps_norm, new_circuit, new_circuit_inds, N, max_chi, max_trunc_err)

    return new_circuit, new_circuit_inds, fid
end
