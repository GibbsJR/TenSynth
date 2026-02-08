# Layer addition protocols with multiple initialization strategies
# Adapted from MPS2Circuit/src/protocols/layer_addition.jl
# Key changes: Vector{Array{ComplexF64,3}} → FiniteMPS{ComplexF64},
#              DecompositionLogger → no-op stubs via Union{Nothing}

using LinearAlgebra

# ==============================================================================
# Layer Initialization Strategy
# ==============================================================================

"""
    LayerInitStrategy

Strategy for initializing new layer gates.

# Values
- `IDENTITY`: Initialize with identity gates
- `RANDOM`: Initialize with small random perturbations
- `PREOPTIMIZED`: Pre-optimize layer against residual state V†|ψ_target⟩
"""
@enum LayerInitStrategy begin
    IDENTITY
    RANDOM
    PREOPTIMIZED
end

"""
    init_strategy_from_symbol(sym::Symbol) -> LayerInitStrategy

Convert a symbol to LayerInitStrategy enum.
"""
function init_strategy_from_symbol(sym::Symbol)::LayerInitStrategy
    if sym == :identity
        return IDENTITY
    elseif sym == :random
        return RANDOM
    elseif sym == :preoptimized
        return PREOPTIMIZED
    else
        throw(ArgumentError("Unknown initialization strategy: $sym. Use :identity, :random, or :preoptimized"))
    end
end

# ==============================================================================
# Gate Initialization Functions
# ==============================================================================

"""
    initialize_layer_gates(layer_inds, strategy, random_strength) -> Vector{Matrix{ComplexF64}}

Create initial gates for a new layer.
"""
function initialize_layer_gates(
    layer_inds::Vector{Tuple{Int,Int}},
    strategy::LayerInitStrategy,
    random_strength::Float64
)::Vector{Matrix{ComplexF64}}
    n_gates = length(layer_inds)

    if strategy == IDENTITY
        return [Matrix{ComplexF64}(I, 4, 4) for _ in 1:n_gates]
    elseif strategy == RANDOM
        return [randU(random_strength, 2) for _ in 1:n_gates]
    elseif strategy == PREOPTIMIZED
        return [Matrix{ComplexF64}(I, 4, 4) for _ in 1:n_gates]
    else
        error("Unknown initialization strategy: $strategy")
    end
end

# ==============================================================================
# Pre-optimization Functions
# ==============================================================================

"""
    preoptimize_new_layer(mps_residual, layer_inds; kwargs...) -> Vector{Matrix{ComplexF64}}

Pre-optimize a new layer against the residual state V†|ψ_target⟩.
Maximizes ⟨0...0|U_new|mps_residual⟩.
"""
function preoptimize_new_layer(
    mps_residual::FiniteMPS{ComplexF64},
    layer_inds::Vector{Tuple{Int,Int}};
    use_chi2_truncation::Bool=false,
    n_sweeps::Int=10,
    slowdown::Float64=0.1,
    max_chi::Int=64,
    max_trunc_err::Float64=1e-10
)::Vector{Matrix{ComplexF64}}
    N = length(mps_residual.tensors)

    if isempty(layer_inds)
        return Matrix{ComplexF64}[]
    end

    # Optionally truncate to chi=2
    mps_work = if use_chi2_truncation
        truncated, _ = truncate_to_chi2(mps_residual)
        truncated
    else
        FiniteMPS{ComplexF64}(deepcopy(mps_residual.tensors))
    end

    # Normalize
    normalize_mps!(mps_work)

    # Target is |0...0⟩
    mps_zero = zeroMPS(N)

    # Initialize layer with identity gates
    layer_gates = [Matrix{ComplexF64}(I, 4, 4) for _ in layer_inds]

    # Optimize the layer to maximize ⟨0|U_layer|mps_residual⟩
    optimized_gates = optimize_single_layer(
        mps_zero, mps_work, layer_gates, layer_inds;
        n_sweeps=n_sweeps, slowdown=slowdown,
        max_chi=max_chi, max_trunc_err=max_trunc_err
    )

    return optimized_gates
end

# ==============================================================================
# Layer Addition with Optimization
# ==============================================================================

"""
    add_optimized_layer!(circuit, circuit_inds, mps_target, layer_inds; kwargs...) -> Float64

Add a new layer to the circuit and optimize all layers jointly.

1. Compute residual: V†|ψ_target⟩
2. Initialize new layer gates using specified strategy
3. If PREOPTIMIZED: pre-optimize the new layer against the residual
4. Prepend the new layer to the circuit
5. Optimize ALL gates using optimize_circuit_direct
6. Return the achieved fidelity
"""
function add_optimized_layer!(
    circuit::Vector{Vector{Matrix{ComplexF64}}},
    circuit_inds::Vector{Vector{Tuple{Int,Int}}},
    mps_target::FiniteMPS{ComplexF64},
    layer_inds::Vector{Tuple{Int,Int}};
    init_strategy::LayerInitStrategy=PREOPTIMIZED,
    random_strength::Float64=0.01,
    use_chi2_truncation::Bool=false,
    max_chi::Int=64,
    max_trunc_err::Float64=1e-10,
    n_sweeps::Int=50,
    n_layer_sweeps::Int=5,
    slowdown::Float64=0.1,
    converge_threshold::Float64=1e-8,
    verbose::Bool=false
)::Float64
    N = length(mps_target.tensors)

    if isempty(layer_inds)
        if isempty(circuit)
            return 0.0
        end
        mps_circuit = Ansatz2MPS(N, circuit, circuit_inds, max_chi, max_trunc_err)
        return fidelity(mps_target, mps_circuit)
    end

    # Normalize target
    mps_targ_norm = FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors))
    normalize_mps!(mps_targ_norm)

    # Initialize new layer gates based on strategy
    local new_layer_gates::Vector{Matrix{ComplexF64}}

    if init_strategy == PREOPTIMIZED
        # Compute residual: V†|ψ_target⟩
        if isempty(circuit)
            mps_residual = FiniteMPS{ComplexF64}(deepcopy(mps_targ_norm.tensors))
        else
            mps_residual = _apply_inverse_circuit(mps_targ_norm, circuit, circuit_inds, max_chi, max_trunc_err)
        end

        # Pre-optimize new layer
        preopt_gates = preoptimize_new_layer(
            mps_residual, layer_inds;
            use_chi2_truncation=use_chi2_truncation,
            n_sweeps=n_layer_sweeps,
            slowdown=slowdown,
            max_chi=max_chi,
            max_trunc_err=max_trunc_err
        )

        # Invert the pre-optimized gates before prepending
        new_layer_gates = [Matrix{ComplexF64}(gate') for gate in preopt_gates]

        if verbose
            println("  Pre-optimized new layer with $(length(layer_inds)) gates (inverted for prepending)")
        end
    else
        # IDENTITY or RANDOM initialization
        new_layer_gates = initialize_layer_gates(layer_inds, init_strategy, random_strength)

        if verbose
            strategy_name = init_strategy == IDENTITY ? "identity" : "random"
            println("  Initialized new layer with $strategy_name gates")
        end
    end

    # Prepend new layer to circuit
    pushfirst!(circuit, new_layer_gates)
    pushfirst!(circuit_inds, layer_inds)

    # Optimize ALL layers jointly using direct optimization
    opt_circuit, _ = optimize_circuit_direct(
        mps_targ_norm, circuit, circuit_inds;
        max_chi=max_chi,
        max_trunc_err=max_trunc_err,
        n_sweeps=n_sweeps,
        n_layer_sweeps=n_layer_sweeps,
        slowdown=slowdown,
        converge_threshold=converge_threshold,
        verbose=verbose
    )

    # Update circuit in place with optimized gates
    for i in 1:length(circuit)
        circuit[i] = opt_circuit[i]
    end

    # Compute final fidelity
    mps_circuit = Ansatz2MPS(N, circuit, circuit_inds, max_chi, max_trunc_err)
    final_fid = fidelity(mps_targ_norm, mps_circuit)

    if verbose
        println("  Layer added. New fidelity: $(round(final_fid, digits=6))")
    end

    return final_fid
end

"""
    iterative_layer_addition(mps_target, topology; kwargs...)
        -> (circuit, circuit_inds, fidelity_history)

Perform iterative layer addition decomposition.

Adds layers one at a time using the specified topology and initialization strategy,
optimizing all layers jointly after each addition.
"""
function iterative_layer_addition(
    mps_target::FiniteMPS{ComplexF64},
    topology::LayerTopology;
    target_fidelity::Float64=0.95,
    max_layers::Int=10,
    init_strategy::LayerInitStrategy=PREOPTIMIZED,
    random_strength::Float64=0.01,
    use_chi2_truncation::Bool=false,
    max_chi::Int=64,
    max_trunc_err::Float64=1e-10,
    n_sweeps::Int=50,
    n_layer_sweeps::Int=5,
    slowdown::Float64=0.1,
    converge_threshold::Float64=1e-8,
    verbose::Bool=false
)::Tuple{Vector{Vector{Matrix{ComplexF64}}}, Vector{Vector{Tuple{Int,Int}}}, Vector{Float64}}
    N = length(mps_target.tensors)

    if verbose
        strategy_name = init_strategy == IDENTITY ? "identity" :
                       init_strategy == RANDOM ? "random" : "preoptimized"
        println("Iterative layer addition decomposition:")
        println("  Qubits: $N, Target fidelity: $target_fidelity")
        println("  Topology: $topology, Init strategy: $strategy_name")
        println("  Max layers: $max_layers")
    end

    # Initialize empty circuit
    circuit = Vector{Vector{Matrix{ComplexF64}}}()
    circuit_inds = Vector{Vector{Tuple{Int,Int}}}()
    fidelity_history = Float64[]

    current_fid = 0.0
    prev_fid = 0.0

    for layer_num in 1:max_layers
        # Generate layer indices based on topology
        layer_inds = generate_layer_indices(N, topology, layer_num)

        if isempty(layer_inds)
            continue
        end

        # Add and optimize the layer
        current_fid = add_optimized_layer!(
            circuit, circuit_inds, mps_target, layer_inds;
            init_strategy=init_strategy,
            random_strength=random_strength,
            use_chi2_truncation=use_chi2_truncation,
            max_chi=max_chi,
            max_trunc_err=max_trunc_err,
            n_sweeps=n_sweeps,
            n_layer_sweeps=n_layer_sweeps,
            slowdown=slowdown,
            converge_threshold=converge_threshold,
            verbose=verbose
        )

        push!(fidelity_history, current_fid)

        if verbose
            improvement = layer_num > 1 ? current_fid - prev_fid : current_fid
            println("  Layer $layer_num: fidelity=$(round(current_fid, digits=6)), delta=$(round(improvement, digits=6))")
        end

        # Check if target reached
        if current_fid >= target_fidelity && current_fid >= 0.999
            if verbose
                println("  Reached excellent fidelity at layer $layer_num")
            end
            break
        end

        # Early stopping
        if length(fidelity_history) >= 4
            recent_improvement = fidelity_history[end] - fidelity_history[end-3]
            if recent_improvement < 1e-8 && current_fid >= target_fidelity
                if verbose
                    println("  Stopping early: minimal improvement and target reached")
                end
                break
            end
        end

        prev_fid = current_fid
    end

    if verbose
        n_gates = sum(length.(circuit); init=0)
        println("  Final: fidelity=$(round(current_fid, digits=6)), depth=$(length(circuit)), gates=$n_gates")
    end

    return circuit, circuit_inds, fidelity_history
end
