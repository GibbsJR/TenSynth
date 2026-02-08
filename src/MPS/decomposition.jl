# High-level decomposition API for MPS → Quantum Circuit
# Adapted from MPS2Circuit/src/decomposition.jl
# Key changes: Vector{Array{ComplexF64,3}} → FiniteMPS{ComplexF64},
#              Riemannian methods EXCLUDED, DecompositionLogger → no-op stubs,
#              optimize(hp, nothing, nothing, ...) → optimize_layered(hp, ...)

using LinearAlgebra

# ==============================================================================
# Result Types
# ==============================================================================

"""
    DecompositionResult

Result of decomposing an MPS into a quantum circuit.

# Fields
- `circuit::Vector{Vector{Matrix{ComplexF64}}}`: Layered gates
- `circuit_inds::Vector{Vector{Tuple{Int,Int}}}`: Site indices for each gate
- `fidelity::Float64`: Final fidelity |⟨ψ_circuit|ψ_target⟩|²
- `depth::Int`: Circuit depth (number of layers)
- `n_gates::Int`: Total number of 2-qubit gates
- `n_qubits::Int`: Number of qubits
- `method::Symbol`: Decomposition method used
- `optimization_history::Vector{Float64}`: Fidelity at each step
- `max_gates_per_bond::Int`: Maximum number of SU(4) gates on any single bond
"""
struct DecompositionResult
    circuit::Vector{Vector{Matrix{ComplexF64}}}
    circuit_inds::Vector{Vector{Tuple{Int,Int}}}
    fidelity::Float64
    depth::Int
    n_gates::Int
    n_qubits::Int
    method::Symbol
    optimization_history::Vector{Float64}
    max_gates_per_bond::Int
end

function Base.show(io::IO, result::DecompositionResult)
    print(io, "DecompositionResult(fidelity=$(round(result.fidelity, digits=6)), ",
          "depth=$(result.depth), n_gates=$(result.n_gates), ",
          "max_gates_per_bond=$(result.max_gates_per_bond), method=:$(result.method))")
end

function Base.show(io::IO, ::MIME"text/plain", result::DecompositionResult)
    println(io, "DecompositionResult:")
    println(io, "  Fidelity:         $(round(result.fidelity, digits=8))")
    println(io, "  Depth:            $(result.depth) layers")
    println(io, "  Gates:            $(result.n_gates) two-qubit gates")
    println(io, "  Max gates/bond:   $(result.max_gates_per_bond)")
    println(io, "  Qubits:           $(result.n_qubits)")
    println(io, "  Method:           :$(result.method)")
    if !isempty(result.optimization_history)
        println(io, "  History:          $(length(result.optimization_history)) optimization steps")
    end
end

# ==============================================================================
# Helper Functions
# ==============================================================================

"""
    _compute_max_gates_per_bond(circuit_inds) -> Int

Compute the maximum number of SU(4) gates applied to any single bond.
"""
function _compute_max_gates_per_bond(circuit_inds::Vector{Vector{Tuple{Int,Int}}})::Int
    if isempty(circuit_inds)
        return 0
    end

    bond_counts = Dict{Tuple{Int,Int}, Int}()

    for layer_inds in circuit_inds
        for (i, j) in layer_inds
            bond = i < j ? (i, j) : (j, i)
            bond_counts[bond] = get(bond_counts, bond, 0) + 1
        end
    end

    return isempty(bond_counts) ? 0 : maximum(values(bond_counts))
end

"""
    _compute_circuit_fidelity(mps_target, circuit, circuit_inds, N, max_chi, max_trunc_err) -> Float64

Compute fidelity between target MPS and circuit applied to |0...0⟩.
"""
function _compute_circuit_fidelity(mps_target::FiniteMPS{ComplexF64},
                                   circuit::Vector{Vector{Matrix{ComplexF64}}},
                                   circuit_inds::Vector{Vector{Tuple{Int,Int}}},
                                   N::Int, max_chi::Int, max_trunc_err::Float64)::Float64
    mps_circuit = Ansatz2MPS(N, circuit, circuit_inds, max_chi, max_trunc_err)
    return fidelity(mps_target, mps_circuit)
end

"""
    _create_brick_layer_inds(N, parity) -> Vector{Tuple{Int,Int}}
"""
function _create_brick_layer_inds(N::Int, parity::Int)::Vector{Tuple{Int,Int}}
    inds = Tuple{Int,Int}[]
    start = parity == 0 ? 1 : 2
    for i in start:2:N-1
        push!(inds, (i, i+1))
    end
    return inds
end

"""
    _create_initial_ansatz(N, n_layers) -> (circuit, circuit_inds)

Create an initial circuit ansatz with identity gates in a brick-wall pattern.
"""
function _create_initial_ansatz(N::Int, n_layers::Int)
    circuit = Vector{Vector{Matrix{ComplexF64}}}()
    circuit_inds = Vector{Vector{Tuple{Int,Int}}}()

    for layer in 1:n_layers
        parity = (layer - 1) % 2
        layer_inds = _create_brick_layer_inds(N, parity)

        if !isempty(layer_inds)
            layer_gates = [Matrix{ComplexF64}(I, 4, 4) for _ in layer_inds]
            push!(circuit, layer_gates)
            push!(circuit_inds, layer_inds)
        end
    end

    return circuit, circuit_inds
end

"""
    _create_random_ansatz(N, n_layers) -> (circuit, circuit_inds)

Create an initial circuit ansatz with random unitary gates in a brick-wall pattern.
"""
function _create_random_ansatz(N::Int, n_layers::Int)
    circuit = Vector{Vector{Matrix{ComplexF64}}}()
    circuit_inds = Vector{Vector{Tuple{Int,Int}}}()

    for layer in 1:n_layers
        parity = (layer - 1) % 2
        layer_inds = _create_brick_layer_inds(N, parity)

        if !isempty(layer_inds)
            layer_gates = [randU(1.0, 2) for _ in layer_inds]
            push!(circuit, layer_gates)
            push!(circuit_inds, layer_inds)
        end
    end

    return circuit, circuit_inds
end

# ==============================================================================
# Decomposition Methods
# ==============================================================================

"""
    _decompose_analytical(mps_target, max_layers, max_chi, max_trunc_err, verbose)
        -> DecompositionResult

Analytical (SVD-based) decomposition with optimization polish.
"""
function _decompose_analytical(mps_target::FiniteMPS{ComplexF64},
                               max_layers::Int,
                               max_chi::Int,
                               max_trunc_err::Float64,
                               verbose::Bool)::DecompositionResult
    N = length(mps_target.tensors)

    if verbose
        println("Analytical decomposition:")
        println("  Target qubits: $N")
        println("  Max layers: $max_layers")
    end

    # Use the analytical decomposition from protocols/analytical.jl
    circuit, circuit_inds, history = analytical_decomposition(
        mps_target,
        max_layers=max_layers,
        max_chi=max_chi,
        max_trunc_err=max_trunc_err,
        verbose=verbose
    )

    # Polish with a few optimization sweeps
    if !isempty(circuit)
        hp = HyperParams(N=N, max_chi=max_chi, max_trunc_err=max_trunc_err,
                         n_sweeps=10, n_layer_sweeps=5, slowdown=0.1,
                         print_every=0)
        circuit, circuit_inds = optimize_layered(hp, mps_target, circuit, circuit_inds; print_info=false)
    end

    # Compute final fidelity
    final_fid = isempty(circuit) ? 0.0 : _compute_circuit_fidelity(mps_target, circuit, circuit_inds, N, max_chi, max_trunc_err)
    push!(history, final_fid)

    n_gates = sum(length.(circuit); init=0)
    depth = length(circuit)

    if verbose
        println("  Final fidelity: $(round(final_fid, digits=6))")
        println("  Depth: $depth, Gates: $n_gates")
    end

    max_gpb = _compute_max_gates_per_bond(circuit_inds)
    return DecompositionResult(circuit, circuit_inds, final_fid, depth, n_gates, N, :analytical, history, max_gpb)
end

"""
    _decompose_direct(mps_target, n_layers, target_fidelity, max_chi, max_trunc_err, n_sweeps, verbose;
                      circuit_inds=nothing) -> DecompositionResult

Direct optimization decomposition with fixed-depth circuit.
"""
function _decompose_direct(mps_target::FiniteMPS{ComplexF64},
                           n_layers::Int,
                           target_fidelity::Float64,
                           max_chi::Int,
                           max_trunc_err::Float64,
                           n_sweeps::Int,
                           verbose::Bool;
                           circuit_inds::Union{Nothing, Vector{Vector{Tuple{Int,Int}}}}=nothing)::DecompositionResult
    N = length(mps_target.tensors)

    # Determine circuit structure
    local actual_circuit_inds::Vector{Vector{Tuple{Int,Int}}}
    local circuit::Vector{Vector{Matrix{ComplexF64}}}

    if circuit_inds === nothing
        if verbose
            println("Direct optimization decomposition:")
            println("  Target qubits: $N, Layers: $n_layers")
            println("  Circuit topology: brick-wall (nearest-neighbor)")
            println("  Target fidelity: $target_fidelity")
        end
        circuit, actual_circuit_inds = _create_random_ansatz(N, n_layers)
    else
        actual_circuit_inds = circuit_inds
        if verbose
            has_long_range = any(layer -> any(((i, j),) -> j > i + 1, layer), actual_circuit_inds)
            println("Direct optimization decomposition:")
            println("  Target qubits: $N, Layers: $(length(actual_circuit_inds))")
            println("  Circuit topology: user-specified $(has_long_range ? "(with long-range gates)" : "(nearest-neighbor)")")
            println("  Target fidelity: $target_fidelity")
        end
        circuit = [[randU(1.0, 2) for _ in layer] for layer in actual_circuit_inds]
    end

    if isempty(circuit)
        return DecompositionResult(circuit, actual_circuit_inds, 0.0, 0, 0, N, :direct, Float64[], 0)
    end

    history = Float64[]

    # Compute initial fidelity
    init_fid = _compute_circuit_fidelity(mps_target, circuit, actual_circuit_inds, N, max_chi, max_trunc_err)
    push!(history, init_fid)

    if verbose
        println("  Initial fidelity: $(round(init_fid, digits=6))")
    end

    # Run optimization
    hp = HyperParams(N=N, max_chi=max_chi, max_trunc_err=max_trunc_err,
                     n_sweeps=n_sweeps, n_layer_sweeps=5, slowdown=0.1,
                     converge_threshold=1e-8, print_every=verbose ? 10 : 0)

    circuit, actual_circuit_inds = optimize_layered(hp, mps_target, circuit, actual_circuit_inds; print_info=verbose)

    # Compute final fidelity
    final_fid = _compute_circuit_fidelity(mps_target, circuit, actual_circuit_inds, N, max_chi, max_trunc_err)
    push!(history, final_fid)

    n_gates = sum(length.(circuit))
    depth = length(circuit)

    if verbose
        println("  Final fidelity: $(round(final_fid, digits=6))")
        println("  Depth: $depth, Gates: $n_gates")
    end

    max_gpb = _compute_max_gates_per_bond(actual_circuit_inds)
    return DecompositionResult(circuit, actual_circuit_inds, final_fid, depth, n_gates, N, :direct, history, max_gpb)
end

"""
    _decompose_iterative(mps_target, max_layers, target_fidelity, max_chi, max_trunc_err,
                         sweeps_per_layer, verbose; circuit_inds=nothing) -> DecompositionResult

Iterative (Iter[DiOall]) decomposition — highest quality method.
"""
function _decompose_iterative(mps_target::FiniteMPS{ComplexF64},
                              max_layers::Int,
                              target_fidelity::Float64,
                              max_chi::Int,
                              max_trunc_err::Float64,
                              sweeps_per_layer::Int,
                              verbose::Bool;
                              circuit_inds::Union{Nothing, Vector{Vector{Tuple{Int,Int}}}}=nothing)::DecompositionResult
    N = length(mps_target.tensors)

    circuit, result_circuit_inds, history = iterative_decomposition(
        mps_target,
        target_fidelity=target_fidelity,
        max_layers=max_layers,
        sweeps_per_layer=sweeps_per_layer,
        max_chi=max_chi,
        max_trunc_err=max_trunc_err,
        use_analytical_init=(circuit_inds === nothing),
        verbose=verbose,
        circuit_inds=circuit_inds
    )

    # Compute final fidelity
    final_fid = isempty(circuit) ? 0.0 : _compute_circuit_fidelity(mps_target, circuit, result_circuit_inds, N, max_chi, max_trunc_err)

    n_gates = sum(length.(circuit); init=0)
    depth = length(circuit)

    max_gpb = _compute_max_gates_per_bond(result_circuit_inds)
    return DecompositionResult(circuit, result_circuit_inds, final_fid, depth, n_gates, N, :iterative, history, max_gpb)
end

# ==============================================================================
# Layer Addition Protocol Methods
# ==============================================================================

"""
    _decompose_with_layer_addition(mps_target, max_layers, target_fidelity, ...) -> DecompositionResult

Iterative decomposition using the layer addition protocol.
"""
function _decompose_with_layer_addition(
    mps_target::FiniteMPS{ComplexF64},
    max_layers::Int,
    target_fidelity::Float64,
    max_chi::Int,
    max_trunc_err::Float64,
    n_sweeps::Int,
    converge_threshold::Float64,
    verbose::Bool,
    topology::LayerTopology,
    init_strategy::LayerInitStrategy,
    random_strength::Float64,
    use_chi2_truncation::Bool,
    circuit_inds::Union{Nothing, Vector{Vector{Tuple{Int,Int}}}},
    method::Symbol
)::DecompositionResult
    N = length(mps_target.tensors)

    use_custom_inds = circuit_inds !== nothing

    if verbose
        topo_name = use_custom_inds ? "custom" : string(topology)
        init_name = init_strategy == IDENTITY ? "identity" :
                   init_strategy == RANDOM ? "random" : "preoptimized"
        println("Layer addition decomposition:")
        println("  Topology: $topo_name, Init: $init_name")
        println("  Max layers: $max_layers, Target fidelity: $target_fidelity")
    end

    local result_circuit::Vector{Vector{Matrix{ComplexF64}}}
    local result_circuit_inds::Vector{Vector{Tuple{Int,Int}}}
    local history::Vector{Float64}

    if use_custom_inds
        result_circuit = Vector{Vector{Matrix{ComplexF64}}}()
        result_circuit_inds = Vector{Vector{Tuple{Int,Int}}}()
        history = Float64[]

        current_fid = 0.0
        for (layer_num, layer_inds) in enumerate(circuit_inds)
            if layer_num > max_layers
                break
            end

            current_fid = add_optimized_layer!(
                result_circuit, result_circuit_inds, mps_target, layer_inds;
                init_strategy=init_strategy,
                random_strength=random_strength,
                use_chi2_truncation=use_chi2_truncation,
                max_chi=max_chi,
                max_trunc_err=max_trunc_err,
                n_sweeps=n_sweeps,
                converge_threshold=converge_threshold,
                verbose=verbose
            )
            push!(history, current_fid)

            if verbose
                println("  Layer $layer_num: fidelity=$(round(current_fid, digits=6))")
            end

            if current_fid >= target_fidelity && current_fid >= 0.999
                if verbose
                    println("  Reached target fidelity")
                end
                break
            end
        end
    else
        result_circuit, result_circuit_inds, history = iterative_layer_addition(
            mps_target, topology;
            target_fidelity=target_fidelity,
            max_layers=max_layers,
            init_strategy=init_strategy,
            random_strength=random_strength,
            use_chi2_truncation=use_chi2_truncation,
            max_chi=max_chi,
            max_trunc_err=max_trunc_err,
            n_sweeps=n_sweeps,
            converge_threshold=converge_threshold,
            verbose=verbose
        )
    end

    # Compute final fidelity
    final_fid = isempty(history) ? 0.0 : history[end]
    n_gates = sum(length.(result_circuit); init=0)
    depth = length(result_circuit)

    if verbose
        println("  Final: fidelity=$(round(final_fid, digits=6)), depth=$depth, gates=$n_gates")
    end

    max_gpb = _compute_max_gates_per_bond(result_circuit_inds)
    return DecompositionResult(result_circuit, result_circuit_inds, final_fid, depth, n_gates, N, method, history, max_gpb)
end

"""
    _decompose_fixed_depth(mps_target, max_layers, ...) -> DecompositionResult

Fixed-depth decomposition with topology-based circuit structure.
"""
function _decompose_fixed_depth(
    mps_target::FiniteMPS{ComplexF64},
    max_layers::Int,
    target_fidelity::Float64,
    max_chi::Int,
    max_trunc_err::Float64,
    n_sweeps::Int,
    converge_threshold::Float64,
    verbose::Bool,
    topology::LayerTopology,
    init_strategy::LayerInitStrategy,
    random_strength::Float64,
    circuit_inds::Union{Nothing, Vector{Vector{Tuple{Int,Int}}}},
    method::Symbol
)::DecompositionResult
    N = length(mps_target.tensors)

    # Determine circuit structure
    local actual_circuit_inds::Vector{Vector{Tuple{Int,Int}}}

    if circuit_inds !== nothing
        actual_circuit_inds = circuit_inds
    else
        actual_circuit_inds = generate_circuit_indices(N, topology, max_layers)
    end

    if verbose
        topo_name = circuit_inds !== nothing ? "custom" : string(topology)
        init_name = init_strategy == IDENTITY ? "identity" :
                   init_strategy == RANDOM ? "random" : "preoptimized"
        println("Fixed-depth decomposition:")
        println("  Topology: $topo_name, Init: $init_name")
        println("  Layers: $(length(actual_circuit_inds)), Target fidelity: $target_fidelity")
    end

    if isempty(actual_circuit_inds)
        return DecompositionResult(Vector{Vector{Matrix{ComplexF64}}}(), actual_circuit_inds,
                                   0.0, 0, 0, N, method, Float64[], 0)
    end

    # Initialize circuit gates based on strategy
    circuit = Vector{Vector{Matrix{ComplexF64}}}()
    for layer_inds in actual_circuit_inds
        layer_gates = initialize_layer_gates(layer_inds, init_strategy, random_strength)
        push!(circuit, layer_gates)
    end

    # Compute initial fidelity
    history = Float64[]
    mps_init = Ansatz2MPS(N, circuit, actual_circuit_inds, max_chi, max_trunc_err)
    init_fid = fidelity(mps_target, mps_init)
    push!(history, init_fid)

    if verbose
        println("  Initial fidelity: $(round(init_fid, digits=6))")
    end

    # Optimize using optimize_circuit_direct
    opt_circuit, opt_history = optimize_circuit_direct(
        mps_target, circuit, actual_circuit_inds;
        max_chi=max_chi,
        max_trunc_err=max_trunc_err,
        n_sweeps=n_sweeps,
        n_layer_sweeps=5,
        slowdown=0.1,
        converge_threshold=converge_threshold,
        verbose=verbose
    )

    if length(opt_history) > 1
        append!(history, opt_history[2:end])
    end

    # Compute final fidelity
    mps_final = Ansatz2MPS(N, opt_circuit, actual_circuit_inds, max_chi, max_trunc_err)
    final_fid = fidelity(mps_target, mps_final)

    n_gates = sum(length.(opt_circuit))
    depth = length(opt_circuit)

    if verbose
        println("  Final fidelity: $(round(final_fid, digits=6))")
        println("  Depth: $depth, Gates: $n_gates")
    end

    max_gpb = _compute_max_gates_per_bond(actual_circuit_inds)
    return DecompositionResult(opt_circuit, actual_circuit_inds, final_fid, depth, n_gates, N, method, history, max_gpb)
end

# ==============================================================================
# Input Validation Helpers
# ==============================================================================

function _validate_mps(mps::FiniteMPS{ComplexF64})
    N = length(mps.tensors)
    N >= 2 || throw(ArgumentError("MPS must have at least 2 sites, got $N"))
    for (i, t) in enumerate(mps.tensors)
        ndims(t) == 3 || throw(ArgumentError("MPS tensor at site $i must have 3 dimensions, got $(ndims(t))"))
    end
end

function _validate_circuit_inds(circuit_inds::Vector{Vector{Tuple{Int,Int}}}, N::Int)
    for (layer_idx, layer) in enumerate(circuit_inds)
        for (gate_idx, (i, j)) in enumerate(layer)
            1 <= i <= N || throw(ArgumentError("Gate ($i,$j) in layer $layer_idx: site $i out of range [1,$N]"))
            1 <= j <= N || throw(ArgumentError("Gate ($i,$j) in layer $layer_idx: site $j out of range [1,$N]"))
            i != j || throw(ArgumentError("Gate ($i,$j) in layer $layer_idx: sites must be distinct"))
        end
    end
end

# ==============================================================================
# Main Entry Point
# ==============================================================================

"""
    decompose(mps; kwargs...) -> DecompositionResult

Decompose an MPS into a quantum circuit.

Main entry point for MPS-to-circuit decomposition. Supports three methods:
- `:analytical` - Fast SVD-based (lower quality)
- `:direct` - Direct optimization (medium quality)
- `:iterative` - Iterative layer addition (best quality, default)

# Arguments
- `mps::FiniteMPS{ComplexF64}`: Target MPS
- `target_fidelity::Float64=0.95`: Target fidelity
- `max_layers::Int=10`: Maximum circuit layers
- `method::Symbol=:iterative`: Decomposition method
- `max_chi::Int=64`: Maximum bond dimension
- `max_trunc_err::Float64=1e-10`: SVD truncation threshold
- `n_sweeps::Int=50`: Optimization sweeps
- `converge_threshold::Float64=1e-8`: Convergence threshold
- `verbose::Bool=false`: Print progress
- `circuit_inds`: Optional user-specified circuit structure
- `topology::Symbol=:brickwork`: Circuit topology (:staircase, :brickwork, :custom)
- `layer_init::Symbol=:preoptimized`: Layer initialization (:identity, :random, :preoptimized)
- `random_strength::Float64=0.01`: Perturbation strength for :random init
- `use_chi2_truncation::Bool=false`: Truncate residual to chi=2 for preoptimization
- `add_layers_iteratively::Bool=true`: Add layers one-by-one vs fixed depth
"""
function decompose(mps::FiniteMPS{ComplexF64};
                   target_fidelity::Float64=0.95,
                   max_layers::Int=10,
                   method::Symbol=:iterative,
                   max_chi::Int=64,
                   max_trunc_err::Float64=1e-10,
                   n_sweeps::Int=50,
                   converge_threshold::Float64=1e-8,
                   verbose::Bool=false,
                   circuit_inds::Union{Nothing, Vector{Vector{Tuple{Int,Int}}}}=nothing,
                   # Layer addition parameters
                   topology::Symbol=:brickwork,
                   layer_init::Symbol=:preoptimized,
                   random_strength::Float64=0.01,
                   use_chi2_truncation::Bool=false,
                   add_layers_iteratively::Bool=true)::DecompositionResult

    # Validate inputs
    _validate_mps(mps)
    N = length(mps.tensors)

    if method ∉ [:analytical, :direct, :iterative]
        throw(ArgumentError("method must be :analytical, :direct, or :iterative, got :$method"))
    end

    if topology ∉ [:staircase, :brickwork, :custom]
        throw(ArgumentError("topology must be :staircase, :brickwork, or :custom, got :$topology"))
    end

    if layer_init ∉ [:identity, :random, :preoptimized]
        throw(ArgumentError("layer_init must be :identity, :random, or :preoptimized, got :$layer_init"))
    end

    if circuit_inds !== nothing
        if method == :analytical
            throw(ArgumentError("Custom circuit_inds not supported for :analytical method"))
        end
        _validate_circuit_inds(circuit_inds, N)
    end

    if topology == :custom && circuit_inds === nothing
        throw(ArgumentError("topology=:custom requires circuit_inds to be provided"))
    end

    if method == :analytical && topology != :brickwork
        throw(ArgumentError(":analytical method does not support custom topology"))
    end

    # Convert symbols to enums for layer addition
    init_strategy = init_strategy_from_symbol(layer_init)
    topo = topology == :custom ? CUSTOM : topology_from_symbol(topology)

    # Normalize the target MPS
    mps_normalized = FiniteMPS{ComplexF64}(deepcopy(mps.tensors))
    normalize_mps!(mps_normalized)

    if verbose
        println("=" ^ 50)
        println("MPS to Circuit Decomposition")
        println("=" ^ 50)
        println("Input MPS: $N qubits, max bond dim $(max_bond_dim(mps_normalized))")
        println("Topology: $topology, Layer init: $layer_init")
        println("Add layers iteratively: $add_layers_iteratively")
    end

    # Dispatch to appropriate method
    result = if method == :analytical
        _decompose_analytical(mps_normalized, max_layers, max_chi, max_trunc_err, verbose)
    elseif add_layers_iteratively
        _decompose_with_layer_addition(mps_normalized, max_layers, target_fidelity, max_chi, max_trunc_err,
                                       n_sweeps, converge_threshold, verbose, topo, init_strategy, random_strength,
                                       use_chi2_truncation, circuit_inds, method)
    else
        _decompose_fixed_depth(mps_normalized, max_layers, target_fidelity, max_chi, max_trunc_err,
                               n_sweeps, converge_threshold, verbose, topo, init_strategy, random_strength,
                               circuit_inds, method)
    end

    if verbose
        println("=" ^ 50)
        println("Decomposition complete!")
        println("=" ^ 50)
    end

    return result
end

# ==============================================================================
# Utility Functions for Working with Results
# ==============================================================================

"""
    apply_decomposition(result; mps_start=nothing, max_chi=64, max_trunc_err=1e-10)
        -> FiniteMPS{ComplexF64}

Apply a decomposition result to get the output MPS.
"""
function apply_decomposition(result::DecompositionResult;
                             mps_start::Union{Nothing, FiniteMPS{ComplexF64}}=nothing,
                             max_chi::Int=64,
                             max_trunc_err::Float64=1e-10)::FiniteMPS{ComplexF64}
    N = result.n_qubits

    if mps_start === nothing
        return Ansatz2MPS(N, result.circuit, result.circuit_inds, max_chi, max_trunc_err)
    else
        return Ansatz2MPS(mps_start, result.circuit, result.circuit_inds, max_chi, max_trunc_err)
    end
end

"""
    circuit_to_flat(result) -> (gates, indices)

Flatten the layered circuit into a single list of gates and indices.
"""
function circuit_to_flat(result::DecompositionResult)
    gates = Matrix{ComplexF64}[]
    inds = Tuple{Int,Int}[]

    for (layer_gates, layer_inds) in zip(result.circuit, result.circuit_inds)
        append!(gates, layer_gates)
        append!(inds, layer_inds)
    end

    return gates, inds
end

"""
    verify_decomposition(mps_target, result; max_chi=128) -> Float64

Verify the fidelity of a decomposition with higher precision.
"""
function verify_decomposition(mps_target::FiniteMPS{ComplexF64},
                              result::DecompositionResult;
                              max_chi::Int=128)::Float64
    N = length(mps_target.tensors)

    # Apply circuit with high precision
    mps_circuit = Ansatz2MPS(N, result.circuit, result.circuit_inds, max_chi, 1e-14)

    # Normalize both for fair comparison
    mps_target_norm = FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors))
    normalize_mps!(mps_target_norm)
    normalize_mps!(mps_circuit)

    return fidelity(mps_target_norm, mps_circuit)
end

# ==============================================================================
# Benchmarking
# ==============================================================================

"""
    BenchmarkResult

Result of benchmarking a decomposition method.
"""
struct BenchmarkResult
    method::Symbol
    fidelity::Float64
    depth::Int
    n_gates::Int
    time::Float64
    n_iterations::Int
    history::Vector{Float64}
end

function Base.show(io::IO, r::BenchmarkResult)
    print(io, "BenchmarkResult(:$(r.method), fid=$(round(r.fidelity, digits=6)), ",
          "depth=$(r.depth), gates=$(r.n_gates), time=$(round(r.time, digits=3))s)")
end

"""
    benchmark_methods(mps; methods=[:analytical, :iterative], kwargs...) -> Vector{BenchmarkResult}

Benchmark different decomposition methods on the same MPS.
"""
function benchmark_methods(mps::FiniteMPS{ComplexF64};
                           methods::Vector{Symbol}=[:analytical, :iterative],
                           n_layers::Int=6,
                           target_fidelity::Float64=0.99,
                           max_chi::Int=64,
                           max_trunc_err::Float64=1e-10,
                           n_sweeps::Int=20,
                           verbose::Bool=true)::Vector{BenchmarkResult}
    results = BenchmarkResult[]
    N = length(mps.tensors)

    if verbose
        println("=" ^ 60)
        println("Benchmarking Decomposition Methods")
        println("=" ^ 60)
        println("MPS: $N qubits, max bond dim $(max_bond_dim(mps))")
        println("Target layers: $n_layers, Target fidelity: $target_fidelity")
        println("-" ^ 60)
    end

    for method in methods
        if verbose
            print("Running $method... ")
            flush(stdout)
        end

        t = @elapsed result = decompose(mps,
            method=method,
            max_layers=n_layers,
            target_fidelity=target_fidelity,
            max_chi=max_chi,
            max_trunc_err=max_trunc_err,
            n_sweeps=n_sweeps,
            verbose=false
        )

        push!(results, BenchmarkResult(
            method,
            result.fidelity,
            result.depth,
            result.n_gates,
            t,
            length(result.optimization_history),
            result.optimization_history
        ))

        if verbose
            println("fid=$(round(result.fidelity, digits=6)), time=$(round(t, digits=3))s")
        end
    end

    if verbose
        println("-" ^ 60)
        println("Summary:")
        for r in results
            println("  $(r.method): fidelity=$(round(r.fidelity, digits=6)), ",
                   "depth=$(r.depth), gates=$(r.n_gates), time=$(round(r.time, digits=3))s")
        end
        println("=" ^ 60)
    end

    return results
end
