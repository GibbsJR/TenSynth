# Main optimization loop
# Adapted from MPO2Circuit/src/optimization/optimize.jl

using LinearAlgebra

"""
    OptimizerConfig

Configuration for MPO circuit optimization.
"""
struct OptimizerConfig
    max_chi::Int
    max_trunc_err::Float64
    n_sweeps::Int
    n_layer_sweeps::Int
    slowdown::Float64
    verbose::Bool
end

function OptimizerConfig(;
    max_chi::Int=128,
    max_trunc_err::Float64=1e-14,
    n_sweeps::Int=20,
    n_layer_sweeps::Int=2,
    slowdown::Float64=0.0,
    verbose::Bool=true
)
    return OptimizerConfig(max_chi, max_trunc_err, n_sweeps, n_layer_sweeps, slowdown, verbose)
end

"""
    OptimizationResult

Result of MPO circuit optimization.
"""
struct OptimizationResult
    initial_cost::Float64
    final_cost::Float64
    cost_history::Vector{Float64}
end

"""
    ansatz_to_mpo(n_qubits, layers; max_chi, max_trunc_err) -> FiniteMPO{ComplexF64}

Convert circuit ansatz (vector of layers) to MPO representation.
"""
function ansatz_to_mpo(n_qubits::Int, layers::Vector{GateLayer};
                        max_chi::Int=128, max_trunc_err::Float64=1e-14)::FiniteMPO{ComplexF64}
    mpo = identity_mpo(n_qubits)

    for layer in layers
        apply_layer!(mpo, layer; max_chi=max_chi, max_trunc_err=max_trunc_err)
    end

    return mpo
end

"""
    apply_inverse_layer!(mpo, layer; max_chi, max_trunc_err)

Apply the inverse of a layer to an MPO (applies inverse gates in reverse order).
"""
function apply_inverse_layer!(mpo::FiniteMPO{ComplexF64}, layer::GateLayer;
                               max_chi::Int=128, max_trunc_err::Float64=1e-14)
    for i in length(layer.gates):-1:1
        gate = layer.gates[i]
        (site1, site2) = layer.indices[i]
        inv_gate = GateMatrix(Matrix(gate.matrix'), gate.name * "_inv", gate.params)

        if site2 == site1 + 1
            apply_gate!(mpo, inv_gate, site1, site2; max_chi=max_chi, max_trunc_err=max_trunc_err)
        else
            apply_gate_long_range!(mpo, inv_gate, site1, site2;
                                    max_chi=max_chi, max_trunc_err=max_trunc_err)
        end
    end
end

"""
    optimize!(target::FiniteMPO{ComplexF64}, circuit::LayeredCircuit,
              config::OptimizerConfig) -> OptimizationResult

Optimize a circuit to approximate a target MPO.

The algorithm uses alternating backward/forward sweeps:
1. Backward sweep: layers from end to 2 — peel off and optimize
2. Forward sweep: layers from 1 to end — build up and optimize
3. Repeat for multiple sweeps until convergence

The circuit is modified in place with the optimized gates.
"""
function optimize!(target::FiniteMPO{ComplexF64}, circuit::LayeredCircuit,
                    config::OptimizerConfig)::OptimizationResult
    n = circuit.n_qubits
    n_layers = length(circuit.layers)

    if n_layers == 0
        c = hst_cost(target, identity_mpo(n))
        return OptimizationResult(c, c, [c])
    end

    max_chi = config.max_chi
    max_trunc_err = config.max_trunc_err
    n_sweeps = config.n_sweeps
    n_layer_sweeps = config.n_layer_sweeps
    slowdown = config.slowdown
    verbose = config.verbose

    # Compute initial cost
    mpo_test = ansatz_to_mpo(n, circuit.layers; max_chi=2*max_chi, max_trunc_err=max_trunc_err)
    initial_cost = hst_cost(target, mpo_test)
    costs = [initial_cost]

    if verbose
        println()
        println("Optimization: $(n_layers) layers, $(n) qubits")
        println("Initial cost: $(round(initial_cost, sigdigits=8))")
        flush(stdout)
    end

    # Working copies
    mpo_target = FiniteMPO{ComplexF64}([copy(t) for t in target.tensors])

    for sweep in 1:n_sweeps
        # Backward sweep: layers from end to 2
        for layer_idx in n_layers:-1:2
            apply_inverse_layer!(mpo_test, circuit.layers[layer_idx];
                                  max_chi=max_chi, max_trunc_err=max_trunc_err)

            circuit.layers[layer_idx] = layer_sweep!(
                mpo_target, mpo_test, circuit.layers[layer_idx];
                n_sweeps=n_layer_sweeps, slowdown=slowdown,
                max_chi=max_chi, max_trunc_err=max_trunc_err
            )

            apply_inverse_layer!(mpo_target, circuit.layers[layer_idx];
                                  max_chi=max_chi, max_trunc_err=max_trunc_err)
        end

        # Apply inverse of layer 1 to target
        apply_inverse_layer!(mpo_target, circuit.layers[1];
                              max_chi=max_chi, max_trunc_err=max_trunc_err)

        # Reset test to identity for forward sweep
        mpo_test = identity_mpo(n)

        # Forward sweep: layers from 1 to end
        for layer_idx in 1:n_layers
            apply_layer!(mpo_target, circuit.layers[layer_idx];
                          max_chi=max_chi, max_trunc_err=max_trunc_err)

            circuit.layers[layer_idx] = layer_sweep!(
                mpo_target, mpo_test, circuit.layers[layer_idx];
                n_sweeps=n_layer_sweeps, slowdown=slowdown,
                max_chi=max_chi, max_trunc_err=max_trunc_err
            )

            apply_layer!(mpo_test, circuit.layers[layer_idx];
                          max_chi=max_chi, max_trunc_err=max_trunc_err)
        end

        # Reset target for next sweep
        mpo_target = FiniteMPO{ComplexF64}([copy(t) for t in target.tensors])

        # Compute and record cost
        c = hst_cost(target, mpo_test)
        push!(costs, c)

        if verbose && sweep % 5 == 0
            println("Sweep $sweep: cost = $(round(c, sigdigits=8))")
            flush(stdout)
        end
    end

    final_cost = costs[end]

    if verbose
        println("Final cost: $(round(final_cost, sigdigits=8))")
        flush(stdout)
    end

    return OptimizationResult(initial_cost, final_cost, costs)
end

"""
    optimize_simple!(target, circuit, config) -> OptimizationResult

Simplified optimization that only does forward sweeps.
Useful for testing and debugging.
"""
function optimize_simple!(target::FiniteMPO{ComplexF64}, circuit::LayeredCircuit,
                           config::OptimizerConfig)::OptimizationResult
    n = circuit.n_qubits
    n_layers = length(circuit.layers)

    if n_layers == 0
        c = hst_cost(target, identity_mpo(n))
        return OptimizationResult(c, c, [c])
    end

    max_chi = config.max_chi
    max_trunc_err = config.max_trunc_err
    n_sweeps = config.n_sweeps
    n_layer_sweeps = config.n_layer_sweeps
    slowdown = config.slowdown
    verbose = config.verbose

    mpo_test = ansatz_to_mpo(n, circuit.layers; max_chi=2*max_chi, max_trunc_err=max_trunc_err)
    initial_cost = hst_cost(target, mpo_test)
    costs = [initial_cost]

    if verbose
        println("Simple optimization: $(n_layers) layers")
        println("Initial cost: $(round(initial_cost, sigdigits=8))")
    end

    for sweep in 1:n_sweeps
        mpo_test = identity_mpo(n)

        for layer_idx in 1:n_layers
            circuit.layers[layer_idx] = layer_sweep!(
                target, mpo_test, circuit.layers[layer_idx];
                n_sweeps=n_layer_sweeps, slowdown=slowdown,
                max_chi=max_chi, max_trunc_err=max_trunc_err
            )

            apply_layer!(mpo_test, circuit.layers[layer_idx];
                          max_chi=max_chi, max_trunc_err=max_trunc_err)
        end

        c = hst_cost(target, mpo_test)
        push!(costs, c)

        if verbose && sweep % 5 == 0
            println("Sweep $sweep: cost = $(round(c, sigdigits=8))")
        end
    end

    final_cost = costs[end]

    if verbose
        println("Final cost: $(round(final_cost, sigdigits=8))")
    end

    return OptimizationResult(initial_cost, final_cost, costs)
end
