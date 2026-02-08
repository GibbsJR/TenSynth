# Gradient computation for quantum circuit optimization
# Adapted from iMPS2Circuit/src/Algorithms/Gradients.jl

using LinearAlgebra

# ============================================================================
# Finite Difference Gradients
# ============================================================================

"""
    compute_gradient_fd!(grad::Vector{Float64}, circuit::ParameterizedCircuit,
                         psi_init::iMPSType, psi_target::iMPSType, config::BondConfig;
                         delta::Float64=1e-6, tol::Float64=1e-10)

Compute the gradient of the infidelity cost using central finite differences.
"""
function compute_gradient_fd!(grad::Vector{Float64}, circuit::ParameterizedCircuit,
                               psi_init::iMPSType{T}, psi_target::iMPSType{T}, config::BondConfig;
                               delta::Float64=1e-6, tol::Float64=1e-10) where T
    params = get_params(circuit)
    n = length(params)

    if length(grad) != n
        throw(ArgumentError("Gradient vector size mismatch: expected $n, got $(length(grad))"))
    end

    for i in 1:n
        params_plus = copy(params)
        params_plus[i] += delta
        set_params!(circuit, params_plus)

        psi_plus = deepcopy(psi_init)
        apply_circuit!(psi_plus, circuit, config)
        cost_plus = infidelity(psi_plus, psi_target; tol=tol)

        params_minus = copy(params)
        params_minus[i] -= delta
        set_params!(circuit, params_minus)

        psi_minus = deepcopy(psi_init)
        apply_circuit!(psi_minus, circuit, config)
        cost_minus = infidelity(psi_minus, psi_target; tol=tol)

        grad[i] = (cost_plus - cost_minus) / (2.0 * delta)
    end

    set_params!(circuit, params)

    return grad
end

function compute_gradient_fd(circuit::ParameterizedCircuit, psi_init::iMPSType{T},
                              psi_target::iMPSType{T}, config::BondConfig;
                              delta::Float64=1e-6, tol::Float64=1e-10) where T
    grad = zeros(Float64, n_params(circuit))
    compute_gradient_fd!(grad, circuit, psi_init, psi_target, config; delta=delta, tol=tol)
    return grad
end

# ============================================================================
# Backward Pass Gradient
# ============================================================================

"""
    compute_gradient_backward!(grad::Vector{Float64}, circuit::ParameterizedCircuit,
                               psi_init::iMPSType, psi_target::iMPSType, config::BondConfig;
                               delta::Float64=1e-6, tol::Float64=1e-10)

Compute gradients using the backward pass algorithm.
"""
function compute_gradient_backward!(grad::Vector{Float64}, circuit::ParameterizedCircuit,
                                     psi_init::iMPSType{T}, psi_target::iMPSType{T},
                                     config::BondConfig;
                                     delta::Float64=1e-6, tol::Float64=1e-10) where T
    params = get_params(circuit)
    n_total = length(params)

    if length(grad) != n_total
        throw(ArgumentError("Gradient vector size mismatch"))
    end

    psi_evolved = deepcopy(psi_init)
    apply_circuit!(psi_evolved, circuit, config)

    psi_target_current = deepcopy(psi_target)

    param_indices = circuit._param_indices
    if circuit._dirty
        get_params(circuit)  # triggers rebuild
        param_indices = circuit._param_indices
    end

    n_gates_count = n_gates(circuit)
    for g_idx in n_gates_count:-1:1
        gate = circuit.gates[g_idx]

        gate_inv_matrix = inverse(gate)

        apply_gate!(psi_evolved, gate_inv_matrix, gate.qubits, config)

        idx_range = param_indices[g_idx]
        n_gate_params = length(idx_range)

        if n_gate_params > 0
            gate_params = gate.params

            for (local_idx, global_idx) in enumerate(idx_range)
                params_plus = copy(gate_params)
                params_plus[local_idx] += delta
                gate_plus = ParameterizedGate(gate.parameterization, gate.qubits, params_plus)

                psi_test = deepcopy(psi_evolved)
                apply_gate!(psi_test, to_matrix(gate_plus), gate.qubits, config)
                cost_plus = infidelity(psi_test, psi_target_current; tol=tol)

                params_minus = copy(gate_params)
                params_minus[local_idx] -= delta
                gate_minus = ParameterizedGate(gate.parameterization, gate.qubits, params_minus)

                psi_test = deepcopy(psi_evolved)
                apply_gate!(psi_test, to_matrix(gate_minus), gate.qubits, config)
                cost_minus = infidelity(psi_test, psi_target_current; tol=tol)

                grad[global_idx] = (cost_plus - cost_minus) / (2.0 * delta)
            end
        end

        apply_gate!(psi_target_current, gate_inv_matrix, gate.qubits, config)
    end

    return grad
end

function compute_gradient_backward(circuit::ParameterizedCircuit, psi_init::iMPSType{T},
                                    psi_target::iMPSType{T}, config::BondConfig;
                                    delta::Float64=1e-6, tol::Float64=1e-10) where T
    grad = zeros(Float64, n_params(circuit))
    compute_gradient_backward!(grad, circuit, psi_init, psi_target, config; delta=delta, tol=tol)
    return grad
end

# ============================================================================
# Batch Gradient Computation
# ============================================================================

function compute_gradient_batch!(grad::Vector{Float64}, circuit::ParameterizedCircuit,
                                  psi_inits::Vector{iMPSType{T}}, psi_targets::Vector{iMPSType{T}},
                                  config::BondConfig;
                                  method::Symbol=:backward,
                                  delta::Float64=1e-6, tol::Float64=1e-10) where T
    n_pairs = length(psi_inits)
    if n_pairs != length(psi_targets)
        throw(ArgumentError("Number of initial and target states must match"))
    end

    fill!(grad, 0.0)
    grad_single = zeros(Float64, length(grad))

    for i in 1:n_pairs
        if method == :backward
            compute_gradient_backward!(grad_single, circuit, psi_inits[i], psi_targets[i],
                                       config; delta=delta, tol=tol)
        else
            compute_gradient_fd!(grad_single, circuit, psi_inits[i], psi_targets[i],
                                 config; delta=delta, tol=tol)
        end
        grad .+= grad_single
    end

    grad ./= n_pairs
    return grad
end

function compute_gradient_batch(circuit::ParameterizedCircuit,
                                 psi_inits::Vector{iMPSType{T}}, psi_targets::Vector{iMPSType{T}},
                                 config::BondConfig; kwargs...) where T
    grad = zeros(Float64, n_params(circuit))
    compute_gradient_batch!(grad, circuit, psi_inits, psi_targets, config; kwargs...)
    return grad
end

# ============================================================================
# Cost Function Computation
# ============================================================================

function compute_cost(circuit::ParameterizedCircuit, psi_init::iMPSType{T},
                       psi_target::iMPSType{T}, config::BondConfig;
                       tol::Float64=1e-10) where T
    psi = deepcopy(psi_init)
    apply_circuit!(psi, circuit, config)
    return infidelity(psi, psi_target; tol=tol)
end

function compute_cost_batch(circuit::ParameterizedCircuit,
                             psi_inits::Vector{iMPSType{T}}, psi_targets::Vector{iMPSType{T}},
                             config::BondConfig; tol::Float64=1e-10) where T
    n_pairs = length(psi_inits)
    if n_pairs != length(psi_targets)
        throw(ArgumentError("Number of initial and target states must match"))
    end

    total_cost = 0.0
    for i in 1:n_pairs
        total_cost += compute_cost(circuit, psi_inits[i], psi_targets[i], config; tol=tol)
    end

    return total_cost / n_pairs
end

# ============================================================================
# Gradient Verification
# ============================================================================

function verify_gradient(circuit::ParameterizedCircuit, psi_init::iMPSType{T},
                          psi_target::iMPSType{T}, config::BondConfig;
                          delta::Float64=1e-6, tol::Float64=1e-6) where T
    grad_fd = compute_gradient_fd(circuit, psi_init, psi_target, config; delta=delta)
    grad_bw = compute_gradient_backward(circuit, psi_init, psi_target, config; delta=delta)

    max_diff = maximum(abs.(grad_fd - grad_bw))
    return max_diff < tol, max_diff, grad_fd, grad_bw
end
