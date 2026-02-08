# Direct circuit optimization with smart dispatch
# Adapted from MPS2Circuit/src/optimization/direct.jl
# Key changes: raw tensors → FiniteMPS wrappers, DecompositionLogger deferred to Phase E.5

using LinearAlgebra

"""
    optimize_single_layer(mps_target, mps_test, layer_gates, layer_inds; kwargs...)
        -> Vector{Matrix{ComplexF64}}

Optimize a single layer of gates to maximize overlap ⟨mps_target|U_layer|mps_test⟩.
Automatically dispatches: LayerSweep_general for non-overlapping, gate-by-gate for overlapping.
"""
function optimize_single_layer(
    mps_target::FiniteMPS{ComplexF64},
    mps_test::FiniteMPS{ComplexF64},
    layer_gates::Vector{Matrix{ComplexF64}},
    layer_inds::Vector{Tuple{Int,Int}};
    n_sweeps::Int=5,
    slowdown::Float64=0.1,
    max_chi::Int=64,
    max_trunc_err::Float64=1e-10
)::Vector{Matrix{ComplexF64}}

    if isempty(layer_gates)
        return Matrix{ComplexF64}[]
    end

    if is_non_overlapping_layer(layer_inds)
        return LayerSweep_general(mps_target, mps_test, layer_gates, layer_inds, n_sweeps, slowdown)
    else
        return _optimize_layer_gatewise(mps_target, mps_test, layer_gates, layer_inds,
                                        n_sweeps, slowdown, max_chi, max_trunc_err)
    end
end

"""
    _optimize_layer_gatewise(...)

Internal: optimize a layer gate-by-gate using SingleGateEnv_cached.
Used for overlapping layers where LayerSweep_general cannot be applied.
"""
function _optimize_layer_gatewise(
    mps_target::FiniteMPS{ComplexF64},
    mps_test::FiniteMPS{ComplexF64},
    layer_gates::Vector{Matrix{ComplexF64}},
    layer_inds::Vector{Tuple{Int,Int}},
    n_sweeps::Int,
    slowdown::Float64,
    max_chi::Int,
    max_trunc_err::Float64
)::Vector{Matrix{ComplexF64}}

    N = length(mps_target.tensors)
    n_gates = length(layer_gates)
    optimized_gates = copy(layer_gates)

    cache = GateEnvCache(N)

    for sweep in 1:n_sweeps
        # Forward sweep
        mps_work = FiniteMPS{ComplexF64}(deepcopy(mps_test.tensors))

        for gate_idx in 1:n_gates
            inds = layer_inds[gate_idx]
            update_cache!(cache, mps_target, mps_work)
            optimized_gates[gate_idx] = SingleGateEnv_cached(cache, optimized_gates[gate_idx], inds, slowdown)

            mps_work, _ = ApplyGateLayers(mps_work, [optimized_gates[gate_idx]], [inds],
                                           max_chi, max_trunc_err, inds)
        end

        # Backward sweep
        mps_work = FiniteMPS{ComplexF64}(deepcopy(mps_test.tensors))
        for g_idx in 1:n_gates
            mps_work, _ = ApplyGateLayers(mps_work, [optimized_gates[g_idx]], [layer_inds[g_idx]],
                                           max_chi, max_trunc_err, layer_inds[g_idx])
        end

        for gate_idx in n_gates:-1:1
            inds = layer_inds[gate_idx]
            mps_work, _ = ApplyGateLayers(mps_work, [inv(optimized_gates[gate_idx])], [inds],
                                           max_chi, max_trunc_err, inds)

            update_cache!(cache, mps_target, mps_work)
            optimized_gates[gate_idx] = SingleGateEnv_cached(cache, optimized_gates[gate_idx], inds, slowdown)

            mps_work, _ = ApplyGateLayers(mps_work, [optimized_gates[gate_idx]], [inds],
                                           max_chi, max_trunc_err, inds)
        end
    end

    return optimized_gates
end

"""
    optimize_circuit_direct(mps_target, circuit, circuit_inds; kwargs...)
        -> (Vector{Vector{Matrix{ComplexF64}}}, Vector{Float64})

Direct optimization of a circuit to maximize fidelity with target MPS.
Uses smart dispatch: LayerSweep_general for non-overlapping, gate-by-gate for overlapping.
"""
function optimize_circuit_direct(
    mps_target::FiniteMPS{ComplexF64},
    circuit::Vector{Vector{Matrix{ComplexF64}}},
    circuit_inds::Vector{Vector{Tuple{Int,Int}}};
    max_chi::Int=64,
    max_trunc_err::Float64=1e-10,
    n_sweeps::Int=50,
    n_layer_sweeps::Int=5,
    slowdown::Float64=0.1,
    converge_threshold::Float64=1e-8,
    verbose::Bool=false
)::Tuple{Vector{Vector{Matrix{ComplexF64}}}, Vector{Float64}}

    N = length(mps_target.tensors)
    n_layers = length(circuit)

    if n_layers == 0
        return (circuit, Float64[])
    end

    has_overlapping = any(!is_non_overlapping_layer(inds) for inds in circuit_inds)

    if has_overlapping
        return _optimize_circuit_gatewise(mps_target, circuit, circuit_inds;
                                          max_chi=max_chi, max_trunc_err=max_trunc_err,
                                          n_sweeps=n_sweeps, slowdown=slowdown,
                                          converge_threshold=converge_threshold, verbose=verbose)
    end

    return _optimize_circuit_layerwise(mps_target, circuit, circuit_inds;
                                       max_chi=max_chi, max_trunc_err=max_trunc_err,
                                       n_sweeps=n_sweeps, n_layer_sweeps=n_layer_sweeps,
                                       slowdown=slowdown, converge_threshold=converge_threshold,
                                       verbose=verbose)
end

"""
Internal: Optimize circuit with overlapping layers using gate-by-gate optimization.
"""
function _optimize_circuit_gatewise(
    mps_target::FiniteMPS{ComplexF64},
    circuit::Vector{Vector{Matrix{ComplexF64}}},
    circuit_inds::Vector{Vector{Tuple{Int,Int}}};
    max_chi::Int=64,
    max_trunc_err::Float64=1e-10,
    n_sweeps::Int=50,
    slowdown::Float64=0.1,
    converge_threshold::Float64=1e-8,
    verbose::Bool=false
)::Tuple{Vector{Vector{Matrix{ComplexF64}}}, Vector{Float64}}

    N = length(mps_target.tensors)

    # Flatten circuit
    flat_gates = Matrix{ComplexF64}[]
    flat_inds = Tuple{Int,Int}[]
    layer_boundaries = Int[0]

    for (layer_gates, layer_inds) in zip(circuit, circuit_inds)
        for (gate, inds) in zip(layer_gates, layer_inds)
            push!(flat_gates, gate)
            push!(flat_inds, inds)
        end
        push!(layer_boundaries, length(flat_gates))
    end

    n_gates = length(flat_gates)
    opt_gates = copy(flat_gates)
    fidelity_history = Float64[]

    # Initial fidelity
    mps_test = canonicalise(zeroMPS(N), 1)
    prev_inds_test = (1, 2)
    mps_test, prev_inds_test = ApplyGateLayers(mps_test, opt_gates, flat_inds, max_chi, max_trunc_err, prev_inds_test)
    init_fid = abs(inner(mps_target, mps_test))^2
    push!(fidelity_history, init_fid)

    if verbose
        println("Initial fidelity: $(round(init_fid, digits=6))")
    end

    prev_fid = init_fid
    mps_targ_init = FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors))

    mps_test_orth = OrthMPS(mps_test, prev_inds_test[2])
    mps_targ_orth = OrthMPS(FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors)), 0)
    gate_env_cache = GateEnvCache(N)

    for sweep_num in 1:n_sweeps
        target_site = flat_inds[end][1]
        ensure_canonical!(mps_test_orth, target_site)
        ensure_canonical!(mps_targ_orth, target_site)

        prev_inds_test = flat_inds[end]
        prev_inds_targ = flat_inds[end]

        # Backward sweep
        for ii in n_gates:-1:2
            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, [inv(opt_gates[ii])], [flat_inds[ii]],
                max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])

            update_cache!(gate_env_cache, mps_targ_orth.mps, mps_test_orth.mps)
            opt_gates[ii] = SingleGateEnv_cached(gate_env_cache, opt_gates[ii], flat_inds[ii], slowdown)

            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, [inv(opt_gates[ii])], [flat_inds[ii]],
                max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])
        end

        mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
            mps_targ_orth.mps, [inv(opt_gates[1])], [flat_inds[1]],
            max_chi, max_trunc_err, prev_inds_targ)
        set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

        # Reset test MPS
        mps_test_orth.mps = zeroMPS(N)
        ensure_canonical!(mps_test_orth, 1)
        set_orth_centre!(mps_test_orth, 1)
        prev_inds_test = (1, 2)

        # Forward sweep
        for ii in 1:n_gates
            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, [opt_gates[ii]], [flat_inds[ii]],
                max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

            update_cache!(gate_env_cache, mps_targ_orth.mps, mps_test_orth.mps)
            opt_gates[ii] = SingleGateEnv_cached(gate_env_cache, opt_gates[ii], flat_inds[ii], slowdown)

            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, [opt_gates[ii]], [flat_inds[ii]],
                max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])
        end

        mps_targ_orth.mps = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))
        invalidate_orth_centre!(mps_targ_orth)
        current_fid = abs(inner(mps_targ_orth.mps, mps_test_orth.mps))^2
        push!(fidelity_history, current_fid)

        if verbose && sweep_num % 1 == 0
            println("Sweep $sweep_num: fidelity = $(round(current_fid, digits=6))")
        end

        rel_change = abs(current_fid - prev_fid) / max(abs(current_fid), 1e-10)
        if rel_change < converge_threshold || current_fid > 1.0 - 1e-10
            if verbose
                println("Converged after $sweep_num sweeps. Final fidelity: $(round(current_fid, digits=6))")
            end
            break
        end
        prev_fid = current_fid
    end

    # Unflatten gates back into layers
    opt_circuit = Vector{Vector{Matrix{ComplexF64}}}()
    for layer_idx in 1:length(circuit)
        start_idx = layer_boundaries[layer_idx] + 1
        end_idx = layer_boundaries[layer_idx + 1]
        push!(opt_circuit, opt_gates[start_idx:end_idx])
    end

    return (opt_circuit, fidelity_history)
end

"""
Internal: Optimize circuit with non-overlapping layers using LayerSweep_general.
"""
function _optimize_circuit_layerwise(
    mps_target::FiniteMPS{ComplexF64},
    circuit::Vector{Vector{Matrix{ComplexF64}}},
    circuit_inds::Vector{Vector{Tuple{Int,Int}}};
    max_chi::Int=64,
    max_trunc_err::Float64=1e-10,
    n_sweeps::Int=50,
    n_layer_sweeps::Int=5,
    slowdown::Float64=0.1,
    converge_threshold::Float64=1e-8,
    verbose::Bool=false
)::Tuple{Vector{Vector{Matrix{ComplexF64}}}, Vector{Float64}}

    N = length(mps_target.tensors)
    n_layers = length(circuit)

    opt_circuit = deepcopy(circuit)
    fidelity_history = Float64[]

    mps_test = Ansatz2MPS(N, opt_circuit, circuit_inds, max_chi, max_trunc_err)
    init_fid = abs(inner(mps_target, mps_test))^2
    push!(fidelity_history, init_fid)

    if verbose
        println("Initial fidelity: $(round(init_fid, digits=6))")
    end

    prev_fid = init_fid
    mps_targ_init = FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors))

    mps_test_orth = OrthMPS(mps_test, 0)
    mps_targ_orth = OrthMPS(FiniteMPS{ComplexF64}(deepcopy(mps_target.tensors)), 0)

    for sweep_num in 1:n_sweeps
        target_site = circuit_inds[end][1][1]
        ensure_canonical!(mps_test_orth, target_site)
        ensure_canonical!(mps_targ_orth, target_site)

        prev_inds_test = circuit_inds[end][1]
        prev_inds_targ = circuit_inds[end][1]

        # Backward sweep through layers
        for layer_idx in n_layers:-1:2
            inv_gates = [inv(g) for g in opt_circuit[layer_idx]]
            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, inv_gates, circuit_inds[layer_idx],
                max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])

            opt_circuit[layer_idx] = LayerSweep_general(
                mps_targ_orth.mps, mps_test_orth.mps,
                opt_circuit[layer_idx], circuit_inds[layer_idx], n_layer_sweeps, slowdown)

            inv_gates_opt = [inv(g) for g in opt_circuit[layer_idx]]
            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, inv_gates_opt, circuit_inds[layer_idx],
                max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])
        end

        inv_gates_1 = [inv(g) for g in opt_circuit[1]]
        mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
            mps_targ_orth.mps, inv_gates_1, circuit_inds[1],
            max_chi, max_trunc_err, prev_inds_targ)
        set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

        # Reset and forward sweep
        mps_test_orth.mps = zeroMPS(N)
        ensure_canonical!(mps_test_orth, 1)
        set_orth_centre!(mps_test_orth, 1)
        ensure_canonical!(mps_targ_orth, 1)

        prev_inds_targ = (1, 1)
        prev_inds_test = (1, 1)

        for layer_idx in 1:n_layers
            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, opt_circuit[layer_idx], circuit_inds[layer_idx],
                max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

            opt_circuit[layer_idx] = LayerSweep_general(
                mps_targ_orth.mps, mps_test_orth.mps,
                opt_circuit[layer_idx], circuit_inds[layer_idx], n_layer_sweeps, slowdown)

            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, opt_circuit[layer_idx], circuit_inds[layer_idx],
                max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])
        end

        ensure_canonical!(mps_test_orth, 1)
        mps_targ_orth.mps = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))
        invalidate_orth_centre!(mps_targ_orth)

        current_fid = abs(inner(mps_targ_orth.mps, mps_test_orth.mps))^2
        push!(fidelity_history, current_fid)

        if verbose && sweep_num % 5 == 0
            println("Sweep $sweep_num: fidelity = $(round(current_fid, digits=6))")
        end

        rel_change = abs(current_fid - prev_fid) / max(abs(current_fid), 1e-10)
        if rel_change < converge_threshold || current_fid > 1.0 - 1e-10
            if verbose
                println("Converged after $sweep_num sweeps. Final fidelity: $(round(current_fid, digits=6))")
            end
            break
        end
        prev_fid = current_fid
    end

    return (opt_circuit, fidelity_history)
end
