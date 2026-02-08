# Shared types and internal functions for compilation
# Adapted from iMPS2Circuit/src/Algorithms/CompilationCommon.jl

using LinearAlgebra

# ============================================================================
# Compilation Result
# ============================================================================

"""
    CompilationResult{T}

Result of circuit compilation (both state and unitary compilation).
"""
struct CompilationResult{T}
    circuit::ParameterizedCircuit
    train_fidelity::Float64
    test_fidelity::Float64
    converged::Bool
    train_history::Vector{Float64}
    test_history::Vector{Float64}
end

# ============================================================================
# Internal Compilation Function
# ============================================================================

"""
    _compile_with_states(ansatz, psi_inits_train, psi_targets_train,
                         psi_inits_test, psi_targets_test, bond_config; ...)

Internal function that performs the actual compilation.
Shared between state compilation and unitary compilation.
"""
function _compile_with_states(ansatz::ParameterizedCircuit,
                               psi_inits_train::Vector{iMPSType{T}},
                               psi_targets_train::Vector{iMPSType{T}},
                               psi_inits_test::Vector{iMPSType{T}},
                               psi_targets_test::Vector{iMPSType{T}},
                               bond_config::BondConfig;
                               optimizer::Symbol,
                               max_iter::Int,
                               converge_tol::Float64,
                               verbose::Bool) where T

    problem = CompilationProblem(
        ansatz,
        psi_inits_train, psi_targets_train,
        psi_inits_test, psi_targets_test,
        bond_config;
        delta=1e-6, tol=1e-10
    )

    if optimizer == :lbfgs
        result = optimize!(problem; method=:lbfgs, max_iter=max_iter,
                           g_tol=converge_tol, show_trace=verbose)
    elseif optimizer == :adam
        result = optimize_adam!(problem; max_iter=max_iter,
                                tol=converge_tol, verbose=verbose)
    elseif optimizer == :gd
        result = optimize_gd!(problem; max_iter=max_iter,
                              tol=converge_tol, verbose=verbose)
    else
        throw(ArgumentError("Unknown optimizer: $(optimizer)"))
    end

    train_fidelity = 1.0 - result.final_train_cost
    test_fidelity = 1.0 - result.final_test_cost

    return CompilationResult{T}(
        ansatz,
        train_fidelity,
        test_fidelity,
        result.converged,
        result.train_history,
        result.test_history
    )
end
