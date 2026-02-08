# State preparation compilation: find circuit U such that U|ψ_init⟩ ≈ |ψ_target⟩
# Adapted from iMPS2Circuit/src/Algorithms/StateCompilation.jl

using LinearAlgebra

# ============================================================================
# State Compilation Configuration
# ============================================================================

struct StateCompilationConfig
    max_chi::Int
    max_trunc_err::Float64
    optimizer::Symbol
    max_iter::Int
    converge_tol::Float64
    verbose::Bool
end

function StateCompilationConfig(; max_chi::Int=32, max_trunc_err::Float64=1e-10,
                                  optimizer::Symbol=:lbfgs, max_iter::Int=100,
                                  converge_tol::Float64=1e-6, verbose::Bool=true)
    return StateCompilationConfig(max_chi, max_trunc_err, optimizer,
                                   max_iter, converge_tol, verbose)
end

# ============================================================================
# State Preparation Compilation
# ============================================================================

"""
    compile_state_preparation(psi_init::iMPSType, psi_target::iMPSType, ansatz::ParameterizedCircuit;
                               config::StateCompilationConfig=StateCompilationConfig())

Compile a circuit that prepares a target state from a specific initial state.

Typically psi_init is the |0...0⟩ product state and psi_target is a ground state from iTEBD.
"""
function compile_state_preparation(psi_init::iMPSType{T}, psi_target::iMPSType{T},
                                    ansatz::ParameterizedCircuit;
                                    config::StateCompilationConfig=StateCompilationConfig()) where T

    bond_config = BondConfig(config.max_chi, config.max_trunc_err)

    psi_inits = [deepcopy(psi_init)]
    psi_targets = [deepcopy(psi_target)]

    return _compile_with_states(ansatz, psi_inits, psi_targets,
                                iMPSType{T}[], iMPSType{T}[], bond_config;
                                optimizer=config.optimizer,
                                max_iter=config.max_iter,
                                converge_tol=config.converge_tol,
                                verbose=config.verbose)
end
