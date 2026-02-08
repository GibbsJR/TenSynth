# Unitary compilation: find circuit U(θ) ≈ target unitary e^{-iHt}
# Adapted from iMPS2Circuit/src/Algorithms/UnitaryCompilation.jl

using LinearAlgebra
using Random

# ============================================================================
# Unitary Compilation Configuration
# ============================================================================

struct UnitaryCompilationConfig
    n_train::Int
    n_test::Int
    max_chi::Int
    max_trunc_err::Float64
    optimizer::Symbol
    max_iter::Int
    converge_tol::Float64
    verbose::Bool
end

function UnitaryCompilationConfig(; n_train::Int=8, n_test::Int=16,
                                    max_chi::Int=32, max_trunc_err::Float64=1e-10,
                                    optimizer::Symbol=:lbfgs, max_iter::Int=100,
                                    converge_tol::Float64=1e-5, verbose::Bool=true)
    return UnitaryCompilationConfig(n_train, n_test, max_chi, max_trunc_err,
                                     optimizer, max_iter, converge_tol, verbose)
end

# ============================================================================
# State Generation
# ============================================================================

"""
    generate_training_states(unit_cell::Int, n_states::Int; chi::Int=2) -> Vector{iMPSType{ComplexF64}}

Generate random product states for training.
"""
function generate_training_states(unit_cell::Int, n_states::Int;
                                   chi::Int=2)::Vector{iMPSType{ComplexF64}}
    states = Vector{iMPSType{ComplexF64}}(undef, n_states)
    for i in 1:n_states
        states[i] = random_product_state(unit_cell)
    end
    return states
end

"""
    generate_target_states(psi_inits, target_circuit, config) -> Vector{iMPS}

Generate target states by applying target circuit to initial states.
"""
function generate_target_states(psi_inits::Vector{iMPSType{T}},
                                 target_circuit::ParameterizedCircuit,
                                 config::BondConfig) where T
    n = length(psi_inits)
    targets = Vector{iMPSType{T}}(undef, n)

    for i in 1:n
        targets[i] = deepcopy(psi_inits[i])
        apply_circuit!(targets[i], target_circuit, config)
    end

    return targets
end

"""
    generate_target_states(psi_inits, target_unitary, sites, config) -> Vector{iMPS}

Generate target states by applying a unitary to initial states.
"""
function generate_target_states(psi_inits::Vector{iMPSType{T}},
                                 target_unitary::Matrix{ComplexF64},
                                 sites::Tuple{Int,Int},
                                 config::BondConfig) where T
    n = length(psi_inits)
    targets = Vector{iMPSType{T}}(undef, n)

    for i in 1:n
        targets[i] = deepcopy(psi_inits[i])
        apply_gate_nn!(targets[i], target_unitary, sites, config)
    end

    return targets
end

# ============================================================================
# Main Compilation Functions
# ============================================================================

"""
    compile_unitary(target_circuit::ParameterizedCircuit, ansatz::ParameterizedCircuit,
                    unit_cell::Int; config::UnitaryCompilationConfig=UnitaryCompilationConfig())

Compile a target circuit into an ansatz circuit by optimization.
Trains on random product states and tests on unseen random product states.
"""
function compile_unitary(target_circuit::ParameterizedCircuit, ansatz::ParameterizedCircuit,
                         unit_cell::Int; config::UnitaryCompilationConfig=UnitaryCompilationConfig())

    bond_config = BondConfig(config.max_chi, config.max_trunc_err)

    psi_inits_train = generate_training_states(unit_cell, config.n_train)
    psi_inits_test = generate_training_states(unit_cell, config.n_test)

    psi_targets_train = generate_target_states(psi_inits_train, target_circuit, bond_config)
    psi_targets_test = generate_target_states(psi_inits_test, target_circuit, bond_config)

    return _compile_with_states(ansatz, psi_inits_train, psi_targets_train,
                                psi_inits_test, psi_targets_test, bond_config;
                                optimizer=config.optimizer,
                                max_iter=config.max_iter,
                                converge_tol=config.converge_tol,
                                verbose=config.verbose)
end

"""
    compile_unitary(target_unitary::Matrix, sites, ansatz, unit_cell; config)

Compile a target unitary matrix into an ansatz circuit.
"""
function compile_unitary(target_unitary::Matrix{ComplexF64}, sites::Tuple{Int,Int},
                         ansatz::ParameterizedCircuit, unit_cell::Int;
                         config::UnitaryCompilationConfig=UnitaryCompilationConfig())

    bond_config = BondConfig(config.max_chi, config.max_trunc_err)

    psi_inits_train = generate_training_states(unit_cell, config.n_train)
    psi_inits_test = generate_training_states(unit_cell, config.n_test)

    psi_targets_train = generate_target_states(psi_inits_train, target_unitary,
                                                sites, bond_config)
    psi_targets_test = generate_target_states(psi_inits_test, target_unitary,
                                               sites, bond_config)

    return _compile_with_states(ansatz, psi_inits_train, psi_targets_train,
                                psi_inits_test, psi_targets_test, bond_config;
                                optimizer=config.optimizer,
                                max_iter=config.max_iter,
                                converge_tol=config.converge_tol,
                                verbose=config.verbose)
end

"""
    compile_time_evolution(H::SpinLatticeHamiltonian, t::Float64, ansatz::ParameterizedCircuit;
                           trotter_order::TrotterOrder=SECOND_ORDER,
                           n_trotter_steps::Int=10,
                           config::UnitaryCompilationConfig=UnitaryCompilationConfig())

Compile time evolution exp(-iHt) into a parameterized circuit.
"""
function compile_time_evolution(H::SpinLatticeHamiltonian, t::Float64,
                                 ansatz::ParameterizedCircuit;
                                 trotter_order::TrotterOrder=SECOND_ORDER,
                                 n_trotter_steps::Int=10,
                                 config::UnitaryCompilationConfig=UnitaryCompilationConfig())

    target_circuit = trotterize(H, t, trotter_order, n_trotter_steps)

    return compile_unitary(target_circuit, ansatz, H.unit_cell; config=config)
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    compile_tfim_evolution(unit_cell, t, J, h, n_layers; ...)

Convenience function to compile time evolution of TFIM.
"""
function compile_tfim_evolution(unit_cell::Int, t::Float64, J::Float64, h::Float64,
                                 n_layers::Int;
                                 parameterization::AbstractParameterization=DressedZZParameterization(),
                                 config::UnitaryCompilationConfig=UnitaryCompilationConfig())
    H = TFIMHamiltonian(unit_cell; J=J, h=h)
    ansatz = nearest_neighbour_ansatz(unit_cell, n_layers, parameterization)

    return compile_time_evolution(H, t, ansatz; config=config)
end

"""
    compile_xxz_evolution(unit_cell, t, Jxy, Jz, n_layers; ...)

Convenience function to compile time evolution of XXZ model.
"""
function compile_xxz_evolution(unit_cell::Int, t::Float64, Jxy::Float64, Jz::Float64,
                                n_layers::Int;
                                parameterization::AbstractParameterization=PauliGeneratorParameterization(),
                                config::UnitaryCompilationConfig=UnitaryCompilationConfig())
    H = XXZHamiltonian(unit_cell; Jxy=Jxy, Jz=Jz)
    ansatz = nearest_neighbour_ansatz(unit_cell, n_layers, parameterization)

    return compile_time_evolution(H, t, ansatz; config=config)
end

"""
    verify_compilation(result::CompilationResult, target_circuit::ParameterizedCircuit,
                       unit_cell::Int; n_states::Int=100) -> Float64

Verify compilation by computing average fidelity on fresh random states.
"""
function verify_compilation(result::CompilationResult, target_circuit::ParameterizedCircuit,
                            unit_cell::Int; n_states::Int=100)

    bond_config = BondConfig(32, 1e-10)

    total_fidelity = 0.0
    for _ in 1:n_states
        psi = random_product_state(unit_cell)
        psi_target = deepcopy(psi)

        apply_circuit!(psi, result.circuit, bond_config)
        apply_circuit!(psi_target, target_circuit, bond_config)

        total_fidelity += local_fidelity(psi, psi_target)
    end

    return total_fidelity / n_states
end
