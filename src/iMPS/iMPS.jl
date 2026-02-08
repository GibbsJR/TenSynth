module iMPS

using LinearAlgebra
using Random
using TensorOperations
using LinearMaps
using Arpack
using Optim
using LineSearches

# Import from Core
using ..Core
using ..Core: AbstractParameterization, AbstractiMPS, AbstractHamiltonian
using ..Core: iMPS as iMPSType, ParameterizedGate, ParameterizedCircuit
using ..Core: BondConfig, GateMatrix
using ..Core: PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
using ..Core: Id, X, Y, Z
using ..Core: H_GATE, S_GATE, T_GATE, CNOT, SWAP, FSWAP
using ..Core: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
using ..Core: RX, RY, RZ, RZZ
using ..Core: PauliGeneratorParameterization, DressedZZParameterization
using ..Core: SingleQubitXYZParameterization, ZZParameterization, ZZXParameterization
using ..Core: KAKParameterization
# import (not using) for functions that iMPS extends with new methods
import ..Core: n_params, to_matrix
using ..Core: is_unitary
using ..Core: robust_svd, polar_unitary, reverse_qubits

# ============================================================================
# Include files in dependency order
# ============================================================================

# States
include("states/imps_type.jl")
include("states/initialization.jl")
include("states/normalization.jl")
include("states/transfer_matrix.jl")

# Operators
include("operators/gate_application.jl")
include("operators/hamiltonians.jl")
include("operators/impo.jl")

# Circuits
include("circuits/circuit_structure.jl")
include("circuits/swap_networks.jl")
include("circuits/trotterization.jl")

# Algorithms
include("algorithms/gradients.jl")
include("algorithms/optimization.jl")
include("algorithms/itebd.jl")
include("algorithms/compilation_common.jl")
include("algorithms/state_compilation.jl")
include("algorithms/unitary_compilation.jl")

# ============================================================================
# Exports
# ============================================================================

# --- States ---
# bond_dimensions NOT exported to avoid conflict with TenSynth.MPO.bond_dimensions.
export max_bond_dimension
export invalidate_cache!, get_site_tensor

# --- Initialization ---
export random_product_state!, random_product_state
export random_imps!, random_imps
export product_state!, zero_state!, plus_state!, neel_state!

# --- Normalization ---
export absorb_bonds!, truncate!, canonicalize!, normalize_state!
export norm_squared, check_canonical_form

# --- Transfer matrix ---
export transfer_matrix_action!, transfer_matrix_eigenvalue
export local_fidelity, infidelity, infidelity_average

# --- Gate application ---
# apply_gate! NOT exported to avoid conflict with TenSynth.MPO.apply_gate!.
export apply_gate_single!, apply_gate_nn!
export apply_gates!

# --- Hamiltonians ---
# Hamiltonian types/functions NOT exported to avoid conflicts with TenSynth.Hamiltonians.
# Use qualified access (e.g. TenSynth.iMPS.TFIMHamiltonian) or import from TenSynth.Hamiltonians.

# --- iMPO ---
export iMPO
export gate_to_impo, apply_impo!, apply_gate_impo!

# --- Circuit structure ---
export get_params, set_params!
export n_gates, add_gate!, insert_gate!, remove_gate!
export apply_circuit!, depth
export is_single_qubit, is_two_qubit
export two_qubit_gate_count, single_qubit_gate_count
export random_gate, inverse
export nearest_neighbour_ansatz, random_circuit

# --- Swap networks ---
export SwapRoute
export compute_swap_route, route_distance
export expand_with_swaps, apply_gate_with_swaps!
export analyze_routing

# --- Trotterization ---
export TrotterOrder, FIRST_ORDER, SECOND_ORDER, FOURTH_ORDER
# trotterize, trotterize_tfim, trotterize_xxz, trotter_error_bound NOT exported
# to avoid conflicts with TenSynth.Hamiltonians. Use qualified access.

# --- Gradients ---
export compute_gradient_fd!, compute_gradient_fd
export compute_gradient_backward!, compute_gradient_backward
export compute_gradient_batch!, compute_gradient_batch
export compute_cost, compute_cost_batch
export verify_gradient

# --- Optimization ---
# optimize! and OptimizationResult NOT exported to avoid conflicts with TenSynth.MPO.
export CompilationProblem
export optimize_gd!, optimize_adam!

# --- iTEBD ---
export iTEBDConfig, iTEBDResult
export itebd_ground_state, itebd_ground_state!
export apply_itebd_step!, apply_itebd_step_fourth_order!
export compute_energy
export itebd_tfim_ground_state, itebd_xxz_ground_state, itebd_heisenberg_ground_state

# --- Compilation ---
# CompilationResult NOT exported to avoid conflict with TenSynth.MPS.CompilationResult.
# Use TenSynth.iMPS.CompilationResult for qualified access.
export StateCompilationConfig, compile_state_preparation
export UnitaryCompilationConfig
export generate_training_states, generate_target_states
export compile_unitary, compile_time_evolution
export compile_tfim_evolution, compile_xxz_evolution
export verify_compilation

end # module iMPS
