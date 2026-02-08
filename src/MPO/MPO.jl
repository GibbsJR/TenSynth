module MPO

# MPO backend module: finite MPO circuit compilation (from MPO2Circuit).
# Convention: tensors are [left_bond, phys_out, phys_in, right_bond].

using LinearAlgebra
using TensorOperations

# Import Core types and utilities
using ..Core

# Core MPO/MPS operations
include("core.jl")
include("conversion.jl")
include("inner.jl")
include("canonicalize.jl")
include("compression.jl")

# Gate application
include("gates/application.jl")
include("gates/layers.jl")

# Circuit construction
include("circuit/ansatz.jl")

# Optimization
include("optimization/environments.jl")
include("optimization/gate_update.jl")
include("optimization/layer_sweep.jl")
include("optimization/optimize.jl")

# --- Exports ---

# Core operations
export identity_mpo
export n_sites, bond_dimensions, trace_mpo

# Conversion
export mpo_to_mps, mps_to_mpo
export mpo_to_mps_direct, mps_to_mpo_direct

# Inner products
export inner_mps, inner_mpo, hst_cost

# Canonicalization (not exported to avoid conflict with MPS.canonicalise)
# Use MPO.canonicalise or MPO.canonicalise_from_to for qualified access.

# Compression
export mpo_svd_truncate

# Gate application
export apply_gate!, apply_gate_to_tensors
export apply_gate_long_range!
export gate_to_mpo, mpo_mpo_contract!
export random_unitary

# Layers
export apply_layer!, apply_layers!
export brick_wall_indices, identity_layer, random_layer

# Circuit
export circuit_to_mpo
export identity_circuit, brick_wall_circuit
export circuit_depth, circuit_gate_count

# Environments
export LayerEnvironments, TwoLayerEnvironments
export init_layer_environments, init_two_layer_environments
export update_left_environments!, update_right_environments!
export update_two_layer_left!, update_two_layer_right!

# Gate update
export mpo_polar_unitary, rsvd_update
export compute_gate_environment_nn, compute_gate_environment_lr
export compute_optimal_gate

# Layer sweep
export layer_sweep!, layer_to_mpo

# Optimization
export OptimizerConfig, OptimizationResult
export optimize!, optimize_simple!
export ansatz_to_mpo, apply_inverse_layer!

end # module MPO
