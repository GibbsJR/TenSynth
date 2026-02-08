module MPS

# MPS backend module: finite MPS circuit compilation (from MPS2Circuit).
# Provides MPS construction, canonicalization, gate application, circuit optimization,
# decomposition protocols, synthesis pipeline, and export utilities.

using LinearAlgebra
using TensorOperations
using ..Core
import ITensors
import ITensorMPS

# ==============================================================================
# MPS Core: construction, canonicalization, inner products, truncation, conversions
# ==============================================================================

include("construction.jl")
include("canonicalization.jl")
include("inner_products.jl")
include("truncation.jl")
include("conversions.jl")

# ==============================================================================
# Gates: single-qubit, two-qubit (KAK), decomposition, application, layers
# ==============================================================================

include("gates/single_qubit.jl")
include("gates/two_qubit.jl")
include("gates/decomposition.jl")
include("gates/application.jl")
include("gates/layers.jl")

# ==============================================================================
# Environments: MPS-MPO-MPS contractions for optimization
# ==============================================================================

include("environments.jl")

# ==============================================================================
# Optimization: config, topologies, compression, layer_sweep, variational,
#               direct, gradient, constrained
# ==============================================================================

include("optimization/config.jl")
include("optimization/topologies.jl")
include("optimization/compression.jl")
include("optimization/layer_sweep.jl")
include("optimization/variational.jl")
include("optimization/direct.jl")
include("optimization/gradient.jl")
include("optimization/constrained.jl")

# ==============================================================================
# Synthesis: gate counting, redundancy removal, synthesis pipeline
# ==============================================================================

include("synthesis/gate_counting.jl")
include("synthesis/redundancy.jl")
include("synthesis/pipeline.jl")

# ==============================================================================
# Protocols: analytical, iterative, layer addition
# ==============================================================================

include("protocols/analytical.jl")
include("protocols/iterative.jl")
include("protocols/layer_addition.jl")

# ==============================================================================
# High-level API: decomposition entry point, compilation pipeline
# ==============================================================================

include("decomposition.jl")
include("pipeline.jl")

# ==============================================================================
# Exports
# ==============================================================================

# --- MPS construction ---
export zeroMPS, randMPS, productMPS, ghzMPS
export IdentityMPO
export normalize_mps!, fidelity, max_bond_dim

# --- Canonicalization ---
export canonicalise, canonicalise_FromTo

# --- Inner products ---
export inner

# --- Truncation ---
export truncate_mps, SVD_truncate, pad_zeros

# --- Gates ---
export Thetas2SU2, SU22Thetas
export Thetas2SU4, SU42Thetas, RZs2SU4
export XZXXZX_2_XZX
export apply1Q, apply_gate_efficient
export ApplyGateLayers, Ansatz2MPS, ApplyGatesLR

# --- Optimization ---
export HyperParams
export LayerTopology, STAIRCASE, BRICKWORK, CUSTOM
export generate_layer_indices, generate_circuit_indices, topology_from_symbol, is_non_overlapping_layer
export VarCompress_mps_1site, AdaptVarCompress
export LayerSweep_general
export optimize_layered, optimize_flat, SingleGateEnv
export optimize_single_layer, optimize_circuit_direct
export optimize_gradient
export optimize_constrain

# --- Synthesis ---
export Char2M, GSChars2U, NumTGates
export remove_redundant_rotations, GetRMGaugedCircuit
export SynthesisResult, synthesize, synthesize_rz, synthesize_su2, synthesize_su4
export estimate_t_gates, summarize_synthesis, mps_to_clifford_t

# --- Protocols ---
export truncate_to_chi2
export analytical_decomposition, analytical_decomposition_staircase
export extract_gate_ran2020, extract_staircase_gates, extract_gate_from_bond
export mps_to_staircase_layer, mps_to_layer
export iterative_decomposition, optimize_existing_circuit, add_layer_and_optimize
export LayerInitStrategy, IDENTITY, RANDOM, PREOPTIMIZED
export init_strategy_from_symbol, initialize_layer_gates
export preoptimize_new_layer, add_optimized_layer!, iterative_layer_addition

# --- Decomposition ---
export DecompositionResult, decompose
export apply_decomposition, circuit_to_flat, verify_decomposition
export BenchmarkResult, benchmark_methods

# --- Pipeline ---
export CompilationResult, compile
export to_openqasm, to_qiskit, show_circuit
export circuit_stats, save_circuit

# --- PyCall extension hooks ---
# These functions have no methods by default. When the TenSynthPyCallExt
# extension loads (via `using PyCall`), it adds concrete methods.
# The synthesis pipeline uses try/catch to gracefully handle the no-method case.
function ApproxRZ end
function ApproxSU4 end
function ApproxSU2_trasyn end
function ApproxSU4_trasyn end

export ApproxRZ, ApproxSU4, ApproxSU2_trasyn, ApproxSU4_trasyn

end # module MPS
