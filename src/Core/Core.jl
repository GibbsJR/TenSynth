module Core

using LinearAlgebra

# Include submodules in dependency order
include("types.jl")
include("config.jl")
include("constants.jl")
include("linalg.jl")
include("tensor_utils.jl")
include("parameterizations.jl")
include("cost.jl")

# --- Export abstract types ---
export AbstractTensorNetwork
export AbstractQuantumState, AbstractMPS, AbstractMPO, AbstractiMPS
export AbstractGate, AbstractSingleQubitGate, AbstractTwoQubitGate
export AbstractParameterization
export AbstractCircuit, AbstractLayeredCircuit, AbstractParameterizedCircuit
export AbstractHamiltonian, AbstractOptimizer

# --- Export concrete types ---
export FiniteMPS, FiniteMPO, iMPS
export GateMatrix, ParameterizedGate
export GateLayer, LayeredCircuit, ParameterizedCircuit

# --- Export config types ---
export BondConfig, OptimConfig

# --- Export gate constants ---
export PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
export Id, X, Y, Z
export P0, P1
export H_GATE, Hadamard, S_GATE, T_GATE, W
export II, IX, IY, IZ
export XI, XX, XY, XZ
export YI, YX, YY, YZ
export ZI, ZX, ZY, ZZ
export HI, IH, HH
export CNOT, SWAP, FSWAP, XYZ
export Proj11, U_singlet
export PAULI_1Q, PAULI_1Q_NAMES
export PAULI_2Q, PAULI_2Q_NAMES

# --- Export rotation functions ---
export RX, RY, RZ, RZZ, U_gate

# --- Export linalg ---
export robust_svd, polar_unitary, randU, randU_sym

# --- Export tensor utilities ---
export reverse_qubits
export tensor_to_matrix, matrix_to_tensor
export contract_tensors
export apply_matrix_to_tensor!, apply_matrix_to_tensor
export svd_truncate
export is_unitary, is_hermitian

# --- Export parameterizations ---
export KAKParameterization, PauliGeneratorParameterization
export DressedZZParameterization, SingleQubitXYZParameterization
export ZZParameterization, ZZXParameterization
export n_params, to_matrix

# --- Export cost ---
export cost

end # module Core
