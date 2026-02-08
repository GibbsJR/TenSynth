# Concrete parameterization subtypes and to_matrix() dispatch
#
# KAKParameterization: 15-param RZ-based decomposition (from MPS2Circuit)
# PauliGeneratorParameterization: exp(i * Σ p_j P_j) (from iMPS2Circuit)
# DressedZZParameterization: 7-param dressed ZZ (from iMPS2Circuit)
# SingleQubitXYZParameterization: 3-param single-qubit (from iMPS2Circuit)
# ZZParameterization: 1-param ZZ (from iMPS2Circuit)
# ZZXParameterization: 3-param ZZX (from iMPS2Circuit)

# ==========================================================================
# Concrete parameterization types
# ==========================================================================

struct KAKParameterization <: AbstractParameterization end
struct PauliGeneratorParameterization <: AbstractParameterization end
struct DressedZZParameterization <: AbstractParameterization end
struct SingleQubitXYZParameterization <: AbstractParameterization end
struct ZZParameterization <: AbstractParameterization end
struct ZZXParameterization <: AbstractParameterization end

"""
    n_params(p::AbstractParameterization) -> Int

Return the number of parameters for a given parameterization.
"""
n_params(::KAKParameterization) = 15
n_params(::PauliGeneratorParameterization) = 15
n_params(::DressedZZParameterization) = 7
n_params(::SingleQubitXYZParameterization) = 3
n_params(::ZZParameterization) = 1
n_params(::ZZXParameterization) = 3

# ==========================================================================
# to_matrix dispatch
# ==========================================================================

"""
    to_matrix(gate::ParameterizedGate) -> Matrix{ComplexF64}

Convert a ParameterizedGate to its unitary matrix representation.
Two-qubit gates return 4×4 matrices; single-qubit gates return 2×2 matrices.
"""
function to_matrix end

"""
KAK parameterization: 15-parameter RZ-based SU(4) decomposition.
Uses H-RZ-H-RZ-H-RZ-H blocks for single-qubit rotations and
CNOT-based circuits for XX, YY, ZZ interactions.
"""
function to_matrix(gate::ParameterizedGate{KAKParameterization})::Matrix{ComplexF64}
    thetas = gate.params
    length(thetas) == 15 || throw(ArgumentError("KAK requires 15 parameters, got $(length(thetas))"))

    RZs = [RZ(Float64(θ)) for θ in thetas]

    U = kron(PAULI_I, PAULI_I)
    ind = 1

    # Leading single-qubit on qubit 2
    U *= kron(PAULI_I, H_GATE)
    U *= kron(PAULI_I, RZs[ind]); ind += 1
    U *= kron(PAULI_I, H_GATE)
    U *= kron(PAULI_I, RZs[ind]); ind += 1
    U *= kron(PAULI_I, H_GATE)
    U *= kron(PAULI_I, RZs[ind]); ind += 1
    U *= kron(PAULI_I, H_GATE)

    # Leading single-qubit on qubit 1
    U *= kron(H_GATE, PAULI_I)
    U *= kron(RZs[ind], PAULI_I); ind += 1
    U *= kron(H_GATE, PAULI_I)
    U *= kron(RZs[ind], PAULI_I); ind += 1
    U *= kron(H_GATE, PAULI_I)
    U *= kron(RZs[ind], PAULI_I); ind += 1
    U *= kron(H_GATE, PAULI_I)

    # XX interaction: H⊗H CNOT (I⊗RZ) CNOT H⊗H
    U *= kron(H_GATE, H_GATE)
    U *= CNOT
    U *= kron(PAULI_I, RZs[ind]); ind += 1
    U *= CNOT
    U *= kron(H_GATE, H_GATE)

    # YY interaction: S⊗S H⊗H CNOT (I⊗RZ) CNOT H⊗H S†⊗S†
    U *= kron(S_GATE, S_GATE)
    U *= kron(H_GATE, H_GATE)
    U *= CNOT
    U *= kron(PAULI_I, RZs[ind]); ind += 1
    U *= CNOT
    U *= kron(H_GATE, H_GATE)
    U *= kron(S_GATE', S_GATE')

    # ZZ interaction: CNOT (I⊗RZ) CNOT
    U *= CNOT
    U *= kron(PAULI_I, RZs[ind]); ind += 1
    U *= CNOT

    # Trailing single-qubit on qubit 2
    U *= kron(PAULI_I, H_GATE)
    U *= kron(PAULI_I, RZs[ind]); ind += 1
    U *= kron(PAULI_I, H_GATE)
    U *= kron(PAULI_I, RZs[ind]); ind += 1
    U *= kron(PAULI_I, H_GATE)
    U *= kron(PAULI_I, RZs[ind]); ind += 1
    U *= kron(PAULI_I, H_GATE)

    # Trailing single-qubit on qubit 1
    U *= kron(H_GATE, PAULI_I)
    U *= kron(RZs[ind], PAULI_I); ind += 1
    U *= kron(H_GATE, PAULI_I)
    U *= kron(RZs[ind], PAULI_I); ind += 1
    U *= kron(H_GATE, PAULI_I)
    U *= kron(RZs[ind], PAULI_I); ind += 1
    U *= kron(H_GATE, PAULI_I)

    return reverse_qubits(U)
end

"""
Pauli generator parameterization: U = exp(i * Σ_j p_j P_j) with 15 Pauli generators.
Order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ.
"""
function to_matrix(gate::ParameterizedGate{PauliGeneratorParameterization})::Matrix{ComplexF64}
    p = gate.params
    length(p) == 15 || throw(ArgumentError("PauliGenerator requires 15 parameters, got $(length(p))"))

    Gen = zeros(ComplexF64, 4, 4)
    Gen += p[1] * IX
    Gen += p[2] * IY
    Gen += p[3] * IZ
    Gen += p[4] * XI
    Gen += p[5] * XX
    Gen += p[6] * XY
    Gen += p[7] * XZ
    Gen += p[8] * YI
    Gen += p[9] * YX
    Gen += p[10] * YY
    Gen += p[11] * YZ
    Gen += p[12] * ZI
    Gen += p[13] * ZX
    Gen += p[14] * ZY
    Gen += p[15] * ZZ

    U = exp(im * Gen)
    return reverse_qubits(U)
end

"""
Dressed ZZ: single-qubit rotations followed by ZZ rotation (7 parameters).
U = exp(i*ZZ*p[7]) * kron(exp(i*(p[4]X+p[5]Y+p[6]Z)), I) * kron(I, exp(i*(p[1]X+p[2]Y+p[3]Z)))
"""
function to_matrix(gate::ParameterizedGate{DressedZZParameterization})::Matrix{ComplexF64}
    p = gate.params
    length(p) == 7 || throw(ArgumentError("DressedZZ requires 7 parameters, got $(length(p))"))

    U = Matrix{ComplexF64}(I, 4, 4)
    U *= kron(PAULI_I, exp(im * (p[1] * PAULI_X + p[2] * PAULI_Y + p[3] * PAULI_Z)))
    U *= kron(exp(im * (p[4] * PAULI_X + p[5] * PAULI_Y + p[6] * PAULI_Z)), PAULI_I)
    U *= exp(im * ZZ * p[7])

    return reverse_qubits(U)
end

"""
Single-qubit XYZ rotation: U = exp(i*(p[1]X + p[2]Y + p[3]Z)) (3 parameters).
Returns a 2×2 matrix.
"""
function to_matrix(gate::ParameterizedGate{SingleQubitXYZParameterization})::Matrix{ComplexF64}
    p = gate.params
    length(p) == 3 || throw(ArgumentError("SingleQubitXYZ requires 3 parameters, got $(length(p))"))

    return exp(im * (p[1] * PAULI_X + p[2] * PAULI_Y + p[3] * PAULI_Z))
end

"""
ZZ rotation via CNOT circuit: CNOT * (I⊗RZ(-2θ)) * CNOT (1 parameter).
"""
function to_matrix(gate::ParameterizedGate{ZZParameterization})::Matrix{ComplexF64}
    p = gate.params
    length(p) == 1 || throw(ArgumentError("ZZ requires 1 parameter, got $(length(p))"))

    U = Matrix{ComplexF64}(I, 4, 4)
    U *= CNOT
    U *= kron(PAULI_I, RZ(-2.0 * p[1]))
    U *= CNOT

    return reverse_qubits(U)
end

"""
ZZX: ZZ rotation followed by X rotations on both qubits (3 parameters).
"""
function to_matrix(gate::ParameterizedGate{ZZXParameterization})::Matrix{ComplexF64}
    p = gate.params
    length(p) == 3 || throw(ArgumentError("ZZX requires 3 parameters, got $(length(p))"))

    U = Matrix{ComplexF64}(I, 4, 4)
    # RZZ via CNOT circuit
    U *= CNOT
    U *= kron(PAULI_I, RZ(-2.0 * p[1]))
    U *= CNOT
    # RX on each qubit via HZH decomposition
    U *= kron(PAULI_I, H_GATE * RZ(-2.0 * p[2]) * H_GATE)
    U *= kron(H_GATE * RZ(-2.0 * p[3]) * H_GATE, PAULI_I)

    return reverse_qubits(U)
end
