# Unified gate constants and rotation functions
# Base: iMPS2Circuit's Constants.jl, augmented with MPS2Circuit extras.

# ==========================================================================
# Single-qubit Pauli matrices
# ==========================================================================

const PAULI_I = ComplexF64[1 0; 0 1]
const PAULI_X = ComplexF64[0 1; 1 0]
const PAULI_Y = ComplexF64[0 -im; im 0]
const PAULI_Z = ComplexF64[1 0; 0 -1]

# Aliases
const Id = PAULI_I
const X = PAULI_X
const Y = PAULI_Y
const Z = PAULI_Z

# Projectors
const P0 = ComplexF64[1 0; 0 0]
const P1 = ComplexF64[0 0; 0 1]

# ==========================================================================
# Common single-qubit gates
# ==========================================================================

const H_GATE = ComplexF64(1/sqrt(2)) * ComplexF64[1 1; 1 -1]
const Hadamard = H_GATE

const S_GATE = ComplexF64[1 0; 0 im]
const T_GATE = ComplexF64[1 0; 0 exp(im*π/4)]
const W = ComplexF64[exp(im*π/4) 0; 0 exp(im*π/4)]

# ==========================================================================
# Two-qubit Pauli basis (16 elements)
# ==========================================================================

const II = kron(PAULI_I, PAULI_I)
const IX = kron(PAULI_I, PAULI_X)
const IY = kron(PAULI_I, PAULI_Y)
const IZ = kron(PAULI_I, PAULI_Z)
const XI = kron(PAULI_X, PAULI_I)
const XX = kron(PAULI_X, PAULI_X)
const XY = kron(PAULI_X, PAULI_Y)
const XZ = kron(PAULI_X, PAULI_Z)
const YI = kron(PAULI_Y, PAULI_I)
const YX = kron(PAULI_Y, PAULI_X)
const YY = kron(PAULI_Y, PAULI_Y)
const YZ = kron(PAULI_Y, PAULI_Z)
const ZI = kron(PAULI_Z, PAULI_I)
const ZX = kron(PAULI_Z, PAULI_X)
const ZY = kron(PAULI_Z, PAULI_Y)
const ZZ = kron(PAULI_Z, PAULI_Z)

# Two-qubit Hadamard combinations
const HI = kron(H_GATE, PAULI_I)
const IH = kron(PAULI_I, H_GATE)
const HH = kron(H_GATE, H_GATE)

# Heisenberg interaction term
const XYZ = XX + YY + ZZ

# ==========================================================================
# Standard two-qubit gates
# ==========================================================================

const CNOT = ComplexF64[
    1 0 0 0;
    0 1 0 0;
    0 0 0 1;
    0 0 1 0
]

const SWAP = ComplexF64[
    1 0 0 0;
    0 0 1 0;
    0 1 0 0;
    0 0 0 1
]

const FSWAP = ComplexF64[
    1 0 0 0;
    0 0 1 0;
    0 1 0 0;
    0 0 0 -1
]

const Proj11 = ComplexF64[
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 1
]

const U_singlet = ComplexF64[
     0.0        1.0        0.0       0.0;
     1/sqrt(2)  0.0       -1/sqrt(2) 0.0;
    -1/sqrt(2)  0.0       -1/sqrt(2) 0.0;
     0.0        0.0        0.0      -1.0
]

# ==========================================================================
# Pauli collections for iteration
# ==========================================================================

const PAULI_1Q = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
const PAULI_1Q_NAMES = ["I", "X", "Y", "Z"]

const PAULI_2Q = [II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]
const PAULI_2Q_NAMES = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ",
                         "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]

# ==========================================================================
# Rotation functions
# ==========================================================================

"""
    RZ(theta::Real) -> Matrix{ComplexF64}

Rotation around Z-axis: exp(-iθ/2 Z) = diag(exp(-iθ/2), exp(iθ/2)).
"""
function RZ(theta::Real)::Matrix{ComplexF64}
    return ComplexF64[exp(-im*theta/2) 0; 0 exp(im*theta/2)]
end

"""
    RX(theta::Real) -> Matrix{ComplexF64}

Rotation around X-axis: exp(-iθ/2 X).
"""
function RX(theta::Real)::Matrix{ComplexF64}
    c = cos(theta/2)
    s = sin(theta/2)
    return ComplexF64[c -im*s; -im*s c]
end

"""
    RY(theta::Real) -> Matrix{ComplexF64}

Rotation around Y-axis: exp(-iθ/2 Y).
"""
function RY(theta::Real)::Matrix{ComplexF64}
    c = cos(theta/2)
    s = sin(theta/2)
    return ComplexF64[c -s; s c]
end

"""
    RZZ(theta::Real) -> Matrix{ComplexF64}

Two-qubit ZZ rotation: exp(-iθ/2 ZZ).
"""
function RZZ(theta::Real)::Matrix{ComplexF64}
    return exp(-im * theta/2 * ZZ)
end

"""
    U_gate(theta::Real, lam::Real, phi::Real) -> Matrix{ComplexF64}

Standard parameterized single-qubit gate (U3 gate in IBM notation).
"""
function U_gate(theta::Real, lam::Real, phi::Real)::Matrix{ComplexF64}
    return ComplexF64[
        cos(theta/2)                 -exp(im*lam)*sin(theta/2);
        exp(im*phi)*sin(theta/2)      exp(im*(phi+lam))*cos(theta/2)
    ]
end
