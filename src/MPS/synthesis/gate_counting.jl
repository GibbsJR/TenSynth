# Gate counting utilities
# Adapted from MPS2Circuit/src/synthesis/gate_counting.jl
# Key changes: Id → PAULI_I, Hadamard → H_GATE, S → S_GATE, T → T_GATE, X → PAULI_X, W → W

"""
    Char2M(c::Char) -> Matrix{ComplexF64}

Convert a single character representing a Clifford+T gate to its matrix representation.

Supports both gridsynth format (uppercase) and trasyn format (lowercase):
'I'/'i' → PAULI_I, 'H'/'h' → H_GATE, 'S'/'s' → S_GATE,
'X'/'x' → PAULI_X, 'T'/'t' → T_GATE, 'W'/'w' → W
"""
function Char2M(c::Char)::Matrix{ComplexF64}
    if c == 'I' || c == 'i'
        return PAULI_I
    elseif c == 'H' || c == 'h'
        return H_GATE
    elseif c == 'S' || c == 's'
        return S_GATE
    elseif c == 'X' || c == 'x'
        return PAULI_X
    elseif c == 'T' || c == 't'
        return T_GATE
    elseif c == 'W' || c == 'w'
        return W
    else
        throw(ArgumentError("Unrecognized gate character: $c"))
    end
end

"""
    GSChars2U(gates::String) -> Matrix{ComplexF64}

Convert a gate string (sequence of Clifford+T gate characters) to a unitary matrix.
Gates are applied left-to-right (first character applied first to the state).
"""
function GSChars2U(gates::String)::Matrix{ComplexF64}
    U = Matrix{ComplexF64}(I, 2, 2)
    for g in gates
        U = Char2M(g) * U
    end
    return U
end

"""
    NumTGates(gates::String) -> Int

Count T gates in a gate string. Handles both gridsynth ('T') and trasyn ('t') formats.
"""
function NumTGates(gates::String)::Int
    return count(c -> c == 'T' || c == 't', gates)
end
