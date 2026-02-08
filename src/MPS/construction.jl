# MPS construction functions
# Adapted from MPS2Circuit/src/mps/construction.jl
# Key change: Vector{Array{ComplexF64, 3}} → FiniteMPS{ComplexF64} (uses .tensors field)

using LinearAlgebra

"""
    zeroMPS(N::Int) -> FiniteMPS{ComplexF64}

Create an MPS representing the all-zeros state |00...0⟩.
"""
function zeroMPS(N::Int)::FiniteMPS{ComplexF64}
    tensors = [reshape(ComplexF64[1.0, 0.0], 1, 2, 1) for _ in 1:N]
    return FiniteMPS{ComplexF64}(tensors)
end

"""
    randMPS(N::Int, D::Int) -> FiniteMPS{ComplexF64}

Create a random MPS in right-canonical form with maximum bond dimension D.
"""
function randMPS(N::Int, D::Int)::FiniteMPS{ComplexF64}
    tensors = Vector{Array{ComplexF64, 3}}()

    for i in 1:N-1
        right_dim = Int(min(D, 2.0^i))
        left_dim = Int(min(D, 2.0^(i-1)))

        M = rand(ComplexF64, right_dim, 2 * left_dim)
        F = svd(M)
        prepend!(tensors, [reshape(F.Vt, right_dim, 2, left_dim)])
    end

    # First site: left bond dimension is 1
    last_right_dim = Int(min(D, 2.0^(N-1)))
    M = rand(ComplexF64, 1, 2, last_right_dim)
    prepend!(tensors, [M ./ norm(M)])

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    IdentityMPO(N::Int) -> FiniteMPO{ComplexF64}

Create an MPO representing the identity operator.
Convention: [left_bond, phys_out, phys_in, right_bond]
"""
function IdentityMPO(N::Int)::FiniteMPO{ComplexF64}
    tensors = [reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1) for _ in 1:N]
    return FiniteMPO{ComplexF64}(tensors)
end

# ============================================================================
# Convenience Constructors
# ============================================================================

"""
    productMPS(states::Vector{Int}) -> FiniteMPS{ComplexF64}

Create an MPS representing a product state from a vector of qubit states.
Each element specifies the state of that qubit: 0 for |0⟩, 1 for |1⟩.
"""
function productMPS(states::Vector{Int})::FiniteMPS{ComplexF64}
    isempty(states) && throw(ArgumentError("states vector cannot be empty"))

    for (i, s) in enumerate(states)
        s in (0, 1) || throw(ArgumentError("states[$i] = $s is invalid; must be 0 or 1"))
    end

    tensors = Vector{Array{ComplexF64, 3}}(undef, length(states))
    for (i, s) in enumerate(states)
        if s == 0
            tensors[i] = reshape(ComplexF64[1.0, 0.0], 1, 2, 1)
        else
            tensors[i] = reshape(ComplexF64[0.0, 1.0], 1, 2, 1)
        end
    end
    return FiniteMPS{ComplexF64}(tensors)
end

"""
    productMPS(bitstring::String) -> FiniteMPS{ComplexF64}

Create an MPS representing a product state from a bitstring.
"""
function productMPS(bitstring::String)::FiniteMPS{ComplexF64}
    isempty(bitstring) && throw(ArgumentError("bitstring cannot be empty"))

    for (i, c) in enumerate(bitstring)
        c in ('0', '1') || throw(ArgumentError("bitstring[$i] = '$c' is invalid; must be '0' or '1'"))
    end

    states = [c == '1' ? 1 : 0 for c in bitstring]
    return productMPS(states)
end

"""
    ghzMPS(N::Int) -> FiniteMPS{ComplexF64}

Create an MPS representing the GHZ state (|00...0⟩ + |11...1⟩)/√2.
"""
function ghzMPS(N::Int)::FiniteMPS{ComplexF64}
    N >= 2 || throw(ArgumentError("GHZ state requires at least 2 qubits, got N=$N"))

    tensors = Vector{Array{ComplexF64, 3}}(undef, N)

    first = zeros(ComplexF64, 1, 2, 2)
    first[1, 1, 1] = 1.0 / sqrt(2)
    first[1, 2, 2] = 1.0 / sqrt(2)
    tensors[1] = first

    for i in 2:N-1
        middle = zeros(ComplexF64, 2, 2, 2)
        middle[1, 1, 1] = 1.0
        middle[2, 2, 2] = 1.0
        tensors[i] = middle
    end

    last = zeros(ComplexF64, 2, 2, 1)
    last[1, 1, 1] = 1.0
    last[2, 2, 1] = 1.0
    tensors[N] = last

    return FiniteMPS{ComplexF64}(tensors)
end

"""
    wMPS(N::Int) -> FiniteMPS{ComplexF64}

Create an MPS representing the W state (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√N.
"""
function wMPS(N::Int)::FiniteMPS{ComplexF64}
    N >= 2 || throw(ArgumentError("W state requires at least 2 qubits, got N=$N"))

    tensors = Vector{Array{ComplexF64, 3}}(undef, N)

    first = zeros(ComplexF64, 1, 2, 2)
    first[1, 1, 1] = 1.0
    first[1, 2, 2] = 1.0 / sqrt(N)
    tensors[1] = first

    for i in 2:N-1
        middle = zeros(ComplexF64, 2, 2, 2)
        middle[1, 1, 1] = 1.0
        middle[1, 2, 2] = 1.0 / sqrt(N)
        middle[2, 1, 2] = 1.0
        tensors[i] = middle
    end

    last = zeros(ComplexF64, 2, 2, 1)
    last[1, 2, 1] = 1.0 / sqrt(N)
    last[2, 1, 1] = 1.0
    tensors[N] = last

    return FiniteMPS{ComplexF64}(tensors)
end

# ============================================================================
# MPS Inspection Utilities
# ============================================================================

"""
    bond_dimensions(mps::FiniteMPS) -> Vector{Int}

Get the internal bond dimensions of an MPS.
"""
function bond_dimensions(mps::FiniteMPS)::Vector{Int}
    t = mps.tensors
    isempty(t) && return Int[]
    length(t) == 1 && return Int[]
    return [size(t[i], 3) for i in 1:length(t)-1]
end

"""
    is_normalized(mps::FiniteMPS; tol::Float64=1e-10) -> Bool

Check if an MPS is normalized (⟨ψ|ψ⟩ ≈ 1).
"""
function is_normalized(mps::FiniteMPS; tol::Float64=1e-10)::Bool
    norm_sq = real(inner(mps, mps))
    return abs(norm_sq - 1.0) < tol
end

"""
    normalize_mps!(mps::FiniteMPS) -> FiniteMPS

Normalize an MPS in-place so that ⟨ψ|ψ⟩ = 1.
"""
function normalize_mps!(mps::FiniteMPS)::FiniteMPS
    norm_sq = real(inner(mps, mps))
    if norm_sq > 0
        mps.tensors[1] = mps.tensors[1] / sqrt(norm_sq)
    end
    return mps
end

"""
    entanglement_entropy(mps::FiniteMPS, site::Int) -> Float64

Compute the von Neumann entanglement entropy at a bipartition.
"""
function entanglement_entropy(mps::FiniteMPS, site::Int)::Float64
    N = length(mps.tensors)
    (site < 1 || site >= N) && throw(BoundsError("site must be in 1:$(N-1), got $site"))

    mps_canon = canonicalise(mps, site)

    T = mps_canon.tensors[site]
    T_r = reshape(T, size(T, 1) * size(T, 2), size(T, 3))

    F = svd(T_r)
    singular_values = F.S

    sv_squared = singular_values.^2
    sv_squared = sv_squared / sum(sv_squared)

    entropy = 0.0
    for p in sv_squared
        if p > 1e-15
            entropy -= p * log(p)
        end
    end

    return entropy
end

"""
    fidelity(mps1::FiniteMPS, mps2::FiniteMPS) -> Float64

Compute the fidelity |⟨ψ₁|ψ₂⟩|² / (⟨ψ₁|ψ₁⟩⟨ψ₂|ψ₂⟩) between two MPS.
"""
function fidelity(mps1::FiniteMPS, mps2::FiniteMPS)::Float64
    overlap = inner(mps1, mps2)
    norm1_sq = real(inner(mps1, mps1))
    norm2_sq = real(inner(mps2, mps2))

    if norm1_sq < 1e-15 || norm2_sq < 1e-15
        return 0.0
    end

    return abs(overlap)^2 / (norm1_sq * norm2_sq)
end

"""
    max_bond_dim(mps::FiniteMPS) -> Int

Get the maximum bond dimension of an MPS.
"""
function max_bond_dim(mps::FiniteMPS)::Int
    dims = bond_dimensions(mps)
    isempty(dims) && return 0
    return maximum(dims)
end
