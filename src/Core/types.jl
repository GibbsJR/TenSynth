# Abstract type hierarchy and concrete types for TenSynth

# Top-level abstract type
abstract type AbstractTensorNetwork end

# Quantum states
abstract type AbstractQuantumState <: AbstractTensorNetwork end
abstract type AbstractMPS <: AbstractQuantumState end
abstract type AbstractMPO <: AbstractQuantumState end
abstract type AbstractiMPS <: AbstractQuantumState end

# Gates
abstract type AbstractGate <: AbstractTensorNetwork end
abstract type AbstractSingleQubitGate <: AbstractGate end
abstract type AbstractTwoQubitGate <: AbstractGate end

# Parameterization schemes
abstract type AbstractParameterization <: AbstractTensorNetwork end

# Circuits
abstract type AbstractCircuit <: AbstractTensorNetwork end
abstract type AbstractLayeredCircuit <: AbstractCircuit end
abstract type AbstractParameterizedCircuit <: AbstractCircuit end

# Hamiltonians and optimizers
abstract type AbstractHamiltonian <: AbstractTensorNetwork end
abstract type AbstractOptimizer <: AbstractTensorNetwork end

# ==========================================================================
# Concrete tensor network types
# ==========================================================================

struct FiniteMPS{T<:Number} <: AbstractMPS
    tensors::Vector{Array{T, 3}}  # [left_bond, physical, right_bond]
end

struct FiniteMPO{T<:Number} <: AbstractMPO
    tensors::Vector{Array{T, 4}}  # [left_bond, phys_out, phys_in, right_bond]
end

mutable struct iMPS{T<:Number} <: AbstractiMPS
    gamma::Vector{Array{T, 3}}
    lambda::Vector{Matrix{T}}
    unit_cell::Int
    physical_dim::Int
    normalized::Bool
    _gamma_absorbed::Union{Nothing, Vector{Array{T,3}}}

    function iMPS{T}(
        gamma::Vector{Array{T,3}},
        lambda::Vector{Matrix{T}},
        unit_cell::Int,
        physical_dim::Int,
        normalized::Bool,
        _gamma_absorbed::Union{Nothing, Vector{Array{T,3}}}
    ) where T
        length(gamma) == unit_cell || throw(ArgumentError("gamma length must equal unit_cell"))
        length(lambda) == unit_cell || throw(ArgumentError("lambda length must equal unit_cell"))
        unit_cell > 0 || throw(ArgumentError("unit_cell must be positive"))
        physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
        new{T}(gamma, lambda, unit_cell, physical_dim, normalized, _gamma_absorbed)
    end
end

# ==========================================================================
# Gate types
# ==========================================================================

struct GateMatrix <: AbstractTwoQubitGate
    matrix::Matrix{ComplexF64}
    name::String
    params::Dict{Symbol, Any}
end

mutable struct ParameterizedGate{P<:AbstractParameterization} <: AbstractTwoQubitGate
    parameterization::P
    qubits::Tuple{Int, Int}
    params::Vector{Float64}
end

# ==========================================================================
# Circuit types
# ==========================================================================

struct GateLayer
    gates::Vector{GateMatrix}
    indices::Vector{Tuple{Int, Int}}
end

mutable struct LayeredCircuit <: AbstractLayeredCircuit
    n_qubits::Int
    layers::Vector{GateLayer}
end

mutable struct ParameterizedCircuit <: AbstractParameterizedCircuit
    gates::Vector{ParameterizedGate}
    n_qubits::Int
    _params_flat::Vector{Float64}
    _param_indices::Vector{UnitRange{Int}}
    _dirty::Bool
end
