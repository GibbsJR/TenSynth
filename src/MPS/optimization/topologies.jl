# Circuit layer topology definitions for MPS decomposition
# Adapted from MPS2Circuit/src/optimization/topologies.jl

"""
    LayerTopology

Enum for circuit layer topologies.
- `STAIRCASE`: Sequential gates (1,2), (2,3), ... - overlapping
- `BRICKWORK`: Alternating even/odd non-overlapping layers
- `CUSTOM`: User-specified gate indices
"""
@enum LayerTopology begin
    STAIRCASE
    BRICKWORK
    CUSTOM
end

"""
    generate_layer_indices(N, topology, layer_num) -> Vector{Tuple{Int,Int}}

Generate gate indices for a single layer based on the specified topology.
"""
function generate_layer_indices(N::Int, topology::LayerTopology, layer_num::Int)::Vector{Tuple{Int,Int}}
    N >= 2 || throw(ArgumentError("N must be at least 2, got N=$N"))
    layer_num >= 1 || throw(ArgumentError("layer_num must be positive"))

    if topology == STAIRCASE
        return [(i, i+1) for i in 1:(N-1)]
    elseif topology == BRICKWORK
        if isodd(layer_num)
            return [(i, i+1) for i in 1:2:(N-1)]
        else
            return [(i, i+1) for i in 2:2:(N-1)]
        end
    elseif topology == CUSTOM
        throw(ArgumentError("CUSTOM topology requires explicit indices"))
    else
        throw(ArgumentError("Unknown topology: $topology"))
    end
end

"""
    is_non_overlapping_layer(layer_inds) -> Bool

Check if a layer has no overlapping gates (no shared qubits between gates).
"""
function is_non_overlapping_layer(layer_inds::Vector{Tuple{Int,Int}})::Bool
    if length(layer_inds) <= 1
        return true
    end
    qubits_used = Set{Int}()
    for (i, j) in layer_inds
        if i in qubits_used || j in qubits_used
            return false
        end
        push!(qubits_used, i)
        push!(qubits_used, j)
    end
    return true
end

"""
    generate_circuit_indices(N, topology, n_layers) -> Vector{Vector{Tuple{Int,Int}}}

Generate gate indices for an entire circuit.
"""
function generate_circuit_indices(N::Int, topology::LayerTopology, n_layers::Int)::Vector{Vector{Tuple{Int,Int}}}
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    n_layers >= 0 || throw(ArgumentError("n_layers must be non-negative"))
    if topology == CUSTOM
        throw(ArgumentError("CUSTOM topology requires explicit indices"))
    end
    return [generate_layer_indices(N, topology, layer) for layer in 1:n_layers]
end

"""
    topology_from_symbol(sym::Symbol) -> LayerTopology

Convert a symbol to a LayerTopology enum value.
"""
function topology_from_symbol(sym::Symbol)::LayerTopology
    if sym == :staircase
        return STAIRCASE
    elseif sym == :brickwork
        return BRICKWORK
    elseif sym == :custom
        return CUSTOM
    else
        throw(ArgumentError("Unknown topology symbol: $sym"))
    end
end

gates_per_staircase_layer(N::Int) = N - 1

function gates_per_brickwork_layer(N::Int, layer_num::Int)::Int
    return isodd(layer_num) ? N ÷ 2 : (N - 1) ÷ 2
end
