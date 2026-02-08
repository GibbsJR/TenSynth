# Circuit to MPO conversion
# Adapted from MPO2Circuit/src/circuit/ansatz.jl

using LinearAlgebra

"""
    circuit_to_mpo(circuit::LayeredCircuit; max_chi::Int=128, max_trunc_err::Float64=1e-14) -> FiniteMPO{ComplexF64}

Convert a quantum circuit to its MPO representation.
"""
function circuit_to_mpo(circuit::LayeredCircuit; max_chi::Int=128, max_trunc_err::Float64=1e-14)::FiniteMPO{ComplexF64}
    mpo = identity_mpo(circuit.n_qubits)
    for layer in circuit.layers
        apply_layer!(mpo, layer; max_chi=max_chi, max_trunc_err=max_trunc_err)
    end
    return mpo
end

"""
    identity_circuit(n_qubits::Int) -> LayeredCircuit
"""
function identity_circuit(n_qubits::Int)::LayeredCircuit
    return LayeredCircuit(n_qubits, GateLayer[])
end

"""
    brick_wall_circuit(n_qubits::Int, depth::Int; random::Bool=true) -> LayeredCircuit

Create a brick-wall structured circuit.
"""
function brick_wall_circuit(n_qubits::Int, depth::Int; random::Bool=true)::LayeredCircuit
    layers = GateLayer[]

    for d in 1:depth
        offset = (d - 1) % 2
        indices = brick_wall_indices(n_qubits, offset)

        if isempty(indices)
            continue
        end

        if random
            push!(layers, random_layer(indices))
        else
            push!(layers, identity_layer(indices))
        end
    end

    return LayeredCircuit(n_qubits, layers)
end

"""
    circuit_depth(circuit::LayeredCircuit) -> Int
"""
circuit_depth(circuit::LayeredCircuit)::Int = length(circuit.layers)

"""
    circuit_gate_count(circuit::LayeredCircuit) -> Int
"""
function circuit_gate_count(circuit::LayeredCircuit)::Int
    return sum(length(layer.gates) for layer in circuit.layers; init=0)
end
