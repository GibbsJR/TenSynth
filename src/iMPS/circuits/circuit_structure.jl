# Circuit structure for iMPS — ParameterizedCircuit operations
# Adapted from iMPS2Circuit/src/Circuits/CircuitStructure.jl
# Key refactoring: QuantumCircuit → ParameterizedCircuit (from Core/types.jl)
#                  GateType enum → ParameterizedGate{P} type dispatch (from Core)

# ============================================================================
# Constructors
# ============================================================================

"""
    ParameterizedCircuit(gates::Vector{ParameterizedGate}, n_qubits::Int)

Construct a circuit from a list of gates.
"""
function ParameterizedCircuit(gates::Vector{<:ParameterizedGate}, n_qubits::Int)
    for gate in gates
        max_qubit = max(gate.qubits...)
        if max_qubit > n_qubits
            throw(ArgumentError("Gate references qubit $max_qubit but circuit has only $n_qubits qubits"))
        end
        if min(gate.qubits...) < 1
            throw(ArgumentError("Qubit indices must be >= 1"))
        end
    end

    circuit = ParameterizedCircuit(Vector{ParameterizedGate}(gates), n_qubits, Float64[], UnitRange{Int}[], true)
    _rebuild_param_cache!(circuit)
    return circuit
end

"""
    ParameterizedCircuit(n_qubits::Int)

Construct an empty circuit with the given number of qubits.
"""
function ParameterizedCircuit(n_qubits::Int)
    return ParameterizedCircuit(ParameterizedGate[], n_qubits, Float64[], UnitRange{Int}[], false)
end

# ============================================================================
# Parameter caching
# ============================================================================

function _rebuild_param_cache!(circuit::ParameterizedCircuit)
    total_params = sum(length(g.params) for g in circuit.gates; init=0)
    circuit._params_flat = Vector{Float64}(undef, total_params)
    circuit._param_indices = Vector{UnitRange{Int}}(undef, length(circuit.gates))

    idx = 1
    for (i, gate) in enumerate(circuit.gates)
        np = length(gate.params)
        range = idx:(idx + np - 1)
        circuit._param_indices[i] = range
        circuit._params_flat[range] .= gate.params
        idx += np
    end

    circuit._dirty = false
    return nothing
end

# ============================================================================
# Parameter access
# ============================================================================

"""
    get_params(circuit::ParameterizedCircuit) -> Vector{Float64}

Get all circuit parameters as a flat vector. Returns a copy.
"""
function get_params(circuit::ParameterizedCircuit)::Vector{Float64}
    if circuit._dirty
        _rebuild_param_cache!(circuit)
    end
    return copy(circuit._params_flat)
end

"""
    set_params!(circuit::ParameterizedCircuit, params::Vector{Float64})

Set all circuit parameters from a flat vector.
"""
function set_params!(circuit::ParameterizedCircuit, params::Vector{Float64})
    if circuit._dirty
        _rebuild_param_cache!(circuit)
    end

    if length(params) != length(circuit._params_flat)
        throw(ArgumentError("Expected $(length(circuit._params_flat)) parameters, got $(length(params))"))
    end

    circuit._params_flat .= params

    for (i, gate) in enumerate(circuit.gates)
        gate.params .= params[circuit._param_indices[i]]
    end

    return nothing
end

"""
    n_params(circuit::ParameterizedCircuit) -> Int

Get the total number of parameters in the circuit.
"""
function n_params(circuit::ParameterizedCircuit)::Int
    if circuit._dirty
        _rebuild_param_cache!(circuit)
    end
    return length(circuit._params_flat)
end

"""
    n_gates(circuit::ParameterizedCircuit) -> Int
"""
function n_gates(circuit::ParameterizedCircuit)::Int
    return length(circuit.gates)
end

# ============================================================================
# Circuit modification
# ============================================================================

"""
    add_gate!(circuit::ParameterizedCircuit, gate::ParameterizedGate)
"""
function add_gate!(circuit::ParameterizedCircuit, gate::ParameterizedGate)
    max_qubit = max(gate.qubits...)
    if max_qubit > circuit.n_qubits
        throw(ArgumentError("Gate references qubit $max_qubit but circuit has only $(circuit.n_qubits) qubits"))
    end

    push!(circuit.gates, gate)
    circuit._dirty = true
    return nothing
end

"""
    insert_gate!(circuit::ParameterizedCircuit, index::Int, gate::ParameterizedGate)
"""
function insert_gate!(circuit::ParameterizedCircuit, index::Int, gate::ParameterizedGate)
    max_qubit = max(gate.qubits...)
    if max_qubit > circuit.n_qubits
        throw(ArgumentError("Gate references qubit $max_qubit but circuit has only $(circuit.n_qubits) qubits"))
    end

    insert!(circuit.gates, index, gate)
    circuit._dirty = true
    return nothing
end

"""
    remove_gate!(circuit::ParameterizedCircuit, index::Int) -> ParameterizedGate
"""
function remove_gate!(circuit::ParameterizedCircuit, index::Int)
    gate = popat!(circuit.gates, index)
    circuit._dirty = true
    return gate
end

# ============================================================================
# Circuit application
# ============================================================================

"""
    apply_circuit!(psi::iMPSType, circuit::ParameterizedCircuit, config::BondConfig)

Apply all gates in the circuit to the iMPS state.
"""
function apply_circuit!(psi::iMPSType, circuit::ParameterizedCircuit, config::BondConfig)
    for gate in circuit.gates
        apply_gate!(psi, gate, config)
    end
    return nothing
end

function apply_circuit!(psi::iMPSType, circuit::ParameterizedCircuit;
                        max_chi::Int=32, max_trunc_err::Float64=1e-10)
    config = BondConfig(max_chi, max_trunc_err)
    apply_circuit!(psi, circuit, config)
    return nothing
end

# ============================================================================
# Circuit properties
# ============================================================================

"""
    depth(circuit::ParameterizedCircuit) -> Int

Compute the circuit depth (number of layers when gates are parallelized).
"""
function depth(circuit::ParameterizedCircuit)::Int
    if isempty(circuit.gates)
        return 0
    end

    qubit_counts = zeros(Int, circuit.n_qubits)
    for gate in circuit.gates
        for q in unique(gate.qubits)
            qubit_counts[q] += 1
        end
    end

    return maximum(qubit_counts)
end

"""
    is_single_qubit(gate::ParameterizedGate) -> Bool
"""
function is_single_qubit(gate::ParameterizedGate)::Bool
    return gate.qubits[1] == gate.qubits[2]
end

"""
    is_two_qubit(gate::ParameterizedGate) -> Bool
"""
function is_two_qubit(gate::ParameterizedGate)::Bool
    return gate.qubits[1] != gate.qubits[2]
end

"""
    two_qubit_gate_count(circuit::ParameterizedCircuit) -> Int
"""
function two_qubit_gate_count(circuit::ParameterizedCircuit)::Int
    return count(is_two_qubit, circuit.gates)
end

"""
    single_qubit_gate_count(circuit::ParameterizedCircuit) -> Int
"""
function single_qubit_gate_count(circuit::ParameterizedCircuit)::Int
    return count(is_single_qubit, circuit.gates)
end

# ============================================================================
# Copy and iteration
# ============================================================================

function Base.copy(circuit::ParameterizedCircuit)
    return ParameterizedCircuit(copy(circuit.gates), circuit.n_qubits)
end

function Base.deepcopy(circuit::ParameterizedCircuit)
    gates_copy = [deepcopy(g) for g in circuit.gates]
    return ParameterizedCircuit(gates_copy, circuit.n_qubits)
end

Base.iterate(circuit::ParameterizedCircuit) = iterate(circuit.gates)
Base.iterate(circuit::ParameterizedCircuit, state) = iterate(circuit.gates, state)
Base.length(circuit::ParameterizedCircuit) = length(circuit.gates)
Base.getindex(circuit::ParameterizedCircuit, i) = circuit.gates[i]
Base.lastindex(circuit::ParameterizedCircuit) = lastindex(circuit.gates)
Base.firstindex(circuit::ParameterizedCircuit) = firstindex(circuit.gates)

function Base.show(io::IO, circuit::ParameterizedCircuit)
    print(io, "ParameterizedCircuit(n_qubits=$(circuit.n_qubits), n_gates=$(n_gates(circuit)), n_params=$(n_params(circuit)))")
end

# ============================================================================
# Gate constructors
# ============================================================================

"""
    ParameterizedGate(parameterization::P, qubits::Tuple{Int,Int}) where P<:AbstractParameterization

Construct a gate with zero parameters.
"""
function ParameterizedGate(parameterization::P, qubits::Tuple{Int,Int}) where P<:AbstractParameterization
    np = n_params(parameterization)
    return ParameterizedGate{P}(parameterization, qubits, zeros(Float64, np))
end

# Note: ParameterizedGate(parameterization, qubits, params) constructor is provided
# by Core's struct definition. Validation is done at the call sites that need it.

"""
    random_gate(parameterization::P, qubits::Tuple{Int,Int}; scale::Float64=0.3) where P

Create a gate with random parameters.
"""
function random_gate(parameterization::P, qubits::Tuple{Int,Int}; scale::Float64=0.3) where P<:AbstractParameterization
    np = n_params(parameterization)
    params = scale .* randn(np)
    return ParameterizedGate(parameterization, qubits, params)
end

"""
    inverse(gate::ParameterizedGate) -> Matrix{ComplexF64}

Return the inverse (adjoint) of the gate's unitary matrix.
"""
function inverse(gate::ParameterizedGate)::Matrix{ComplexF64}
    return collect(to_matrix(gate)')
end

# ============================================================================
# Ansatz generators
# ============================================================================

"""
    nearest_neighbour_ansatz(n_qubits::Int, n_layers::Int,
                             parameterization::P=PauliGeneratorParameterization()) -> ParameterizedCircuit

Create a nearest-neighbour circuit ansatz with periodic boundary conditions.
"""
function nearest_neighbour_ansatz(n_qubits::Int, n_layers::Int,
                                   parameterization::P=PauliGeneratorParameterization()) where P<:AbstractParameterization
    if n_qubits < 2
        throw(ArgumentError("Need at least 2 qubits for nearest-neighbour ansatz"))
    end

    gates = ParameterizedGate[]

    for layer in 1:n_layers
        for i in 1:(n_qubits-1)
            push!(gates, ParameterizedGate(parameterization, (i, i+1)))
        end
        push!(gates, ParameterizedGate(parameterization, (n_qubits, 1)))
    end

    return ParameterizedCircuit(gates, n_qubits)
end

"""
    random_circuit(n_qubits::Int, n_gates_count::Int,
                   parameterization::P=PauliGeneratorParameterization();
                   scale::Float64=0.3) -> ParameterizedCircuit

Create a circuit with random gates on random adjacent qubit pairs.
"""
function random_circuit(n_qubits::Int, n_gates_count::Int,
                        parameterization::P=PauliGeneratorParameterization();
                        scale::Float64=0.3) where P<:AbstractParameterization
    if n_qubits < 2
        throw(ArgumentError("Need at least 2 qubits"))
    end

    gates = ParameterizedGate[]

    for _ in 1:n_gates_count
        i = rand(1:(n_qubits-1))
        gate = random_gate(parameterization, (i, i+1); scale=scale)
        push!(gates, gate)
    end

    return ParameterizedCircuit(gates, n_qubits)
end
