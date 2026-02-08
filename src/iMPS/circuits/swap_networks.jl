# SWAP-based gate routing for non-adjacent qubits
# Adapted from iMPS2Circuit/src/Circuits/SwapNetworks.jl

"""
    SwapRoute

Represents a SWAP network route for applying a gate to non-adjacent qubits.
"""
struct SwapRoute
    swaps_before::Vector{Tuple{Int,Int}}
    target_sites::Tuple{Int,Int}
    swaps_after::Vector{Tuple{Int,Int}}
end

# ============================================================================
# Route computation
# ============================================================================

"""
    compute_swap_route(sites::Tuple{Int,Int}, unit_cell::Int) -> SwapRoute

Compute the SWAP network needed to apply a gate to non-adjacent sites.
"""
function compute_swap_route(sites::Tuple{Int,Int}, unit_cell::Int)::SwapRoute
    i1, i2 = sites

    if i1 == i2
        return SwapRoute(Tuple{Int,Int}[], (i1, i2), Tuple{Int,Int}[])
    end

    if i1 > i2
        i1, i2 = i2, i1
    end

    if i2 == i1 + 1 || (i1 == 1 && i2 == unit_cell)
        return SwapRoute(Tuple{Int,Int}[], sites, Tuple{Int,Int}[])
    end

    swaps_before = Tuple{Int,Int}[]
    current_pos = i2

    while current_pos > i1 + 1
        swap_pair = (current_pos - 1, current_pos)
        push!(swaps_before, swap_pair)
        current_pos -= 1
    end

    target_sites = (i1, i1 + 1)
    swaps_after = reverse(swaps_before)

    return SwapRoute(swaps_before, target_sites, swaps_after)
end

"""
    route_distance(route::SwapRoute) -> Int
"""
function route_distance(route::SwapRoute)::Int
    return length(route.swaps_before) + length(route.swaps_after)
end

# ============================================================================
# Circuit with SWAP routing
# ============================================================================

"""
    expand_with_swaps(circuit::ParameterizedCircuit) -> ParameterizedCircuit

Create a new circuit where all non-adjacent two-qubit gates are expanded
into SWAP networks with the gate applied to adjacent qubits.
"""
function expand_with_swaps(circuit::ParameterizedCircuit)::ParameterizedCircuit
    expanded_gates = ParameterizedGate[]

    for gate in circuit.gates
        if is_single_qubit(gate)
            push!(expanded_gates, gate)
            continue
        end

        i1, i2 = gate.qubits

        if abs(i1 - i2) == 1 || (min(i1, i2) == 1 && max(i1, i2) == circuit.n_qubits)
            push!(expanded_gates, gate)
            continue
        end

        route = compute_swap_route(gate.qubits, circuit.n_qubits)

        for swap_sites in route.swaps_before
            swap_gate = _create_swap_gate(swap_sites)
            push!(expanded_gates, swap_gate)
        end

        target_gate = ParameterizedGate(gate.parameterization, route.target_sites, copy(gate.params))
        push!(expanded_gates, target_gate)

        for swap_sites in route.swaps_after
            swap_gate = _create_swap_gate(swap_sites)
            push!(expanded_gates, swap_gate)
        end
    end

    return ParameterizedCircuit(expanded_gates, circuit.n_qubits)
end

"""
    _create_swap_gate(sites::Tuple{Int,Int}) -> ParameterizedGate

Create a SWAP gate as SU4: SWAP = exp(i*π/4*(XX + YY + ZZ)).
"""
function _create_swap_gate(sites::Tuple{Int,Int})
    params = zeros(Float64, 15)
    params[7] = -π/4   # XX
    params[11] = -π/4  # YY
    params[15] = -π/4  # ZZ

    return ParameterizedGate(PauliGeneratorParameterization(), sites, params)
end

# ============================================================================
# SWAP network application to iMPS
# ============================================================================

"""
    apply_gate_with_swaps!(psi::iMPSType, U::Matrix{ComplexF64}, sites::Tuple{Int,Int}, config::BondConfig)

Apply a gate to non-adjacent sites using a SWAP network.
"""
function apply_gate_with_swaps!(psi::iMPSType, U::Matrix{ComplexF64}, sites::Tuple{Int,Int}, config::BondConfig)
    route = compute_swap_route(sites, psi.unit_cell)

    for swap_sites in route.swaps_before
        apply_gate_nn!(psi, SWAP, swap_sites, config)
    end

    apply_gate_nn!(psi, U, route.target_sites, config)

    for swap_sites in route.swaps_after
        apply_gate_nn!(psi, SWAP, swap_sites, config)
    end

    return nothing
end

function apply_gate_with_swaps!(psi::iMPSType, gate::ParameterizedGate, config::BondConfig)
    U = to_matrix(gate)
    apply_gate_with_swaps!(psi, U, gate.qubits, config)
    return nothing
end

# ============================================================================
# Routing analysis
# ============================================================================

"""
    analyze_routing(circuit::ParameterizedCircuit) -> Dict{String, Any}

Analyze the SWAP routing requirements of a circuit.
"""
function analyze_routing(circuit::ParameterizedCircuit)::Dict{String, Any}
    n_swaps = 0
    non_adjacent = 0
    max_dist = 0

    for gate in circuit.gates
        if is_single_qubit(gate)
            continue
        end

        i1, i2 = gate.qubits
        dist = abs(i1 - i2)

        if min(i1, i2) == 1 && max(i1, i2) == circuit.n_qubits
            dist = 1
        end

        max_dist = max(max_dist, dist)

        if dist > 1
            non_adjacent += 1
            route = compute_swap_route(gate.qubits, circuit.n_qubits)
            n_swaps += route_distance(route)
        end
    end

    return Dict{String, Any}(
        "n_swaps_needed" => n_swaps,
        "non_adjacent_gates" => non_adjacent,
        "max_distance" => max_dist,
        "total_gates" => n_gates(circuit),
        "two_qubit_gates" => two_qubit_gate_count(circuit)
    )
end
