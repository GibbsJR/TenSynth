# Suzuki-Trotter decomposition for time evolution
# Adapted from iMPS2Circuit/src/Circuits/Trotterization.jl
# Key refactoring: GateType enum → AbstractParameterization type dispatch

using LinearAlgebra

# ============================================================================
# Trotter order enum
# ============================================================================

@enum TrotterOrder begin
    FIRST_ORDER   # O(dt^2) error per step
    SECOND_ORDER  # O(dt^3) error per step (symmetric Trotter)
    FOURTH_ORDER  # O(dt^5) error per step (Suzuki formula)
end

# ============================================================================
# Trotter decomposition for Hamiltonians
# ============================================================================

"""
    trotterize(H::SpinLatticeHamiltonian, t::Float64, order::TrotterOrder, n_steps::Int;
               parameterization::P=DressedZZParameterization()) -> ParameterizedCircuit

Create a Trotter circuit for time evolution e^{-iHt}.
"""
function trotterize(H::SpinLatticeHamiltonian, t::Float64, order::TrotterOrder, n_steps::Int;
                    parameterization::AbstractParameterization=DressedZZParameterization())
    dt = t / n_steps

    gates = ParameterizedGate[]

    if order == FIRST_ORDER
        for _ in 1:n_steps
            append!(gates, _first_order_step(H, dt, parameterization))
        end
    elseif order == SECOND_ORDER
        append!(gates, _second_order_circuit(H, dt, n_steps, parameterization))
    elseif order == FOURTH_ORDER
        for _ in 1:n_steps
            append!(gates, _fourth_order_step(H, dt, parameterization))
        end
    end

    return ParameterizedCircuit(gates, H.unit_cell)
end

function trotterize(H::SpinLatticeHamiltonian, t::Float64;
                    order::TrotterOrder=SECOND_ORDER, n_steps::Int=10,
                    parameterization::AbstractParameterization=DressedZZParameterization())
    return trotterize(H, t, order, n_steps; parameterization=parameterization)
end

# ============================================================================
# First-order Trotter
# ============================================================================

function _first_order_step(H::SpinLatticeHamiltonian, dt::Float64,
                           parameterization::AbstractParameterization)
    gates = ParameterizedGate[]

    for term in get_two_site_terms(H)
        gate = _term_to_gate(term, dt, parameterization)
        push!(gates, gate)
    end

    for term in get_single_site_terms(H)
        gate = _single_site_term_to_gate(term, dt)
        push!(gates, gate)
    end

    return gates
end

# ============================================================================
# Second-order Trotter (symmetric)
# ============================================================================

function _second_order_circuit(H::SpinLatticeHamiltonian, dt::Float64, n_steps::Int,
                                parameterization::AbstractParameterization)
    gates = ParameterizedGate[]

    single_terms = get_single_site_terms(H)
    two_terms = get_two_site_terms(H)

    # Initial half-step of single-site terms
    for term in single_terms
        gate = _single_site_term_to_gate(term, dt/2)
        push!(gates, gate)
    end

    # Middle steps
    for step in 1:(n_steps-1)
        for term in two_terms
            gate = _term_to_gate(term, dt, parameterization)
            push!(gates, gate)
        end

        for term in single_terms
            gate = _single_site_term_to_gate(term, dt)
            push!(gates, gate)
        end
    end

    # Final two-site step
    for term in two_terms
        gate = _term_to_gate(term, dt, parameterization)
        push!(gates, gate)
    end

    # Final half-step of single-site terms
    for term in single_terms
        gate = _single_site_term_to_gate(term, dt/2)
        push!(gates, gate)
    end

    return gates
end

# ============================================================================
# Fourth-order Trotter (Suzuki formula)
# ============================================================================

function _fourth_order_step(H::SpinLatticeHamiltonian, dt::Float64,
                            parameterization::AbstractParameterization)
    gates = ParameterizedGate[]

    u = 1.0 / (4.0 - 4.0^(1/3))

    single_terms = get_single_site_terms(H)
    two_terms = get_two_site_terms(H)

    append!(gates, _second_order_step(single_terms, two_terms, u*dt, parameterization))
    append!(gates, _second_order_step(single_terms, two_terms, u*dt, parameterization))
    append!(gates, _second_order_step(single_terms, two_terms, (1-4*u)*dt, parameterization))
    append!(gates, _second_order_step(single_terms, two_terms, u*dt, parameterization))
    append!(gates, _second_order_step(single_terms, two_terms, u*dt, parameterization))

    return gates
end

function _second_order_step(single_terms::Vector{LocalTerm}, two_terms::Vector{LocalTerm},
                            dt::Float64, parameterization::AbstractParameterization)
    gates = ParameterizedGate[]

    for term in single_terms
        gate = _single_site_term_to_gate(term, dt/2)
        push!(gates, gate)
    end

    for term in two_terms
        gate = _term_to_gate(term, dt, parameterization)
        push!(gates, gate)
    end

    for term in single_terms
        gate = _single_site_term_to_gate(term, dt/2)
        push!(gates, gate)
    end

    return gates
end

# ============================================================================
# Term to gate conversion
# ============================================================================

function _term_to_gate(term::LocalTerm, dt::Float64,
                       parameterization::AbstractParameterization)
    coeff = term.coefficient
    op = term.operator

    if parameterization isa ZZParameterization
        zz_coeff = _extract_zz_coefficient(op)
        params = [coeff * zz_coeff * dt]
        return ParameterizedGate(ZZParameterization(), term.sites, params)

    elseif parameterization isa DressedZZParameterization
        params = _operator_to_dressed_zz_params(op, coeff * dt)
        return ParameterizedGate(DressedZZParameterization(), term.sites, params)

    elseif parameterization isa PauliGeneratorParameterization
        params = _operator_to_su4_params(op, coeff * dt)
        return ParameterizedGate(PauliGeneratorParameterization(), term.sites, params)

    else
        throw(ArgumentError("Unsupported parameterization: $(typeof(parameterization))"))
    end
end

function _single_site_term_to_gate(term::LocalTerm, dt::Float64)
    coeff = term.coefficient
    op = term.operator

    x_coeff = real(tr(op * PAULI_X)) / 2
    y_coeff = real(tr(op * PAULI_Y)) / 2
    z_coeff = real(tr(op * PAULI_Z)) / 2

    params = [coeff * x_coeff * dt, coeff * y_coeff * dt, coeff * z_coeff * dt]
    site = term.sites[1]
    return ParameterizedGate(SingleQubitXYZParameterization(), (site, site), params)
end

# ============================================================================
# Pauli decomposition helpers
# ============================================================================

function _extract_zz_coefficient(op::Matrix)::Float64
    return real(tr(ZZ * op)) / 4
end

function _operator_to_dressed_zz_params(op::Matrix, scale::Float64)::Vector{Float64}
    x1 = real(tr(XI * op)) / 4
    y1 = real(tr(YI * op)) / 4
    z1 = real(tr(ZI * op)) / 4

    x2 = real(tr(IX * op)) / 4
    y2 = real(tr(IY * op)) / 4
    z2 = real(tr(IZ * op)) / 4

    zz = real(tr(ZZ * op)) / 4

    return [x1 * scale, y1 * scale, z1 * scale, x2 * scale, y2 * scale, z2 * scale, zz * scale]
end

function _operator_to_su4_params(op::Matrix, scale::Float64)::Vector{Float64}
    paulis = [IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]

    params = zeros(Float64, 15)
    for (i, P) in enumerate(paulis)
        params[i] = real(tr(P * op)) / 4 * scale
    end

    return params
end

# ============================================================================
# Specialized Trotter circuits
# ============================================================================

"""
    trotterize_tfim(unit_cell::Int, t::Float64, J::Float64, h::Float64;
                    order::TrotterOrder=SECOND_ORDER, n_steps::Int=10) -> ParameterizedCircuit
"""
function trotterize_tfim(unit_cell::Int, t::Float64, J::Float64, h::Float64;
                         order::TrotterOrder=SECOND_ORDER, n_steps::Int=10)
    if order != SECOND_ORDER
        H = TFIMHamiltonian(unit_cell; J=J, h=h)
        return trotterize(H, t, order, n_steps; parameterization=DressedZZParameterization())
    end

    return _trotterize_tfim_second_order(unit_cell, t, J, h, n_steps)
end

function _trotterize_tfim_second_order(unit_cell::Int, t::Float64, J::Float64, h::Float64,
                                        n_steps::Int)
    dt = t / n_steps
    gates = ParameterizedGate[]

    zz_param = -J * dt
    x_half = -h * dt / 2
    x_full = -h * dt

    for step in 1:n_steps
        if step == 1
            x_param = x_half
        else
            x_param = x_full
        end

        for bond_idx in 1:unit_cell
            site1 = bond_idx
            site2 = mod1(bond_idx + 1, unit_cell)

            if bond_idx == 1
                params = [x_param, 0.0, 0.0, x_param, 0.0, 0.0, zz_param]
            else
                params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zz_param]
            end

            push!(gates, ParameterizedGate(DressedZZParameterization(), (site1, site2), params))
        end
    end

    # Trailing half X rotations
    for site in 1:unit_cell
        params = [x_half, 0.0, 0.0]
        push!(gates, ParameterizedGate(SingleQubitXYZParameterization(), (site, site), params))
    end

    return ParameterizedCircuit(gates, unit_cell)
end

"""
    trotterize_xxz(unit_cell::Int, t::Float64, Jxy::Float64, Jz::Float64;
                   order::TrotterOrder=SECOND_ORDER, n_steps::Int=10) -> ParameterizedCircuit
"""
function trotterize_xxz(unit_cell::Int, t::Float64, Jxy::Float64, Jz::Float64;
                        order::TrotterOrder=SECOND_ORDER, n_steps::Int=10)
    H = XXZHamiltonian(unit_cell; Jxy=Jxy, Jz=Jz, h=0.0)
    return trotterize(H, t, order, n_steps; parameterization=PauliGeneratorParameterization())
end

# ============================================================================
# Trotter error bound
# ============================================================================

function trotter_error_bound(H::SpinLatticeHamiltonian, t::Float64,
                             order::TrotterOrder, n_steps::Int)::Float64
    dt = t / n_steps

    total_norm = 0.0
    for term in H.terms
        total_norm += abs(term.coefficient) * opnorm(term.operator)
    end

    if order == FIRST_ORDER
        return t * dt * total_norm^2
    elseif order == SECOND_ORDER
        return t * dt^2 * total_norm^3
    elseif order == FOURTH_ORDER
        return t * dt^4 * total_norm^5
    else
        return Inf
    end
end
