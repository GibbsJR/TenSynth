# Suzuki-Trotter decomposition for time evolution
# Merged from: iMPS2Circuit/src/Circuits/Trotterization.jl (ParameterizedCircuit output)
#              MPO2Circuit/src/hamiltonians/tfim.jl (LayeredCircuit output)
# Key changes: GateType enum → AbstractParameterization dispatch,
#              QuantumCircuit → ParameterizedCircuit, Circuit → LayeredCircuit

using LinearAlgebra

# ==========================================================================
# Helper: build ParameterizedCircuit from gate list
# ==========================================================================

"""
    _build_parameterized_circuit(gates, n_qubits) -> ParameterizedCircuit

Build a ParameterizedCircuit from a vector of ParameterizedGate instances.
"""
function _build_parameterized_circuit(gates::Vector{ParameterizedGate}, n_qubits::Int)
    params_flat = Float64[]
    param_indices = UnitRange{Int}[]
    for gate in gates
        start = length(params_flat) + 1
        append!(params_flat, gate.params)
        push!(param_indices, start:length(params_flat))
    end
    return ParameterizedCircuit(gates, n_qubits, params_flat, param_indices, false)
end

# ==========================================================================
# Trotter decomposition: ParameterizedCircuit output (iMPS-style)
# ==========================================================================

"""
    trotterize(H::SpinLatticeHamiltonian, t::Float64;
               order::Symbol=:second, n_steps::Int=10,
               parameterization::AbstractParameterization=DressedZZParameterization())
        -> ParameterizedCircuit

Create a Trotter circuit for time evolution e^{-iHt}.

Returns a ParameterizedCircuit with gates acting on the Hamiltonian's unit cell sites.

# Arguments
- `H`: The Hamiltonian to evolve under
- `t`: Total evolution time
- `order`: Trotter order — `:first`, `:second`, or `:fourth`
- `n_steps`: Number of Trotter steps
- `parameterization`: Gate parameterization for two-qubit gates
"""
function trotterize(H::SpinLatticeHamiltonian, t::Float64;
                    order::Symbol=:second, n_steps::Int=10,
                    parameterization::AbstractParameterization=DressedZZParameterization())
    dt = t / n_steps

    gates = ParameterizedGate[]

    if order == :first
        for _ in 1:n_steps
            append!(gates, _first_order_step(H, dt, parameterization))
        end
    elseif order == :second
        append!(gates, _second_order_circuit(H, dt, n_steps, parameterization))
    elseif order == :fourth
        for _ in 1:n_steps
            append!(gates, _fourth_order_step(H, dt, parameterization))
        end
    else
        throw(ArgumentError("order must be :first, :second, or :fourth, got $order"))
    end

    return _build_parameterized_circuit(gates, H.unit_cell)
end

# ==========================================================================
# First-order Trotter
# ==========================================================================

"""
Generate gates for one first-order Trotter step: e^{-iHdt} ≈ ∏_k e^{-ih_k dt}
"""
function _first_order_step(H::SpinLatticeHamiltonian, dt::Float64,
                           param::AbstractParameterization)
    gates = ParameterizedGate[]

    for term in get_two_site_terms(H)
        push!(gates, _term_to_gate(term, dt, param))
    end

    for term in get_single_site_terms(H)
        push!(gates, _single_site_term_to_gate(term, dt))
    end

    return gates
end

# ==========================================================================
# Second-order Trotter (symmetric)
# ==========================================================================

"""
Generate gates for full second-order Trotter circuit.
e^{-iHdt} ≈ e^{-iA dt/2} e^{-iB dt} e^{-iA dt/2}
"""
function _second_order_circuit(H::SpinLatticeHamiltonian, dt::Float64, n_steps::Int,
                               param::AbstractParameterization)
    gates = ParameterizedGate[]

    single_terms = get_single_site_terms(H)
    two_terms = get_two_site_terms(H)

    # Initial half-step of single-site terms
    for term in single_terms
        push!(gates, _single_site_term_to_gate(term, dt/2))
    end

    # Middle steps
    for step in 1:(n_steps-1)
        for term in two_terms
            push!(gates, _term_to_gate(term, dt, param))
        end
        for term in single_terms
            push!(gates, _single_site_term_to_gate(term, dt))
        end
    end

    # Final two-site step
    for term in two_terms
        push!(gates, _term_to_gate(term, dt, param))
    end

    # Final half-step of single-site terms
    for term in single_terms
        push!(gates, _single_site_term_to_gate(term, dt/2))
    end

    return gates
end

# ==========================================================================
# Fourth-order Trotter (Suzuki formula)
# ==========================================================================

"""
Generate gates for one fourth-order Trotter step using Suzuki's formula.
S4(dt) = S2(u*dt) S2(u*dt) S2((1-4u)*dt) S2(u*dt) S2(u*dt)
where u = 1/(4 - 4^(1/3))
"""
function _fourth_order_step(H::SpinLatticeHamiltonian, dt::Float64,
                            param::AbstractParameterization)
    gates = ParameterizedGate[]

    u = 1.0 / (4.0 - 4.0^(1/3))

    single_terms = get_single_site_terms(H)
    two_terms = get_two_site_terms(H)

    append!(gates, _second_order_step(single_terms, two_terms, u*dt, param))
    append!(gates, _second_order_step(single_terms, two_terms, u*dt, param))
    append!(gates, _second_order_step(single_terms, two_terms, (1-4*u)*dt, param))
    append!(gates, _second_order_step(single_terms, two_terms, u*dt, param))
    append!(gates, _second_order_step(single_terms, two_terms, u*dt, param))

    return gates
end

"""
Single second-order Trotter step for use in higher-order formulas.
"""
function _second_order_step(single_terms::Vector{LocalTerm}, two_terms::Vector{LocalTerm},
                            dt::Float64, param::AbstractParameterization)
    gates = ParameterizedGate[]

    for term in single_terms
        push!(gates, _single_site_term_to_gate(term, dt/2))
    end

    for term in two_terms
        push!(gates, _term_to_gate(term, dt, param))
    end

    for term in single_terms
        push!(gates, _single_site_term_to_gate(term, dt/2))
    end

    return gates
end

# ==========================================================================
# Term → gate conversion (dispatch on parameterization)
# ==========================================================================

"""
    _term_to_gate(term::LocalTerm, dt::Float64, param::AbstractParameterization) -> ParameterizedGate

Convert a two-site Hamiltonian term to a gate implementing e^{-i * term * dt}.
"""
function _term_to_gate(term::LocalTerm, dt::Float64, param::DressedZZParameterization)
    is_two_site(term) || throw(ArgumentError("Expected two-site term"))
    params = _operator_to_dressed_zz_params(term.operator, term.coefficient * dt)
    return ParameterizedGate(DressedZZParameterization(), term.sites, params)
end

function _term_to_gate(term::LocalTerm, dt::Float64, param::ZZParameterization)
    is_two_site(term) || throw(ArgumentError("Expected two-site term"))
    zz_coeff = real(tr(ZZ * term.operator)) / 4
    params = [term.coefficient * zz_coeff * dt]
    return ParameterizedGate(ZZParameterization(), term.sites, params)
end

function _term_to_gate(term::LocalTerm, dt::Float64, param::PauliGeneratorParameterization)
    is_two_site(term) || throw(ArgumentError("Expected two-site term"))
    params = _operator_to_su4_params(term.operator, term.coefficient * dt)
    return ParameterizedGate(PauliGeneratorParameterization(), term.sites, params)
end

"""
    _single_site_term_to_gate(term::LocalTerm, dt::Float64) -> ParameterizedGate

Convert a single-site Hamiltonian term to a gate implementing e^{-i * term * dt}.
"""
function _single_site_term_to_gate(term::LocalTerm, dt::Float64)
    is_single_site(term) || throw(ArgumentError("Expected single-site term"))

    coeff = term.coefficient
    op = term.operator

    x_coeff = real(tr(op * X)) / 2
    y_coeff = real(tr(op * Y)) / 2
    z_coeff = real(tr(op * Z)) / 2

    params = [coeff * x_coeff * dt, coeff * y_coeff * dt, coeff * z_coeff * dt]
    site = term.sites[1]
    return ParameterizedGate(SingleQubitXYZParameterization(), (site, site), params)
end

# ==========================================================================
# Pauli decomposition helpers
# ==========================================================================

"""
Convert a 4x4 operator to DressedZZ parameters [x1, y1, z1, x2, y2, z2, zz].
"""
function _operator_to_dressed_zz_params(op::Matrix, scale::Float64)::Vector{Float64}
    x1 = real(tr(XI * op)) / 4
    y1 = real(tr(YI * op)) / 4
    z1 = real(tr(ZI * op)) / 4
    x2 = real(tr(IX * op)) / 4
    y2 = real(tr(IY * op)) / 4
    z2 = real(tr(IZ * op)) / 4
    zz = real(tr(ZZ * op)) / 4
    return [x1*scale, y1*scale, z1*scale, x2*scale, y2*scale, z2*scale, zz*scale]
end

"""
Convert a 4x4 Hermitian operator to SU4 generator parameters (15 parameters).
"""
function _operator_to_su4_params(op::Matrix, scale::Float64)::Vector{Float64}
    paulis = [XI, YI, ZI, IX, IY, IZ, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ]
    params = zeros(Float64, 15)
    for (i, P) in enumerate(paulis)
        params[i] = real(tr(P * op)) / 4 * scale
    end
    return params
end

# ==========================================================================
# Specialized Trotter circuits (ParameterizedCircuit output)
# ==========================================================================

"""
    trotterize_tfim(unit_cell::Int, t::Float64, J::Float64, h::Float64;
                    order::Symbol=:second, n_steps::Int=10) -> ParameterizedCircuit

Convenience function for TFIM Trotter circuit.
"""
function trotterize_tfim(unit_cell::Int, t::Float64, J::Float64, h::Float64;
                         order::Symbol=:second, n_steps::Int=10)
    H = TFIMHamiltonian(unit_cell; J=J, h=h)
    return trotterize(H, t; order=order, n_steps=n_steps, parameterization=DressedZZParameterization())
end

"""
    trotterize_xxz(unit_cell::Int, t::Float64, Jxy::Float64, Jz::Float64;
                   order::Symbol=:second, n_steps::Int=10) -> ParameterizedCircuit

Convenience function for XXZ Trotter circuit.
"""
function trotterize_xxz(unit_cell::Int, t::Float64, Jxy::Float64, Jz::Float64;
                        order::Symbol=:second, n_steps::Int=10)
    H = XXZHamiltonian(unit_cell; Jxy=Jxy, Jz=Jz, h=0.0)
    return trotterize(H, t; order=order, n_steps=n_steps, parameterization=PauliGeneratorParameterization())
end

# ==========================================================================
# Finite-chain Trotter circuit: LayeredCircuit output (MPO-style)
# Adapted from MPO2Circuit/src/hamiltonians/tfim.jl
# ==========================================================================

"""
    tfim_trotter_circuit(n_qubits::Int, dt::Float64, h::Float64;
                         order::Symbol=:second, n_steps::Int=1) -> LayeredCircuit

Create a Trotter circuit for TFIM time evolution on a finite chain.

Returns a LayeredCircuit with explicit unitary gate matrices, suitable for
the MPO backend.

# Arguments
- `n_qubits`: Number of qubits (must be even)
- `dt`: Total time step
- `h`: Transverse field strength
- `order`: Trotter order — `:first`, `:second`, or `:fourth`
- `n_steps`: Number of Trotter steps

# Notes
The TFIM Hamiltonian H = -∑ Z_i Z_{i+1} - h ∑ X_i is split into:
- H_odd: ZZ terms on bonds (1,2),(3,4),... plus transverse field
- H_even: ZZ terms on bonds (2,3),(4,5),...
"""
function tfim_trotter_circuit(n_qubits::Int, dt::Float64, h::Float64;
                              order::Symbol=:second, n_steps::Int=1)
    n_qubits % 2 == 0 || throw(ArgumentError("n_qubits must be even, got $n_qubits"))
    n_steps >= 1 || throw(ArgumentError("n_steps must be >= 1"))

    H_odd = ZZ + h * (XI + IX)
    H_even = ComplexF64.(ZZ)  # ensure ComplexF64

    layers = GateLayer[]

    if order == :first
        _add_tfim_first_order!(layers, n_qubits, dt, n_steps, H_odd, H_even)
    elseif order == :second
        _add_tfim_second_order!(layers, n_qubits, dt, n_steps, H_odd, H_even)
    elseif order == :fourth
        _add_tfim_fourth_order!(layers, n_qubits, dt, n_steps, H_odd, H_even)
    else
        throw(ArgumentError("order must be :first, :second, or :fourth, got $order"))
    end

    return LayeredCircuit(n_qubits, layers)
end

# --- First-order Trotter (LayeredCircuit) ---

function _add_tfim_first_order!(layers::Vector{GateLayer}, n_qubits::Int, dt::Float64,
                                n_steps::Int, H_odd::Matrix{ComplexF64}, H_even::Matrix{ComplexF64})
    dt_step = dt / n_steps

    U_odd = exp(-im * dt_step * H_odd)
    U_even = exp(-im * dt_step * H_even)

    odd_indices = [(2i-1, 2i) for i in 1:(n_qubits÷2)]
    even_indices = [(2i, 2i+1) for i in 1:(n_qubits÷2 - 1)]

    for _ in 1:n_steps
        push!(layers, GateLayer(
            [GateMatrix(Matrix(U_odd), "TFIM_odd", Dict{Symbol,Any}(:dt => dt_step)) for _ in odd_indices],
            odd_indices
        ))
        if !isempty(even_indices)
            push!(layers, GateLayer(
                [GateMatrix(Matrix(U_even), "TFIM_even", Dict{Symbol,Any}(:dt => dt_step)) for _ in even_indices],
                even_indices
            ))
        end
    end
end

# --- Second-order Trotter (LayeredCircuit) ---

function _add_tfim_second_order!(layers::Vector{GateLayer}, n_qubits::Int, dt::Float64,
                                 n_steps::Int, H_odd::Matrix{ComplexF64}, H_even::Matrix{ComplexF64})
    dt_step = dt / n_steps

    U_odd_half = exp(-im * (dt_step / 2) * H_odd)
    U_even_half = exp(-im * (dt_step / 2) * H_even)

    odd_indices = [(2i-1, 2i) for i in 1:(n_qubits÷2)]
    even_indices = [(2i, 2i+1) for i in 1:(n_qubits÷2 - 1)]

    for _ in 1:n_steps
        push!(layers, GateLayer(
            [GateMatrix(Matrix(U_odd_half), "TFIM_odd_half", Dict{Symbol,Any}(:dt => dt_step/2)) for _ in odd_indices],
            odd_indices
        ))
        if !isempty(even_indices)
            push!(layers, GateLayer(
                [GateMatrix(Matrix(U_even_half), "TFIM_even_half", Dict{Symbol,Any}(:dt => dt_step/2)) for _ in even_indices],
                even_indices
            ))
            push!(layers, GateLayer(
                [GateMatrix(Matrix(U_even_half), "TFIM_even_half", Dict{Symbol,Any}(:dt => dt_step/2)) for _ in even_indices],
                even_indices
            ))
        end
        push!(layers, GateLayer(
            [GateMatrix(Matrix(U_odd_half), "TFIM_odd_half", Dict{Symbol,Any}(:dt => dt_step/2)) for _ in odd_indices],
            odd_indices
        ))
    end
end

# --- Fourth-order Trotter (LayeredCircuit) ---

function _add_tfim_fourth_order!(layers::Vector{GateLayer}, n_qubits::Int, dt::Float64,
                                 n_steps::Int, H_odd::Matrix{ComplexF64}, H_even::Matrix{ComplexF64})
    p2 = 1 / (4 - 4^(1/3))
    dt_step = dt / n_steps
    dt_edge = p2 * dt_step
    dt_middle = (1 - 4*p2) * dt_step

    for _ in 1:n_steps
        _add_single_s2!(layers, n_qubits, dt_edge, H_odd, H_even)
        _add_single_s2!(layers, n_qubits, dt_edge, H_odd, H_even)
        _add_single_s2!(layers, n_qubits, dt_middle, H_odd, H_even)
        _add_single_s2!(layers, n_qubits, dt_edge, H_odd, H_even)
        _add_single_s2!(layers, n_qubits, dt_edge, H_odd, H_even)
    end
end

"""
Add a single S2(dt) block (4 layers) to the circuit.
"""
function _add_single_s2!(layers::Vector{GateLayer}, n_qubits::Int, dt::Float64,
                         H_odd::Matrix{ComplexF64}, H_even::Matrix{ComplexF64})
    U_odd_half = exp(-im * (dt / 2) * H_odd)
    U_even_half = exp(-im * (dt / 2) * H_even)

    odd_indices = [(2i-1, 2i) for i in 1:(n_qubits÷2)]
    even_indices = [(2i, 2i+1) for i in 1:(n_qubits÷2 - 1)]

    push!(layers, GateLayer(
        [GateMatrix(Matrix(U_odd_half), "TFIM_odd", Dict{Symbol,Any}(:dt => dt/2)) for _ in odd_indices],
        odd_indices
    ))
    if !isempty(even_indices)
        push!(layers, GateLayer(
            [GateMatrix(Matrix(U_even_half), "TFIM_even", Dict{Symbol,Any}(:dt => dt/2)) for _ in even_indices],
            even_indices
        ))
        push!(layers, GateLayer(
            [GateMatrix(Matrix(U_even_half), "TFIM_even", Dict{Symbol,Any}(:dt => dt/2)) for _ in even_indices],
            even_indices
        ))
    end
    push!(layers, GateLayer(
        [GateMatrix(Matrix(U_odd_half), "TFIM_odd", Dict{Symbol,Any}(:dt => dt/2)) for _ in odd_indices],
        odd_indices
    ))
end

# ==========================================================================
# Trotter error bound
# ==========================================================================

"""
    trotter_error_bound(H::SpinLatticeHamiltonian, t::Float64;
                        order::Symbol=:second, n_steps::Int=10) -> Float64

Estimate the Trotter error bound (rough upper bound).
"""
function trotter_error_bound(H::SpinLatticeHamiltonian, t::Float64;
                             order::Symbol=:second, n_steps::Int=10)
    dt = t / n_steps

    total_norm = 0.0
    for term in H.terms
        total_norm += abs(term.coefficient) * opnorm(term.operator)
    end

    if order == :first
        return t * dt * total_norm^2
    elseif order == :second
        return t * dt^2 * total_norm^3
    elseif order == :fourth
        return t * dt^4 * total_norm^5
    else
        return Inf
    end
end
