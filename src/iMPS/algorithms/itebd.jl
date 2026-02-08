# Imaginary Time Evolving Block Decimation (iTEBD) for iMPS ground states
# Adapted from iMPS2Circuit/src/Algorithms/iTEBD.jl

using LinearAlgebra

# ============================================================================
# iTEBD Configuration and Result
# ============================================================================

struct iTEBDConfig
    max_chi::Int
    max_trunc_err::Float64
    dt_schedule::Vector{Float64}
    n_steps_per_dt::Int
    converge_threshold::Float64
    verbose::Bool
end

function iTEBDConfig(; max_chi::Int=64, max_trunc_err::Float64=1e-10,
                      dt_schedule::Vector{Float64}=Float64[],
                      n_steps_per_dt::Int=20, converge_threshold::Float64=1e-5,
                      verbose::Bool=true)
    if isempty(dt_schedule)
        dt_schedule = [10.0^(-i) for i in 1.0:0.5:4.0]
    end
    return iTEBDConfig(max_chi, max_trunc_err, dt_schedule, n_steps_per_dt,
                       converge_threshold, verbose)
end

struct iTEBDResult{T}
    psi::iMPSType{T}
    energy::Float64
    converged::Bool
    energy_history::Vector{Float64}
end

# ============================================================================
# iTEBD Ground State
# ============================================================================

"""
    itebd_ground_state(H::SpinLatticeHamiltonian; config::iTEBDConfig=iTEBDConfig())

Find the ground state of a Hamiltonian using imaginary time evolution (iTEBD).
"""
function itebd_ground_state(H::SpinLatticeHamiltonian; config::iTEBDConfig=iTEBDConfig())
    unit_cell = H.unit_cell
    psi = random_product_state(unit_cell)
    plus_state!(psi)

    return itebd_ground_state!(psi, H; config=config)
end

"""
    itebd_ground_state!(psi::iMPSType, H::SpinLatticeHamiltonian; config::iTEBDConfig=iTEBDConfig())

Find ground state starting from existing iMPS (in-place).
"""
function itebd_ground_state!(psi::iMPSType{T}, H::SpinLatticeHamiltonian;
                              config::iTEBDConfig=iTEBDConfig()) where T
    energy_history = Float64[]
    converged = false
    energy = Inf

    bond_config = BondConfig(config.max_chi, config.max_trunc_err)

    for dt in config.dt_schedule
        if config.verbose
            println("\n--- dt = $dt ---")
        end

        prev_energy = Inf

        for step in 1:config.n_steps_per_dt
            apply_itebd_step!(psi, H, dt, bond_config)

            energy = compute_energy(psi, H)
            push!(energy_history, energy)

            if config.verbose && step % 5 == 1
                chi_avg = _mean([size(psi.lambda[i], 1) for i in 1:psi.unit_cell])
                @info "Step $step: E = $energy, χ_avg = $chi_avg"
            end

            if abs(energy - prev_energy) / max(abs(prev_energy), 1e-10) < config.converge_threshold / 10
                if config.verbose
                    @info "Converged at step $step"
                end
                converged = true
                break
            end

            prev_energy = energy
        end
    end

    absorb_bonds!(psi)
    energy = compute_energy(psi, H)

    return iTEBDResult{T}(psi, energy, converged, energy_history)
end

# ============================================================================
# iTEBD Step Application
# ============================================================================

"""
    apply_itebd_step!(psi::iMPSType, H::SpinLatticeHamiltonian, dt::Float64, config::BondConfig)

Apply one second-order Suzuki-Trotter imaginary time evolution step.
"""
function apply_itebd_step!(psi::iMPSType{T}, H::SpinLatticeHamiltonian, dt::Float64,
                            config::BondConfig) where T
    two_site_terms = get_two_site_terms(H)
    single_site_terms = get_single_site_terms(H)

    # First half step: forward sweep
    for term in two_site_terms
        U = exp(-dt/2 * get_matrix(term))
        apply_gate_nn!(psi, U, term.sites, config)
    end

    for term in single_site_terms
        U = exp(-dt/2 * get_matrix(term))
        apply_gate_single!(psi, U, term.sites[1])
    end

    # Second half step: backward sweep
    for term in reverse(single_site_terms)
        U = exp(-dt/2 * get_matrix(term))
        apply_gate_single!(psi, U, term.sites[1])
    end

    for term in reverse(two_site_terms)
        U = exp(-dt/2 * get_matrix(term))
        apply_gate_nn!(psi, U, term.sites, config)
    end

    psi.normalized = false
    absorb_bonds!(psi)

    return psi
end

"""
    apply_itebd_step_fourth_order!(psi::iMPSType, H::SpinLatticeHamiltonian, dt::Float64, config::BondConfig)

Apply one fourth-order Suzuki-Trotter imaginary time evolution step.
"""
function apply_itebd_step_fourth_order!(psi::iMPSType{T}, H::SpinLatticeHamiltonian,
                                         dt::Float64, config::BondConfig) where T
    p = 1.0 / (4.0 - 4.0^(1/3))

    for _ in 1:2
        apply_itebd_step!(psi, H, p * dt, config)
    end
    apply_itebd_step!(psi, H, (1 - 4*p) * dt, config)
    for _ in 1:2
        apply_itebd_step!(psi, H, p * dt, config)
    end

    return psi
end

# ============================================================================
# Energy Computation
# ============================================================================

"""
    compute_energy(psi::iMPSType, H::SpinLatticeHamiltonian) -> Float64

Compute the energy per site of an iMPS with respect to a Hamiltonian.
"""
function compute_energy(psi::iMPSType{T}, H::SpinLatticeHamiltonian) where T
    n = psi.unit_cell
    total_energy = 0.0

    I4 = Matrix{ComplexF64}(I, 4, 4)

    two_site_terms = get_two_site_terms(H)

    for term in two_site_terms
        e_raw = _expectation_two_site_canonical(psi, get_matrix(term), term.sites)
        norm_val = _expectation_two_site_canonical(psi, I4, term.sites)
        total_energy += real(e_raw) / real(norm_val)
    end

    return total_energy / n
end

"""
    _expectation_two_site_canonical(psi::iMPSType, op::Matrix, sites::Tuple{Int,Int}) -> ComplexF64

Compute expectation value of a two-site operator using the canonical form formula.
"""
function _expectation_two_site_canonical(psi::iMPSType{T}, op::Matrix{ComplexF64},
                                          sites::Tuple{Int,Int}) where T
    n = psi.unit_cell
    i1, i2 = sites

    if i2 == mod1(i1 + 1, n)
        return _compute_bond_expectation(psi, op, i1, i2)
    elseif i1 == mod1(i2 + 1, n)
        d = 2
        op_swapped = _swap_operator_indices(op, d)
        return _compute_bond_expectation(psi, op_swapped, i2, i1)
    else
        @warn "Non-adjacent two-site operator not supported: sites=$sites"
        return zero(ComplexF64)
    end
end

function _swap_operator_indices(op::Matrix{ComplexF64}, d::Int)
    op_t = reshape(op, d, d, d, d)
    op_swapped = permutedims(op_t, (2, 1, 4, 3))
    return reshape(op_swapped, d*d, d*d)
end

"""
    _compute_bond_expectation(psi::iMPSType, op::Matrix, i1::Int, i2::Int) -> ComplexF64

Compute expectation value of a two-site operator on bond (i1, i2) where i2 = i1+1.
"""
function _compute_bond_expectation(psi::iMPSType{T}, op::Matrix{ComplexF64},
                                    i1::Int, i2::Int) where T
    n = psi.unit_cell
    d = 2

    λ_left = psi.lambda[mod1(i1-1, n)]
    γ1 = psi.gamma[i1]
    λ_mid = psi.lambda[i1]
    γ2 = psi.gamma[i2]
    λ_right = psi.lambda[i2]

    op_t = reshape(op, d, d, d, d)

    χL = size(γ1, 1)
    χM = size(γ1, 3)
    χR = size(γ2, 3)

    result = zero(ComplexF64)

    @inbounds for α in 1:χL
        λL_sq = abs2(λ_left[α, α])

        for γ_idx in 1:χR
            λR_sq = abs2(λ_right[γ_idx, γ_idx])
            weight = λL_sq * λR_sq

            for i in 1:d, j in 1:d, ip in 1:d, jp in 1:d
                bra_val = zero(ComplexF64)
                for β in 1:χM
                    λM = λ_mid[β, β]
                    bra_val += conj(γ1[α, ip, β] * λM * γ2[β, jp, γ_idx])
                end

                ket_val = zero(ComplexF64)
                for β in 1:χM
                    λM = λ_mid[β, β]
                    ket_val += γ1[α, i, β] * λM * γ2[β, j, γ_idx]
                end

                result += weight * bra_val * op_t[ip, jp, i, j] * ket_val
            end
        end
    end

    return result
end

# ============================================================================
# Convenience Functions
# ============================================================================

function itebd_tfim_ground_state(unit_cell::Int, J::Float64, h::Float64;
                                  max_chi::Int=64, verbose::Bool=true)
    H = TFIMHamiltonian(unit_cell; J=J, h=h)
    config = iTEBDConfig(max_chi=max_chi, verbose=verbose)
    return itebd_ground_state(H; config=config)
end

function itebd_xxz_ground_state(unit_cell::Int, Jxy::Float64, Jz::Float64;
                                 h::Float64=0.0, max_chi::Int=64, verbose::Bool=true)
    H = XXZHamiltonian(unit_cell; Jxy=Jxy, Jz=Jz, h=h)
    config = iTEBDConfig(max_chi=max_chi, verbose=verbose)
    return itebd_ground_state(H; config=config)
end

function itebd_heisenberg_ground_state(unit_cell::Int; J::Float64=1.0,
                                        max_chi::Int=64, verbose::Bool=true)
    H = HeisenbergHamiltonian(unit_cell; J=J)
    config = iTEBDConfig(max_chi=max_chi, verbose=verbose)
    return itebd_ground_state(H; config=config)
end

_mean(x) = sum(x) / length(x)
