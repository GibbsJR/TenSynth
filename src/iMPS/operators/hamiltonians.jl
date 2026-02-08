# Spin lattice Hamiltonians for iMPS

using LinearAlgebra

struct LocalTerm
    operator::Matrix{ComplexF64}
    sites::Tuple{Int, Int}
    coefficient::Float64
end

function get_matrix(term::LocalTerm)
    return term.coefficient * term.operator
end

struct SpinLatticeHamiltonian <: AbstractHamiltonian
    terms::Vector{LocalTerm}
    unit_cell::Int
end

SpinLatticeHamiltonian(unit_cell::Int) = SpinLatticeHamiltonian(LocalTerm[], unit_cell)

function add_term!(H::SpinLatticeHamiltonian, term::LocalTerm)
    push!(H.terms, term)
    return H
end

function get_two_site_terms(H::SpinLatticeHamiltonian)
    return [t for t in H.terms if t.sites[1] != t.sites[2]]
end

function get_single_site_terms(H::SpinLatticeHamiltonian)
    return [t for t in H.terms if t.sites[1] == t.sites[2]]
end

function get_interaction_sites(H::SpinLatticeHamiltonian)
    sites = Set{Tuple{Int,Int}}()
    for t in H.terms
        if t.sites[1] != t.sites[2]
            push!(sites, t.sites)
        end
    end
    return collect(sites)
end

function get_local_hamiltonian(H::SpinLatticeHamiltonian, sites::Tuple{Int,Int})
    result = zeros(ComplexF64, 4, 4)
    for t in H.terms
        if t.sites == sites
            result .+= get_matrix(t)
        end
    end
    return result
end

function get_single_site_hamiltonian(H::SpinLatticeHamiltonian, site::Int)
    result = zeros(ComplexF64, 2, 2)
    for t in H.terms
        if t.sites == (site, site)
            result .+= get_matrix(t)
        end
    end
    return result
end

# Standard Hamiltonians

function TFIMHamiltonian(unit_cell::Int; J::Float64=1.0, h::Float64=1.0)
    H = SpinLatticeHamiltonian(unit_cell)
    for i in 1:unit_cell
        j = mod1(i + 1, unit_cell)
        add_term!(H, LocalTerm(kron(PAULI_Z, PAULI_Z), (i, j), -J))
        add_term!(H, LocalTerm(kron(PAULI_X, PAULI_I), (i, j), -h / 2))
        add_term!(H, LocalTerm(kron(PAULI_I, PAULI_X), (i, j), -h / 2))
    end
    return H
end

function XXZHamiltonian(unit_cell::Int; Jxy::Float64=1.0, Jz::Float64=1.0, h::Float64=0.0)
    H = SpinLatticeHamiltonian(unit_cell)
    for i in 1:unit_cell
        j = mod1(i + 1, unit_cell)
        add_term!(H, LocalTerm(kron(PAULI_X, PAULI_X), (i, j), Jxy))
        add_term!(H, LocalTerm(kron(PAULI_Y, PAULI_Y), (i, j), Jxy))
        add_term!(H, LocalTerm(kron(PAULI_Z, PAULI_Z), (i, j), Jz))
        if abs(h) > 0
            add_term!(H, LocalTerm(kron(PAULI_Z, PAULI_I), (i, j), -h / 2))
            add_term!(H, LocalTerm(kron(PAULI_I, PAULI_Z), (i, j), -h / 2))
        end
    end
    return H
end

function HeisenbergHamiltonian(unit_cell::Int; J::Float64=1.0, h::Float64=0.0)
    return XXZHamiltonian(unit_cell; Jxy=J, Jz=J, h=h)
end

function XYHamiltonian(unit_cell::Int; J::Float64=1.0, h::Float64=0.0)
    return XXZHamiltonian(unit_cell; Jxy=J, Jz=0.0, h=h)
end
