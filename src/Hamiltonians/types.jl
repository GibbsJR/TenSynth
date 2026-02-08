# Hamiltonian types: LocalTerm and SpinLatticeHamiltonian
# Adapted from iMPS2Circuit/src/Operators/Hamiltonians.jl
# Key changes: uses TenSynth.Core types

using LinearAlgebra

"""
    LocalTerm

A single local term in a spin Hamiltonian.

# Fields
- `operator::Matrix{ComplexF64}`: The local operator matrix (2x2 for single-site, 4x4 for two-site)
- `sites::Tuple{Int, Int}`: Site indices where the operator acts (1-based, within unit cell)
- `coefficient::Float64`: Real coefficient multiplying this term
"""
struct LocalTerm
    operator::Matrix{ComplexF64}
    sites::Tuple{Int, Int}
    coefficient::Float64

    function LocalTerm(
        operator::Matrix{ComplexF64},
        sites::Tuple{Int, Int},
        coefficient::Float64
    )
        d = size(operator, 1)
        d == size(operator, 2) || throw(ArgumentError("Operator must be square"))
        (d == 2 || d == 4) || throw(ArgumentError("Operator must be 2x2 (single-site) or 4x4 (two-site)"))
        sites[1] > 0 || throw(ArgumentError("Site indices must be positive"))
        sites[2] > 0 || throw(ArgumentError("Site indices must be positive"))
        new(operator, sites, coefficient)
    end
end

"""
    LocalTerm(operator::Matrix{ComplexF64}, site::Int, coefficient::Float64=1.0)

Convenience constructor for single-site terms.
"""
function LocalTerm(operator::Matrix{ComplexF64}, site::Int, coefficient::Float64=1.0)
    size(operator) == (2, 2) || throw(ArgumentError("Single-site operator must be 2x2"))
    LocalTerm(operator, (site, site), coefficient)
end

"""
    LocalTerm(operator::Matrix{ComplexF64}, sites::Tuple{Int, Int})

Convenience constructor with default coefficient of 1.0.
"""
function LocalTerm(operator::Matrix{ComplexF64}, sites::Tuple{Int, Int})
    LocalTerm(operator, sites, 1.0)
end

"""
    is_single_site(term::LocalTerm) -> Bool

Check if this term acts on a single site.
"""
is_single_site(term::LocalTerm) = term.sites[1] == term.sites[2]

"""
    is_two_site(term::LocalTerm) -> Bool

Check if this term acts on two sites.
"""
is_two_site(term::LocalTerm) = term.sites[1] != term.sites[2]

"""
    get_matrix(term::LocalTerm) -> Matrix{ComplexF64}

Get the full operator matrix including the coefficient.
"""
get_matrix(term::LocalTerm) = term.coefficient * term.operator

# ==========================================================================
# SpinLatticeHamiltonian
# ==========================================================================

"""
    SpinLatticeHamiltonian <: AbstractHamiltonian

A Hamiltonian for a spin lattice system with periodic boundary conditions.

# Fields
- `terms::Vector{LocalTerm}`: Collection of local terms
- `unit_cell::Int`: Number of sites in the unit cell
"""
struct SpinLatticeHamiltonian <: AbstractHamiltonian
    terms::Vector{LocalTerm}
    unit_cell::Int

    function SpinLatticeHamiltonian(terms::Vector{LocalTerm}, unit_cell::Int)
        unit_cell > 0 || throw(ArgumentError("Unit cell size must be positive"))
        for term in terms
            max_site = max(term.sites[1], term.sites[2])
            max_site <= unit_cell || throw(ArgumentError(
                "Site index $max_site exceeds unit cell size $unit_cell"
            ))
        end
        new(terms, unit_cell)
    end
end

"""
    SpinLatticeHamiltonian(unit_cell::Int)

Create an empty Hamiltonian with the given unit cell size.
"""
SpinLatticeHamiltonian(unit_cell::Int) = SpinLatticeHamiltonian(LocalTerm[], unit_cell)

"""
    add_term!(H::SpinLatticeHamiltonian, term::LocalTerm) -> SpinLatticeHamiltonian

Add a term to the Hamiltonian. Returns the modified Hamiltonian.
"""
function add_term!(H::SpinLatticeHamiltonian, term::LocalTerm)
    max_site = max(term.sites[1], term.sites[2])
    max_site <= H.unit_cell || throw(ArgumentError(
        "Site index $max_site exceeds unit cell size $(H.unit_cell)"
    ))
    push!(H.terms, term)
    return H
end

"""
    get_two_site_terms(H::SpinLatticeHamiltonian) -> Vector{LocalTerm}

Get all two-site interaction terms.
"""
get_two_site_terms(H::SpinLatticeHamiltonian) = filter(is_two_site, H.terms)

"""
    get_single_site_terms(H::SpinLatticeHamiltonian) -> Vector{LocalTerm}

Get all single-site (field) terms.
"""
get_single_site_terms(H::SpinLatticeHamiltonian) = filter(is_single_site, H.terms)

"""
    get_interaction_sites(H::SpinLatticeHamiltonian) -> Vector{Tuple{Int, Int}}

Get all unique pairs of interacting sites.
"""
function get_interaction_sites(H::SpinLatticeHamiltonian)
    sites = Tuple{Int, Int}[]
    for term in H.terms
        if is_two_site(term) && !(term.sites in sites)
            push!(sites, term.sites)
        end
    end
    return sites
end

"""
    get_local_hamiltonian(H::SpinLatticeHamiltonian, sites::Tuple{Int, Int}) -> Matrix{ComplexF64}

Get the total two-site Hamiltonian matrix acting on the given pair of sites.
"""
function get_local_hamiltonian(H::SpinLatticeHamiltonian, sites::Tuple{Int, Int})::Matrix{ComplexF64}
    result = zeros(ComplexF64, 4, 4)
    for term in H.terms
        if term.sites == sites && is_two_site(term)
            result .+= get_matrix(term)
        end
    end
    return result
end

"""
    get_single_site_hamiltonian(H::SpinLatticeHamiltonian, site::Int) -> Matrix{ComplexF64}

Get the total single-site Hamiltonian matrix acting on the given site.
"""
function get_single_site_hamiltonian(H::SpinLatticeHamiltonian, site::Int)::Matrix{ComplexF64}
    result = zeros(ComplexF64, 2, 2)
    for term in H.terms
        if is_single_site(term) && term.sites[1] == site
            result .+= get_matrix(term)
        end
    end
    return result
end

function Base.show(io::IO, H::SpinLatticeHamiltonian)
    n_single = count(is_single_site, H.terms)
    n_two = count(is_two_site, H.terms)
    print(io, "SpinLatticeHamiltonian(unit_cell=$(H.unit_cell), single_site_terms=$n_single, two_site_terms=$n_two)")
end
