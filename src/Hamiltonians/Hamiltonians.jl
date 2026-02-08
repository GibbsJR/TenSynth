module Hamiltonians

# Hamiltonians module: unified Hamiltonian definitions and Trotter circuit generation.
# Merges iMPS2Circuit's Hamiltonians + Trotterization with MPO2Circuit's Trotter circuits.

using LinearAlgebra
using ..Core

# ==========================================================================
# Types: LocalTerm, SpinLatticeHamiltonian
# ==========================================================================

include("types.jl")

# ==========================================================================
# Standard Hamiltonian models
# ==========================================================================

include("tfim.jl")
include("xxz.jl")
include("heisenberg.jl")
include("xy.jl")

# ==========================================================================
# Trotter circuit generation
# ==========================================================================

include("trotter.jl")

# ==========================================================================
# Exports
# ==========================================================================

# --- Types ---
export LocalTerm, SpinLatticeHamiltonian

# --- Type utilities ---
export is_single_site, is_two_site, get_matrix
export add_term!, get_two_site_terms, get_single_site_terms
export get_interaction_sites, get_local_hamiltonian, get_single_site_hamiltonian

# --- Hamiltonian constructors ---
export TFIMHamiltonian, XXZHamiltonian, HeisenbergHamiltonian, XYHamiltonian

# --- Trotter circuits ---
export trotterize, trotterize_tfim, trotterize_xxz
export tfim_trotter_circuit
export trotter_error_bound

end # module Hamiltonians
