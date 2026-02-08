# Transverse Field Ising Model (TFIM) Hamiltonian
# Adapted from iMPS2Circuit/src/Operators/Hamiltonians.jl

"""
    TFIMHamiltonian(unit_cell::Int; J::Float64=1.0, h::Float64=1.0) -> SpinLatticeHamiltonian

Create a Transverse Field Ising Model (TFIM) Hamiltonian.

    H = -J ∑_⟨i,j⟩ Z_i Z_j - h ∑_i X_i

# Arguments
- `unit_cell::Int`: Number of sites in the unit cell
- `J::Float64`: Ising coupling strength (default: 1.0)
- `h::Float64`: Transverse field strength (default: 1.0)

# Notes
Each bond Hamiltonian is: -J*ZZ - h/2*(XI + IX), embedding single-site terms
into the two-site operators (split evenly between bonds touching each site).
This follows the convention in Vidal's iTEBD paper.
"""
function TFIMHamiltonian(unit_cell::Int; J::Float64=1.0, h::Float64=1.0)
    terms = LocalTerm[]

    for i in 1:unit_cell
        j = mod1(i + 1, unit_cell)
        H_bond = -J * ZZ - (h/2) * (XI + IX)
        push!(terms, LocalTerm(H_bond, (i, j), 1.0))
    end

    return SpinLatticeHamiltonian(terms, unit_cell)
end
