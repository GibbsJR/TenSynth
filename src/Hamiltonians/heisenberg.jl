# Isotropic Heisenberg Hamiltonian
# Adapted from iMPS2Circuit/src/Operators/Hamiltonians.jl

"""
    HeisenbergHamiltonian(unit_cell::Int; J::Float64=1.0, h::Float64=0.0) -> SpinLatticeHamiltonian

Create an isotropic Heisenberg Hamiltonian (XXZ with Jxy = Jz = J).

    H = J ∑_⟨i,j⟩ (X_i X_j + Y_i Y_j + Z_i Z_j) + h ∑_i Z_i
"""
function HeisenbergHamiltonian(unit_cell::Int; J::Float64=1.0, h::Float64=0.0)
    return XXZHamiltonian(unit_cell; Jxy=J, Jz=J, h=h)
end
