# XY Model Hamiltonian
# Adapted from iMPS2Circuit/src/Operators/Hamiltonians.jl

"""
    XYHamiltonian(unit_cell::Int; J::Float64=1.0, h::Float64=0.0) -> SpinLatticeHamiltonian

Create an XY model Hamiltonian (XXZ with Jz = 0).

    H = J ∑_⟨i,j⟩ (X_i X_j + Y_i Y_j) + h ∑_i Z_i
"""
function XYHamiltonian(unit_cell::Int; J::Float64=1.0, h::Float64=0.0)
    return XXZHamiltonian(unit_cell; Jxy=J, Jz=0.0, h=h)
end
