# XXZ Heisenberg Hamiltonian
# Adapted from iMPS2Circuit/src/Operators/Hamiltonians.jl

"""
    XXZHamiltonian(unit_cell::Int; Jxy::Float64=1.0, Jz::Float64=1.0, h::Float64=0.0) -> SpinLatticeHamiltonian

Create an XXZ Heisenberg Hamiltonian.

    H = Jxy ∑_⟨i,j⟩ (X_i X_j + Y_i Y_j) + Jz ∑_⟨i,j⟩ Z_i Z_j + h ∑_i Z_i

# Arguments
- `unit_cell::Int`: Number of sites in the unit cell
- `Jxy::Float64`: XY coupling strength (default: 1.0)
- `Jz::Float64`: Z coupling strength (default: 1.0)
- `h::Float64`: Longitudinal field strength (default: 0.0)

# Notes
- Jxy = Jz gives the isotropic Heisenberg model
- Jz = 0 gives the XX model
- Jxy = 0 gives the Ising model
"""
function XXZHamiltonian(unit_cell::Int; Jxy::Float64=1.0, Jz::Float64=1.0, h::Float64=0.0)
    terms = LocalTerm[]

    for i in 1:unit_cell
        j = mod1(i + 1, unit_cell)
        if i < j || (i == unit_cell && j == 1)
            if abs(Jxy) > eps()
                push!(terms, LocalTerm(XX, (i, j), Jxy))
                push!(terms, LocalTerm(YY, (i, j), Jxy))
            end
            if abs(Jz) > eps()
                push!(terms, LocalTerm(ZZ, (i, j), Jz))
            end
        end
    end

    if abs(h) > eps()
        for i in 1:unit_cell
            push!(terms, LocalTerm(Z, i, h))
        end
    end

    return SpinLatticeHamiltonian(terms, unit_cell)
end
