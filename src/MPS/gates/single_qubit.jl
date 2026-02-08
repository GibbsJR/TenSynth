# Single-qubit parameterized gates and decomposition
# Adapted from MPS2Circuit/src/gates/single_qubit.jl
# Note: RZ, U_gate are already defined in Core/constants.jl
# This file adds MPS-specific single-qubit utilities: XZX_gate, SU2 decomposition

using LinearAlgebra
using Optim

"""
    RemovePhase(U::Matrix{ComplexF64}) -> Matrix{ComplexF64}

Remove the global phase from a unitary matrix by making det(U)^(1/n) = 1.
"""
function RemovePhase(U::Matrix{ComplexF64})::Matrix{ComplexF64}
    n = size(U, 1)
    d = det(U)
    phase = d^(1/n)
    return U / phase
end

"""
    XZX_gate(theta1, theta2, theta3) -> Matrix{ComplexF64}

Single-qubit gate decomposed as Rx(theta1) Rz(theta2) Rx(theta3).
Implemented using the identity Rx(theta) = H Rz(theta) H.
"""
function XZX_gate(theta1::Real, theta2::Real, theta3::Real)::Matrix{ComplexF64}
    return Hadamard * RZ(theta3) * Hadamard * RZ(theta2) * Hadamard * RZ(theta1) * Hadamard
end

"""
    RZs2SU2(RZs::Vector) -> Matrix{ComplexF64}

Compose three RZ gates with Hadamards to form an arbitrary SU(2) element.
Constructs: H * RZ_3 * H * RZ_2 * H * RZ_1 * H
"""
function RZs2SU2(RZs::Vector)::Matrix{ComplexF64}
    U = Matrix{ComplexF64}(Id)
    ind = 1
    U *= Hadamard
    U *= RZs[ind]; ind += 1
    U *= Hadamard
    U *= RZs[ind]; ind += 1
    U *= Hadamard
    U *= RZs[ind]; ind += 1
    U *= Hadamard
    return U
end

"""
    Thetas2SU2(thetas::Vector{Float64}) -> Matrix{ComplexF64}

Convert three rotation angles to an SU(2) matrix via H-RZ-H decomposition.
"""
function Thetas2SU2(thetas::Vector{Float64})::Matrix{ComplexF64}
    RZs = [RZ(theta) for theta in thetas]
    return RZs2SU2(RZs)
end

"""
    SU22Thetas(U_targ::Matrix{ComplexF64}; max_iters=1000, tol=1e-8) -> Vector{Float64}

Decompose an SU(2) matrix into three RZ rotation angles using LBFGS optimization.
"""
function SU22Thetas(U_targ::Matrix{ComplexF64}; max_iters::Int=1000, tol::Float64=1e-8)::Vector{Float64}
    function cost(thetas)
        return 1 - abs(tr(U_targ' * Thetas2SU2(thetas)))^2 / 4
    end

    for i in 1:max_iters
        thetas_init = randn(Float64, 3)
        res = Optim.optimize(cost, thetas_init, Optim.LBFGS(m=30),
                             Optim.Options(x_abstol=1e-6, f_abstol=1e-6, g_tol=1e-12,
                                          iterations=10^5, show_trace=false))
        if Optim.minimum(res) < tol
            return Optim.minimizer(res)
        end
    end

    throw(ArgumentError("SU22Thetas: optimization did not converge within $max_iters iterations"))
end

"""
    Rz2Theta(U::Matrix{ComplexF64}) -> Float64

Extract the rotation angle from an RZ gate matrix.
"""
function Rz2Theta(U::Matrix{ComplexF64})::Float64
    U_temp = RemovePhase(U)
    return real(-im * log(U_temp[2, 2]))
end
