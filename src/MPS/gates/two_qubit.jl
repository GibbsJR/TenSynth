# Two-qubit gate decomposition via 15-parameter SU(4) representation
# Adapted from MPS2Circuit/src/gates/two_qubit.jl
# Key changes: S → S_GATE, Id → PAULI_I (aliased), RSVD → polar_unitary

using LinearAlgebra
using Optim

"""
    RZs2SU4(RZs::Vector) -> Matrix{ComplexF64}

Compose 15 RZ gates with Hadamards and CNOTs to form an arbitrary SU(4) element.
The 15-parameter decomposition provides reliable optimization convergence.
"""
function RZs2SU4(RZs::Vector)::Matrix{ComplexF64}
    U = kron(Id, Id)
    ind = 1

    # Leading single-qubit on qubit 2
    U *= kron(Id, Hadamard)
    U *= kron(Id, RZs[ind]); ind += 1
    U *= kron(Id, Hadamard)
    U *= kron(Id, RZs[ind]); ind += 1
    U *= kron(Id, Hadamard)
    U *= kron(Id, RZs[ind]); ind += 1
    U *= kron(Id, Hadamard)

    # Leading single-qubit on qubit 1
    U *= kron(Hadamard, Id)
    U *= kron(RZs[ind], Id); ind += 1
    U *= kron(Hadamard, Id)
    U *= kron(RZs[ind], Id); ind += 1
    U *= kron(Hadamard, Id)
    U *= kron(RZs[ind], Id); ind += 1
    U *= kron(Hadamard, Id)

    # XX interaction
    U *= kron(Hadamard, Hadamard)
    U *= CNOT
    U *= kron(Id, RZs[ind]); ind += 1
    U *= CNOT
    U *= kron(Hadamard, Hadamard)

    # YY interaction
    U *= kron(S_GATE, S_GATE)
    U *= kron(Hadamard, Hadamard)
    U *= CNOT
    U *= kron(Id, RZs[ind]); ind += 1
    U *= CNOT
    U *= kron(Hadamard, Hadamard)
    U *= kron(S_GATE', S_GATE')

    # ZZ interaction
    U *= CNOT
    U *= kron(Id, RZs[ind]); ind += 1
    U *= CNOT

    # Trailing single-qubit on qubit 2
    U *= kron(Id, Hadamard)
    U *= kron(Id, RZs[ind]); ind += 1
    U *= kron(Id, Hadamard)
    U *= kron(Id, RZs[ind]); ind += 1
    U *= kron(Id, Hadamard)
    U *= kron(Id, RZs[ind]); ind += 1
    U *= kron(Id, Hadamard)

    # Trailing single-qubit on qubit 1
    U *= kron(Hadamard, Id)
    U *= kron(RZs[ind], Id); ind += 1
    U *= kron(Hadamard, Id)
    U *= kron(RZs[ind], Id); ind += 1
    U *= kron(Hadamard, Id)
    U *= kron(RZs[ind], Id); ind += 1
    U *= kron(Hadamard, Id)

    # Permutation to match qubit ordering convention
    return reshape(permutedims(reshape(U, 2, 2, 2, 2), (2, 1, 4, 3)), 4, 4)
end

"""
    Thetas2SU4(thetas::Vector{Float64}) -> Matrix{ComplexF64}

Convert 15 rotation angles to an SU(4) matrix.
"""
function Thetas2SU4(thetas::Vector{Float64})::Matrix{ComplexF64}
    RZs = [RZ(theta) for theta in thetas]
    return RZs2SU4(RZs)
end

"""
    Cost_SU4(thetas::Vector{Float64}, U_targ) -> Float64

Cost function for SU(4) decomposition: 1 - |tr(U†U_targ)|²/16.
"""
function Cost_SU4(thetas::Vector{Float64}, U_targ)::Float64
    return 1 - abs(tr(Thetas2SU4(thetas) * U_targ'))^2 / 16
end

function Cost_SU4(U::Matrix{ComplexF64}, U_targ)::Float64
    return 1 - abs(tr(U * U_targ'))^2 / 16
end

"""
    Grad_SU4(thetas::Vector{Float64}, U_targ) -> Vector{Float64}

Compute the gradient of Cost_SU4 via finite differences.
"""
function Grad_SU4(thetas::Vector{Float64}, U_targ)::Vector{Float64}
    eps = 1e-7
    grad = zeros(Float64, 15)
    for i in 1:15
        thetas_plus = copy(thetas)
        thetas_minus = copy(thetas)
        thetas_plus[i] += eps
        thetas_minus[i] -= eps
        grad[i] = (Cost_SU4(thetas_plus, U_targ) - Cost_SU4(thetas_minus, U_targ)) / (2 * eps)
    end
    return grad
end

"""
    SU42Thetas_single(U_targ) -> Tuple{Float64, Vector{Float64}}

Single optimization attempt to decompose SU(4) into 15 angles.
"""
function SU42Thetas_single(U_targ)::Tuple{Float64, Vector{Float64}}
    thetas_init = randn(Float64, 15)
    res = Optim.optimize(x -> log10(Cost_SU4(x, U_targ)), thetas_init, Optim.LBFGS(m=60),
                         Optim.Options(x_abstol=1e-12, f_reltol=1e-3, g_tol=1e-12,
                                      iterations=10^5, show_trace=false))
    return Optim.minimum(res), Optim.minimizer(res)
end

"""
    SU42Thetas(U_targ; max_iters=1000, tol=1e-6) -> Vector{Float64}

Decompose an SU(4) matrix into 15 RZ rotation angles with multiple restarts.
"""
function SU42Thetas(U_targ; max_iters::Int=1000, tol::Float64=1e-6)::Vector{Float64}
    for i in 1:max_iters
        opt_f, opt_x = SU42Thetas_single(U_targ)
        if opt_f < log10(tol)
            return opt_x
        end
    end
    throw(ArgumentError("SU42Thetas: optimization did not converge within $max_iters iterations"))
end

"""
    XZXXZX_2_XZX(theta1, theta2) -> Vector{Float64}

Combine two consecutive single-qubit gates into one.
"""
function XZXXZX_2_XZX(theta1::Vector{Float64}, theta2::Vector{Float64})::Vector{Float64}
    U1 = Thetas2SU2(theta1)
    U2 = Thetas2SU2(theta2)
    U_targ = U2 * U1

    function loss_XZX(p, U_target)
        return 1 - abs(tr(U_target' * Thetas2SU2(p)))^2 / 4
    end

    theta_0 = 2pi * rand(3)
    loss_train = x -> loss_XZX(x, U_targ)

    res = Optim.optimize(loss_train, theta_0, Optim.LBFGS(m=32),
                         Optim.Options(x_abstol=1e-8, f_abstol=1e-8, g_tol=1e-8,
                                      iterations=10^5, show_trace=false))
    return Optim.minimizer(res)
end
