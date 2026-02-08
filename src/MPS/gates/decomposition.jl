# CNOT decomposition utilities
# Adapted from MPS2Circuit/src/gates/decomposition.jl
# Key changes: Zygote.gradient → finite-difference gradients, RSVD → polar_unitary

using LinearAlgebra

"""
    Dressed1CNOT(unis) -> Matrix{ComplexF64}

Construct a two-qubit unitary from 4 single-qubit gates and 1 CNOT.
Circuit structure: (U1 kron U2) * CNOT * (U3 kron U4)
"""
function Dressed1CNOT(unis)
    M = kron(unis[1], unis[2])
    M *= CNOT
    M *= kron(unis[3], unis[4])
    return M
end

"""
    Dressed2CNOT(unis) -> Matrix{ComplexF64}

Construct a two-qubit unitary from 6 single-qubit gates and 2 CNOTs.
"""
function Dressed2CNOT(unis)
    M = kron(unis[1], unis[2])
    M *= CNOT
    M *= kron(unis[3], unis[4])
    M *= CNOT
    M *= kron(unis[5], unis[6])
    return M
end

"""
    Dressed3CNOT(unis) -> Matrix{ComplexF64}

Construct a two-qubit unitary from 8 single-qubit gates and 3 CNOTs.
Any SU(4) can be decomposed into at most 3 CNOTs.
"""
function Dressed3CNOT(unis)
    M = kron(unis[1], unis[2])
    M *= CNOT
    M *= kron(unis[3], unis[4])
    M *= CNOT
    M *= kron(unis[5], unis[6])
    M *= CNOT
    M *= kron(unis[7], unis[8])
    return M
end

function D1CNOT_cost(M, unis)
    return 1 - abs(tr(M' * Dressed1CNOT(unis))) / abs(tr(M' * M))
end

function D2CNOT_cost(M, unis)
    return 1 - abs(tr(M' * Dressed2CNOT(unis))) / abs(tr(M' * M))
end

function D3CNOT_cost(M, unis)
    return 1 - abs(tr(M' * Dressed3CNOT(unis))) / abs(tr(M' * M))
end

# Riemannian gradient for unitary manifold optimization
function _rgrad_cnot(uni, G)
    return G - 0.5 * uni * (uni' * G + G' * uni)
end

function _VecTransport(X, Y)
    return Y - 0.5 * X * (Y' * X + X' * Y)
end

"""
    _fd_gradient_unis(cost_fn, unis) -> Vector{Matrix{ComplexF64}}

Compute gradient of a real-valued cost function with respect to a vector of complex matrices
using finite differences (Wirtinger conjugate gradient).
Replaces Zygote.gradient for this use case.
"""
function _fd_gradient_unis(cost_fn, unis)
    eps = 1e-7
    grads = [zeros(ComplexF64, size(u)) for u in unis]
    base_cost = cost_fn(unis)

    for k in 1:length(unis)
        for i in 1:size(unis[k], 1), j in 1:size(unis[k], 2)
            # Real part derivative
            unis_p = [copy(u) for u in unis]
            unis_p[k][i, j] += eps
            df_dre = (cost_fn(unis_p) - base_cost) / eps

            # Imaginary part derivative
            unis_p = [copy(u) for u in unis]
            unis_p[k][i, j] += eps * im
            df_dim = (cost_fn(unis_p) - base_cost) / eps

            # Wirtinger conjugate gradient: df/dz_bar = (df/dx + i*df/dy)/2
            grads[k][i, j] = (df_dre + im * df_dim) / 2
        end
    end
    return grads
end

"""
    Uni2CNOTs(U_targ; n_cnots=3, IfPrint=false) -> Matrix{ComplexF64}

Decompose a two-qubit unitary into a CNOT-based circuit using
gradient descent on the unitary manifold.
"""
function Uni2CNOTs(U_targ; n_cnots=3, IfPrint=false)
    if n_cnots == 0
        return Matrix{ComplexF64}(I, 4, 4)
    elseif n_cnots > 3
        n_cnots = 3
    end

    # Initialize random unitaries
    unis = [randU(1e-0, 1) for _ in 1:2*(1+n_cnots)]
    momentums = [zeros(ComplexF64, 2, 2) for _ in 1:length(unis)]

    cost = 1.0
    prev_cost = 1.0
    lr = 1.0

    for i in 1:10^4
        # Compute gradient via finite differences
        if n_cnots == 1
            G = _fd_gradient_unis(x -> D1CNOT_cost(U_targ, x), unis)
        elseif n_cnots == 2
            G = _fd_gradient_unis(x -> D2CNOT_cost(U_targ, x), unis)
        elseif n_cnots == 3
            G = _fd_gradient_unis(x -> D3CNOT_cost(U_targ, x), unis)
        end

        # Momentum-based update on unitary manifold
        beta1 = 0.9
        momentums = [beta1 * m + (1 - beta1) * _rgrad_cnot(u, g) for (m, u, g) in zip(momentums, unis, G)]
        unis = [polar_unitary(u .- lr * m) for (u, m) in zip(unis, momentums)]
        momentums = [_VecTransport(u, m) for (u, m) in zip(unis, momentums)]

        if i % 100 == 0
            if n_cnots == 1
                cost = D1CNOT_cost(U_targ, unis)
            elseif n_cnots == 2
                cost = D2CNOT_cost(U_targ, unis)
            elseif n_cnots == 3
                cost = D3CNOT_cost(U_targ, unis)
            end

            if cost < 1e-10 || (abs(cost - prev_cost) / max(abs(prev_cost), 1e-15) < 1e-4 && i >= 20)
                break
            end
            prev_cost = cost
        end
    end

    if n_cnots == 1
        return Dressed1CNOT(unis)
    elseif n_cnots == 2
        return Dressed2CNOT(unis)
    elseif n_cnots == 3
        return Dressed3CNOT(unis)
    end
end
