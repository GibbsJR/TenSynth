# Initialization functions for iMPS states

using LinearAlgebra
using Random

function random_product_state!(psi::iMPSType{T}; rng::AbstractRNG=Random.default_rng()) where T
    d = psi.physical_dim
    for i in 1:psi.unit_cell
        state = randn(rng, T, d) + im * randn(rng, real(T), d)
        state ./= norm(state)
        psi.gamma[i] = reshape(state, 1, d, 1)
        psi.lambda[i] = reshape(T[1], 1, 1)
    end
    psi.normalized = true
    psi._gamma_absorbed = nothing
    return psi
end

function random_product_state(unit_cell::Int; physical_dim::Int=2,
                              T::Type=ComplexF64, rng::AbstractRNG=Random.default_rng())
    psi = iMPSType{T}(unit_cell; physical_dim=physical_dim)
    random_product_state!(psi; rng=rng)
    return psi
end

function random_imps!(psi::iMPSType{T}, chi::Int; rng::AbstractRNG=Random.default_rng()) where T
    chi > 0 || throw(ArgumentError("chi must be positive"))
    d = psi.physical_dim
    for i in 1:psi.unit_cell
        psi.gamma[i] = randn(rng, T, chi, d, chi) + im * randn(rng, real(T), chi, d, chi)
        psi.gamma[i] ./= norm(psi.gamma[i])
        s = abs.(randn(rng, real(T), chi)) .+ 0.1
        s ./= norm(s)
        psi.lambda[i] = diagm(T.(s))
    end
    psi.normalized = false
    psi._gamma_absorbed = nothing
    return psi
end

function random_imps(unit_cell::Int, chi::Int; physical_dim::Int=2,
                     T::Type=ComplexF64, rng::AbstractRNG=Random.default_rng())
    psi = iMPSType{T}(unit_cell; physical_dim=physical_dim)
    random_imps!(psi, chi; rng=rng)
    return psi
end

function product_state!(psi::iMPSType{T}, states::Vector{Vector{S}}) where {T, S}
    length(states) == psi.unit_cell || throw(ArgumentError("Number of states must equal unit_cell"))
    d = psi.physical_dim
    for i in 1:psi.unit_cell
        length(states[i]) == d || throw(ArgumentError("State $i has wrong dimension"))
        state = convert(Vector{T}, states[i])
        state ./= norm(state)
        psi.gamma[i] = reshape(state, 1, d, 1)
        psi.lambda[i] = reshape(T[1], 1, 1)
    end
    psi.normalized = true
    psi._gamma_absorbed = nothing
    return psi
end

function zero_state!(psi::iMPSType{T}) where T
    d = psi.physical_dim
    for i in 1:psi.unit_cell
        state = zeros(T, d)
        state[1] = one(T)
        psi.gamma[i] = reshape(state, 1, d, 1)
        psi.lambda[i] = reshape(T[1], 1, 1)
    end
    psi.normalized = true
    psi._gamma_absorbed = nothing
    return psi
end

function plus_state!(psi::iMPSType{T}) where T
    d = psi.physical_dim
    d == 2 || throw(ArgumentError("plus_state requires physical_dim=2"))
    for i in 1:psi.unit_cell
        state = T[1, 1] / sqrt(T(2))
        psi.gamma[i] = reshape(state, 1, d, 1)
        psi.lambda[i] = reshape(T[1], 1, 1)
    end
    psi.normalized = true
    psi._gamma_absorbed = nothing
    return psi
end

function neel_state!(psi::iMPSType{T}) where T
    d = psi.physical_dim
    d == 2 || throw(ArgumentError("neel_state requires physical_dim=2"))
    for i in 1:psi.unit_cell
        state = zeros(T, d)
        state[mod1(i, 2) == 1 ? 1 : 2] = one(T)
        psi.gamma[i] = reshape(state, 1, d, 1)
        psi.lambda[i] = reshape(T[1], 1, 1)
    end
    psi.normalized = true
    psi._gamma_absorbed = nothing
    return psi
end
