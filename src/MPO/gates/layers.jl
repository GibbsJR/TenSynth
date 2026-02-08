# Layer application to MPO
# Adapted from MPO2Circuit/src/gates/layers.jl

using LinearAlgebra

"""
    apply_layer!(mpo::FiniteMPO{ComplexF64}, layer::GateLayer;
                 max_chi::Int=128, max_trunc_err::Float64=1e-14,
                 canonicalize::Bool=true) -> Float64

Apply a layer of gates to an MPO. Returns total truncation error.
"""
function apply_layer!(mpo::FiniteMPO{ComplexF64}, layer::GateLayer;
                      max_chi::Int=128, max_trunc_err::Float64=1e-14,
                      canonicalize::Bool=true)::Float64
    n = n_sites(mpo)
    total_err = 0.0

    gate_order = sortperm(layer.indices, by=x->x[1])

    if canonicalize
        mpo_canon = canonicalise(mpo, 1)
        for i in 1:n
            mpo.tensors[i] = mpo_canon.tensors[i]
        end
    end

    prev_bond = (1, 2)

    for idx in gate_order
        gate = layer.gates[idx]
        site1, site2 = layer.indices[idx]

        (1 <= site1 < site2 <= n) ||
            throw(ArgumentError("Invalid gate indices ($site1, $site2) for MPO with $n sites"))

        if site2 == site1 + 1
            if canonicalize && prev_bond != (site1, site2)
                mpo_canon = canonicalise_from_to(mpo, prev_bond, (site1, site2))
                for i in 1:n
                    mpo.tensors[i] = mpo_canon.tensors[i]
                end
            end

            err = apply_gate!(mpo, gate, site1, site2;
                             max_chi=max_chi, max_trunc_err=max_trunc_err)
            total_err += err
            prev_bond = (site1, site2)
        else
            err = apply_gate_long_range!(mpo, gate, site1, site2;
                                         max_chi=max_chi, max_trunc_err=max_trunc_err,
                                         canonicalize=canonicalize)
            total_err += err
            prev_bond = (site1, site1+1)
        end
    end

    return total_err
end

"""
    apply_layers!(mpo::FiniteMPO{ComplexF64}, layers::Vector{GateLayer};
                  max_chi::Int=128, max_trunc_err::Float64=1e-14,
                  canonicalize::Bool=true) -> Float64
"""
function apply_layers!(mpo::FiniteMPO{ComplexF64}, layers::Vector{GateLayer};
                       max_chi::Int=128, max_trunc_err::Float64=1e-14,
                       canonicalize::Bool=true)::Float64
    total_err = 0.0
    for layer in layers
        err = apply_layer!(mpo, layer;
                          max_chi=max_chi, max_trunc_err=max_trunc_err,
                          canonicalize=canonicalize)
        total_err += err
    end
    return total_err
end

"""
    brick_wall_indices(n_qubits::Int, offset::Int) -> Vector{Tuple{Int,Int}}
"""
function brick_wall_indices(n_qubits::Int, offset::Int)::Vector{Tuple{Int,Int}}
    indices = Tuple{Int,Int}[]
    start = 1 + offset
    for i in start:2:(n_qubits-1)
        push!(indices, (i, i+1))
    end
    return indices
end

"""
    identity_layer(indices::Vector{Tuple{Int,Int}}) -> GateLayer
"""
function identity_layer(indices::Vector{Tuple{Int,Int}})::GateLayer
    n_gates = length(indices)
    gates = [GateMatrix(Matrix{ComplexF64}(I, 4, 4), "", Dict{Symbol,Any}()) for _ in 1:n_gates]
    return GateLayer(gates, indices)
end

"""
    random_layer(indices::Vector{Tuple{Int,Int}}) -> GateLayer
"""
function random_layer(indices::Vector{Tuple{Int,Int}})::GateLayer
    n_gates = length(indices)
    gates = [GateMatrix(random_unitary(4), "", Dict{Symbol,Any}()) for _ in 1:n_gates]
    return GateLayer(gates, indices)
end
