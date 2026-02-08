# Inner product and expectation value functions
# Adapted from MPS2Circuit/src/mps/inner_products.jl
# Key change: Vector{Array{ComplexF64, 3}} → FiniteMPS, Vector{Array{ComplexF64, 4}} → FiniteMPO

using LinearAlgebra
using TensorOperations

"""
    inner(mps1::FiniteMPS, mps2::FiniteMPS) -> ComplexF64

Compute the inner product ⟨mps1|mps2⟩ between two MPS.
"""
function inner(mps1::FiniteMPS, mps2::FiniteMPS)::ComplexF64
    T = ones(ComplexF64, 1, 1)

    for ii in 1:length(mps1.tensors)
        T1 = conj(mps1.tensors[ii])
        T2 = mps2.tensors[ii]

        @tensoropt Tnew[l, m] := T[i, j] * T1[i, k, l] * T2[j, k, m]
        T = copy(Tnew)
    end
    return T[1, 1]
end

"""
    inner(mpo1::FiniteMPO, mpo2::FiniteMPO) -> ComplexF64

Compute the inner product ⟨mpo1|mpo2⟩ between two MPOs (Frobenius inner product).
Convention: tensors are [left, phys_out, phys_in, right].
"""
function inner(mpo1::FiniteMPO, mpo2::FiniteMPO)::ComplexF64
    T = ones(ComplexF64, 1, 1)

    for ii in 1:length(mpo1.tensors)
        T1 = conj(mpo1.tensors[ii])
        T2 = mpo2.tensors[ii]

        # Convention: [left, phys_out, phys_in, right]
        # Tr(A†B): contract A's phys_out(k) with B's phys_in(k), A's phys_in(a) with B's phys_out(a)
        @tensoropt Tnew[l, m] := T[i, j] * T1[i, k, a, l] * T2[j, a, k, m]
        T = copy(Tnew)
    end

    return T[1, 1]
end

"""
    GetExps(mps::FiniteMPS, ExpList::Vector{<:AbstractMatrix{ComplexF64}}, ExpList_inds::Vector{}) -> Vector

Compute expectation values of operators for a given MPS.
Supports single-site (2×2) and nearest-neighbor two-site (4×4) operators.
"""
function GetExps(mps::FiniteMPS, ExpList::Vector{<:AbstractMatrix{ComplexF64}}, ExpList_inds::Vector{})
    t = mps.tensors

    if length(ExpList) != length(ExpList_inds)
        throw(ArgumentError("GetExps: ExpList and ExpList_inds must have same length"))
    end

    for inds in ExpList_inds
        if length(inds) == 2 && inds[1] + 1 != inds[2]
            throw(ArgumentError("GetExps: two-site operators must act on adjacent sites"))
        end
        if length(inds) > 2
            throw(ArgumentError("GetExps: operators with more than 2 sites not supported"))
        end
    end

    N = length(t)

    Lenvs = [ones(ComplexF64, 1, 1) for _ in 1:N]
    Renvs = [ones(ComplexF64, 1, 1) for _ in 1:N]

    # Initialize right environments
    R = ones(ComplexF64, 1, 1)
    for ii in N:-1:1
        T1 = conj(t[ii])
        T3 = t[ii]
        @tensoropt R_new[i, p] := T1[i, k, l] * T3[p, k, q] * R[l, q]
        R = R_new ./ norm(R_new)
        Renvs[ii] = R
    end

    # Initialize left environments
    L = ones(ComplexF64, 1, 1)
    for ii in 1:N
        T1 = conj(t[ii])
        T3 = t[ii]
        @tensoropt L_new[l, q] := L[i, p] * T1[i, k, l] * T3[p, k, q]
        L = L_new ./ norm(L_new)
        Lenvs[ii] = L
    end

    arr_exps = []

    for i in 1:length(ExpList)
        inds = ExpList_inds[i]
        O = ExpList[i]

        if length(inds) == 1
            L_env = inds[1] == 1 ? ones(ComplexF64, 1, 1) : Lenvs[inds[1]-1]
            R_env = inds[1] == N ? ones(ComplexF64, 1, 1) : Renvs[inds[1]+1]

            @tensoropt num[] := L_env[i, a] * conj(t[inds[1]])[i, l, k] * O[l, b] * t[inds[1]][a, b, c] * R_env[k, c]
            @tensoropt denom[] := L_env[i, a] * conj(t[inds[1]])[i, l, k] * t[inds[1]][a, l, c] * R_env[k, c]

        elseif length(inds) == 2
            O_4d = reshape(O, 2, 2, 2, 2)

            L_env = inds[1] == 1 ? ones(ComplexF64, 1, 1) : Lenvs[inds[1]-1]
            R_env = inds[2] == N ? ones(ComplexF64, 1, 1) : Renvs[inds[2]+1]

            @tensoropt num[] := L_env[i, a] * conj(t[inds[1]])[i, j, k] * t[inds[1]][a, b, c] * O_4d[j, l, b, d] * conj(t[inds[2]])[k, l, m] * t[inds[2]][c, d, e] * R_env[m, e]
            @tensoropt denom[] := L_env[i, a] * conj(t[inds[1]])[i, j, k] * t[inds[1]][a, j, c] * conj(t[inds[2]])[k, l, m] * t[inds[2]][c, l, e] * R_env[m, e]
        end

        append!(arr_exps, num[1] / denom[1])
    end

    return arr_exps
end
