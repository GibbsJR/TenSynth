# Inner products for MPS and MPO
# Adapted from MPO2Circuit/src/mpo/inner.jl
# INDEX SWAP applied: TenSynth uses [left, phys_out, phys_in, right]
# Original MPO2Circuit: T1[i, a, j, k] with a=phys_in, j=phys_out
# TenSynth:            T1[i, j, a, k] with j=phys_out, a=phys_in
# In contractions: where original has T[i, a, j, k], we use T[i, j, a, k]
# and swap the roles of the physical index labels accordingly.

using TensorOperations

"""
    inner_mps(mps1::FiniteMPS{ComplexF64}, mps2::FiniteMPS{ComplexF64}) -> ComplexF64

Compute the inner product <mps1|mps2>. The first MPS is conjugated.
"""
function inner_mps(mps1::FiniteMPS{ComplexF64}, mps2::FiniteMPS{ComplexF64})::ComplexF64
    n1 = n_sites(mps1)
    n2 = n_sites(mps2)
    n1 == n2 || throw(ArgumentError("MPS must have same number of sites: $n1 vs $n2"))

    T = ones(ComplexF64, 1, 1)

    for i in 1:n1
        A = conj(mps1.tensors[i])  # [left1, phys, right1]
        B = mps2.tensors[i]        # [left2, phys, right2]

        @tensor Tnew[l, m] := T[i, j] * A[i, k, l] * B[j, k, m]
        T = Tnew
    end

    return T[1, 1]
end

"""
    inner_mpo(mpo1::FiniteMPO{ComplexF64}, mpo2::FiniteMPO{ComplexF64}) -> ComplexF64

Compute the normalized Hilbert-Schmidt inner product Tr(mpo1† * mpo2) / 2^n.
The first MPO is conjugated.

Convention: tensors are [left, phys_out, phys_in, right].
For Tr(A†B), we contract: A's phys_out with B's phys_in, and A's phys_in with B's phys_out.
"""
function inner_mpo(mpo1::FiniteMPO{ComplexF64}, mpo2::FiniteMPO{ComplexF64})::ComplexF64
    n1 = n_sites(mpo1)
    n2 = n_sites(mpo2)
    n1 == n2 || throw(ArgumentError("MPO must have same number of sites: $n1 vs $n2"))

    T = ones(ComplexF64, 1, 1)

    for i in 1:n1
        # A = conj(mpo1): [left, phys_out, phys_in, right]
        # B = mpo2:       [left, phys_out, phys_in, right]
        # For Tr(A†B): contract A's phys_out(j) with B's phys_in(j=a), and A's phys_in(a) with B's phys_out(a=k)
        # Original (MPO2Circuit with [left,phys_in,phys_out,right]):
        #   @tensor Tnew[l, m] := T[i, j] * A[i, k, a, l] * B[j, a, k, m]
        # After index swap to [left,phys_out,phys_in,right]:
        #   A[i, a, k, l] means A[left, phys_out, phys_in, right]
        #   B[j, k, a, m] means B[left, phys_out, phys_in, right]
        #   Contract: A's phys_out(a) with B's phys_in(a), A's phys_in(k) with B's phys_out(k)
        A = conj(mpo1.tensors[i])
        B = mpo2.tensors[i]

        @tensor Tnew[l, m] := T[i, j] * A[i, a, k, l] * B[j, k, a, m]
        T = Tnew
    end

    return T[1, 1] / (2.0^n1)
end

"""
    hst_cost(mpo1::FiniteMPO{ComplexF64}, mpo2::FiniteMPO{ComplexF64}) -> Float64

Compute the Hilbert-Schmidt Test (HST) cost: 1 - |<mpo1|mpo2>|² / (<mpo1|mpo1> * <mpo2|mpo2>).
Computed via normalized MPS inner products for numerical stability.
"""
function hst_cost(mpo1::FiniteMPO{ComplexF64}, mpo2::FiniteMPO{ComplexF64})::Float64
    mps1, _ = mpo_to_mps(mpo1)
    mps2, _ = mpo_to_mps(mpo2)

    overlap = inner_mps(mps1, mps2)

    return 1.0 - abs(overlap)^2
end
