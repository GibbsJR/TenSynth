# Tensor manipulation utilities
# Adapted from iMPS2Circuit/src/Core/TensorUtils.jl

"""
    reverse_qubits(M::Matrix{T}) where T -> Matrix{T}

Reverse qubit ordering in a 2-qubit gate matrix.
Converts between |q0 q1⟩ and |q1 q0⟩ conventions.
This is an involution: reverse_qubits(reverse_qubits(M)) == M.
"""
function reverse_qubits(M::Matrix{T}) where T
    size(M) == (4, 4) || throw(ArgumentError("Matrix must be 4×4 for 2-qubit operations"))
    perm = [1, 3, 2, 4]
    return M[perm, perm]
end

"""
    tensor_to_matrix(T::Array{ComplexF64, 4}) -> Matrix{ComplexF64}

Reshape a 4-index tensor to a matrix by grouping (d1,d2) as rows and (d3,d4) as columns.
"""
function tensor_to_matrix(T::Array{ComplexF64, 4})
    d1, d2, d3, d4 = size(T)
    return reshape(T, d1*d2, d3*d4)
end

"""
    matrix_to_tensor(M::Matrix{ComplexF64}, dims::NTuple{4,Int}) -> Array{ComplexF64, 4}

Reshape a matrix back to a 4-index tensor with specified dimensions.
"""
function matrix_to_tensor(M::Matrix{ComplexF64}, dims::NTuple{4,Int})
    return reshape(M, dims)
end

"""
    contract_tensors(A::Array{T,3}, B::Array{T,3}) where T -> Array{T,4}

Contract two 3-index tensors along a shared bond index.
A[χ_L, d_A, χ_M] contracted with B[χ_M, d_B, χ_R] → result[χ_L, d_A, d_B, χ_R].
"""
function contract_tensors(A::Array{T,3}, B::Array{T,3}) where T
    χ_L, d_A, χ_M = size(A)
    χ_M2, d_B, χ_R = size(B)

    χ_M == χ_M2 || throw(DimensionMismatch("Bond dimensions must match: $χ_M vs $χ_M2"))

    result = zeros(T, χ_L, d_A, d_B, χ_R)

    @inbounds for r in 1:χ_R, j in 1:d_B, i in 1:d_A, l in 1:χ_L
        for m in 1:χ_M
            result[l, i, j, r] += A[l, i, m] * B[m, j, r]
        end
    end

    return result
end

"""
    apply_matrix_to_tensor!(result::Array{T,3}, M::Matrix{T}, A::Array{T,3}) where T

Apply a matrix M to the physical index of a 3-index tensor A (in-place).
"""
function apply_matrix_to_tensor!(result::Array{T,3}, M::Matrix{T}, A::Array{T,3}) where T
    χ_L, d, χ_R = size(A)
    d_out = size(M, 1)

    size(M, 2) == d || throw(DimensionMismatch("Matrix columns must match physical dimension"))
    size(result) == (χ_L, d_out, χ_R) || throw(DimensionMismatch("Result tensor has wrong dimensions"))

    fill!(result, zero(T))

    @inbounds for r in 1:χ_R, j in 1:d_out, l in 1:χ_L
        for i in 1:d
            result[l, j, r] += M[j, i] * A[l, i, r]
        end
    end

    return result
end

"""
    apply_matrix_to_tensor(M::Matrix{T}, A::Array{T,3}) where T -> Array{T,3}

Non-mutating version: apply matrix M to the physical index of tensor A.
"""
function apply_matrix_to_tensor(M::Matrix{T}, A::Array{T,3}) where T
    χ_L, d, χ_R = size(A)
    d_out = size(M, 1)
    result = zeros(T, χ_L, d_out, χ_R)
    apply_matrix_to_tensor!(result, M, A)
    return result
end

"""
    svd_truncate(M::Matrix{T}, max_chi::Int, max_trunc_err::Float64) where T

SVD with truncation to at most max_chi singular values.
Returns (U, S, Vt, actual_chi).
"""
function svd_truncate(M::Matrix{T}, max_chi::Int, max_trunc_err::Float64) where T
    F = svd(M)

    S = F.S
    total_weight = sum(abs2, S)

    if total_weight < eps(Float64)
        return F.U[:, 1:1], [zero(real(T))], F.Vt[1:1, :], 1
    end

    cumulative_discarded = 0.0
    chi = min(length(S), max_chi)

    for k in length(S):-1:1
        if k <= max_chi
            err = cumulative_discarded / total_weight
            if err <= max_trunc_err
                chi = k
                break
            end
        end
        cumulative_discarded += abs2(S[k])
    end

    chi = max(1, chi)

    return F.U[:, 1:chi], S[1:chi], F.Vt[1:chi, :], chi
end

"""
    is_unitary(M::Matrix{T}; tol::Float64=1e-10) where T -> Bool

Check if a matrix is unitary: M†M = MM† = I.
"""
function is_unitary(M::Matrix{T}; tol::Float64=1e-10) where T
    n = size(M, 1)
    size(M, 2) == n || return false

    MdM = M' * M
    MMd = M * M'
    I_n = Matrix{T}(I, n, n)

    return norm(MdM - I_n) < tol && norm(MMd - I_n) < tol
end

"""
    is_hermitian(M::Matrix{T}; tol::Float64=1e-10) where T -> Bool

Check if a matrix is Hermitian: M = M†.
"""
function is_hermitian(M::Matrix{T}; tol::Float64=1e-10) where T
    return norm(M - M') < tol
end
