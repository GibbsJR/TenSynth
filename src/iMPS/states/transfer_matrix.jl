# Transfer matrix operations for iMPS overlaps

using LinearAlgebra
using LinearMaps
using Arpack
using TensorOperations

function transfer_matrix_action!(y::AbstractVector{T}, x::AbstractVector{T},
                                  psi::iMPSType{T}, phi::iMPSType{T};
                                  uni::Matrix{ComplexF64}=Matrix{ComplexF64}(I, 4, 4),
                                  uni_inds::Tuple{Int,Int}=(1,2)) where T
    n = psi.unit_cell
    n == phi.unit_cell || throw(DimensionMismatch("Unit cell sizes must match"))

    uni_inds[1] >= 1 && uni_inds[1] <= n || throw(ArgumentError("uni_inds[1] out of range"))
    uni_inds[2] >= 1 && uni_inds[2] <= n || throw(ArgumentError("uni_inds[2] out of range"))
    uni_inds[1] <= uni_inds[2] || throw(ArgumentError("uni_inds[1] must be <= uni_inds[2]"))

    if psi._gamma_absorbed === nothing
        absorb_bonds!(psi)
    end
    if phi._gamma_absorbed === nothing
        absorb_bonds!(phi)
    end

    χ_L_psi = size(psi._gamma_absorbed[1], 1)
    χ_L_phi = size(phi._gamma_absorbed[1], 1)

    length(x) == χ_L_psi * χ_L_phi || throw(DimensionMismatch("Input vector wrong size"))
    length(y) == χ_L_psi * χ_L_phi || throw(DimensionMismatch("Output vector wrong size"))

    is_identity_uni = uni ≈ Matrix{ComplexF64}(I, size(uni)...)

    X_curr = reshape(convert(Array{T}, copy(x)), χ_L_psi, χ_L_phi)

    if is_identity_uni
        for i in 1:n
            γ_psi = psi._gamma_absorbed[i]
            γ_phi_conj = conj.(phi._gamma_absorbed[i])
            @tensor X_new[-1, -2] := X_curr[1, 2] * γ_phi_conj[2, 3, -2] * γ_psi[1, 3, -1]
            X_curr = X_new
        end
    else
        uni_4d = reshape(uni, 2, 2, 2, 2)

        # Sites before uni_inds[1]
        for i in 1:(uni_inds[1] - 1)
            γ_psi = psi._gamma_absorbed[i]
            γ_phi_conj = conj.(phi._gamma_absorbed[i])
            @tensor X_new[-1, -2] := X_curr[1, 2] * γ_phi_conj[2, 3, -2] * γ_psi[1, 3, -1]
            X_curr = X_new
        end

        # First unitary site — keep physical indices open
        i = uni_inds[1]
        γ_psi = psi._gamma_absorbed[i]
        γ_phi_conj = conj.(phi._gamma_absorbed[i])
        @tensor X_open[-1, -2, -3, -4] := X_curr[1, 2] * γ_phi_conj[2, -1, -4] * γ_psi[1, -2, -3]
        X_curr = X_open

        # Sites between unitary sites
        for i in (uni_inds[1] + 1):(uni_inds[2] - 1)
            γ_psi = psi._gamma_absorbed[i]
            γ_phi_conj = conj.(phi._gamma_absorbed[i])
            @tensor X_new[-1, -2, -3, -4] := X_curr[-1, -2, 1, 2] * γ_phi_conj[2, 3, -4] * γ_psi[1, 3, -3]
            X_curr = X_new
        end

        # Second unitary site — apply unitary and close physical indices
        i = uni_inds[2]
        γ_psi = psi._gamma_absorbed[i]
        γ_phi_conj = conj.(phi._gamma_absorbed[i])
        @tensor X_new[-1, -2] := X_curr[11, 13, 1, 2] * γ_phi_conj[2, 12, -2] * γ_psi[1, 14, -1] * uni_4d[11, 12, 13, 14]
        X_curr = X_new

        # Sites after uni_inds[2]
        for i in (uni_inds[2] + 1):n
            γ_psi = psi._gamma_absorbed[i]
            γ_phi_conj = conj.(phi._gamma_absorbed[i])
            @tensor X_new[-1, -2] := X_curr[1, 2] * γ_phi_conj[2, 3, -2] * γ_psi[1, 3, -1]
            X_curr = X_new
        end
    end

    Y = reshape(y, χ_L_psi, χ_L_phi)
    Y .= X_curr
    return y
end

function transfer_matrix_eigenvalue(psi::iMPSType{T}, phi::iMPSType{T};
                                    uni::Matrix{ComplexF64}=Matrix{ComplexF64}(I, 4, 4),
                                    uni_inds::Tuple{Int,Int}=(1,2),
                                    tol::Float64=1e-8,
                                    v0::Union{Nothing, Vector{T}}=nothing) where T
    if psi._gamma_absorbed === nothing
        absorb_bonds!(psi)
    end
    if phi._gamma_absorbed === nothing
        absorb_bonds!(phi)
    end

    χ_psi = size(psi._gamma_absorbed[1], 1)
    χ_phi = size(phi._gamma_absorbed[1], 1)
    N = χ_psi * χ_phi

    if N <= 16
        return _transfer_matrix_eigenvalue_direct(psi, phi; uni=uni, uni_inds=uni_inds)
    end

    function tm_action!(y, x)
        transfer_matrix_action!(y, x, psi, phi; uni=uni, uni_inds=uni_inds)
    end

    TM = LinearMap{T}(tm_action!, N; ismutating=true)

    if v0 === nothing
        v0 = randn(T, N)
        v0 ./= norm(v0)
    end

    try
        vals, vecs, info = eigs(TM, nev=1, which=:LM, tol=tol, v0=v0, maxiter=300)
        if info != 0
            return _transfer_matrix_eigenvalue_direct(psi, phi; uni=uni, uni_inds=uni_inds)
        end
        return abs(vals[1]), vecs[:, 1]
    catch
        return _transfer_matrix_eigenvalue_direct(psi, phi; uni=uni, uni_inds=uni_inds)
    end
end

function _transfer_matrix_eigenvalue_direct(psi::iMPSType{T}, phi::iMPSType{T};
                                            uni::Matrix{ComplexF64}=Matrix{ComplexF64}(I, 4, 4),
                                            uni_inds::Tuple{Int,Int}=(1,2)) where T
    if psi._gamma_absorbed === nothing
        absorb_bonds!(psi)
    end
    if phi._gamma_absorbed === nothing
        absorb_bonds!(phi)
    end

    χ_psi = size(psi._gamma_absorbed[1], 1)
    χ_phi = size(phi._gamma_absorbed[1], 1)
    N = χ_psi * χ_phi

    TM = zeros(T, N, N)
    for j in 1:N
        x = zeros(T, N)
        x[j] = one(T)
        y = zeros(T, N)
        transfer_matrix_action!(y, x, psi, phi; uni=uni, uni_inds=uni_inds)
        TM[:, j] = y
    end

    F = eigen(TM)
    idx = argmax(abs.(F.values))
    return abs(F.values[idx]), F.vectors[:, idx]
end

function local_fidelity(psi::iMPSType{T}, phi::iMPSType{T}; tol::Float64=1e-10) where T
    eig, _ = transfer_matrix_eigenvalue(psi, phi; tol=tol)
    return eig^(2.0 / psi.unit_cell)
end

function infidelity(psi::iMPSType{T}, phi::iMPSType{T}; tol::Float64=1e-10) where T
    return 1.0 - local_fidelity(psi, phi; tol=tol)
end

function infidelity_average(psi_vec::Vector{iMPSType{T}}, phi_vec::Vector{iMPSType{T}};
                            tol::Float64=1e-10) where T
    length(psi_vec) == length(phi_vec) || throw(ArgumentError("Vectors must have same length"))
    n = length(psi_vec)
    total = 0.0
    for i in 1:n
        total += infidelity(psi_vec[i], phi_vec[i]; tol=tol)
    end
    return total / n
end
