# Normalization and canonical form for iMPS

using LinearAlgebra

function absorb_bonds!(psi::iMPSType{T}) where T
    n = psi.unit_cell
    if psi._gamma_absorbed === nothing
        psi._gamma_absorbed = Vector{Array{T,3}}(undef, n)
    end

    for i in 1:n
        i_prev = mod1(i - 1, n)
        lambda_left = psi.lambda[i_prev]
        gamma_i = psi.gamma[i]

        χ_L_λ, χ_R_λ = size(lambda_left)
        χ_L, d, χ_R = size(gamma_i)
        χ_R_λ == χ_L || throw(DimensionMismatch("Bond dimension mismatch at site $i"))

        gamma_m = zeros(T, χ_L_λ, d, χ_R)
        @inbounds for r in 1:χ_R, p in 1:d, l in 1:χ_L_λ
            val = zero(T)
            for m in 1:χ_L
                val += lambda_left[l, m] * gamma_i[m, p, r]
            end
            gamma_m[l, p, r] = val
        end

        psi._gamma_absorbed[i] = gamma_m
    end
    return psi
end

function truncate!(psi::iMPSType{T}, max_chi::Int; max_trunc_err::Float64=1e-10) where T
    n = psi.unit_cell
    for i in 1:n
        λ = psi.lambda[i]
        χ_curr = size(λ, 1)
        if χ_curr > max_chi
            psi.lambda[i] = λ[1:max_chi, 1:max_chi]
            γ = psi.gamma[i]
            χ_L, d, χ_R = size(γ)
            new_χ_L = min(χ_L, max_chi)
            new_χ_R = min(χ_R, max_chi)
            psi.gamma[i] = γ[1:new_χ_L, :, 1:new_χ_R]

            i_next = mod1(i + 1, n)
            γ_next = psi.gamma[i_next]
            χ_L_next = size(γ_next, 1)
            if χ_L_next > max_chi
                psi.gamma[i_next] = γ_next[1:max_chi, :, :]
            end
        end
    end
    psi.normalized = false
    psi._gamma_absorbed = nothing
    return psi
end

function canonicalize!(psi::iMPSType{T}; max_iterations::Int=100, tol::Float64=1e-12) where T
    n = psi.unit_cell
    d = psi.physical_dim

    for iter in 1:max_iterations
        max_change = 0.0

        for i in 1:n
            i_next = mod1(i + 1, n)
            γ = psi.gamma[i]
            λ = psi.lambda[i]
            χ_L, _, χ_R = size(γ)

            θ = zeros(T, χ_L, d, χ_R)
            @inbounds for r in 1:χ_R, p in 1:d, l in 1:χ_L
                val = zero(T)
                for m in 1:size(λ, 1)
                    val += γ[l, p, m] * λ[m, r]
                end
                θ[l, p, r] = val
            end

            θ_mat = reshape(θ, χ_L * d, χ_R)
            F = svd(θ_mat)
            χ_new = length(F.S)
            new_gamma = reshape(F.U, χ_L, d, χ_new)
            new_lambda = diagm(F.S)

            old_s = diag(psi.lambda[i])
            new_s = F.S[1:min(length(old_s), length(F.S))]
            if length(old_s) == length(new_s)
                change = norm(old_s - new_s)
                max_change = max(max_change, change)
            else
                max_change = 1.0
            end

            psi.gamma[i] = new_gamma
            psi.lambda[i] = new_lambda

            γ_next = psi.gamma[i_next]
            χ_L_next, d_next, χ_R_next = size(γ_next)
            Vd = F.Vt

            if size(Vd, 2) == χ_L_next
                new_gamma_next = zeros(T, χ_new, d_next, χ_R_next)
                @inbounds for r in 1:χ_R_next, p in 1:d_next, l in 1:χ_new
                    val = zero(T)
                    for m in 1:χ_L_next
                        val += Vd[l, m] * γ_next[m, p, r]
                    end
                    new_gamma_next[l, p, r] = val
                end
                psi.gamma[i_next] = new_gamma_next
            end
        end

        if max_change < tol
            break
        end
    end

    for i in 1:n
        s = diag(psi.lambda[i])
        norm_s = norm(s)
        if norm_s > eps(Float64)
            psi.lambda[i] = diagm(s ./ norm_s)
        end
    end

    psi.normalized = true
    psi._gamma_absorbed = nothing
    absorb_bonds!(psi)
    return psi
end

function normalize_state!(psi::iMPSType{T}) where T
    if psi.normalized
        return psi
    end

    n = psi.unit_cell
    eig, _ = transfer_matrix_eigenvalue(psi, psi; tol=1e-8)

    if abs(eig) > eps(Float64) && abs(eig - 1.0) > 1e-14
        scale_factor = eig^(0.5 / n)
        for i in 1:n
            psi.gamma[i] ./= scale_factor
        end
    end

    psi._gamma_absorbed = nothing
    absorb_bonds!(psi)
    psi.normalized = true
    return psi
end

function norm_squared(psi::iMPSType{T}) where T
    eig, _ = transfer_matrix_eigenvalue(psi, psi; tol=1e-8)
    return eig
end

function check_canonical_form(psi::iMPSType{T}; tol::Float64=1e-10) where T
    n = psi.unit_cell
    for i in 1:n
        γ = psi.gamma[i]
        λ = psi.lambda[i]
        χ_L, d, χ_R = size(γ)
        λ_sq = λ * λ'
        result = zeros(T, χ_L, χ_L)
        for p in 1:d
            γ_p = γ[:, p, :]
            result .+= γ_p * λ_sq * γ_p'
        end
        I_χ = Matrix{T}(I, χ_L, χ_L)
        if norm(result - I_χ) > tol * χ_L
            return false
        end
    end
    return true
end
