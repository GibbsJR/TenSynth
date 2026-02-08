# Variational MPS compression routines
# Adapted from MPS2Circuit/src/optimization/compression.jl
# Key changes: Vector{Array{...}} → FiniteMPS/FiniteMPO, SVD() → robust_svd()

using LinearAlgebra
using TensorOperations

"""
    VarCompress_mps_1site(mps_targ::FiniteMPS, mps_init::FiniteMPS, chi_trunc::Int)
        -> (infidelity, compressed_mps, s_norms)

Variationally compress an MPS to a target bond dimension using 1-site sweeps.
"""
function VarCompress_mps_1site(mps_targ::FiniteMPS{ComplexF64}, mps_init::FiniteMPS{ComplexF64},
                                chi_trunc::Int)::Tuple{Float64, FiniteMPS{ComplexF64}, Vector{Float64}}
    N = length(mps_targ.tensors)
    mps_var = copy(mps_init.tensors)

    # Initialize right environments
    T1 = conj(mps_targ.tensors[end])[:, :, 1]
    T2 = mps_var[end][:, :, 1]
    @tensor temp[i, k] := T1[i, j] * T2[k, j]
    Renvs = Vector{Array{ComplexF64,2}}([temp])

    for ii in length(mps_var)-1:-1:1
        T1 = conj(mps_targ.tensors[ii])
        T2 = mps_var[ii]
        R = Renvs[1]
        @tensoropt temp[l, i] := T1[l, j, m] * T2[i, j, k] * R[m, k]
        prepend!(Renvs, [temp ./ norm(temp)])
    end

    Lenvs = Vector{Array{ComplexF64,2}}([zeros(ComplexF64, 1, 1) for _ in 1:N])
    s_norms = zeros(Float64, N)
    n_sweeps = Int(1e4)
    prev_cost = 2.0

    for s in 1:n_sweeps
        # Right sweep
        for ii in 1:N-1
            L = ii > 1 ? Lenvs[ii-1] : ones(ComplexF64, 1, 1)
            R = ii < N ? Renvs[ii+1] : ones(ComplexF64, 1, 1)
            T = conj(mps_targ.tensors[ii])

            @tensoropt env[j, k, m] := L[i, j] * T[i, k, l] * R[l, m]

            F = robust_svd(reshape(conj(env), size(env, 1) * size(env, 2), size(env, 3)))
            s_norms[ii] = sqrt(sum(F.S .^ 2))

            mps_var[ii] = reshape(F.U, size(env, 1), size(env, 2), size(F.U, 2))[:, :, 1:min(chi_trunc, size(F.U, 2))]

            T1 = diagm(F.S) * F.Vt
            T2 = mps_var[ii+1]
            @tensor temp2[i, k, l] := T1[i, j] * T2[j, k, l]
            mps_var[ii+1] = temp2[1:min(chi_trunc, size(temp2, 1)), :, :]
            mps_var[ii+1] = mps_var[ii+1] ./ norm(mps_var[ii+1])

            # Update left environment
            T1 = conj(mps_targ.tensors[ii])
            T2 = mps_var[ii]
            @tensoropt temp3[l, m] := L[i, j] * T1[i, k, l] * T2[j, k, m]
            Lenvs[ii] = temp3 ./ norm(temp3)
        end

        # Update rightmost Renv
        T1 = conj(mps_targ.tensors[end])
        T2 = mps_var[end]
        R = ones(1, 1)
        @tensoropt temp4[i, j] := T1[i, k, l] * T2[j, k, m] * R[l, m]
        Renvs[end] = temp4

        # Left sweep
        for ii in N:-1:2
            L = ii > 1 ? Lenvs[ii-1] : ones(ComplexF64, 1, 1)
            R = ii < N ? Renvs[ii+1] : ones(ComplexF64, 1, 1)
            T = conj(mps_targ.tensors[ii])

            @tensoropt env[j, k, m] := L[i, j] * T[i, k, l] * R[l, m]

            F = robust_svd(reshape(conj(env), size(env, 1), size(env, 2) * size(env, 3)))
            s_norms[ii] = sqrt(sum(F.S .^ 2))

            mps_var[ii] = reshape(F.Vt, size(F.Vt, 1), size(env, 2), size(env, 3))[1:min(chi_trunc, size(F.Vt, 1)), :, :]

            T1 = mps_var[ii-1]
            T2 = F.U * diagm(F.S)
            @tensor temp5[i, j, l] := T1[i, j, k] * T2[k, l]
            mps_var[ii-1] = temp5[:, :, 1:min(chi_trunc, size(temp5, 3))]
            mps_var[ii-1] = mps_var[ii-1] ./ norm(mps_var[ii-1])

            # Update right environment
            T1 = conj(mps_targ.tensors[ii])
            T2 = mps_var[ii]
            @tensoropt temp6[i, j] := T1[i, k, l] * T2[j, k, m] * R[l, m]
            Renvs[ii] = temp6 ./ norm(temp6)
        end

        # Check convergence
        cost = 1 - abs(inner(mps_targ, FiniteMPS{ComplexF64}(mps_var)))^2
        if abs(prev_cost - cost) / max(prev_cost, 1e-10) < 0.0001
            break
        end
        prev_cost = cost
    end

    mps_result = canonicalise(FiniteMPS{ComplexF64}(mps_var), 1)
    return 1 - abs(inner(mps_targ, mps_result))^2, mps_result, s_norms
end

"""
    AdaptVarCompress(mpo_targ::FiniteMPO, init_chi, factor, n_diff, ifPrint=true) -> FiniteMPO

Adaptively compress an MPO to achieve target infidelity.
"""
function AdaptVarCompress(mpo_targ::FiniteMPO{ComplexF64}, init_chi::Int, factor::Int,
                          n_diff::Float64, ifPrint::Bool=true)::FiniteMPO{ComplexF64}
    mps_targ, s_norms = mpo2mps(mpo_targ)
    mps_trunc = canonicalise(truncate_mps(mps_targ, init_chi), 1)

    new_mpo = mpo_targ

    for i in init_chi:factor:1024
        if i != init_chi
            mps_trunc = pad_zeros(mps_trunc, factor)
        end

        infidelity, mps_trunc, _ = VarCompress_mps_1site(mps_targ, mps_trunc, i)
        new_mpo = mps2mpo(mps_trunc, s_norms)

        if ifPrint
            println("Chi $i 1-f $(infidelity)")
            flush(stdout)
        end

        if infidelity < n_diff
            break
        end
    end

    return new_mpo
end

"""
    AdaptVarCompress(mps_targ::FiniteMPS, init_chi, factor, n_diff, ifPrint=true) -> FiniteMPS

Adaptively compress an MPS to achieve target infidelity.
"""
function AdaptVarCompress(mps_targ::FiniteMPS{ComplexF64}, init_chi::Int, factor::Int,
                          n_diff::Float64, ifPrint::Bool=true)::FiniteMPS{ComplexF64}
    mps_trunc = canonicalise(truncate_mps(mps_targ, init_chi), 1)

    new_mps = mps_targ

    for i in init_chi:factor:1024
        if i != init_chi
            mps_trunc = pad_zeros(mps_trunc, factor)
        end

        infidelity, mps_trunc, _ = VarCompress_mps_1site(mps_targ, mps_trunc, i)
        new_mps = FiniteMPS{ComplexF64}(deepcopy(mps_trunc.tensors))

        if ifPrint
            println("Chi $i 1-f $(infidelity)")
            flush(stdout)
        end

        if infidelity < n_diff
            break
        end
    end

    return new_mps
end
