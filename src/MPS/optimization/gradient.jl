# Gradient-based optimization functions
# Adapted from MPS2Circuit/src/optimization/gradient.jl
# Key changes: RSVD() → polar_unitary(), SVD() → robust_svd(),
#              Vector{Array} → FiniteMPS/FiniteMPO wrappers

using LinearAlgebra
using TensorOperations

"""
    RGrad(X, G) -> Matrix{ComplexF64}

Riemannian gradient on the unitary manifold: projects Euclidean gradient G
onto the tangent space at point X.
"""
function RGrad(X, G)
    return G - X * G' * X
end

"""
    VectorTransport(X, Y) -> Matrix{ComplexF64}

Parallel transport of tangent vector Y along the geodesic at point X on the unitary manifold.
"""
function VectorTransport(X, Y)
    return Y - 0.5 * X * (Y' * X + X' * Y)
end

"""
    LayerGrads(mps_targ, mps_test, layer_ansatz, layer_ansatz_inds) -> Vector{Matrix{ComplexF64}}

Compute gradients for a layer of two-qubit gates by building environment tensors
and extracting the gradient as -inv(SVD(env)) for each gate.
"""
function LayerGrads(mps_targ_in::FiniteMPS{ComplexF64}, mps_test_in::FiniteMPS{ComplexF64},
                    layer_ansatz_in::Vector{Matrix{ComplexF64}}, layer_ansatz_inds::Vector{Tuple{Int,Int}})

    mps_targ = mps_targ_in.tensors
    mps_test = mps_test_in.tensors
    layer_ansatz = copy(layer_ansatz_in)
    N = length(mps_targ)

    layer_Lenvs = Vector{Array{ComplexF64,3}}([ones(ComplexF64, 1, 1, 1) for _ in 1:N])
    layer_Renvs = Vector{Array{ComplexF64,3}}([ones(ComplexF64, 1, 1, 1) for _ in 1:N])
    grads = Vector{Matrix{ComplexF64}}([zeros(ComplexF64, 4, 4) for _ in 1:length(layer_ansatz_in)])

    # Build gate layer MPO
    gate_mpo = Vector{Array{ComplexF64,4}}()
    for i in 1:length(layer_ansatz)
        start_site = i == 1 ? 1 : layer_ansatz_inds[i-1][2] + 1
        for kk in start_site:layer_ansatz_inds[i][1]-1
            push!(gate_mpo, reshape(Matrix(ComplexF64(1.0) * I, 2, 2), 1, 2, 2, 1))
        end
        append!(gate_mpo, LRgate2mpo(layer_ansatz[i], layer_ansatz_inds[i], 8, 1e-8))
    end
    for kk in layer_ansatz_inds[end][2]+1:N
        push!(gate_mpo, reshape(Matrix(ComplexF64(1.0) * I, 2, 2), 1, 2, 2, 1))
    end

    # Initialize right environments
    R = ones(ComplexF64, 1, 1, 1)
    for ii in N:-1:1
        T1 = conj(mps_targ[ii])
        T2 = gate_mpo[ii]
        T3 = mps_test[ii]
        @tensoropt R[i, m, p] := T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q] * R[l, o, q]
        layer_Renvs[ii] = R ./ norm(R)
    end

    # Initialize left environments up to first gate
    L = ones(ComplexF64, 1, 1, 1)
    for ii in 1:layer_ansatz_inds[1][1]-1
        T1 = conj(mps_targ[ii])
        T2 = gate_mpo[ii]
        T3 = mps_test[ii]
        @tensoropt L[l, o, q] := L[i, m, p] * T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q]
        layer_Lenvs[ii] = L ./ norm(L)
    end

    # Compute gradients via right sweep
    for ii in 1:length(layer_ansatz)
        if ii != 1
            L = layer_ansatz_inds[ii-1][1] - 1 >= 1 ? copy(layer_Lenvs[layer_ansatz_inds[ii-1][1]-1]) : ones(ComplexF64, 1, 1, 1)
            for jj in layer_ansatz_inds[ii-1][1]:layer_ansatz_inds[ii][1]-1
                T1 = conj(mps_targ[jj])
                T2 = gate_mpo[jj]
                T3 = mps_test[jj]
                @tensoropt L[l, o, q] := L[i, m, p] * T1[i, k, l] * T2[m, k, n, o] * T3[p, n, q]
                layer_Lenvs[jj] = L ./ norm(L)
            end
        end

        if layer_ansatz_inds[ii][1] == layer_ansatz_inds[ii][2] - 1
            L_env = layer_ansatz_inds[ii][1] > 1 ? copy(layer_Lenvs[layer_ansatz_inds[ii][1]-1][:, 1, :]) : ones(ComplexF64, 1, 1)
            R_env = layer_ansatz_inds[ii][2] < N ? copy(layer_Renvs[layer_ansatz_inds[ii][2]+1][:, 1, :]) : ones(ComplexF64, 1, 1)

            T1L = conj(mps_targ[layer_ansatz_inds[ii][1]])
            T1R = conj(mps_targ[layer_ansatz_inds[ii][2]])
            T2L = mps_test[layer_ansatz_inds[ii][1]]
            T2R = mps_test[layer_ansatz_inds[ii][2]]

            @tensoropt ans[q, s, l, o] := L_env[i, j] * T1L[i, l, m] * T1R[m, o, p] * T2L[j, q, r] * T2R[r, s, t] * R_env[p, t]

            env_matrix = reshape(ans, 4, 4)
            F = svd(env_matrix)
            Uni = F.U * F.Vt
            grads[ii] = -inv(Uni)
        else
            @warn "Long-range gates not supported in LayerGrads, skipping gate $ii"
        end
    end

    return grads
end

"""
    LayerGrads(mpo_targ, mpo_test, layer_ansatz, layer_ansatz_inds) -> Vector{Matrix{ComplexF64}}

MPO version: converts to MPS and calls the MPS version.
"""
function LayerGrads(mpo_targ::FiniteMPO{ComplexF64}, mpo_test::FiniteMPO{ComplexF64},
                    layer_ansatz::Vector{Matrix{ComplexF64}}, layer_ansatz_inds::Vector{Tuple{Int,Int}})
    mps_targ, _ = mpo2mps(mpo_targ)
    mps_test, _ = mpo2mps(mpo_test)
    return LayerGrads(mps_targ, mps_test, layer_ansatz, layer_ansatz_inds)
end

# ============================================================================
# RAdam optimizer on unitary manifold
# ============================================================================

"""
    RAdam_Step(iter, lr, ansatz, gradient, momentum, variance)

RAdam step for layered ansatz structure. Updates gates on the unitary manifold.
"""
function RAdam_Step(iii::Int, lr::Float64,
                    ansatz::Vector{Vector{Matrix{ComplexF64}}},
                    gradient::Vector{Vector{Matrix{ComplexF64}}},
                    momentum_in::Vector{Vector{Matrix{ComplexF64}}},
                    variance_in::Vector{Vector{Matrix{ComplexF64}}})

    beta1 = 0.9
    beta2 = 0.999

    ansatz_new = deepcopy(ansatz)
    momentum_temp = deepcopy(momentum_in)
    variance_temp = deepcopy(variance_in)

    for ii in 1:length(ansatz)
        for j in 1:length(ansatz[ii])
            RG = RGrad(variance_temp[ii][j], gradient[ii][j])
            momentum_temp[ii][j] = beta1 * momentum_temp[ii][j] + (1 - beta1) * RG
            variance_temp[ii][j] = beta2 * variance_temp[ii][j] + (1 - beta2) * abs.(RG).^2

            bias_correction = sqrt(1 - beta2^iii) / (1 - beta1^iii)
            search_dir = (lr * bias_correction) * momentum_temp[ii][j] ./ (sqrt.(variance_temp[ii][j]) .+ 1e-8)

            ansatz_new[ii][j] = polar_unitary(ansatz_new[ii][j] - search_dir, ansatz_new[ii][j], 0.0)
            variance_temp[ii][j] = polar_unitary(variance_temp[ii][j] - search_dir, ansatz_new[ii][j], 0.0)
            momentum_temp[ii][j] = VectorTransport(variance_temp[ii][j], momentum_temp[ii][j])
        end
    end

    return ansatz_new, momentum_temp, variance_temp
end

"""
    RAdam_Step(iter, lr, ansatz, gradient, momentum, variance)

RAdam step for flat ansatz structure (single layer).
"""
function RAdam_Step(iii::Int, lr::Float64,
                    ansatz::Vector{Matrix{ComplexF64}},
                    gradient::Vector{Matrix{ComplexF64}},
                    momentum_in::Vector{Matrix{ComplexF64}},
                    variance_in::Vector{Matrix{ComplexF64}})

    beta1 = 0.9
    beta2 = 0.999

    ansatz_new = deepcopy(ansatz)
    momentum_temp = deepcopy(momentum_in)
    variance_temp = deepcopy(variance_in)

    for j in 1:length(ansatz)
        RG = RGrad(variance_temp[j], gradient[j])
        momentum_temp[j] = beta1 * momentum_temp[j] + (1 - beta1) * RG
        variance_temp[j] = beta2 * variance_temp[j] + (1 - beta2) * abs.(RG).^2

        bias_correction = sqrt(1 - beta2^iii) / (1 - beta1^iii)
        search_dir = (lr * bias_correction) * momentum_temp[j] ./ (sqrt.(variance_temp[j]) .+ 1e-8)

        ansatz_new[j] = polar_unitary(ansatz_new[j] - search_dir, ansatz_new[j], 0.0)
        variance_temp[j] = polar_unitary(variance_temp[j] - search_dir, ansatz_new[j], 0.0)
        momentum_temp[j] = VectorTransport(variance_temp[j], momentum_temp[j])
    end

    return ansatz_new, momentum_temp, variance_temp
end

# ============================================================================
# Gradient descent optimization — MPO target
# ============================================================================

"""
    optimizeGD(HP, mpo_targ, ansatz, ansatz_inds; kwargs...) -> Vector{Vector{Matrix{ComplexF64}}}

Gradient descent optimization of circuit ansatz to match target MPO.
Uses RAdam optimizer with adaptive learning rate.
"""
function optimizeGD(HP::HyperParams, mpo_targ_init::FiniteMPO{ComplexF64},
                    ansatz::Vector{Vector{Matrix{ComplexF64}}},
                    ansatz_inds::Vector{Vector{Tuple{Int,Int}}};
                    verbose::Bool=true,
                    n_sweeps::Int=10000,
                    n_warmstart::Int=0,
                    lr_init::Float64=0.01)

    N = HP.N
    max_chi = HP.max_chi
    max_trunc_err = HP.max_trunc_err
    n_layer_sweeps = HP.n_layer_sweeps
    slowdown = HP.slowdown

    lr = lr_init
    lrs = [lr]

    # Initialize gradient, momentum, variance
    gradient = deepcopy(ansatz)
    momentum = deepcopy(ansatz)
    variance = deepcopy(ansatz)
    for i in 1:length(ansatz)
        for j in 1:length(ansatz[i])
            gradient[i][j] = zeros(ComplexF64, 4, 4)
            momentum[i][j] = zeros(ComplexF64, 4, 4)
            variance[i][j] = zeros(ComplexF64, 4, 4)
        end
    end

    mpo_targ_tensors = mpo_targ_init.tensors

    # Build initial test MPO
    mpo_test_tensors = [reshape(Matrix(ComplexF64(1.0) * I, 2, 2), 1, 2, 2, 1) for _ in 1:N]
    for i in 1:length(ansatz)
        for j in 1:length(ansatz[i])
            idx1 = ansatz_inds[i][j][1]
            idx2 = ansatz_inds[i][j][2]
            mpo_test_tensors[idx1], mpo_test_tensors[idx2], _ =
                apply_2q_gate(ansatz[i][j], mpo_test_tensors[idx1], mpo_test_tensors[idx2], max_chi, max_trunc_err)
        end
    end

    mps_targ, _ = mpo2mps(mpo_targ_init)
    mps_test, _ = mpo2mps(FiniteMPO{ComplexF64}(mpo_test_tensors))
    cost = 1 - abs(inner(mps_targ, mps_test))^2

    if verbose
        println("\nInit Cost: $cost\n")
        flush(stdout)
    end

    for iii in 1:n_sweeps
        mpo_targ = copy(mpo_targ_tensors)

        # Build test MPO from current ansatz
        mpo_test_t = [reshape(Matrix(ComplexF64(1.0) * I, 2, 2), 1, 2, 2, 1) for _ in 1:N]
        for i in 1:length(ansatz)
            for j in 1:length(ansatz[i])
                idx1 = ansatz_inds[i][j][1]
                idx2 = ansatz_inds[i][j][2]
                mpo_test_t[idx1], mpo_test_t[idx2], _ =
                    apply_2q_gate(ansatz[i][j], mpo_test_t[idx1], mpo_test_t[idx2], max_chi, max_trunc_err)
            end
        end
        mpo_test_compressed = AdaptVarCompress(FiniteMPO{ComplexF64}(mpo_test_t), 20, 10, 1e-8, false)

        mps_targ_curr, _ = mpo2mps(FiniteMPO{ComplexF64}(mpo_targ))
        mps_test_curr, _ = mpo2mps(mpo_test_compressed)
        cost = 1 - abs(inner(mps_targ_curr, mps_test_curr))^2

        if verbose
            println("$iii ", round(cost, sigdigits=8), " lr=", round(lrs[end], sigdigits=3))
            flush(stdout)
        end

        # Backward sweep through layers to compute gradients
        for ii in length(ansatz):-1:1
            for j in 1:length(ansatz[ii])
                idx1 = ansatz_inds[ii][j][1]
                idx2 = ansatz_inds[ii][j][2]
                mpo_test_t[idx1], mpo_test_t[idx2], _ =
                    apply_2q_gate(inv(ansatz[ii][j]), mpo_test_t[idx1], mpo_test_t[idx2], max_chi, max_trunc_err)
            end

            if iii <= n_warmstart
                mps_targ_w, _ = mpo2mps(FiniteMPO{ComplexF64}(mpo_targ))
                mps_test_w, _ = mpo2mps(FiniteMPO{ComplexF64}(mpo_test_t))
                ansatz[ii] = LayerSweep_general(mps_targ_w, mps_test_w, ansatz[ii], ansatz_inds[ii], n_layer_sweeps, slowdown)
            end

            gradient[ii] = LayerGrads(FiniteMPO{ComplexF64}(mpo_targ), FiniteMPO{ComplexF64}(mpo_test_t),
                                      ansatz[ii], ansatz_inds[ii])

            for j in 1:length(ansatz[ii])
                idx1 = ansatz_inds[ii][j][1]
                idx2 = ansatz_inds[ii][j][2]
                mpo_targ[idx1], mpo_targ[idx2], _ =
                    apply_2q_gate(inv(ansatz[ii][j]), mpo_targ[idx1], mpo_targ[idx2], max_chi, max_trunc_err)
            end
        end

        # Line search over learning rates
        if iii <= n_warmstart
            _, momentum, variance = RAdam_Step(iii, lr, deepcopy(ansatz), deepcopy(gradient),
                                               deepcopy(momentum), deepcopy(variance))
        else
            curr_cost = cost
            lr_best = lrs[end] * 10^-0.5

            for lr_temp in [lrs[end] * 10^-0.5, lrs[end] * 10^-0.25, lrs[end], lrs[end] * 10^0.25, lrs[end] * 10^0.5]
                ansatz_temp, _, _ = RAdam_Step(iii, lr_temp, deepcopy(ansatz), deepcopy(gradient),
                                               deepcopy(momentum), deepcopy(variance))

                mpo_test_temp = [reshape(Matrix(ComplexF64(1.0) * I, 2, 2), 1, 2, 2, 1) for _ in 1:N]
                for i in 1:length(ansatz)
                    for j in 1:length(ansatz[i])
                        idx1 = ansatz_inds[i][j][1]
                        idx2 = ansatz_inds[i][j][2]
                        mpo_test_temp[idx1], mpo_test_temp[idx2], _ =
                            apply_2q_gate(ansatz_temp[i][j], mpo_test_temp[idx1], mpo_test_temp[idx2], max_chi, max_trunc_err)
                    end
                end

                mps_t1, _ = mpo2mps(mpo_targ_init)
                mps_t2, _ = mpo2mps(FiniteMPO{ComplexF64}(mpo_test_temp))
                new_cost = 1 - abs(inner(mps_t1, mps_t2))^2

                if new_cost < curr_cost
                    curr_cost = new_cost
                    lr_best = lr_temp
                end
            end

            lr_best = max(lr_best, 1e-5)
            ansatz, momentum, variance = RAdam_Step(iii, lr_best, ansatz, gradient, momentum, variance)
            push!(lrs, lr_best)
        end
    end

    return ansatz
end

"""
    optimizeGD(HP, mps_targ, ansatz, ansatz_inds; kwargs...) -> Vector{Vector{Matrix{ComplexF64}}}

MPS target version: converts to MPO and calls the MPO version.
"""
function optimizeGD(HP::HyperParams, mps_targ::FiniteMPS{ComplexF64},
                    ansatz::Vector{Vector{Matrix{ComplexF64}}},
                    ansatz_inds::Vector{Vector{Tuple{Int,Int}}};
                    kwargs...)
    mpo_targ = mps2mpo(mps_targ)
    return optimizeGD(HP, mpo_targ, ansatz, ansatz_inds; kwargs...)
end

"""
    optimizeGD_flat(HP, mps_targ, ansatz, ansatz_inds; kwargs...) -> Vector{Matrix{ComplexF64}}

Gradient descent with flat ansatz (single layer).
"""
function optimizeGD_flat(HP::HyperParams, mps_targ_in::FiniteMPS{ComplexF64},
                         ansatz::Vector{Matrix{ComplexF64}},
                         ansatz_inds::Vector{Tuple{Int,Int}};
                         verbose::Bool=true,
                         n_sweeps::Int=10000,
                         lr_init::Float64=0.01)

    N = HP.N
    max_chi = HP.max_chi
    max_trunc_err = HP.max_trunc_err

    lr = lr_init
    lrs = [lr]

    gradient = [zeros(ComplexF64, 4, 4) for _ in 1:length(ansatz)]
    momentum = [zeros(ComplexF64, 4, 4) for _ in 1:length(ansatz)]
    variance = [zeros(ComplexF64, 4, 4) for _ in 1:length(ansatz)]

    mps_test = Ansatz2MPS(N, [ansatz], [ansatz_inds], max_chi, max_trunc_err)
    cost = 1 - abs(inner(mps_targ_in, mps_test))^2

    if verbose
        println("\nInit Cost: $cost\n")
        flush(stdout)
    end

    for iii in 1:n_sweeps
        mps_test = Ansatz2MPS(N, [ansatz], [ansatz_inds], max_chi, max_trunc_err)
        cost = 1 - abs(inner(mps_targ_in, mps_test))^2

        if verbose
            println("$iii ", round(cost, sigdigits=8), " lr=", round(lrs[end], sigdigits=3))
            flush(stdout)
        end

        gradient = LayerGrads(mps_targ_in, mps_test, ansatz, ansatz_inds)

        # Line search
        curr_cost = cost
        lr_best = lrs[end] * 10^-0.5

        for lr_temp in [lrs[end] * 10^-0.5, lrs[end] * 10^-0.25, lrs[end], lrs[end] * 10^0.25, lrs[end] * 10^0.5]
            ansatz_temp, _, _ = RAdam_Step(iii, lr_temp, deepcopy(ansatz), deepcopy(gradient),
                                           deepcopy(momentum), deepcopy(variance))
            mps_test_temp = Ansatz2MPS(N, [ansatz_temp], [ansatz_inds], max_chi, max_trunc_err)
            new_cost = 1 - abs(inner(mps_targ_in, mps_test_temp))^2

            if new_cost < curr_cost
                curr_cost = new_cost
                lr_best = lr_temp
            end
        end

        lr_best = max(lr_best, 1e-5)
        ansatz, momentum, variance = RAdam_Step(iii, lr_best, ansatz, gradient, momentum, variance)
        push!(lrs, lr_best)
    end

    return ansatz
end
