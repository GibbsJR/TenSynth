# Constrained optimization functions
# Adapted from MPS2Circuit/src/optimization/constrained.jl
# Key changes: RSVD() → polar_unitary(), raw tensors → FiniteMPS wrappers

using LinearAlgebra

"""
    optimize_constrain(HP, n_cnots, mps_targ, ansatz, ansatz_inds; kwargs...)
        -> (Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}})

Optimize circuit ansatz with constrained CNOT count per two-qubit gate.
Each gate is projected onto the space of unitaries achievable with n_cnots CNOTs
via Uni2CNOTs after each optimization step.
"""
function optimize_constrain(HP::HyperParams, n_cnots::Int,
                            mps_targ_init::FiniteMPS{ComplexF64},
                            ansatz::Vector{Matrix{ComplexF64}},
                            ansatz_inds::Vector{Tuple{Int,Int}};
                            print_info::Bool=true)

    N = HP.N
    max_chi = HP.max_chi
    max_trunc_err = HP.max_trunc_err
    slowdown = HP.slowdown
    n_sweeps = HP.n_sweeps
    converge_threshold = HP.converge_threshold

    mps_targ = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))

    # Build initial test MPS
    mps_test = canonicalise(zeroMPS(N), 1)
    prev_inds_test = (1, 2)
    mps_test, prev_inds_test = ApplyGateLayers(mps_test, ansatz, ansatz_inds, max_chi, max_trunc_err, prev_inds_test)

    cost = 1 - abs(inner(mps_targ_init, mps_test))^2

    if print_info
        println("\nInit Cost: $cost\n")
        println("N gates: $(length(ansatz))\n")
        flush(stdout)
    end

    prev_cost = 1.0

    for iii in 1:n_sweeps
        # Canonicalize to the last gate position
        mps_test = canonicalise(mps_test, ansatz_inds[end][1])
        mps_targ = canonicalise(mps_targ, ansatz_inds[end][1])

        prev_inds_test = ansatz_inds[end]
        prev_inds_targ = ansatz_inds[end]

        # Backward sweep
        for ii in length(ansatz):-1:2
            mps_test, prev_inds_test = ApplyGateLayers(
                mps_test, [inv(ansatz[ii])], [ansatz_inds[ii]],
                max_chi, max_trunc_err, prev_inds_test)

            U_temp = SingleGateEnv(mps_targ, mps_test, ansatz[ii], ansatz_inds[ii], slowdown, false)
            ansatz[ii] = polar_unitary(Uni2CNOTs(U_temp; n_cnots=n_cnots), ansatz[ii], slowdown)

            mps_targ, prev_inds_targ = ApplyGateLayers(
                mps_targ, [inv(ansatz[ii])], [ansatz_inds[ii]],
                max_chi, max_trunc_err, prev_inds_targ)
        end

        # Handle first gate
        mps_targ, prev_inds_targ = ApplyGateLayers(
            mps_targ, [inv(ansatz[1])], [ansatz_inds[1]],
            max_chi, max_trunc_err, prev_inds_targ)

        # Reset test MPS for forward sweep
        mps_test = canonicalise(zeroMPS(N), 1)
        prev_inds_test = (1, 2)

        # Forward sweep
        for ii in 1:length(ansatz)
            mps_targ, prev_inds_targ = ApplyGateLayers(
                mps_targ, [ansatz[ii]], [ansatz_inds[ii]],
                max_chi, max_trunc_err, prev_inds_targ)

            U_temp = SingleGateEnv(mps_targ, mps_test, ansatz[ii], ansatz_inds[ii], slowdown, false)
            ansatz[ii] = polar_unitary(Uni2CNOTs(U_temp; n_cnots=n_cnots), ansatz[ii], slowdown)

            mps_test, prev_inds_test = ApplyGateLayers(
                mps_test, [ansatz[ii]], [ansatz_inds[ii]],
                max_chi, max_trunc_err, prev_inds_test)
        end

        # Reset target MPS
        mps_targ = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))

        cost = 1 - abs(inner(mps_targ, mps_test))^2

        # Full cost with higher precision
        mps_test_full = canonicalise(zeroMPS(N), 1)
        prev_inds_full = (1, 2)
        mps_test_full, prev_inds_full = ApplyGateLayers(
            mps_test_full, ansatz, ansatz_inds, 2 * max_chi, 1e-16, prev_inds_full)
        full_cost = 1 - abs(inner(mps_targ_init, mps_test_full))^2

        if print_info
            println("$iii  $(round(cost, sigdigits=8)) $(round(full_cost, sigdigits=8))")
            flush(stdout)
        end

        if iii > 5 && (abs(prev_cost - full_cost) / max(abs(full_cost), 1e-10) < converge_threshold || full_cost < 1e-8)
            return ansatz, ansatz_inds
        end
        prev_cost = full_cost

        _mygc()
    end

    return ansatz, ansatz_inds
end
