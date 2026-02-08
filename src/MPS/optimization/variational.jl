# Variational optimization functions
# Adapted from MPS2Circuit/src/optimization/variational.jl
# Key changes: Vector{Array} → FiniteMPS/FiniteMPO wrappers, SVD() → robust_svd(),
#              RSVD() → polar_unitary(), OrthMPS wraps FiniteMPS

using LinearAlgebra
using TensorOperations

# GC helper for optimization loops
_mygc() = GC.gc(false)

# ============================================================================
# OrthMPS: MPS with tracked orthogonality center
# ============================================================================

"""
    OrthMPS

MPS with tracked orthogonality center to avoid redundant canonicalization.
Wraps a FiniteMPS and tracks the current orthogonality center position.
"""
mutable struct OrthMPS
    mps::FiniteMPS{ComplexF64}
    orth_centre::Int  # Current orthogonality center (0 = unknown)
end

OrthMPS(mps::FiniteMPS{ComplexF64}) = OrthMPS(mps, 0)

Base.getindex(omps::OrthMPS, i::Int) = omps.mps.tensors[i]
Base.setindex!(omps::OrthMPS, v, i::Int) = (omps.mps.tensors[i] = v)
Base.length(omps::OrthMPS) = length(omps.mps.tensors)
Base.copy(omps::OrthMPS) = OrthMPS(FiniteMPS{ComplexF64}(copy(omps.mps.tensors)), omps.orth_centre)
Base.deepcopy(omps::OrthMPS) = OrthMPS(FiniteMPS{ComplexF64}(deepcopy(omps.mps.tensors)), omps.orth_centre)

"""
    ensure_canonical!(omps::OrthMPS, target_site::Int) -> OrthMPS

Move orthogonality center to target_site using minimal operations.
"""
function ensure_canonical!(omps::OrthMPS, target_site::Int)
    if omps.orth_centre == target_site
        return omps
    end
    if omps.orth_centre == 0
        omps.mps = canonicalise(omps.mps, target_site)
    else
        omps.mps = canonicalise_FromTo(omps.mps, (omps.orth_centre, omps.orth_centre),
                                       (target_site, target_site))
    end
    omps.orth_centre = target_site
    return omps
end

function invalidate_orth_centre!(omps::OrthMPS)
    omps.orth_centre = 0
    return omps
end

function set_orth_centre!(omps::OrthMPS, site::Int)
    omps.orth_centre = site
    return omps
end

# ============================================================================
# GateEnvCache: Cache for environment tensors
# ============================================================================

"""
    GateEnvCache

Cache for precomputed left and right MPS-MPS environment tensors.
After update_cache!, get_gate_env provides O(1) lookup.
"""
mutable struct GateEnvCache
    mps_targ_tensors::Vector{Array{ComplexF64,3}}  # conjugated target
    mps_test_tensors::Vector{Array{ComplexF64,3}}
    Lenvs::Vector{Array{ComplexF64,2}}
    Renvs::Vector{Array{ComplexF64,2}}
    valid::Bool
    N::Int
end

function GateEnvCache(N::Int)
    Lenvs = [ones(ComplexF64, 1, 1) for _ in 1:N+1]
    Renvs = [ones(ComplexF64, 1, 1) for _ in 1:N+1]
    return GateEnvCache(Array{ComplexF64,3}[], Array{ComplexF64,3}[], Lenvs, Renvs, false, N)
end

"""
    update_cache!(cache, mps_targ, mps_test)

Recompute all environments for the given MPS pair.
"""
function update_cache!(cache::GateEnvCache, mps_targ::FiniteMPS{ComplexF64},
                       mps_test::FiniteMPS{ComplexF64})
    N = length(mps_targ.tensors)
    cache.mps_targ_tensors = [conj(t) for t in mps_targ.tensors]
    cache.mps_test_tensors = copy(mps_test.tensors)
    cache.N = N

    if length(cache.Lenvs) != N + 1
        cache.Lenvs = [ones(ComplexF64, 1, 1) for _ in 1:N+1]
        cache.Renvs = [ones(ComplexF64, 1, 1) for _ in 1:N+1]
    end

    # Left environments
    cache.Lenvs[1] = ones(ComplexF64, 1, 1)
    L = ones(ComplexF64, 1, 1)
    for ii in 1:N
        T1 = cache.mps_targ_tensors[ii]
        T3 = cache.mps_test_tensors[ii]
        @tensoropt L_new[l, q] := L[i, p] * T1[i, k, l] * T3[p, k, q]
        L = L_new ./ norm(L_new)
        cache.Lenvs[ii+1] = L
    end

    # Right environments
    cache.Renvs[N+1] = ones(ComplexF64, 1, 1)
    R = ones(ComplexF64, 1, 1)
    for ii in N:-1:1
        T1 = cache.mps_targ_tensors[ii]
        T3 = cache.mps_test_tensors[ii]
        @tensoropt R_new[i, p] := T1[i, k, l] * T3[p, k, q] * R[l, q]
        R = R_new ./ norm(R_new)
        cache.Renvs[ii] = R
    end

    cache.valid = true
end

function invalidate_cache!(cache::GateEnvCache)
    cache.valid = false
end

function get_gate_env(cache::GateEnvCache, inds::Tuple{Int,Int})
    @assert cache.valid "Cache must be initialized with update_cache!"
    return cache.Lenvs[inds[1]], cache.Renvs[inds[2]+1]
end

# ============================================================================
# SingleGateEnv — cached and direct versions
# ============================================================================

"""
    SingleGateEnv_cached(cache, gate, inds, slowdown, JustEnv=false) -> Matrix{ComplexF64}

Compute optimal gate from cached environment tensors. O(χ²) per gate.
"""
function SingleGateEnv_cached(cache::GateEnvCache, gate::Matrix{ComplexF64},
                              inds::Tuple{Int,Int}, slowdown::Float64,
                              JustEnv::Bool=false)::Matrix{ComplexF64}
    @assert cache.valid "Cache must be initialized with update_cache!"

    L, R = get_gate_env(cache, inds)

    if inds[1] == inds[2] - 1
        T1L = cache.mps_targ_tensors[inds[1]]
        T1R = cache.mps_targ_tensors[inds[2]]
        T2L = cache.mps_test_tensors[inds[1]]
        T2R = cache.mps_test_tensors[inds[2]]

        @tensoropt ans[q, s, l, o] := L[i, j] * T1L[i, l, m] * T1R[m, o, p] * T2L[j, q, r] * T2R[r, s, t] * R[p, t]

        @tensoropt denom[] := L[i, j] * T1L[i, q, m] * T1R[m, s, p] * T2L[j, q, r] * T2R[r, s, t] * R[p, t]
        ans = ans ./ denom[1]
    else
        T1 = cache.mps_targ_tensors[inds[1]]
        T2 = cache.mps_test_tensors[inds[1]]
        @tensoropt L_contract[l, q, m, r] := L[i, j] * T1[i, l, m] * T2[j, q, r]

        for iii in inds[1]+1:inds[2]-1
            T1 = cache.mps_targ_tensors[iii]
            T2 = cache.mps_test_tensors[iii]
            @tensoropt L_contract[i, j, n, o] := L_contract[i, j, k, l] * T1[k, m, n] * T2[l, m, o]
            L_contract = L_contract ./ norm(L_contract)
        end

        T1 = cache.mps_targ_tensors[inds[2]]
        T2 = cache.mps_test_tensors[inds[2]]
        @tensoropt ans[j, o, i, l] := L_contract[i, j, k, n] * T1[k, l, m] * T2[n, o, p] * R[m, p]

        T1L = cache.mps_targ_tensors[inds[1]]
        T1R = cache.mps_targ_tensors[inds[2]]
        T2L = cache.mps_test_tensors[inds[1]]
        T2R = cache.mps_test_tensors[inds[2]]
        @tensoropt denom[] := L[i, j] * T1L[i, q, m] * T1R[m, s, p] * T2L[j, q, r] * T2R[r, s, t] * R[p, t]
        ans = ans ./ denom[1]
    end

    F = robust_svd(reshape(ans, 4, 4))

    if !JustEnv
        return polar_unitary(inv(F.U * F.Vt), gate, slowdown)
    else
        return reshape(ans, 4, 4)'
    end
end

"""
    SingleGateEnv(mps_targ, mps_test, gate, inds, slowdown, JustEnv=false) -> Matrix{ComplexF64}

Compute optimal gate from full environment computation. O(N·χ²) per gate.
"""
function SingleGateEnv(mps_targ_in::FiniteMPS{ComplexF64}, mps_test_in::FiniteMPS{ComplexF64},
                       gate::Matrix{ComplexF64}, inds::Tuple{Int,Int}, slowdown::Float64,
                       JustEnv::Bool=false)::Matrix{ComplexF64}

    mps_targ = mps_targ_in.tensors
    mps_test = mps_test_in.tensors
    N = length(mps_targ)

    # Initialize Renvs
    R = ones(ComplexF64, 1, 1)
    for ii in N:-1:inds[2]+1
        T1 = conj(mps_targ[ii])
        T3 = mps_test[ii]
        @tensoropt R[i, p] := T1[i, k, l] * T3[p, k, q] * R[l, q]
        R = R ./ norm(R)
    end

    # Initialize Lenvs
    L = ones(ComplexF64, 1, 1)
    for ii in 1:inds[1]-1
        T1 = conj(mps_targ[ii])
        T3 = mps_test[ii]
        @tensoropt L[l, q] := L[i, p] * T1[i, k, l] * T3[p, k, q]
        L = L ./ norm(L)
    end

    if inds[1] == inds[2] - 1
        T1L = conj(mps_targ[inds[1]])
        T1R = conj(mps_targ[inds[2]])
        T2L = mps_test[inds[1]]
        T2R = mps_test[inds[2]]

        @tensoropt ans[q, s, l, o] := L[i, j] * T1L[i, l, m] * T1R[m, o, p] * T2L[j, q, r] * T2R[r, s, t] * R[p, t]

        @tensoropt denom[] := L[i, j] * T1L[i, q, m] * T1R[m, s, p] * T2L[j, q, r] * T2R[r, s, t] * R[p, t]
        ans = ans ./ denom[1]
    else
        T1 = conj(mps_targ[inds[1]])
        T2 = mps_test[inds[1]]
        @tensoropt L_contract[l, q, m, r] := L[i, j] * T1[i, l, m] * T2[j, q, r]

        for iii in inds[1]+1:inds[2]-1
            T1 = conj(mps_targ[iii])
            T2 = mps_test[iii]
            @tensoropt L_contract[i, j, n, o] := L_contract[i, j, k, l] * T1[k, m, n] * T2[l, m, o]
            L_contract = L_contract ./ norm(L_contract)
        end

        T1 = conj(mps_targ[inds[2]])
        T2 = mps_test[inds[2]]
        @tensoropt ans[j, o, i, l] := L_contract[i, j, k, n] * T1[k, l, m] * T2[n, o, p] * R[m, p]

        T1L = conj(mps_targ[inds[1]])
        T1R = conj(mps_targ[inds[2]])
        T2L = mps_test[inds[1]]
        T2R = mps_test[inds[2]]
        @tensoropt denom[] := L[i, j] * T1L[i, q, m] * T1R[m, s, p] * T2L[j, q, r] * T2R[r, s, t] * R[p, t]
        ans = ans ./ denom[1]
    end

    F = robust_svd(reshape(ans, 4, 4))

    if !JustEnv
        return polar_unitary(inv(F.U * F.Vt), gate, slowdown)
    else
        return reshape(ans, 4, 4)'
    end
end

# ============================================================================
# Layered optimize (non-overlapping layers via LayerSweep_general)
# ============================================================================

"""
    optimize_layered(HP, mps_targ_init, ansatz, ansatz_inds; kwargs...)
        -> (Vector{Vector{Matrix{ComplexF64}}}, Vector{Vector{Tuple{Int,Int}}})

Optimize a layered gate ansatz for MPS state preparation using alternating sweeps.
"""
function optimize_layered(HP::HyperParams, mps_targ_init::FiniteMPS{ComplexF64},
                          ansatz::Vector{Vector{Matrix{ComplexF64}}},
                          ansatz_inds::Vector{Vector{Tuple{Int,Int}}};
                          print_info::Bool=true)

    N = HP.N
    max_chi = HP.max_chi
    max_trunc_err = HP.max_trunc_err
    n_layer_sweeps = HP.n_layer_sweeps
    slowdown = HP.slowdown
    n_sweeps = HP.n_sweeps
    converge_threshold = HP.converge_threshold
    print_every = HP.print_every

    if print_info && print_every != 0
        println("\nN Layers: $(length(ansatz))\n")
        flush(stdout)
    end

    mps_targ = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))
    mps_test = Ansatz2MPS(N, ansatz, ansatz_inds, max_chi, max_trunc_err)

    cost = 1 - abs(inner(mps_targ_init, mps_test))^2

    if print_info && print_every != 0
        println("Init Cost: $cost\n")
        flush(stdout)
    end

    prev_cost = 1.0

    mps_test_orth = OrthMPS(mps_test, 0)
    mps_targ_orth = OrthMPS(mps_targ, 0)

    for iii in 1:n_sweeps
        target_site = ansatz_inds[end][1][1]
        ensure_canonical!(mps_test_orth, target_site)
        ensure_canonical!(mps_targ_orth, target_site)

        prev_inds_test = ansatz_inds[end][1]
        prev_inds_targ = ansatz_inds[end][1]

        # Backward sweep through layers
        for ii in length(ansatz):-1:2
            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, [inv(u) for u in ansatz[ii]],
                ansatz_inds[ii], max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])

            ansatz[ii] = LayerSweep_general(mps_targ_orth.mps, mps_test_orth.mps,
                                            ansatz[ii], ansatz_inds[ii], n_layer_sweeps, slowdown)

            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, [inv(u) for u in ansatz[ii]],
                ansatz_inds[ii], max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])
        end

        # Apply inverse of layer 1 to target
        mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
            mps_targ_orth.mps, [inv(u) for u in ansatz[1]],
            ansatz_inds[1], max_chi, max_trunc_err, prev_inds_targ)
        set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

        # Reset test MPS to |0...0⟩
        mps_test_orth.mps = zeroMPS(N)
        ensure_canonical!(mps_test_orth, 1)
        set_orth_centre!(mps_test_orth, 1)
        ensure_canonical!(mps_targ_orth, 1)

        prev_inds_targ = (1, 1)
        prev_inds_test = (1, 1)

        # Forward sweep through layers
        for ii in 1:length(ansatz)
            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, ansatz[ii], ansatz_inds[ii],
                max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

            ansatz[ii] = LayerSweep_general(mps_targ_orth.mps, mps_test_orth.mps,
                                            ansatz[ii], ansatz_inds[ii], n_layer_sweeps, slowdown)

            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, ansatz[ii], ansatz_inds[ii],
                max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])
        end

        ensure_canonical!(mps_test_orth, 1)
        mps_targ_orth.mps = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))
        invalidate_orth_centre!(mps_targ_orth)

        cost = 1 - abs(inner(mps_targ_orth.mps, mps_test_orth.mps))^2

        if (print_every == 0 && iii % 10 == 0) || (print_every != 0 && iii % print_every == 0)
            mps_test_full = Ansatz2MPS(N, ansatz, ansatz_inds, Int(1.5 * max_chi), 1e-12)
            full_cost = 1 - abs(inner(mps_targ_init, mps_test_full))^2

            if print_info && print_every != 0
                println("$iii  $(round(cost, sigdigits=8)) $(round(full_cost, sigdigits=8))")
                flush(stdout)
            end

            if abs(prev_cost - full_cost) / max(abs(full_cost), 1e-10) < converge_threshold || full_cost < 1e-8
                return ansatz, ansatz_inds
            end
            prev_cost = full_cost
        end

        _mygc()
    end

    return ansatz, ansatz_inds
end

# ============================================================================
# Flat optimize (gate-by-gate via SingleGateEnv_cached)
# ============================================================================

"""
    optimize_flat(HP, mps_targ_init, ansatz, ansatz_inds; kwargs...)
        -> (Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}})

Optimize a flat (non-layered) gate ansatz for MPS state preparation.
"""
function optimize_flat(HP::HyperParams, mps_targ_init::FiniteMPS{ComplexF64},
                       ansatz::Vector{Matrix{ComplexF64}}, ansatz_inds::Vector{Tuple{Int,Int}};
                       print_info::Bool=true)

    N = HP.N
    max_chi = HP.max_chi
    max_trunc_err = HP.max_trunc_err
    slowdown = HP.slowdown
    n_sweeps = HP.n_sweeps
    print_every = HP.print_every
    converge_threshold = HP.converge_threshold

    mps_targ = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))

    mps_test = canonicalise(zeroMPS(N), 1)
    prev_inds_test = (1, 2)
    mps_test, prev_inds_test = ApplyGateLayers(mps_test, ansatz, ansatz_inds, max_chi, max_trunc_err, prev_inds_test)

    cost = 1 - abs(inner(mps_targ_init, mps_test))^2

    if print_info && print_every != 0
        println("Init Cost: $cost\n")
        println("N gates: $(length(ansatz))\n")
        flush(stdout)
    end

    prev_cost = 1.0

    mps_test_orth = OrthMPS(mps_test, prev_inds_test[2])
    mps_targ_orth = OrthMPS(mps_targ, 0)

    gate_env_cache = GateEnvCache(N)

    for iii in 1:n_sweeps
        target_site = ansatz_inds[end][1]
        ensure_canonical!(mps_test_orth, target_site)
        ensure_canonical!(mps_targ_orth, target_site)

        prev_inds_test = ansatz_inds[end]
        prev_inds_targ = ansatz_inds[end]

        # Backward sweep through gates
        for ii in length(ansatz):-1:2
            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, [inv(ansatz[ii])], [ansatz_inds[ii]],
                max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])

            update_cache!(gate_env_cache, mps_targ_orth.mps, mps_test_orth.mps)
            ansatz[ii] = SingleGateEnv_cached(gate_env_cache, ansatz[ii], ansatz_inds[ii], slowdown)

            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, [inv(ansatz[ii])], [ansatz_inds[ii]],
                max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])
        end

        # Apply inverse of gate 1 to target
        mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
            mps_targ_orth.mps, [inv(ansatz[1])], [ansatz_inds[1]],
            max_chi, max_trunc_err, prev_inds_targ)
        set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

        # Reset test MPS to |0...0⟩
        mps_test_orth.mps = zeroMPS(N)
        ensure_canonical!(mps_test_orth, 1)
        set_orth_centre!(mps_test_orth, 1)
        prev_inds_test = (1, 2)

        # Forward sweep through gates
        for ii in 1:length(ansatz)
            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, [ansatz[ii]], [ansatz_inds[ii]],
                max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

            update_cache!(gate_env_cache, mps_targ_orth.mps, mps_test_orth.mps)
            ansatz[ii] = SingleGateEnv_cached(gate_env_cache, ansatz[ii], ansatz_inds[ii], slowdown)

            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, [ansatz[ii]], [ansatz_inds[ii]],
                max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])
        end

        mps_targ_orth.mps = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))
        invalidate_orth_centre!(mps_targ_orth)
        cost = 1 - abs(inner(mps_targ_orth.mps, mps_test_orth.mps))^2

        if iii % max(1, print_every) == 0
            mps_test_full = canonicalise(zeroMPS(N), 1)
            prev_inds_full = (1, 2)
            mps_test_full, prev_inds_full = ApplyGateLayers(mps_test_full, ansatz, ansatz_inds, 2 * max_chi, 1e-16, prev_inds_full)
            full_cost = 1 - abs(inner(mps_targ_init, mps_test_full))^2

            if print_info && print_every != 0
                println("$iii  $(round(cost, sigdigits=8)) $(round(full_cost, sigdigits=8))")
                flush(stdout)
            end

            if (iii >= print_every * 4 && abs(prev_cost - full_cost) / max(abs(full_cost), 1e-10) < converge_threshold) || full_cost < 1e-8
                return ansatz, ansatz_inds
            end
            prev_cost = full_cost
        end

        _mygc()
    end

    return ansatz, ansatz_inds
end

# ============================================================================
# PBC optimize (periodic boundary conditions)
# ============================================================================

"""
    optimize_pbc(HP, mps_targ_init, ansatz, ansatz_inds; kwargs...)
        -> (Vector{Vector{Matrix{ComplexF64}}}, Vector{Vector{Tuple{Int,Int}}})

Optimize a layered gate ansatz with PBC gate support.
PBC gates are identified by having indices (1, N).
"""
function optimize_pbc(HP::HyperParams, mps_targ_init::FiniteMPS{ComplexF64},
                      ansatz::Vector{Vector{Matrix{ComplexF64}}},
                      ansatz_inds::Vector{Vector{Tuple{Int,Int}}};
                      print_info::Bool=true)

    N = HP.N
    max_chi = HP.max_chi
    max_trunc_err = HP.max_trunc_err
    n_layer_sweeps = HP.n_layer_sweeps
    slowdown = HP.slowdown
    n_sweeps = HP.n_sweeps
    converge_threshold = HP.converge_threshold

    mps_targ = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))
    mps_test = Ansatz2MPS(N, ansatz, ansatz_inds, 2 * max_chi, 1e-16)

    cost = 1 - abs(inner(mps_targ_init, mps_test))^2

    if print_info
        println("\nInit Cost: $cost\n")
        println("N Layers: $(length(ansatz))\n")
        flush(stdout)
    end

    prev_cost = 1.0

    mps_test_orth = OrthMPS(mps_test, 0)
    mps_targ_orth = OrthMPS(mps_targ, 0)

    for iii in 1:n_sweeps
        target_site = ansatz_inds[end][1][1]
        ensure_canonical!(mps_test_orth, target_site)
        ensure_canonical!(mps_targ_orth, target_site)

        prev_inds_targ = ansatz_inds[end][1]
        prev_inds_test = ansatz_inds[end][1]

        # Backward sweep through layers
        for ii in length(ansatz):-1:2
            pbc = ansatz_inds[ii][end] == (1, N)

            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, [inv(u) for u in ansatz[ii]][1:end-Int(pbc)],
                ansatz_inds[ii][1:end-Int(pbc)], max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])

            if pbc
                ansatz[ii] = push!(LayerSweep_general(mps_targ_orth.mps, mps_test_orth.mps,
                    ansatz[ii][1:end-1], ansatz_inds[ii][1:end-1], n_layer_sweeps, slowdown), ansatz[ii][end])
            else
                ansatz[ii] = LayerSweep_general(mps_targ_orth.mps, mps_test_orth.mps,
                    ansatz[ii], ansatz_inds[ii], n_layer_sweeps, slowdown)
            end

            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, [inv(u) for u in ansatz[ii]][1:end-Int(pbc)],
                ansatz_inds[ii][1:end-Int(pbc)], max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

            if pbc
                mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                    mps_test_orth.mps, [inv(ansatz[ii][end])], [ansatz_inds[ii][end]],
                    max_chi, max_trunc_err, prev_inds_test)
                set_orth_centre!(mps_test_orth, prev_inds_test[2])

                ansatz[ii][end] = SingleGateEnv(mps_targ_orth.mps, mps_test_orth.mps,
                    ansatz[ii][end], ansatz_inds[ii][end], slowdown)

                mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                    mps_targ_orth.mps, [inv(ansatz[ii][end])], [ansatz_inds[ii][end]],
                    max_chi, max_trunc_err, prev_inds_targ)
                set_orth_centre!(mps_targ_orth, prev_inds_targ[2])
            end
        end

        # Apply inverse of layer 1 to target
        mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
            mps_targ_orth.mps, [inv(u) for u in ansatz[1]],
            ansatz_inds[1], max_chi, max_trunc_err, prev_inds_targ)
        set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

        # Reset test MPS and forward sweep
        mps_test_orth.mps = zeroMPS(N)
        ensure_canonical!(mps_test_orth, 1)
        set_orth_centre!(mps_test_orth, 1)
        ensure_canonical!(mps_targ_orth, 1)

        prev_inds_targ = (1, 1)
        prev_inds_test = (1, 1)

        for ii in 1:length(ansatz)
            pbc = ansatz_inds[ii][end] == (1, N)

            mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                mps_targ_orth.mps, ansatz[ii][1:end-Int(pbc)],
                ansatz_inds[ii][1:end-Int(pbc)], max_chi, max_trunc_err, prev_inds_targ)
            set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

            if pbc
                ansatz[ii] = push!(LayerSweep_general(mps_targ_orth.mps, mps_test_orth.mps,
                    ansatz[ii][1:end-1], ansatz_inds[ii][1:end-1], n_layer_sweeps, slowdown), ansatz[ii][end])
            else
                ansatz[ii] = LayerSweep_general(mps_targ_orth.mps, mps_test_orth.mps,
                    ansatz[ii], ansatz_inds[ii], n_layer_sweeps, slowdown)
            end

            mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                mps_test_orth.mps, ansatz[ii][1:end-Int(pbc)],
                ansatz_inds[ii][1:end-Int(pbc)], max_chi, max_trunc_err, prev_inds_test)
            set_orth_centre!(mps_test_orth, prev_inds_test[2])

            if pbc
                mps_targ_orth.mps, prev_inds_targ = ApplyGateLayers(
                    mps_targ_orth.mps, [ansatz[ii][end]], [ansatz_inds[ii][end]],
                    max_chi, max_trunc_err, prev_inds_targ)
                set_orth_centre!(mps_targ_orth, prev_inds_targ[2])

                ansatz[ii][end] = SingleGateEnv(mps_targ_orth.mps, mps_test_orth.mps,
                    ansatz[ii][end], ansatz_inds[ii][end], slowdown)

                mps_test_orth.mps, prev_inds_test = ApplyGateLayers(
                    mps_test_orth.mps, [ansatz[ii][end]], [ansatz_inds[ii][end]],
                    max_chi, max_trunc_err, prev_inds_test)
                set_orth_centre!(mps_test_orth, prev_inds_test[2])
            end
        end

        ensure_canonical!(mps_test_orth, 1)
        mps_targ_orth.mps = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))
        invalidate_orth_centre!(mps_targ_orth)

        cost = 1 - abs(inner(mps_targ_orth.mps, mps_test_orth.mps))^2

        if iii % 5 == 0
            mps_test_full = Ansatz2MPS(N, ansatz, ansatz_inds, 2 * max_chi, 1e-16)
            full_cost = 1 - abs(inner(mps_targ_init, mps_test_full))^2

            if print_info
                println("$iii  $(round(cost, sigdigits=8)) $(round(full_cost, sigdigits=8))")
                flush(stdout)
            end

            if abs(prev_cost - full_cost) / max(abs(full_cost), 1e-10) < converge_threshold || full_cost < 1e-8
                return ansatz, ansatz_inds
            end
            prev_cost = full_cost
        end

        _mygc()
    end

    return ansatz, ansatz_inds
end

# ============================================================================
# Long-range optimize (via ApplyGatesLR)
# ============================================================================

"""
    optimize_LR(HP, mps_targ_init, ansatz, ansatz_inds; kwargs...)
        -> (Vector{Matrix{ComplexF64}}, Vector{Tuple{Int,Int}}, Float64)

Optimize a flat gate ansatz using long-range gate application (no SWAP decomposition).
"""
function optimize_LR(HP::HyperParams, mps_targ_init::FiniteMPS{ComplexF64},
                     ansatz::Vector{Matrix{ComplexF64}}, ansatz_inds::Vector{Tuple{Int,Int}};
                     mps_start::FiniteMPS{ComplexF64}=zeroMPS(HP.N),
                     print_info::Bool=true)

    N = HP.N
    max_chi = HP.max_chi
    max_trunc_err = HP.max_trunc_err
    slowdown = HP.slowdown
    n_sweeps = HP.n_sweeps
    print_every = HP.print_every
    converge_threshold = HP.converge_threshold

    mps_targ = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))

    mps_test, orth_centre_test = ApplyGatesLR(FiniteMPS{ComplexF64}(deepcopy(mps_start.tensors)),
                                               ansatz, ansatz_inds, max_chi, max_trunc_err)
    mps_targ = canonicalise(mps_targ, 1)

    cost = 1 - abs(inner(mps_targ_init, mps_test))^2
    full_cost = cost

    if print_info && print_every != 0
        println("\nInit Cost: $cost\n")
        println("N gates: $(length(ansatz))\n")
        flush(stdout)
    end

    prev_cost = 1.0

    mps_test_orth = OrthMPS(mps_test, orth_centre_test)
    mps_targ_orth = OrthMPS(mps_targ, 1)

    gate_env_cache = GateEnvCache(N)

    for iii in 1:n_sweeps
        # Backward sweep through gates
        for ii in length(ansatz):-1:2
            mps_test_orth.mps, orth_centre_test = ApplyGatesLR(
                mps_test_orth.mps, [inv(ansatz[ii])], [ansatz_inds[ii]],
                max_chi, max_trunc_err, mps_test_orth.orth_centre)
            set_orth_centre!(mps_test_orth, orth_centre_test)

            update_cache!(gate_env_cache, mps_targ_orth.mps, mps_test_orth.mps)
            ansatz[ii] = SingleGateEnv_cached(gate_env_cache, ansatz[ii], ansatz_inds[ii], slowdown)

            mps_targ_orth.mps, orth_centre_targ = ApplyGatesLR(
                mps_targ_orth.mps, [inv(ansatz[ii])], [ansatz_inds[ii]],
                max_chi, max_trunc_err, mps_targ_orth.orth_centre)
            set_orth_centre!(mps_targ_orth, orth_centre_targ)
        end

        # Apply inverse of gate 1 to target
        mps_targ_orth.mps, orth_centre_targ = ApplyGatesLR(
            mps_targ_orth.mps, [inv(ansatz[1])], [ansatz_inds[1]],
            max_chi, max_trunc_err, mps_targ_orth.orth_centre)
        set_orth_centre!(mps_targ_orth, orth_centre_targ)

        # Reset test MPS to mps_start
        mps_test_orth.mps = FiniteMPS{ComplexF64}(deepcopy(mps_start.tensors))
        ensure_canonical!(mps_test_orth, 1)
        set_orth_centre!(mps_test_orth, 1)

        # Forward sweep through gates
        for ii in 1:length(ansatz)
            mps_targ_orth.mps, orth_centre_targ = ApplyGatesLR(
                mps_targ_orth.mps, [ansatz[ii]], [ansatz_inds[ii]],
                max_chi, max_trunc_err, mps_targ_orth.orth_centre)
            set_orth_centre!(mps_targ_orth, orth_centre_targ)

            update_cache!(gate_env_cache, mps_targ_orth.mps, mps_test_orth.mps)
            ansatz[ii] = SingleGateEnv_cached(gate_env_cache, ansatz[ii], ansatz_inds[ii], slowdown)

            mps_test_orth.mps, orth_centre_test = ApplyGatesLR(
                mps_test_orth.mps, [ansatz[ii]], [ansatz_inds[ii]],
                max_chi, max_trunc_err, mps_test_orth.orth_centre)
            set_orth_centre!(mps_test_orth, orth_centre_test)
        end

        mps_targ_orth.mps = FiniteMPS{ComplexF64}(deepcopy(mps_targ_init.tensors))
        ensure_canonical!(mps_targ_orth, 1)

        cost = 1 - abs(inner(mps_targ_orth.mps, mps_test_orth.mps))^2

        if iii % max(1, print_every) == 0
            mps_test_full, _ = ApplyGatesLR(FiniteMPS{ComplexF64}(deepcopy(mps_start.tensors)),
                                             ansatz, ansatz_inds, 2 * max_chi, 1e-10)
            full_cost = 1 - abs(inner(mps_targ_init, mps_test_full))^2

            if print_info && print_every != 0
                println("$iii  $(round(cost, sigdigits=8)) $(round(full_cost, sigdigits=8))")
                flush(stdout)
            end

            if abs(prev_cost - full_cost) / max(abs(full_cost), 1e-10) < converge_threshold || full_cost < 1e-6
                return ansatz, ansatz_inds, full_cost
            end
            prev_cost = full_cost
        end

        _mygc()
    end

    return ansatz, ansatz_inds, full_cost
end
