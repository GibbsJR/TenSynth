# T-gate synthesis via trasyn (PyCall interface)
# Adapted from MPS2Circuit/src/synthesis/trasyn.jl
# Adds methods to TenSynth.MPS.ApproxSU2_trasyn and TenSynth.MPS.ApproxSU4_trasyn

using PyCall

# Lazy initialization of trasyn Python module
const trasyn_mod = PyNULL()

function __init_trasyn__()
    if ispynull(trasyn_mod)
        copy!(trasyn_mod, pyimport("trasyn"))
    end
end

"""
    ApproxSU2_trasyn(U_target::Matrix{ComplexF64}, max_depth::Int;
                     epsilon=1e-3, num_attempts=5, verbose=false)

Approximate an arbitrary SU(2) gate using Clifford+T gates via trasyn.

Returns `(seq, U_approx, err)`.
"""
function TenSynth.MPS.ApproxSU2_trasyn(U_target::Matrix{ComplexF64}, max_depth::Int;
                                        epsilon::Float64=1e-3, num_attempts::Int=5,
                                        verbose::Bool=false)
    __init_trasyn__()

    result = trasyn_mod.synthesize(U_target, max_depth,
                                   error_threshold=epsilon,
                                   num_attempts=num_attempts,
                                   verbose=verbose)

    seq = string(result[1])
    mat = convert(Matrix{ComplexF64}, result[2])
    err = convert(Float64, result[3])

    return seq, mat, err
end

"""
    ApproxSU2_trasyn(thetas::Vector{Float64}, max_depth::Int; ...)

Approximate an SU(2) gate given by its theta parameterization.
"""
function TenSynth.MPS.ApproxSU2_trasyn(thetas::Vector{Float64}, max_depth::Int;
                                        epsilon::Float64=1e-3, num_attempts::Int=5,
                                        verbose::Bool=false)
    U_target = Thetas2SU2(thetas)
    return TenSynth.MPS.ApproxSU2_trasyn(U_target, max_depth;
                                          epsilon=epsilon, num_attempts=num_attempts,
                                          verbose=verbose)
end

"""
    ApproxSU2_trasyn(theta::Float64, max_depth::Int; ...)

Approximate an RZ(theta) rotation using trasyn.
"""
function TenSynth.MPS.ApproxSU2_trasyn(theta::Float64, max_depth::Int;
                                        epsilon::Float64=1e-3, num_attempts::Int=5,
                                        verbose::Bool=false)
    __init_trasyn__()

    result = trasyn_mod.synthesize(theta, max_depth,
                                   error_threshold=epsilon,
                                   num_attempts=num_attempts,
                                   verbose=verbose)

    seq = string(result[1])
    mat = convert(Matrix{ComplexF64}, result[2])
    err = convert(Float64, result[3])

    return seq, mat, err
end

"""
    ApproxSU4_trasyn(U_targ::Matrix{ComplexF64}, max_depth::Int;
                     epsilon=1e-3, num_attempts=2)

Approximate an arbitrary SU(4) unitary using trasyn for single-qubit gates.

Returns `(U_approx, N_Ts, total_err)`.
"""
function TenSynth.MPS.ApproxSU4_trasyn(U_targ::Matrix{ComplexF64}, max_depth::Int;
                                        epsilon::Float64=1e-3, num_attempts::Int=2)
    thetas = SU42Thetas(U_targ)

    N_Ts = Int[]
    RZs_approx = Matrix{ComplexF64}[]
    total_err = 0.0

    for theta in thetas
        seq, U, err = TenSynth.MPS.ApproxSU2_trasyn(theta, max_depth;
                                                      epsilon=epsilon,
                                                      num_attempts=num_attempts)
        push!(RZs_approx, U)
        push!(N_Ts, NumTGates(seq))
        total_err += err
    end

    U_approx = RZs2SU4(RZs_approx)
    return U_approx, N_Ts, total_err
end
