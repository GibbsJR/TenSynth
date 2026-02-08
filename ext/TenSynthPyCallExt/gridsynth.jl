# T-gate synthesis via gridsynth (PyCall interface)
# Adapted from MPS2Circuit/src/synthesis/gridsynth.jl
# Adds methods to TenSynth.MPS.ApproxRZ and TenSynth.MPS.ApproxSU4

using PyCall

# Lazy initialization of Python modules
const mpmath = PyNULL()
const gridsynth_mod = PyNULL()

function __init_gridsynth__()
    if ispynull(mpmath)
        copy!(mpmath, pyimport("mpmath"))
        mpmath.mp.dps = 128
    end
    if ispynull(gridsynth_mod)
        copy!(gridsynth_mod, pyimport("pygridsynth.gridsynth"))
    end
end

"""
    ApproxRZ(theta::Float64, epsilon::Float64)

Approximate an RZ(θ) rotation using Clifford+T gates via the gridsynth algorithm.

Returns `(gate_string, U_approx)`.
"""
function TenSynth.MPS.ApproxRZ(theta::Float64, epsilon::Float64)
    __init_gridsynth__()

    theta_m = mpmath.mpmathify(string(theta))
    epsilon_m = mpmath.mpmathify(string(epsilon))
    gates = gridsynth_mod.gridsynth_gates(theta=theta_m, epsilon=epsilon_m)

    gate_str = string(gates)
    return gate_str, GSChars2U(gate_str)
end

"""
    ApproxRZ(U::Matrix{ComplexF64}, epsilon::Float64)

Approximate an RZ rotation (given as matrix) using Clifford+T gates.
"""
function TenSynth.MPS.ApproxRZ(U::Matrix{ComplexF64}, epsilon::Float64)
    theta = Rz2Theta(U)
    return TenSynth.MPS.ApproxRZ(theta, epsilon)
end

"""
    ApproxSU4(U_targ::Matrix{ComplexF64}, epsilon::Float64)

Approximate an arbitrary SU(4) (2-qubit) unitary using Clifford+T gates via gridsynth.

Decomposes via 15-parameter KAK, then synthesizes each RZ individually.
Returns `(U_approx, N_Ts)`.
"""
function TenSynth.MPS.ApproxSU4(U_targ::Matrix{ComplexF64}, epsilon::Float64)
    thetas = SU42Thetas(U_targ)

    N_Ts = Int[]
    RZs_approx = Matrix{ComplexF64}[]

    for theta in thetas
        gs, U = TenSynth.MPS.ApproxRZ(theta, epsilon)
        push!(RZs_approx, U)
        push!(N_Ts, NumTGates(gs))
    end

    U_approx = RZs2SU4(RZs_approx)
    return U_approx, N_Ts
end
