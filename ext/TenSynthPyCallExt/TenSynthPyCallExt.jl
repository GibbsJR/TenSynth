module TenSynthPyCallExt

# Package extension: PyCall-backed T-gate synthesis (gridsynth + trasyn)
# Loaded automatically when `using PyCall` alongside `using TenSynth`.
# Adds concrete methods to the stub functions in TenSynth.MPS.

using TenSynth
using TenSynth.Core: RZ, H_GATE, S_GATE, T_GATE, PAULI_I, CNOT
using TenSynth.MPS: SU42Thetas, RZs2SU4, Thetas2SU2, SU22Thetas,
                     Rz2Theta, NumTGates, GSChars2U
using PyCall

include("gridsynth.jl")
include("trasyn.jl")

function __init__()
    # Extension init — Python modules are lazily initialized on first call
end

end # module TenSynthPyCallExt
