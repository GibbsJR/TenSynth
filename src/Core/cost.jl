# Unified cost() interface
# Method stubs — concrete implementations added by backend modules (MPS, MPO, iMPS).

"""
    cost(target, test; kwargs...)

Compute the cost (error metric) between a target and test tensor network.
Dispatches based on type:
- `cost(::FiniteMPS, ::FiniteMPS)` — state infidelity (Phase E)
- `cost(::FiniteMPO, ::FiniteMPO)` — HST cost (Phase C)
- `cost(::iMPS, ::iMPS)` — local infidelity (Phase D)
"""
function cost end

function cost(target::FiniteMPS, test::FiniteMPS; kwargs...)
    # Delegates to MPS.fidelity — requires TenSynth.MPS to be loaded.
    mod = parentmodule(@__MODULE__)  # TenSynth
    if isdefined(mod, :MPS) && isdefined(mod.MPS, :fidelity)
        f = mod.MPS.fidelity(target, test)
        return 1.0 - f
    else
        error("cost(::FiniteMPS, ::FiniteMPS) requires TenSynth.MPS module to be loaded")
    end
end

function cost(target::FiniteMPO, test::FiniteMPO; kwargs...)
    # Delegates to MPO.hst_cost — requires TenSynth.MPO to be loaded.
    mod = parentmodule(@__MODULE__)  # TenSynth
    if isdefined(mod, :MPO) && isdefined(mod.MPO, :hst_cost)
        return mod.MPO.hst_cost(target, test)
    else
        error("cost(::FiniteMPO, ::FiniteMPO) requires TenSynth.MPO module to be loaded")
    end
end

function cost(target::iMPS, test::iMPS; kwargs...)
    # Delegates to iMPS.infidelity — requires TenSynth.iMPS to be loaded.
    mod = parentmodule(@__MODULE__)  # TenSynth
    if isdefined(mod, :iMPS) && isdefined(mod.iMPS, :infidelity)
        return mod.iMPS.infidelity(target, test)
    else
        error("cost(::iMPS, ::iMPS) requires TenSynth.iMPS module to be loaded")
    end
end
