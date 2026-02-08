using Test
using TenSynth
using TenSynth.Core
using LinearAlgebra

@testset "TenSynth" begin
    @testset "Package loads" begin
        @test isdefined(TenSynth, :Core)
        @test isdefined(TenSynth, :MPS)
        @test isdefined(TenSynth, :MPO)
        @test isdefined(TenSynth, :iMPS)
        @test isdefined(TenSynth, :Hamiltonians)
    end

    # Phase B: Core tests
    include("core/test_types.jl")
    include("core/test_constants.jl")
    include("core/test_linalg.jl")
    include("core/test_parameterizations.jl")

    # Phase C: MPO tests
    include("mpo/test_mpo_core.jl")
    include("mpo/test_mpo_inner.jl")
    include("mpo/test_mpo_gates.jl")
    include("mpo/test_mpo_optimization.jl")

    # Phase D: iMPS tests
    include("imps/test_imps_states.jl")
    include("imps/test_imps_gates.jl")
    include("imps/test_imps_circuits.jl")
    include("imps/test_imps_hamiltonians.jl")
    include("imps/test_imps_cost.jl")

    # Phase E: MPS tests
    include("mps/test_mps_basic.jl")
    include("mps/test_mps_gates.jl")
    include("mps/test_mps_decomposition.jl")

    # Phase F: Hamiltonians tests
    include("hamiltonians/test_hamiltonians.jl")

    # Phase H: Integration tests
    include("integration/test_cross_backend.jl")

    # Flagship integration tests (computationally expensive)
    if get(ENV, "TENSYNTH_FULL_TESTS", "false") == "true"
        include("integration/test_flagship_mps.jl")
        include("integration/test_flagship_mpo.jl")
        include("integration/test_flagship_imps.jl")
    end
end
