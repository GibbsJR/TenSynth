@testset "Core Types" begin
    # Abstract type hierarchy
    @test FiniteMPS{ComplexF64} <: AbstractMPS <: AbstractQuantumState
    @test FiniteMPO{ComplexF64} <: AbstractMPO <: AbstractQuantumState
    @test iMPS{ComplexF64} <: AbstractiMPS <: AbstractQuantumState
    @test GateMatrix <: AbstractTwoQubitGate <: AbstractGate
    @test ParameterizedGate{KAKParameterization} <: AbstractTwoQubitGate
    @test AbstractMPS <: AbstractQuantumState <: AbstractTensorNetwork
    @test AbstractLayeredCircuit <: AbstractCircuit <: AbstractTensorNetwork
    @test AbstractParameterizedCircuit <: AbstractCircuit

    # Config types
    @test BondConfig(64, 1e-10) isa BondConfig
    @test BondConfig(32) isa BondConfig
    @test BondConfig(32).max_trunc_err == 1e-10
    @test_throws ArgumentError BondConfig(0, 1e-10)
    @test_throws ArgumentError BondConfig(-1)

    # FiniteMPS construction
    tensors_mps = [randn(ComplexF64, 1, 2, 4), randn(ComplexF64, 4, 2, 4), randn(ComplexF64, 4, 2, 1)]
    mps = FiniteMPS{ComplexF64}(tensors_mps)
    @test length(mps.tensors) == 3
    @test size(mps.tensors[1]) == (1, 2, 4)

    # FiniteMPO construction
    tensors_mpo = [randn(ComplexF64, 1, 2, 2, 4), randn(ComplexF64, 4, 2, 2, 1)]
    mpo = FiniteMPO{ComplexF64}(tensors_mpo)
    @test length(mpo.tensors) == 2
    @test size(mpo.tensors[1]) == (1, 2, 2, 4)

    # iMPS construction
    gamma = [reshape(ComplexF64[1; 0], 1, 2, 1), reshape(ComplexF64[1; 0], 1, 2, 1)]
    lambda = [reshape(ComplexF64[1], 1, 1), reshape(ComplexF64[1], 1, 1)]
    psi = iMPS{ComplexF64}(gamma, lambda, 2, 2, true, nothing)
    @test psi.unit_cell == 2
    @test psi.physical_dim == 2
    @test psi.normalized == true
    @test_throws ArgumentError iMPS{ComplexF64}(gamma, lambda, 3, 2, true, nothing)

    # GateMatrix construction
    g = GateMatrix(ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0], "CNOT", Dict{Symbol,Any}())
    @test g.name == "CNOT"

    # ParameterizedGate construction
    pg = ParameterizedGate(KAKParameterization(), (1, 2), zeros(Float64, 15))
    @test pg.parameterization isa KAKParameterization
    @test pg.qubits == (1, 2)
    @test length(pg.params) == 15

    # GateLayer construction
    gl = GateLayer(GateMatrix[], Tuple{Int,Int}[])
    @test isempty(gl.gates)

    # LayeredCircuit construction
    lc = LayeredCircuit(4, GateLayer[])
    @test lc.n_qubits == 4
end
