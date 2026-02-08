@testset "Cross-Backend Integration" begin
    using TenSynth.Core
    using TenSynth.MPS
    using TenSynth.Hamiltonians
    using LinearAlgebra

    @testset "Unified cost dispatch" begin
        # MPS cost
        mps1 = randMPS(4, 2)
        mps2 = randMPS(4, 2)
        @test 0 <= cost(mps1, mps2) <= 1.0 + 1e-10
        @test cost(mps1, mps1) < 1e-8
    end

    @testset "MPS + Hamiltonians" begin
        # Verify Hamiltonian types are accessible from integration context
        H = TFIMHamiltonian(2; J=1.0, h=1.0)
        @test H isa AbstractHamiltonian

        # Trotter circuit gate unitarity
        circuit = trotterize(H, 0.1; order=:second, n_steps=5)
        @test circuit isa ParameterizedCircuit
        for gate in circuit.gates
            @test is_unitary(to_matrix(gate))
        end
    end

    @testset "LayeredCircuit structure" begin
        circuit = tfim_trotter_circuit(4, 0.1, 1.0; order=:second, n_steps=1)
        @test circuit isa LayeredCircuit
        @test circuit.n_qubits == 4

        # Verify layer indices are valid
        for layer in circuit.layers
            for (i, j) in layer.indices
                @test 1 <= i < j <= circuit.n_qubits
            end
        end
    end

    @testset "Extension hooks" begin
        # PyCall extension stubs should exist
        @test isdefined(TenSynth.MPS, :ApproxRZ)
        @test isdefined(TenSynth.MPS, :ApproxSU4)
        @test isdefined(TenSynth.MPS, :ApproxSU2_trasyn)
        @test isdefined(TenSynth.MPS, :ApproxSU4_trasyn)

        # Without PyCall, synthesis should fall back to estimation
        seq, n_t, err = synthesize_rz(0.5, 1e-3)
        @test occursin("estimated", seq)
        @test n_t > 0
    end

    @testset "MPS decompose + compile" begin
        mps = randMPS(4, 2)

        result = decompose(mps; method=:analytical)
        @test result.fidelity > 0.5

        comp = compile(mps; method=:analytical)
        @test comp.fidelity > 0.5
        @test !isempty(to_openqasm(comp))
    end
end
