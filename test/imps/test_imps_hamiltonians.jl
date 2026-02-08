@testset "iMPS Hamiltonians" begin
    using TenSynth.iMPS

    @testset "TFIM Hamiltonian" begin
        H = TenSynth.iMPS.TFIMHamiltonian(2; J=1.0, h=0.5)
        @test H isa TenSynth.iMPS.SpinLatticeHamiltonian
        @test H.unit_cell == 2
        @test length(H.terms) > 0

        two_site = TenSynth.iMPS.get_two_site_terms(H)
        @test length(two_site) > 0
    end

    @testset "XXZ Hamiltonian" begin
        H = TenSynth.iMPS.XXZHamiltonian(2; Jxy=1.0, Jz=0.5)
        @test H.unit_cell == 2
        @test length(H.terms) > 0
    end

    @testset "Heisenberg Hamiltonian" begin
        H = TenSynth.iMPS.HeisenbergHamiltonian(2; J=1.0)
        @test H.unit_cell == 2
        @test length(H.terms) > 0
    end

    @testset "XY Hamiltonian" begin
        H = TenSynth.iMPS.XYHamiltonian(2; J=1.0)
        @test H.unit_cell == 2
    end

    @testset "Local Hamiltonian extraction" begin
        H = TenSynth.iMPS.TFIMHamiltonian(2; J=1.0, h=1.0)
        h_local = TenSynth.iMPS.get_local_hamiltonian(H, (1, 2))
        @test size(h_local) == (4, 4)
        # Should be Hermitian
        @test h_local ≈ h_local' atol=1e-10
    end

    @testset "Trotterization" begin
        H = TenSynth.iMPS.TFIMHamiltonian(2; J=1.0, h=1.0)
        circuit = TenSynth.iMPS.trotterize(H, 0.1; order=TenSynth.iMPS.SECOND_ORDER, n_steps=5)
        @test circuit isa TenSynth.Core.ParameterizedCircuit
        @test TenSynth.iMPS.n_gates(circuit) > 0

        # All gates should produce unitary matrices
        for gate in circuit.gates
            U = TenSynth.Core.to_matrix(gate)
            @test TenSynth.Core.is_unitary(U)
        end
    end

    @testset "Specialized Trotterization" begin
        circuit = TenSynth.iMPS.trotterize_tfim(2, 0.1, 1.0, 1.0; n_steps=5)
        @test circuit isa TenSynth.Core.ParameterizedCircuit
        @test TenSynth.iMPS.n_gates(circuit) > 0
    end
end
