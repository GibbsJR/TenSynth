@testset "Hamiltonians" begin
    using TenSynth.Hamiltonians
    using TenSynth.Core
    using LinearAlgebra

    @testset "Hamiltonian types" begin
        H = TFIMHamiltonian(2; J=1.0, h=0.5)
        @test H isa AbstractHamiltonian
        @test H isa SpinLatticeHamiltonian
        @test H.unit_cell == 2
        @test length(H.terms) > 0

        H2 = XXZHamiltonian(2; Jxy=1.0, Jz=1.0)
        @test H2 isa AbstractHamiltonian

        H3 = HeisenbergHamiltonian(2; J=1.0)
        @test H3 isa AbstractHamiltonian

        H4 = XYHamiltonian(2; J=1.0)
        @test H4 isa AbstractHamiltonian
    end

    @testset "LocalTerm" begin
        term = LocalTerm(ComplexF64.(PAULI_X), 1, 0.5)
        @test is_single_site(term)
        @test !is_two_site(term)
        @test get_matrix(term) ≈ 0.5 * PAULI_X

        term2 = LocalTerm(ComplexF64.(ZZ), (1, 2))
        @test is_two_site(term2)
        @test term2.coefficient == 1.0
    end

    @testset "Hamiltonian queries" begin
        H = TFIMHamiltonian(4; J=1.0, h=1.0)
        @test length(get_two_site_terms(H)) == 4
        @test length(get_interaction_sites(H)) == 4

        H_local = get_local_hamiltonian(H, (1, 2))
        @test size(H_local) == (4, 4)
        @test norm(H_local) > 0
    end

    @testset "Trotter ParameterizedCircuit" begin
        H = TFIMHamiltonian(2; J=1.0, h=1.0)

        for order in [:first, :second, :fourth]
            circuit = trotterize(H, 0.1; order=order, n_steps=3)
            @test circuit isa ParameterizedCircuit
            @test circuit.n_qubits == 2
            for gate in circuit.gates
                @test is_unitary(to_matrix(gate))
            end
        end
    end

    @testset "Trotter LayeredCircuit" begin
        circuit = tfim_trotter_circuit(6, 0.1, 1.0; order=:second, n_steps=2)
        @test circuit isa LayeredCircuit
        @test circuit.n_qubits == 6

        for layer in circuit.layers
            for gate in layer.gates
                @test is_unitary(gate.matrix)
            end
        end
    end

    @testset "Trotter error scaling" begin
        H = TFIMHamiltonian(2; J=1.0, h=1.0)

        e1 = trotter_error_bound(H, 0.1; order=:first, n_steps=10)
        e2 = trotter_error_bound(H, 0.1; order=:second, n_steps=10)
        e4 = trotter_error_bound(H, 0.1; order=:fourth, n_steps=10)
        @test e4 < e2 < e1
    end
end
