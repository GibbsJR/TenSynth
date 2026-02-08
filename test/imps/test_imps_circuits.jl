@testset "iMPS Circuits" begin
    using TenSynth.iMPS
    using TenSynth.Core: BondConfig, ParameterizedGate, ParameterizedCircuit
    using TenSynth.Core: PauliGeneratorParameterization, DressedZZParameterization
    using TenSynth.Core: is_unitary, to_matrix

    @testset "Circuit construction" begin
        circuit = TenSynth.iMPS.nearest_neighbour_ansatz(2, 3, PauliGeneratorParameterization())
        @test circuit isa ParameterizedCircuit
        @test circuit.n_qubits == 2
        # 2 qubits, 3 layers: each layer has bond (1,2) + bond (2,1) = 2 gates/layer × 3 = 6
        @test TenSynth.iMPS.n_gates(circuit) == 6
        @test TenSynth.iMPS.n_params(circuit) == 6 * 15  # 15 params per SU4 gate
    end

    @testset "Get/set params" begin
        circuit = TenSynth.iMPS.nearest_neighbour_ansatz(2, 1, PauliGeneratorParameterization())
        params = TenSynth.iMPS.get_params(circuit)
        @test length(params) == TenSynth.iMPS.n_params(circuit)
        @test all(params .== 0.0)  # Default is zero

        new_params = randn(length(params))
        TenSynth.iMPS.set_params!(circuit, new_params)
        @test TenSynth.iMPS.get_params(circuit) ≈ new_params
    end

    @testset "Circuit application" begin
        psi = TenSynth.iMPS.random_product_state(2)
        circuit = TenSynth.iMPS.random_circuit(2, 4, PauliGeneratorParameterization(); scale=0.5)
        config = BondConfig(16, 1e-10)

        psi2 = deepcopy(psi)
        TenSynth.iMPS.apply_circuit!(psi2, circuit, config)

        TenSynth.iMPS.absorb_bonds!(psi)
        TenSynth.iMPS.absorb_bonds!(psi2)
        # Random circuit should change the state
        f = TenSynth.iMPS.local_fidelity(psi, psi2)
        @test f < 1.0
    end

    @testset "Circuit properties" begin
        circuit = TenSynth.iMPS.nearest_neighbour_ansatz(3, 2, DressedZZParameterization())
        @test TenSynth.iMPS.n_gates(circuit) > 0
        @test TenSynth.iMPS.two_qubit_gate_count(circuit) == TenSynth.iMPS.n_gates(circuit)
        @test TenSynth.iMPS.single_qubit_gate_count(circuit) == 0
        @test TenSynth.iMPS.depth(circuit) > 0
    end

    @testset "Gate modification" begin
        circuit = ParameterizedCircuit(3)
        gate = ParameterizedGate(PauliGeneratorParameterization(), (1, 2))
        TenSynth.iMPS.add_gate!(circuit, gate)
        @test TenSynth.iMPS.n_gates(circuit) == 1

        gate2 = ParameterizedGate(DressedZZParameterization(), (2, 3))
        TenSynth.iMPS.add_gate!(circuit, gate2)
        @test TenSynth.iMPS.n_gates(circuit) == 2

        removed = TenSynth.iMPS.remove_gate!(circuit, 1)
        @test TenSynth.iMPS.n_gates(circuit) == 1
    end

    @testset "Swap networks" begin
        route = TenSynth.iMPS.compute_swap_route((1, 3), 4)
        @test route isa TenSynth.iMPS.SwapRoute
        @test length(route.swaps_before) > 0

        # Adjacent sites need no swaps
        route_adj = TenSynth.iMPS.compute_swap_route((1, 2), 4)
        @test isempty(route_adj.swaps_before)
        @test isempty(route_adj.swaps_after)
    end
end
