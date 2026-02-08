using TenSynth.MPO
using TenSynth.Core

@testset "MPO Optimization" begin
    @testset "brick_wall_circuit" begin
        circuit = brick_wall_circuit(6, 2; random=true)
        @test circuit.n_qubits == 6
        @test circuit_depth(circuit) == 2
        @test circuit_gate_count(circuit) > 0
    end

    @testset "circuit_to_mpo" begin
        circuit = brick_wall_circuit(4, 2; random=true)
        mpo = circuit_to_mpo(circuit)
        @test n_sites(mpo) == 4
        # Check unitarity via hst_cost (inner_mpo loses norm after multiple SVDs)
        mpo_copy = FiniteMPO{ComplexF64}([copy(t) for t in mpo.tensors])
        @test hst_cost(mpo, mpo_copy) < 1e-6
    end

    @testset "identity circuit" begin
        circuit = brick_wall_circuit(4, 2; random=false)
        mpo = circuit_to_mpo(circuit)
        @test hst_cost(mpo, identity_mpo(4)) < 1e-8
    end

    @testset "ansatz_to_mpo" begin
        circuit = brick_wall_circuit(4, 2; random=true)
        mpo = ansatz_to_mpo(4, circuit.layers)
        mpo2 = circuit_to_mpo(circuit)
        @test hst_cost(mpo, mpo2) < 1e-10
    end

    @testset "layer_sweep! improves cost" begin
        n = 4
        target_circuit = brick_wall_circuit(n, 1; random=true)
        target_mpo = circuit_to_mpo(target_circuit)

        # Random test layer
        indices = brick_wall_indices(n, 0)
        test_layer = random_layer(indices)
        mpo_test = identity_mpo(n)

        # Compute initial cost
        test_mpo_full = layer_to_mpo(test_layer, n)
        initial_cost = hst_cost(target_mpo, test_mpo_full)

        # Optimize layer
        optimized_layer = layer_sweep!(target_mpo, mpo_test, test_layer; n_sweeps=5)

        # Compute optimized cost
        opt_mpo = layer_to_mpo(optimized_layer, n)
        final_cost = hst_cost(target_mpo, opt_mpo)

        @test final_cost <= initial_cost + 1e-6
    end

    @testset "optimize_simple! convergence" begin
        n = 4
        target_circuit = brick_wall_circuit(n, 2; random=true)
        target_mpo = circuit_to_mpo(target_circuit)

        test_circuit = brick_wall_circuit(n, 2; random=true)
        config = OptimizerConfig(n_sweeps=10, verbose=false)
        result = optimize_simple!(target_mpo, test_circuit, config)

        @test result.final_cost <= result.initial_cost + 1e-6
        @test length(result.cost_history) == 11  # initial + 10 sweeps
    end

    @testset "exact recompilation" begin
        n = 4
        # Same structure circuit can be exactly recompiled
        target_circuit = brick_wall_circuit(n, 2; random=true)
        target_mpo = circuit_to_mpo(target_circuit)

        test_circuit = brick_wall_circuit(n, 2; random=true)
        config = OptimizerConfig(n_sweeps=30, verbose=false)
        result = optimize!(target_mpo, test_circuit, config)

        @test result.final_cost < 1e-4
    end

    @testset "unified cost interface" begin
        mpo1 = identity_mpo(4)
        mpo2 = identity_mpo(4)
        @test cost(mpo1, mpo2) < 1e-10

        gate = GateMatrix(CNOT, "CNOT", Dict{Symbol,Any}())
        apply_gate!(mpo2, gate, 1, 2)
        @test cost(mpo1, mpo2) > 0.0
    end
end
