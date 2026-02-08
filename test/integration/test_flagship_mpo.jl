@testset "Flagship: MPO TFIM Time Evolution Depth Scaling" begin
    using Random
    using LinearAlgebra
    using TenSynth.MPO
    using TenSynth.Hamiltonians
    using TenSynth.Core

    Random.seed!(123)

    # --- Parameters ---
    # Use a small system with short evolution time so the target MPO has modest
    # bond dimension and the brick-wall optimizer can converge reliably.
    n_qubits = 4
    h_field = 1.0
    t_total = 0.3
    depths = [2, 4, 8]

    # --- Step 1: Build high-accuracy target MPO ---
    @info "MPO Flagship: Building TFIM target (N=$n_qubits, h=$h_field, t=$t_total)"

    target_circuit = tfim_trotter_circuit(n_qubits, t_total, h_field;
                                           order=:fourth, n_steps=20)
    @test target_circuit isa LayeredCircuit
    @test target_circuit.n_qubits == n_qubits
    @info "  Target circuit: $(length(target_circuit.layers)) layers"

    target_mpo = circuit_to_mpo(target_circuit; max_chi=128)
    @test n_sites(target_mpo) == n_qubits
    @info "  Target MPO bond dims: $(bond_dimensions(target_mpo))"

    # Sanity: target should not be identity
    id_mpo = identity_mpo(n_qubits)
    identity_cost = hst_cost(target_mpo, id_mpo)
    @test identity_cost > 0.01
    @info "  Cost vs identity: $(round(identity_cost, digits=6))"

    # --- Step 2: Optimize at increasing depths ---
    errors = Float64[]

    for d in depths
        @info "  Optimizing depth $d ansatz..."
        Random.seed!(200 + d)

        ansatz = brick_wall_circuit(n_qubits, d; random=true)
        @test circuit_depth(ansatz) == d

        config = OptimizerConfig(;
            max_chi=64,
            n_sweeps=50,
            verbose=false
        )

        result = optimize!(target_mpo, ansatz, config)

        @test result isa OptimizationResult
        push!(errors, result.final_cost)
        @info "    depth=$d: initial_cost=$(round(result.initial_cost, digits=6)), final_cost=$(round(result.final_cost, digits=6))"
    end

    # --- Step 3: Assertions ---
    @info "  Final errors: $errors"

    # Each depth should achieve significant optimization (well below random ~1.0)
    for (i, d) in enumerate(depths)
        @test errors[i] < 0.05  # All depths should converge to low cost
    end

    # The deepest circuit should achieve the best or near-best result
    @test errors[end] <= minimum(errors) + 0.01
end
