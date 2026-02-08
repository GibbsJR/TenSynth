@testset "Flagship: iMPS Unitary Compilation Depth Scaling" begin
    using Random
    using LinearAlgebra
    using TenSynth.iMPS
    using TenSynth.Core

    Random.seed!(456)

    # --- Parameters ---
    # Use t=1.0 so the target unitary is complex enough to require deeper circuits,
    # but not so complex that L-BFGS takes excessively long to converge.
    # Only test 2 and 4 layers (8 layers is too slow for CI).
    unit_cell = 2
    t_evol = 1.0
    J = 1.0
    h_field = 1.0
    layer_counts = [2, 4]

    @info "iMPS Flagship: TFIM unitary compilation (unit_cell=$unit_cell, t=$t_evol, J=$J, h=$h_field)"

    # --- Compile at increasing depths ---
    test_fidelities = Float64[]

    for n_layers in layer_counts
        @info "  Compiling with $n_layers layers..."
        Random.seed!(300 + n_layers)

        config = UnitaryCompilationConfig(;
            n_train=4,
            n_test=4,
            max_chi=16,
            max_iter=150,
            verbose=false
        )

        result = compile_tfim_evolution(unit_cell, t_evol, J, h_field, n_layers;
                                         config=config)

        push!(test_fidelities, result.test_fidelity)
        @info "    n_layers=$n_layers: train_fidelity=$(round(result.train_fidelity, digits=6)), test_fidelity=$(round(result.test_fidelity, digits=6)), converged=$(result.converged)"

        # Basic sanity checks
        @test result.test_fidelity >= 0.0
        @test result.test_fidelity <= 1.0 + 1e-8
        @test result.train_fidelity >= 0.0
        @test result.train_fidelity <= 1.0 + 1e-8
        @test !isempty(result.train_history)
        @test !isempty(result.test_history)
    end

    # --- Assertions ---
    @info "  Test fidelities: $test_fidelities"

    # Deeper circuit should improve fidelity
    @test test_fidelities[2] >= test_fidelities[1] - 1e-3

    # Both should achieve reasonable test fidelity
    @test test_fidelities[1] > 0.8
    @test test_fidelities[2] > 0.95
end
