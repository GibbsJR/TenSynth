@testset "MPS Decomposition" begin
    using TenSynth.MPS
    using TenSynth.Core

    @testset "Analytical decomposition" begin
        mps = randMPS(4, 2)
        result = decompose(mps; method=:analytical)
        @test result isa DecompositionResult
        @test result.fidelity > 0.5
        @test result.depth > 0
        @test result.method == :analytical
    end

    @testset "Decomposition utilities" begin
        mps = randMPS(4, 2)
        result = decompose(mps; method=:analytical)

        # apply_decomposition (keyword-only interface)
        mps_recon = apply_decomposition(result)
        @test mps_recon isa FiniteMPS{ComplexF64}
        @test fidelity(mps, mps_recon) > 0.5

        # circuit_to_flat
        flat_circuit, flat_inds = circuit_to_flat(result)
        @test length(flat_circuit) == length(flat_inds)

        # verify_decomposition
        f = verify_decomposition(mps, result)
        @test f ≈ result.fidelity atol=0.01
    end

    @testset "Compile pipeline" begin
        mps = randMPS(4, 2)
        result = compile(mps; method=:analytical)
        @test result isa CompilationResult
        @test result.fidelity > 0.5
    end

    @testset "Export functions" begin
        mps = randMPS(4, 2)
        result = compile(mps; method=:analytical)

        qasm = to_openqasm(result)
        @test occursin("OPENQASM", qasm)
        @test occursin("qreg", qasm)

        qiskit = to_qiskit(result)
        @test occursin("QuantumCircuit", qiskit)

        stats = circuit_stats(result)
        @test haskey(stats, :n_qubits)
        @test haskey(stats, :circuit_depth)
        @test haskey(stats, :n_2q_gates)
    end

    @testset "Synthesis estimation" begin
        n_t = estimate_t_gates(15, 1e-3)
        @test n_t > 0

        result = SynthesisResult(Vector{String}[], 100, 50, 0.01, 15, :estimated)
        summary = summarize_synthesis(result)
        @test occursin("T-gates", summary)
    end
end
