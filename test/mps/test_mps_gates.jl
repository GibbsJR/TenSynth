@testset "MPS Gates" begin
    using TenSynth.MPS
    using TenSynth.Core

    @testset "Single-qubit application" begin
        mps = zeroMPS(4)
        mps_out = apply1Q(mps, 1, PAULI_X)
        mps_target = productMPS("1000")
        @test abs(inner(mps_out, mps_target)) ≈ 1.0 atol=1e-10
    end

    @testset "KAK round-trip" begin
        for _ in 1:5
            U = randU(1.0, 2)
            θ = SU42Thetas(U)
            U_recon = Thetas2SU4(θ)
            phase = tr(U_recon' * U) / 4
            @test abs(abs(phase) - 1.0) < 1e-8
        end
    end

    @testset "SU2 round-trip" begin
        for _ in 1:5
            U = randU(1.0, 1)
            θ = SU22Thetas(U)
            U_recon = Thetas2SU2(θ)
            phase = tr(U_recon' * U) / 2
            @test abs(abs(phase) - 1.0) < 1e-8
        end
    end

    @testset "Gate layers" begin
        N = 4
        n_layers = 2
        inds = generate_circuit_indices(N, STAIRCASE, n_layers)
        @test length(inds) == n_layers

        inds_bw = generate_circuit_indices(N, BRICKWORK, n_layers)
        @test length(inds_bw) == n_layers
    end
end
