@testset "Core Linalg" begin
    # robust_svd on rectangular matrix
    M = randn(ComplexF64, 8, 4)
    F = TenSynth.Core.robust_svd(M)
    @test M ≈ F.U * Diagonal(F.S) * F.Vt

    # robust_svd on square matrix
    M2 = randn(ComplexF64, 4, 4)
    F2 = TenSynth.Core.robust_svd(M2)
    @test M2 ≈ F2.U * Diagonal(F2.S) * F2.Vt

    # robust_svd NaN rejection
    M_nan = ComplexF64[NaN 0; 0 1]
    @test_throws ErrorException TenSynth.Core.robust_svd(M_nan)

    # polar_unitary on square matrix → unitary
    M3 = randn(ComplexF64, 4, 4)
    U = TenSynth.Core.polar_unitary(M3)
    @test is_unitary(U)

    # randU generates unitaries
    for _ in 1:5
        U1 = TenSynth.Core.randU(1.0)
        @test size(U1) == (4, 4)
        @test is_unitary(U1)

        U2 = TenSynth.Core.randU(0.1, 3)
        @test size(U2) == (8, 8)
        @test is_unitary(U2)
    end

    # randU_sym generates unitaries
    for _ in 1:5
        U3 = TenSynth.Core.randU_sym(1.0)
        @test size(U3) == (4, 4)
        @test is_unitary(U3)
    end
end
