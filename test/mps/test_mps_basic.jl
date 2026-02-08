@testset "MPS Basic" begin
    using TenSynth.MPS
    using TenSynth.Core

    @testset "Construction" begin
        mps = zeroMPS(6)
        @test length(mps.tensors) == 6
        @test size(mps.tensors[1]) == (1, 2, 1)
        @test mps isa FiniteMPS{ComplexF64}

        mps_rand = randMPS(6, 4)
        @test all(size(t, 2) == 2 for t in mps_rand.tensors)
        @test max_bond_dim(mps_rand) <= 4

        mps_ghz = ghzMPS(4)
        @test length(mps_ghz.tensors) == 4
    end

    @testset "Canonicalization" begin
        mps = randMPS(6, 4)
        mps_canon = canonicalise(mps, 3)

        # Left-canonical tensors (sites 1,2)
        for i in 1:2
            A = mps_canon.tensors[i]
            d1, d2, d3 = size(A)
            A_mat = reshape(A, d1*d2, d3)
            @test A_mat' * A_mat ≈ I(d3) atol=1e-10
        end
    end

    @testset "Inner products" begin
        mps1 = randMPS(6, 4)
        overlap = inner(mps1, mps1)
        @test abs(overlap) > 0
        @test imag(inner(mps1, mps1)) ≈ 0 atol=1e-10

        mps2 = randMPS(6, 4)
        @test abs(inner(mps1, mps2)) >= 0
    end

    @testset "Fidelity" begin
        mps = randMPS(4, 2)
        f = fidelity(mps, mps)
        @test f ≈ 1.0 atol=1e-8

        mps2 = randMPS(4, 2)
        f2 = fidelity(mps, mps2)
        @test 0 <= f2 <= 1.0 + 1e-10
    end

    @testset "Truncation" begin
        mps = randMPS(6, 8)
        mps_trunc = truncate_mps(mps, 2)
        @test max_bond_dim(mps_trunc) <= 2
    end
end
