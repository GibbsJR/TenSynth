using TenSynth.MPO

@testset "MPO Core" begin
    @testset "identity_mpo" begin
        mpo = identity_mpo(6)
        @test length(mpo.tensors) == 6
        @test all(size(t) == (1, 2, 2, 1) for t in mpo.tensors)
        # Each tensor should be identity: [1, phys_out, phys_in, 1] = δ_{phys_out, phys_in}
        for t in mpo.tensors
            @test t[1, 1, 1, 1] ≈ 1.0
            @test t[1, 2, 2, 1] ≈ 1.0
            @test t[1, 1, 2, 1] ≈ 0.0
            @test t[1, 2, 1, 1] ≈ 0.0
        end
    end

    @testset "n_sites" begin
        for n in [2, 4, 8]
            mpo = identity_mpo(n)
            @test n_sites(mpo) == n
        end
    end

    @testset "bond_dimensions" begin
        mpo = identity_mpo(6)
        @test bond_dimensions(mpo) == [1, 1, 1, 1, 1]
    end

    @testset "trace_mpo" begin
        # Trace of identity on n sites = 2^n
        for n in [2, 4, 6]
            mpo = identity_mpo(n)
            @test abs(trace_mpo(mpo) - 2.0^n) < 1e-10
        end
    end

    @testset "copy" begin
        mpo = identity_mpo(4)
        mpo2 = copy(mpo)
        mpo2.tensors[1] .*= 2.0
        # Original should be unchanged
        @test mpo.tensors[1][1, 1, 1, 1] ≈ 1.0
    end
end
