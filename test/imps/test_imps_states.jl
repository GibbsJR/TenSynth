@testset "iMPS States" begin
    using TenSynth.iMPS

    @testset "Construction" begin
        psi = TenSynth.iMPS.random_product_state(2)
        @test psi isa TenSynth.Core.iMPS{ComplexF64}
        @test psi.unit_cell == 2
        @test psi.physical_dim == 2
        @test length(psi.gamma) == 2
        @test length(psi.lambda) == 2
    end

    @testset "Zero state" begin
        psi = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.zero_state!(psi)
        @test psi.normalized == true
        @test size(psi.gamma[1]) == (1, 2, 1)
        @test abs(psi.gamma[1][1, 1, 1]) ≈ 1.0
        @test abs(psi.gamma[1][1, 2, 1]) ≈ 0.0
    end

    @testset "Plus state" begin
        psi = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.plus_state!(psi)
        @test psi.normalized == true
        @test abs(psi.gamma[1][1, 1, 1]) ≈ abs(psi.gamma[1][1, 2, 1])
    end

    @testset "Neel state" begin
        psi = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.neel_state!(psi)
        @test abs(psi.gamma[1][1, 1, 1]) ≈ 1.0  # |0⟩ on site 1
        @test abs(psi.gamma[2][1, 2, 1]) ≈ 1.0  # |1⟩ on site 2
    end

    @testset "Bond dimensions" begin
        psi = TenSynth.iMPS.random_product_state(3)
        @test TenSynth.iMPS.bond_dimensions(psi) == [1, 1, 1]
        @test TenSynth.iMPS.max_bond_dimension(psi) == 1
    end

    @testset "Self-fidelity" begin
        psi = TenSynth.iMPS.random_product_state(2)
        TenSynth.iMPS.absorb_bonds!(psi)
        f = TenSynth.iMPS.local_fidelity(psi, psi)
        @test f ≈ 1.0 atol=1e-8
    end

    @testset "Infidelity" begin
        psi = TenSynth.iMPS.random_product_state(2)
        TenSynth.iMPS.absorb_bonds!(psi)
        @test TenSynth.iMPS.infidelity(psi, psi) ≈ 0.0 atol=1e-8
    end

    @testset "Different states have fidelity < 1" begin
        psi1 = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.zero_state!(psi1)
        psi2 = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.plus_state!(psi2)
        TenSynth.iMPS.absorb_bonds!(psi1)
        TenSynth.iMPS.absorb_bonds!(psi2)
        f = TenSynth.iMPS.local_fidelity(psi1, psi2)
        @test f < 1.0
        @test f > 0.0
    end

    @testset "Deepcopy independence" begin
        psi = TenSynth.iMPS.random_product_state(2)
        psi2 = deepcopy(psi)
        psi.gamma[1] .= 0
        @test !all(psi2.gamma[1] .== 0)  # deepcopy is independent
    end
end
