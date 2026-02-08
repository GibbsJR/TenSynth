@testset "iMPS Cost Dispatch" begin
    using TenSynth.iMPS
    using TenSynth.Core: cost

    @testset "cost(iMPS, iMPS) dispatches" begin
        psi = TenSynth.iMPS.random_product_state(2)
        TenSynth.iMPS.absorb_bonds!(psi)

        c = cost(psi, psi)
        @test c ≈ 0.0 atol=1e-8
    end

    @testset "cost between different states" begin
        psi1 = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.zero_state!(psi1)
        psi2 = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.plus_state!(psi2)

        TenSynth.iMPS.absorb_bonds!(psi1)
        TenSynth.iMPS.absorb_bonds!(psi2)

        c = cost(psi1, psi2)
        @test 0 < c < 1.0
    end
end
