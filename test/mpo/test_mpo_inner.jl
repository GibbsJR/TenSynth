using TenSynth.MPO
using TenSynth.Core

@testset "MPO Inner Products" begin
    @testset "inner_mps self-overlap" begin
        mpo = identity_mpo(4)
        mps, _ = mpo_to_mps(mpo)
        overlap = inner_mps(mps, mps)
        @test abs(overlap - 1.0) < 1e-10
    end

    @testset "inner_mpo identity" begin
        for n in [2, 4, 6]
            mpo = identity_mpo(n)
            # inner_mpo normalizes by 2^n, so <I|I>/2^n = Tr(I)/2^n = 2^n/2^n = 1
            @test abs(inner_mpo(mpo, mpo) - 1.0) < 1e-10
        end
    end

    @testset "inner_mpo with gate" begin
        mpo1 = identity_mpo(4)
        mpo2 = identity_mpo(4)
        gate = GateMatrix(CNOT, "CNOT", Dict{Symbol,Any}())
        apply_gate!(mpo2, gate, 1, 2)
        # CNOT is unitary: <I|CNOT> should be Tr(CNOT)/2^n
        # Tr(CNOT) = 1+1+0+0 = 2 (for single gate on 2 qubits within 4-qubit system)
        # Actually, inner_mpo = Tr(I† CNOT)/2^4 and the CNOT only affects 2 sites
        overlap = inner_mpo(mpo1, mpo2)
        @test abs(overlap) < 1.0  # Should not be 1
        @test abs(overlap) > 0.0  # Should not be 0
    end

    @testset "hst_cost identity" begin
        for n in [2, 4, 6]
            mpo = identity_mpo(n)
            @test hst_cost(mpo, mpo) < 1e-10
        end
    end

    @testset "hst_cost different operators" begin
        mpo1 = identity_mpo(4)
        mpo2 = identity_mpo(4)
        gate = GateMatrix(CNOT, "CNOT", Dict{Symbol,Any}())
        apply_gate!(mpo2, gate, 2, 3)
        c = hst_cost(mpo1, mpo2)
        @test c > 0.0
        @test c <= 1.0
    end

    @testset "hst_cost symmetry" begin
        mpo1 = identity_mpo(4)
        mpo2 = identity_mpo(4)
        gate = GateMatrix(CNOT, "CNOT", Dict{Symbol,Any}())
        apply_gate!(mpo2, gate, 1, 2)
        @test abs(hst_cost(mpo1, mpo2) - hst_cost(mpo2, mpo1)) < 1e-10
    end
end
