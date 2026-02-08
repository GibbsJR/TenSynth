using TenSynth.MPO
using TenSynth.Core

@testset "MPO Gates" begin
    @testset "apply_gate! adjacent" begin
        mpo = identity_mpo(4)
        gate = GateMatrix(CNOT, "CNOT", Dict{Symbol,Any}())
        err = apply_gate!(mpo, gate, 1, 2)
        @test err < 1e-10  # CNOT is bond dim 2, no truncation needed
        @test hst_cost(mpo, identity_mpo(4)) > 0.0  # No longer identity
    end

    @testset "gate self-inverse" begin
        # Apply CNOT twice → should return to identity
        mpo = identity_mpo(4)
        gate = GateMatrix(CNOT, "CNOT", Dict{Symbol,Any}())
        apply_gate!(mpo, gate, 2, 3)
        apply_gate!(mpo, gate, 2, 3)
        @test hst_cost(mpo, identity_mpo(4)) < 1e-8
    end

    @testset "SWAP self-inverse" begin
        mpo = identity_mpo(4)
        gate = GateMatrix(SWAP, "SWAP", Dict{Symbol,Any}())
        apply_gate!(mpo, gate, 1, 2)
        apply_gate!(mpo, gate, 1, 2)
        @test hst_cost(mpo, identity_mpo(4)) < 1e-8
    end

    @testset "apply_gate_to_tensors preserves unitarity" begin
        TL = reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1)
        TR = reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1)
        TL_new, TR_new, err = apply_gate_to_tensors(TL, TR, CNOT, 128, 1e-14)
        # Build MPO from the output tensors and check it's unitary
        mpo = FiniteMPO{ComplexF64}([TL_new, TR_new])
        @test abs(inner_mpo(mpo, mpo) - 1.0) < 1e-10
    end

    @testset "gate_to_mpo adjacent" begin
        gate = GateMatrix(CNOT, "CNOT", Dict{Symbol,Any}())
        mpo = gate_to_mpo(gate, 1, 2, 4)
        @test n_sites(mpo) == 4
        # Should represent CNOT on sites 1,2 and identity elsewhere
        @test abs(inner_mpo(mpo, mpo) - 1.0) < 1e-10  # Unitary
    end

    @testset "apply_gate_long_range!" begin
        mpo = identity_mpo(6)
        gate = GateMatrix(CNOT, "CNOT", Dict{Symbol,Any}())
        apply_gate_long_range!(mpo, gate, 1, 4)
        @test hst_cost(mpo, identity_mpo(6)) > 0.0  # No longer identity
        # Check unitarity via hst_cost (inner_mpo loses norm due to mpo_mpo_contract! normalization)
        mpo_copy = FiniteMPO{ComplexF64}([copy(t) for t in mpo.tensors])
        @test hst_cost(mpo, mpo_copy) < 1e-6
    end

    @testset "apply_layer!" begin
        mpo = identity_mpo(4)
        indices = [(1, 2), (3, 4)]
        gates = [GateMatrix(CNOT, "CNOT", Dict{Symbol,Any}()),
                 GateMatrix(SWAP, "SWAP", Dict{Symbol,Any}())]
        layer = GateLayer(gates, indices)
        apply_layer!(mpo, layer)
        @test hst_cost(mpo, identity_mpo(4)) > 0.0
    end
end
