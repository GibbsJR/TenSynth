@testset "iMPS Gates" begin
    using TenSynth.iMPS
    using TenSynth.Core: PAULI_X, CNOT, SWAP, BondConfig, is_unitary
    using TenSynth.Core: PauliGeneratorParameterization, DressedZZParameterization
    using TenSynth.Core: to_matrix, ParameterizedGate

    @testset "Single-qubit gate" begin
        psi = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.zero_state!(psi)
        TenSynth.iMPS.absorb_bonds!(psi)

        psi2 = deepcopy(psi)
        TenSynth.iMPS.apply_gate_single!(psi2, PAULI_X, 1)
        TenSynth.iMPS.absorb_bonds!(psi2)

        # After X on site 1: state changed
        f = TenSynth.iMPS.local_fidelity(psi, psi2)
        @test f < 1.0
    end

    @testset "Two-qubit gate (nearest-neighbor)" begin
        psi = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.zero_state!(psi)
        config = BondConfig(16, 1e-10)

        psi2 = deepcopy(psi)
        TenSynth.iMPS.apply_gate_nn!(psi2, CNOT, (1, 2), config)
        TenSynth.iMPS.absorb_bonds!(psi2)

        # CNOT on |00⟩ should give |00⟩ — fidelity = 1
        TenSynth.iMPS.absorb_bonds!(psi)
        f = TenSynth.iMPS.local_fidelity(psi, psi2)
        @test f ≈ 1.0 atol=1e-6
    end

    @testset "CNOT on |10⟩" begin
        psi = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.zero_state!(psi)
        TenSynth.iMPS.apply_gate_single!(psi, PAULI_X, 1)  # |10⟩
        config = BondConfig(16, 1e-10)

        psi2 = deepcopy(psi)
        TenSynth.iMPS.apply_gate_nn!(psi2, CNOT, (1, 2), config)  # |10⟩ → |11⟩

        psi_ref = TenSynth.Core.iMPS{ComplexF64}(2)
        TenSynth.iMPS.zero_state!(psi_ref)
        TenSynth.iMPS.apply_gate_single!(psi_ref, PAULI_X, 1)
        TenSynth.iMPS.apply_gate_single!(psi_ref, PAULI_X, 2)  # |11⟩

        TenSynth.iMPS.absorb_bonds!(psi2)
        TenSynth.iMPS.absorb_bonds!(psi_ref)
        f = TenSynth.iMPS.local_fidelity(psi2, psi_ref)
        @test f ≈ 1.0 atol=1e-6
    end

    @testset "SWAP preserves state" begin
        psi = TenSynth.iMPS.random_product_state(2)
        config = BondConfig(16, 1e-10)

        psi2 = deepcopy(psi)
        # Apply SWAP twice — should return to original
        TenSynth.iMPS.apply_gate_nn!(psi2, SWAP, (1, 2), config)
        TenSynth.iMPS.apply_gate_nn!(psi2, SWAP, (1, 2), config)

        TenSynth.iMPS.absorb_bonds!(psi)
        TenSynth.iMPS.absorb_bonds!(psi2)
        f = TenSynth.iMPS.local_fidelity(psi, psi2)
        @test f ≈ 1.0 atol=1e-4
    end

    @testset "Parameterized gate application" begin
        psi = TenSynth.iMPS.random_product_state(2)
        config = BondConfig(16, 1e-10)

        gate = ParameterizedGate(PauliGeneratorParameterization(), (1, 2), randn(15))
        @test is_unitary(to_matrix(gate))

        psi2 = deepcopy(psi)
        TenSynth.iMPS.apply_gate!(psi2, gate, config)

        TenSynth.iMPS.absorb_bonds!(psi)
        TenSynth.iMPS.absorb_bonds!(psi2)
        f = TenSynth.iMPS.local_fidelity(psi, psi2)
        # Fidelity should be in valid range
        @test 0.0 <= f <= 1.0
    end
end
