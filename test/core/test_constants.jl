@testset "Core Constants" begin
    I4 = Matrix{ComplexF64}(I, 4, 4)

    # Pauli algebra
    @test PAULI_X * PAULI_X ≈ PAULI_I
    @test PAULI_Y * PAULI_Y ≈ PAULI_I
    @test PAULI_Z * PAULI_Z ≈ PAULI_I
    @test H_GATE * H_GATE ≈ PAULI_I
    @test CNOT * CNOT ≈ I4
    @test SWAP * SWAP ≈ I4
    @test S_GATE * S_GATE ≈ PAULI_Z
    @test T_GATE * T_GATE ≈ S_GATE

    # Cross-check against known matrices
    @test PAULI_X ≈ ComplexF64[0 1; 1 0]
    @test PAULI_Y ≈ ComplexF64[0 -im; im 0]
    @test PAULI_Z ≈ ComplexF64[1 0; 0 -1]
    @test CNOT ≈ ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]

    # Aliases
    @test Id === PAULI_I
    @test X === PAULI_X
    @test Y === PAULI_Y
    @test Z === PAULI_Z
    @test Hadamard === H_GATE

    # Rotation unitarity
    @test is_unitary(RX(0.5))
    @test is_unitary(RY(0.5))
    @test is_unitary(RZ(0.5))
    @test is_unitary(RZZ(0.5))

    # Rotation identities
    @test RX(0.0) ≈ PAULI_I
    @test RY(0.0) ≈ PAULI_I
    @test RZ(0.0) ≈ PAULI_I
    @test RZZ(0.0) ≈ I4

    # U_gate
    @test is_unitary(U_gate(0.5, 0.3, 0.7))

    # Two-qubit Pauli products
    @test XX ≈ kron(PAULI_X, PAULI_X)
    @test ZZ ≈ kron(PAULI_Z, PAULI_Z)
    @test length(PAULI_2Q) == 16
    @test length(PAULI_1Q) == 4

    # FSWAP and special gates
    @test is_unitary(FSWAP)
    @test is_unitary(U_singlet)
    @test Proj11 ≈ ComplexF64[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]
end
