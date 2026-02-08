@testset "Core Parameterizations" begin
    # n_params dispatch
    @test n_params(KAKParameterization()) == 15
    @test n_params(PauliGeneratorParameterization()) == 15
    @test n_params(DressedZZParameterization()) == 7
    @test n_params(SingleQubitXYZParameterization()) == 3
    @test n_params(ZZParameterization()) == 1
    @test n_params(ZZXParameterization()) == 3

    # KAK and PauliGenerator produce unitaries
    for _ in 1:10
        p_kak = randn(15)
        U_kak = to_matrix(ParameterizedGate(KAKParameterization(), (1, 2), p_kak))
        @test size(U_kak) == (4, 4)
        @test is_unitary(U_kak)

        p_pauli = randn(15)
        U_pauli = to_matrix(ParameterizedGate(PauliGeneratorParameterization(), (1, 2), p_pauli))
        @test size(U_pauli) == (4, 4)
        @test is_unitary(U_pauli)
    end

    # Sub-parameterizations produce unitaries
    @test is_unitary(to_matrix(ParameterizedGate(DressedZZParameterization(), (1, 2), randn(7))))
    @test is_unitary(to_matrix(ParameterizedGate(ZZParameterization(), (1, 2), randn(1))))
    @test is_unitary(to_matrix(ParameterizedGate(ZZXParameterization(), (1, 2), randn(3))))

    # SingleQubitXYZ returns 2×2
    @test size(to_matrix(ParameterizedGate(SingleQubitXYZParameterization(), (1, 1), randn(3)))) == (2, 2)
    @test is_unitary(to_matrix(ParameterizedGate(SingleQubitXYZParameterization(), (1, 1), randn(3))))

    # Zero parameters → identity (or close)
    U_kak_zero = to_matrix(ParameterizedGate(KAKParameterization(), (1, 2), zeros(15)))
    @test is_unitary(U_kak_zero)

    U_pauli_zero = to_matrix(ParameterizedGate(PauliGeneratorParameterization(), (1, 2), zeros(15)))
    @test U_pauli_zero ≈ Matrix{ComplexF64}(I, 4, 4) atol=1e-10
end
