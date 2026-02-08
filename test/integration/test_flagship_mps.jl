@testset "Flagship: MPS TFIM Ground State Depth Scaling" begin
    using Random
    using LinearAlgebra
    using ITensors
    using ITensorMPS
    using TenSynth.MPS
    using TenSynth.Core

    Random.seed!(42)

    # --- Parameters ---
    N = 8           # qubits
    h_field = 1.0   # transverse field (critical point)
    chi_dmrg = 16   # DMRG bond dimension
    depths = [2, 4, 8]

    # --- Step 1: Compute TFIM ground state via DMRG ---
    @info "MPS Flagship: Computing TFIM ground state via DMRG (N=$N, h=$h_field, chi=$chi_dmrg)"

    sites = siteinds("S=1/2", N)

    os = OpSum()
    for j in 1:(N - 1)
        os += -1.0, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
        os += -h_field, "Sx", j
    end
    H_itensors = ITensorMPS.MPO(os, sites)

    psi0 = random_mps(sites; linkdims=chi_dmrg)
    sweeps = Sweeps(10)
    setmaxdim!(sweeps, chi_dmrg)
    setcutoff!(sweeps, 1e-10)

    energy, psi = dmrg(H_itensors, psi0, sweeps; outputlevel=0)
    @info "  DMRG energy: $energy"
    @test energy < 0.0  # ground state energy should be negative

    # --- Step 2: Convert to TenSynth FiniteMPS ---
    mps_target = TenSynth.MPS.from_itensors(psi)
    @test mps_target isa FiniteMPS{ComplexF64}
    @test length(mps_target.tensors) == N
    @info "  Converted to FiniteMPS, max bond dim: $(max_bond_dim(mps_target))"

    # --- Step 3: Decompose at increasing depths ---
    fidelities = Float64[]

    for d in depths
        @info "  Decomposing at depth $d..."
        Random.seed!(100 + d)  # reproducible per depth

        result = decompose(mps_target;
            method=:iterative,
            max_layers=d,
            n_sweeps=50,
            target_fidelity=0.99,
            verbose=false
        )

        @test result isa DecompositionResult
        @test result.depth <= d
        push!(fidelities, result.fidelity)
        @info "    depth=$d: fidelity=$(round(result.fidelity, digits=6)), actual_depth=$(result.depth), n_gates=$(result.n_gates)"
    end

    # --- Step 4: Assertions ---
    @info "  Fidelities: $fidelities"

    # Monotonic improvement (allow small numerical noise)
    for i in 2:length(fidelities)
        @test fidelities[i] >= fidelities[i-1] - 1e-6
    end

    # Final fidelity should be decent
    @test fidelities[end] > 0.8
end
