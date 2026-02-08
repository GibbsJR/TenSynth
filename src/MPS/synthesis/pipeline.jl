# Full synthesis pipeline: Circuit → Clifford+T
# Adapted from MPS2Circuit/src/synthesis/pipeline.jl
# Key changes: Uses TenSynth Core constants, PyCall synthesis functions moved to extension

using LinearAlgebra

# ==============================================================================
# Synthesis Result Types
# ==============================================================================

"""
    SynthesisResult

Result of Clifford+T synthesis for a quantum circuit.

# Fields
- `gate_sequences::Vector{Vector{String}}`: Gate sequence strings per 2-qubit gate
- `n_t_gates::Int`: Total T-gate count
- `n_clifford_gates::Int`: Total Clifford gate count (H, S, CNOT)
- `approximation_error::Float64`: Total approximation error
- `n_rz_gates::Int`: Number of RZ rotations before synthesis
- `synthesis_method::Symbol`: Method used (:gridsynth, :trasyn, :estimated)
"""
struct SynthesisResult
    gate_sequences::Vector{Vector{String}}
    n_t_gates::Int
    n_clifford_gates::Int
    approximation_error::Float64
    n_rz_gates::Int
    synthesis_method::Symbol
end

# ==============================================================================
# T-Gate Estimation (without synthesis)
# ==============================================================================

"""
    estimate_t_gates(n_rz::Int, epsilon::Float64) -> Int

Estimate T-gate count using asymptotic formula: N_T ≈ 3 log₂(1/ε) per RZ gate.
"""
function estimate_t_gates(n_rz::Int, epsilon::Float64)::Int
    if n_rz <= 0 || epsilon >= 1.0
        return 0
    end
    t_per_rz = ceil(Int, 3 * log2(1 / epsilon))
    return n_rz * t_per_rz
end

"""
    estimate_t_gates(circuit::Vector{Vector{Matrix{ComplexF64}}}, epsilon::Float64) -> Int

Estimate T-gate count for a circuit. Each 2-qubit gate has 15 RZ in its KAK decomposition.
"""
function estimate_t_gates(circuit::Vector{Vector{Matrix{ComplexF64}}}, epsilon::Float64)::Int
    n_gates = sum(length.(circuit); init=0)
    n_rz = 15 * n_gates
    return estimate_t_gates(n_rz, epsilon)
end

# ==============================================================================
# Clifford Gate Counting
# ==============================================================================

"""
    count_clifford_gates(gate_str::String) -> Int

Count Clifford gates (H, S, X, I, W) in a gate string.
"""
function count_clifford_gates(gate_str::String)::Int
    return count(c -> c in ('H', 'S', 'X', 'I', 'W'), gate_str)
end

# ==============================================================================
# Synthesis Functions
# ==============================================================================

"""
    synthesize_rz(theta, epsilon; use_trasyn=false) -> (gate_sequence, n_t_gates, error)

Synthesize a single RZ rotation to Clifford+T.
Falls back to estimation if gridsynth/trasyn are not available (PyCall extension).
"""
function synthesize_rz(theta::Float64, epsilon::Float64; use_trasyn::Bool=false)::Tuple{String, Int, Float64}
    if is_identity_rotation(theta; tol=epsilon)
        return "I", 0, 0.0
    end

    # Try PyCall-backed synthesis functions if available
    if !use_trasyn && isdefined(@__MODULE__, :ApproxRZ)
        try
            gs, U_approx = ApproxRZ(theta, epsilon)
            U_target = RZ(theta)
            error = 1.0 - abs(tr(U_target' * U_approx)) / 2
            return gs, NumTGates(gs), error
        catch
        end
    end

    if isdefined(@__MODULE__, :ApproxSU2_trasyn)
        try
            max_depth = max(10, ceil(Int, 5 * log2(1 / epsilon)))
            seq, _U_approx, err = ApproxSU2_trasyn(theta, max_depth; epsilon=epsilon)
            return seq, NumTGates(seq), err
        catch
        end
    end

    # Fallback: estimation
    n_t_est = ceil(Int, 3 * log2(1 / epsilon))
    return "[estimated]", n_t_est, epsilon
end

"""
    synthesize_su2(U, epsilon; max_depth=50, num_attempts=5) -> (gate_sequence, n_t_gates, error)

Synthesize a single-qubit SU(2) gate to Clifford+T.
Uses trasyn for direct SU(2) synthesis if available, otherwise decomposes to 3 RZ.
"""
function synthesize_su2(U::Matrix{ComplexF64}, epsilon::Float64;
                        max_depth::Int=50, num_attempts::Int=5)::Tuple{String, Int, Float64}
    if norm(U - I) < epsilon
        return "I", 0, 0.0
    end

    if isdefined(@__MODULE__, :ApproxSU2_trasyn)
        try
            seq, _U_approx, err = ApproxSU2_trasyn(U, max_depth;
                                                     epsilon=epsilon,
                                                     num_attempts=num_attempts)
            return seq, NumTGates(seq), err
        catch
        end
    end

    # Fallback: decompose to 3 RZ angles
    thetas = try
        SU22Thetas(U)
    catch
        return "[su2_failed]", 3 * ceil(Int, 3 * log2(1 / epsilon)), 3 * epsilon
    end

    sequences = String[]
    total_t = 0
    total_err = 0.0

    for theta in thetas
        seq, n_t, err = synthesize_rz(theta, epsilon; use_trasyn=false)
        push!(sequences, seq)
        total_t += n_t
        total_err += err
    end

    combined_seq = "H" * sequences[1] * "H" * sequences[2] * "H" * sequences[3] * "H"
    return combined_seq, total_t, total_err
end

"""
    synthesize_su4(U, epsilon; use_trasyn=false) -> (gate_sequences, total_t_gates, total_error)

Synthesize a 2-qubit gate to Clifford+T via KAK decomposition to 15 RZ angles.
When use_trasyn=true, groups angles into 4 SU(2) blocks for fewer T-gates.
"""
function synthesize_su4(U::Matrix{ComplexF64}, epsilon::Float64;
                        use_trasyn::Bool=false)::Tuple{Vector{String}, Int, Float64}
    if use_trasyn
        return _synthesize_su4_grouped(U, epsilon)
    end

    thetas = try
        SU42Thetas(U; max_iters=500, tol=1e-8)
    catch
        return ["[decomposition_failed]" for _ in 1:15], estimate_t_gates(15, epsilon), epsilon * 15
    end

    sequences = String[]
    total_t = 0
    total_err = 0.0

    for theta in thetas
        seq, n_t, err = synthesize_rz(theta, epsilon; use_trasyn=false)
        push!(sequences, seq)
        total_t += n_t
        total_err += err
    end

    return sequences, total_t, total_err
end

"""
    _synthesize_su4_grouped(U, epsilon) -> (gate_sequences, total_t_gates, total_error)

Group KAK angles into 4 SU(2) blocks + 3 interaction RZ for fewer T-gates via trasyn.
"""
function _synthesize_su4_grouped(U::Matrix{ComplexF64}, epsilon::Float64;
                                  max_depth::Int=50, num_attempts::Int=3)::Tuple{Vector{String}, Int, Float64}
    thetas = try
        SU42Thetas(U; max_iters=500, tol=1e-8)
    catch
        return ["[decomposition_failed]" for _ in 1:7], estimate_t_gates(15, epsilon), epsilon * 15
    end

    sequences = String[]
    total_t = 0
    total_err = 0.0

    # Group 1: Leading 1Q on qubit 2 (angles 1:3)
    su2_1 = Thetas2SU2(thetas[1:3])
    seq1, n_t1, err1 = synthesize_su2(su2_1, epsilon; max_depth=max_depth, num_attempts=num_attempts)
    push!(sequences, seq1); total_t += n_t1; total_err += err1

    # Group 2: Leading 1Q on qubit 1 (angles 4:6)
    su2_2 = Thetas2SU2(thetas[4:6])
    seq2, n_t2, err2 = synthesize_su2(su2_2, epsilon; max_depth=max_depth, num_attempts=num_attempts)
    push!(sequences, seq2); total_t += n_t2; total_err += err2

    # Interaction angles 7:9 (individual RZ)
    for i in 7:9
        seq, n_t, err = synthesize_rz(thetas[i], epsilon; use_trasyn=true)
        push!(sequences, seq); total_t += n_t; total_err += err
    end

    # Group 3: Trailing 1Q on qubit 2 (angles 10:12)
    su2_3 = Thetas2SU2(thetas[10:12])
    seq3, n_t3, err3 = synthesize_su2(su2_3, epsilon; max_depth=max_depth, num_attempts=num_attempts)
    push!(sequences, seq3); total_t += n_t3; total_err += err3

    # Group 4: Trailing 1Q on qubit 1 (angles 13:15)
    su2_4 = Thetas2SU2(thetas[13:15])
    seq4, n_t4, err4 = synthesize_su2(su2_4, epsilon; max_depth=max_depth, num_attempts=num_attempts)
    push!(sequences, seq4); total_t += n_t4; total_err += err4

    return sequences, total_t, total_err
end

# ==============================================================================
# Full Circuit Synthesis
# ==============================================================================

"""
    synthesize(circuit, circuit_inds; epsilon=1e-3, use_trasyn=false,
               remove_redundancy=true, verbose=false) -> SynthesisResult

Synthesize a full circuit to Clifford+T gates.
"""
function synthesize(circuit::Vector{Vector{Matrix{ComplexF64}}},
                   circuit_inds::Vector{Vector{Tuple{Int,Int}}};
                   epsilon::Float64=1e-3,
                   use_trasyn::Bool=false,
                   remove_redundancy::Bool=true,
                   verbose::Bool=false)::SynthesisResult
    if isempty(circuit)
        return SynthesisResult(Vector{String}[], 0, 0, 0.0, 0, :none)
    end

    circuit_opt = circuit
    if remove_redundancy
        if verbose
            println("Applying redundancy removal...")
        end
        circuit_opt, _stats = remove_redundant_rotations(circuit, circuit_inds; verbose=verbose)
    end

    n_gates = sum(length.(circuit_opt); init=0)
    n_rz = 15 * n_gates

    if verbose
        println("Synthesizing $n_gates 2-qubit gates ($n_rz RZ rotations)...")
    end

    all_sequences = Vector{String}[]
    total_t = 0
    total_clifford = 0
    total_error = 0.0
    synthesis_method = :gridsynth

    for (layer_idx, layer) in enumerate(circuit_opt)
        for gate in layer
            seqs, n_t, err = synthesize_su4(gate, epsilon; use_trasyn=use_trasyn)
            push!(all_sequences, seqs)
            total_t += n_t
            total_error += err
            for seq in seqs
                total_clifford += count_clifford_gates(seq)
            end
        end

        if verbose && layer_idx % 5 == 0
            println("  Layer $layer_idx/$(length(circuit_opt)) complete")
        end
    end

    if !isempty(all_sequences) && any(seq -> occursin("[estimated]", join(seq)), all_sequences)
        synthesis_method = :estimated
    elseif use_trasyn
        synthesis_method = :trasyn
    end

    if verbose
        println("Synthesis complete: T-gates=$total_t, Clifford=$total_clifford")
    end

    return SynthesisResult(all_sequences, total_t, total_clifford, total_error, n_rz, synthesis_method)
end

# ==============================================================================
# T-Gate Count Summary
# ==============================================================================

"""
    count_t_gates(result::SynthesisResult) -> Int
"""
function count_t_gates(result::SynthesisResult)::Int
    return result.n_t_gates
end

"""
    summarize_synthesis(result::SynthesisResult) -> String
"""
function summarize_synthesis(result::SynthesisResult)::String
    lines = String[]
    push!(lines, "=== Clifford+T Synthesis Summary ===")
    push!(lines, "Method: $(result.synthesis_method)")
    push!(lines, "RZ rotations: $(result.n_rz_gates)")
    push!(lines, "T-gates: $(result.n_t_gates)")
    push!(lines, "Clifford gates: $(result.n_clifford_gates)")
    push!(lines, "Total gates: $(result.n_t_gates + result.n_clifford_gates)")
    push!(lines, "Approximation error: $(round(result.approximation_error, sigdigits=3))")

    if result.n_rz_gates > 0
        t_per_rz = round(result.n_t_gates / result.n_rz_gates, digits=1)
        push!(lines, "Average T/RZ: $t_per_rz")
    end

    return join(lines, "\n")
end

"""
    mps_to_clifford_t(mps; target_fidelity=0.95, epsilon=1e-3, method=:iterative, verbose=false)
        -> (DecompositionResult, SynthesisResult)

Full pipeline from MPS to Clifford+T gates.
"""
function mps_to_clifford_t(mps::FiniteMPS{ComplexF64};
                           target_fidelity::Float64=0.95,
                           epsilon::Float64=1e-3,
                           method::Symbol=:iterative,
                           verbose::Bool=false)::Tuple{DecompositionResult, SynthesisResult}
    if verbose
        println("Step 1: MPS → Circuit decomposition")
    end

    decomp_result = decompose(mps;
                              method=method,
                              target_fidelity=target_fidelity,
                              verbose=verbose)

    if verbose
        println("  Fidelity: $(round(decomp_result.fidelity, digits=4))")
        println("  Depth: $(decomp_result.depth), Gates: $(decomp_result.n_gates)")
        println("\nStep 2: Circuit → Clifford+T synthesis")
    end

    synth_result = synthesize(decomp_result.circuit, decomp_result.circuit_inds;
                              epsilon=epsilon, verbose=verbose)

    if verbose
        println("\n$(summarize_synthesis(synth_result))")
    end

    return decomp_result, synth_result
end
