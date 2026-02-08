# End-to-End Pipeline: MPS → Quantum Circuit → Clifford+T
# Adapted from MPS2Circuit/src/pipeline.jl
# Key changes: Vector{Array{ComplexF64,3}} → FiniteMPS{ComplexF64},
#              validate_* → inline checks, Printf → string interpolation

using LinearAlgebra

# ==============================================================================
# CompilationResult: Unified result type
# ==============================================================================

"""
    CompilationResult

Unified result of compiling an MPS to a Clifford+T quantum circuit.

# Fields
- `decomposition::DecompositionResult`: Intermediate decomposition result
- `synthesis::Union{SynthesisResult, Nothing}`: Synthesis result
- `n_qubits::Int`: Number of qubits
- `fidelity::Float64`: Final fidelity
- `n_2q_gates::Int`: Number of 2-qubit gates before synthesis
- `circuit_depth::Int`: Circuit depth in layers
- `n_t_gates::Int`: Total T-gate count
- `n_clifford_gates::Int`: Total Clifford gate count
- `approximation_error::Float64`: Total approximation error
- `method::Symbol`: Decomposition method used
- `synthesis_epsilon::Float64`: Precision used for synthesis
- `elapsed_time::Float64`: Total time in seconds
"""
struct CompilationResult
    decomposition::DecompositionResult
    synthesis::Union{SynthesisResult, Nothing}
    n_qubits::Int
    fidelity::Float64
    n_2q_gates::Int
    circuit_depth::Int
    n_t_gates::Int
    n_clifford_gates::Int
    approximation_error::Float64
    method::Symbol
    synthesis_epsilon::Float64
    elapsed_time::Float64
end

function Base.show(io::IO, result::CompilationResult)
    synth_str = result.synthesis === nothing ? "none" : "$(result.n_t_gates) T-gates"
    print(io, "CompilationResult(fidelity=$(round(result.fidelity, digits=6)), ",
          "depth=$(result.circuit_depth), 2Q=$(result.n_2q_gates), $synth_str)")
end

function Base.show(io::IO, ::MIME"text/plain", result::CompilationResult)
    println(io, "CompilationResult:")
    println(io, "  Qubits:            $(result.n_qubits)")
    println(io, "  Fidelity:          $(round(result.fidelity, digits=8))")
    println(io, "  Circuit depth:     $(result.circuit_depth) layers")
    println(io, "  2-qubit gates:     $(result.n_2q_gates)")
    println(io, "  Method:            :$(result.method)")
    if result.synthesis !== nothing
        println(io, "  T-gates:           $(result.n_t_gates)")
        println(io, "  Clifford gates:    $(result.n_clifford_gates)")
        println(io, "  Synthesis error:   $(round(result.approximation_error, digits=8))")
        println(io, "  Synthesis epsilon: $(result.synthesis_epsilon)")
    else
        println(io, "  Synthesis:         not performed")
    end
    println(io, "  Elapsed time:      $(round(result.elapsed_time, digits=3)) seconds")
end

# ==============================================================================
# compile(): Master compilation function
# ==============================================================================

"""
    compile(mps; kwargs...) -> CompilationResult

Compile an MPS to a Clifford+T quantum circuit.

Full pipeline: MPS → 2-qubit gates → Clifford+T.

# Arguments
- `mps::FiniteMPS{ComplexF64}`: Input MPS
- `method::Symbol=:iterative`: Decomposition method
- `target_fidelity::Float64=0.99`: Target fidelity
- `max_layers::Int=20`: Maximum circuit depth
- `n_sweeps::Int=50`: Optimization sweeps per layer
- `synthesize_gates::Bool=true`: Perform Clifford+T synthesis
- `epsilon::Float64=1e-3`: Precision for T-gate synthesis
- `use_trasyn::Bool=false`: Use trasyn instead of gridsynth
- `remove_redundancy::Bool=true`: Remove redundant rotations before synthesis
- `verbose::Bool=false`: Print progress
"""
function compile(mps::FiniteMPS{ComplexF64};
                 method::Symbol=:iterative,
                 target_fidelity::Float64=0.99,
                 max_layers::Int=20,
                 n_sweeps::Int=50,
                 synthesize_gates::Bool=true,
                 epsilon::Float64=1e-3,
                 use_trasyn::Bool=false,
                 remove_redundancy::Bool=true,
                 verbose::Bool=false)::CompilationResult
    # Suppress harmless Julia 1.11 compiler warnings (BoundsError in DomTreeNode
    # during type inference) by redirecting stderr before triggering JIT compilation
    # of the implementation function via invokelatest.
    return redirect_stderr(devnull) do
        Base.invokelatest(_compile_body, mps, method, target_fidelity, max_layers, n_sweeps,
                          synthesize_gates, epsilon, use_trasyn, remove_redundancy, verbose)
    end::CompilationResult
end

function _compile_body(mps, method, target_fidelity, max_layers, n_sweeps,
                       synthesize_gates, epsilon, use_trasyn, remove_redundancy, verbose)

    start_time = time()

    N = length(mps.tensors)
    N >= 2 || throw(ArgumentError("MPS must have at least 2 sites"))
    0 < target_fidelity <= 1.0 || throw(ArgumentError("target_fidelity must be in (0, 1]"))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive"))

    if verbose
        println("=" ^ 60)
        println("MPS -> Quantum Circuit Compilation")
        println("=" ^ 60)
        println("  Qubits: $N")
        println("  Method: :$method")
        println("  Target fidelity: $target_fidelity")
        println("  Synthesize: $synthesize_gates")
        if synthesize_gates
            println("  Epsilon: $epsilon")
        end
        println("-" ^ 60)
    end

    # Step 1: Decomposition
    if verbose
        println("Step 1: Decomposing MPS to quantum circuit...")
    end

    decomp_result = decompose(mps;
                              method=method,
                              target_fidelity=target_fidelity,
                              max_layers=max_layers,
                              n_sweeps=n_sweeps,
                              verbose=verbose)

    if verbose
        println("  Achieved fidelity: $(round(decomp_result.fidelity, digits=8))")
        println("  Circuit depth: $(decomp_result.depth)")
        println("  2-qubit gates: $(decomp_result.n_gates)")
    end

    # Step 2: Synthesis (optional)
    synth_result = nothing
    n_t_gates = 0
    n_clifford_gates = 0
    approximation_error = 0.0

    if synthesize_gates && !isempty(decomp_result.circuit)
        if verbose
            println("-" ^ 60)
            println("Step 2: Synthesizing to Clifford+T...")
        end

        synth_result = synthesize(decomp_result.circuit, decomp_result.circuit_inds;
                                  epsilon=epsilon,
                                  use_trasyn=use_trasyn,
                                  remove_redundancy=remove_redundancy,
                                  verbose=verbose)

        n_t_gates = synth_result.n_t_gates
        n_clifford_gates = synth_result.n_clifford_gates
        approximation_error = synth_result.approximation_error

        if verbose
            println("  T-gates: $n_t_gates")
            println("  Clifford gates: $n_clifford_gates")
            println("  Approximation error: $(round(approximation_error, digits=8))")
        end
    end

    elapsed = time() - start_time

    if verbose
        println("-" ^ 60)
        println("Compilation complete in $(round(elapsed, digits=3)) seconds")
        println("=" ^ 60)
    end

    return CompilationResult(
        decomp_result,
        synth_result,
        N,
        decomp_result.fidelity,
        decomp_result.n_gates,
        decomp_result.depth,
        n_t_gates,
        n_clifford_gates,
        approximation_error,
        method,
        synthesize_gates ? epsilon : 0.0,
        elapsed
    )
end

# ==============================================================================
# to_openqasm(): Export to OpenQASM 2.0 format
# ==============================================================================

"""
    to_openqasm(result; include_header=true) -> String

Export compilation or decomposition result to OpenQASM 2.0 format.
"""
function to_openqasm(result::CompilationResult; include_header::Bool=true)::String
    lines = String[]

    if include_header
        push!(lines, "OPENQASM 2.0;")
        push!(lines, "include \"qelib1.inc\";")
        push!(lines, "")
        push!(lines, "// TenSynth compilation result")
        push!(lines, "// Fidelity: $(round(result.fidelity, digits=8))")
        push!(lines, "// Method: $(result.method)")
        if result.synthesis !== nothing
            push!(lines, "// T-gates: $(result.n_t_gates)")
            push!(lines, "// Synthesis epsilon: $(result.synthesis_epsilon)")
        end
        push!(lines, "")
        push!(lines, "qreg q[$(result.n_qubits)];")
        push!(lines, "")
    end

    decomp = result.decomposition

    for (layer_idx, (layer, inds)) in enumerate(zip(decomp.circuit, decomp.circuit_inds))
        push!(lines, "// Layer $layer_idx")
        for (gate, (qi, qj)) in zip(layer, inds)
            thetas = try
                SU42Thetas(gate; max_iters=100, tol=1e-8)
            catch
                zeros(15)
            end

            # Leading single-qubit on qubit j
            _emit_single_qubit_qasm!(lines, thetas[1:3], qj - 1)

            # Leading single-qubit on qubit i
            _emit_single_qubit_qasm!(lines, thetas[4:6], qi - 1)

            # XX interaction
            push!(lines, "h q[$(qi-1)];")
            push!(lines, "h q[$(qj-1)];")
            push!(lines, "cx q[$(qi-1)],q[$(qj-1)];")
            push!(lines, "rz($(thetas[7])) q[$(qj-1)];")
            push!(lines, "cx q[$(qi-1)],q[$(qj-1)];")
            push!(lines, "h q[$(qi-1)];")
            push!(lines, "h q[$(qj-1)];")

            # YY interaction
            push!(lines, "s q[$(qi-1)];")
            push!(lines, "s q[$(qj-1)];")
            push!(lines, "h q[$(qi-1)];")
            push!(lines, "h q[$(qj-1)];")
            push!(lines, "cx q[$(qi-1)],q[$(qj-1)];")
            push!(lines, "rz($(thetas[8])) q[$(qj-1)];")
            push!(lines, "cx q[$(qi-1)],q[$(qj-1)];")
            push!(lines, "h q[$(qi-1)];")
            push!(lines, "h q[$(qj-1)];")
            push!(lines, "sdg q[$(qi-1)];")
            push!(lines, "sdg q[$(qj-1)];")

            # ZZ interaction
            push!(lines, "cx q[$(qi-1)],q[$(qj-1)];")
            push!(lines, "rz($(thetas[9])) q[$(qj-1)];")
            push!(lines, "cx q[$(qi-1)],q[$(qj-1)];")

            # Trailing single-qubit on qubit j
            _emit_single_qubit_qasm!(lines, thetas[10:12], qj - 1)

            # Trailing single-qubit on qubit i
            _emit_single_qubit_qasm!(lines, thetas[13:15], qi - 1)
        end
        push!(lines, "")
    end

    return join(lines, "\n")
end

"""
    _emit_single_qubit_qasm!(lines, thetas, qubit)

Emit OpenQASM for a single-qubit rotation in XZX form.
"""
function _emit_single_qubit_qasm!(lines::Vector{String}, thetas::AbstractVector, qubit::Int)
    if all(abs.(thetas) .< 1e-10)
        return
    end
    push!(lines, "h q[$qubit];")
    push!(lines, "rz($(thetas[1])) q[$qubit];")
    push!(lines, "h q[$qubit];")
    push!(lines, "rz($(thetas[2])) q[$qubit];")
    push!(lines, "h q[$qubit];")
    push!(lines, "rz($(thetas[3])) q[$qubit];")
    push!(lines, "h q[$qubit];")
end

function to_openqasm(result::DecompositionResult; include_header::Bool=true)::String
    comp_result = CompilationResult(
        result, nothing, result.n_qubits, result.fidelity,
        result.n_gates, result.depth, 0, 0, 0.0, result.method, 0.0, 0.0
    )
    return to_openqasm(comp_result; include_header=include_header)
end

# ==============================================================================
# to_qiskit(): Export to Python Qiskit code
# ==============================================================================

"""
    to_qiskit(result; function_name="create_circuit") -> String

Export to Python Qiskit code.
"""
function to_qiskit(result::CompilationResult; function_name::String="create_circuit")::String
    lines = String[]

    push!(lines, "\"\"\"")
    push!(lines, "TenSynth compilation result exported to Qiskit")
    push!(lines, "Fidelity: $(round(result.fidelity, digits=8))")
    push!(lines, "Method: $(result.method)")
    if result.synthesis !== nothing
        push!(lines, "T-gates: $(result.n_t_gates)")
    end
    push!(lines, "\"\"\"")
    push!(lines, "")
    push!(lines, "from qiskit import QuantumCircuit")
    push!(lines, "import numpy as np")
    push!(lines, "")
    push!(lines, "def $function_name():")
    push!(lines, "    \"\"\"Create the quantum circuit.\"\"\"")
    push!(lines, "    qc = QuantumCircuit($(result.n_qubits))")
    push!(lines, "")

    decomp = result.decomposition

    for (layer_idx, (layer, inds)) in enumerate(zip(decomp.circuit, decomp.circuit_inds))
        push!(lines, "    # Layer $layer_idx")
        for (gate, (qi, qj)) in zip(layer, inds)
            thetas = try
                SU42Thetas(gate; max_iters=100, tol=1e-8)
            catch
                zeros(15)
            end

            _emit_single_qubit_qiskit!(lines, thetas[1:3], qj - 1)
            _emit_single_qubit_qiskit!(lines, thetas[4:6], qi - 1)

            # XX interaction
            push!(lines, "    qc.h($(qi-1))")
            push!(lines, "    qc.h($(qj-1))")
            push!(lines, "    qc.cx($(qi-1), $(qj-1))")
            push!(lines, "    qc.rz($(thetas[7]), $(qj-1))")
            push!(lines, "    qc.cx($(qi-1), $(qj-1))")
            push!(lines, "    qc.h($(qi-1))")
            push!(lines, "    qc.h($(qj-1))")

            # YY interaction
            push!(lines, "    qc.s($(qi-1))")
            push!(lines, "    qc.s($(qj-1))")
            push!(lines, "    qc.h($(qi-1))")
            push!(lines, "    qc.h($(qj-1))")
            push!(lines, "    qc.cx($(qi-1), $(qj-1))")
            push!(lines, "    qc.rz($(thetas[8]), $(qj-1))")
            push!(lines, "    qc.cx($(qi-1), $(qj-1))")
            push!(lines, "    qc.h($(qi-1))")
            push!(lines, "    qc.h($(qj-1))")
            push!(lines, "    qc.sdg($(qi-1))")
            push!(lines, "    qc.sdg($(qj-1))")

            # ZZ interaction
            push!(lines, "    qc.cx($(qi-1), $(qj-1))")
            push!(lines, "    qc.rz($(thetas[9]), $(qj-1))")
            push!(lines, "    qc.cx($(qi-1), $(qj-1))")

            _emit_single_qubit_qiskit!(lines, thetas[10:12], qj - 1)
            _emit_single_qubit_qiskit!(lines, thetas[13:15], qi - 1)
        end
        push!(lines, "")
    end

    push!(lines, "    return qc")
    push!(lines, "")
    push!(lines, "")
    push!(lines, "if __name__ == \"__main__\":")
    push!(lines, "    circuit = $function_name()")
    push!(lines, "    print(circuit)")
    push!(lines, "    print(f\"Depth: {circuit.depth()}\")")
    push!(lines, "    print(f\"Gate count: {circuit.count_ops()}\")")

    return join(lines, "\n")
end

"""
    _emit_single_qubit_qiskit!(lines, thetas, qubit)

Emit Qiskit code for a single-qubit rotation in XZX form.
"""
function _emit_single_qubit_qiskit!(lines::Vector{String}, thetas::AbstractVector, qubit::Int)
    if all(abs.(thetas) .< 1e-10)
        return
    end
    push!(lines, "    qc.h($qubit)")
    push!(lines, "    qc.rz($(thetas[1]), $qubit)")
    push!(lines, "    qc.h($qubit)")
    push!(lines, "    qc.rz($(thetas[2]), $qubit)")
    push!(lines, "    qc.h($qubit)")
    push!(lines, "    qc.rz($(thetas[3]), $qubit)")
    push!(lines, "    qc.h($qubit)")
end

function to_qiskit(result::DecompositionResult; function_name::String="create_circuit")::String
    comp_result = CompilationResult(
        result, nothing, result.n_qubits, result.fidelity,
        result.n_gates, result.depth, 0, 0, 0.0, result.method, 0.0, 0.0
    )
    return to_qiskit(comp_result; function_name=function_name)
end

# ==============================================================================
# show_circuit(): Text-based circuit visualization
# ==============================================================================

"""
    show_circuit(result; max_width=80) -> String

Generate a text-based visualization of the quantum circuit.
"""
function show_circuit(result::CompilationResult; max_width::Int=80)::String
    return show_circuit(result.decomposition; max_width=max_width)
end

function show_circuit(result::DecompositionResult; max_width::Int=80)::String
    N = result.n_qubits
    n_layers = result.depth

    if n_layers == 0
        return "Empty circuit (no gates)"
    end

    label_width = length("q$(N-1): ")

    # Initialize lines for each qubit
    main_lines = [String[] for _ in 1:N]
    conn_lines = [String[] for _ in 1:N]

    for (_layer_idx, (_layer, inds)) in enumerate(zip(result.circuit, result.circuit_inds))
        qubit_roles = Dict{Int, Symbol}()
        for (qi, qj) in inds
            if qi < qj
                qubit_roles[qi] = :top
                qubit_roles[qj] = :bottom
                for q in (qi+1):(qj-1)
                    qubit_roles[q] = :middle
                end
            else
                qubit_roles[qj] = :top
                qubit_roles[qi] = :bottom
                for q in (qj+1):(qi-1)
                    qubit_roles[q] = :middle
                end
            end
        end

        for q in 1:N
            role = get(qubit_roles, q, :none)
            if role == :top
                push!(main_lines[q], "-*-")
                push!(conn_lines[q], " | ")
            elseif role == :bottom
                push!(main_lines[q], "-*-")
                push!(conn_lines[q], "   ")
            elseif role == :middle
                push!(main_lines[q], "-|-")
                push!(conn_lines[q], " | ")
            else
                push!(main_lines[q], "---")
                push!(conn_lines[q], "   ")
            end
        end
    end

    # Build output
    lines = String[]
    for q in 1:N
        label = "q$(q-1): "
        main_str = join(main_lines[q], "")
        conn_str = join(conn_lines[q], "")

        if length(main_str) > max_width - label_width
            main_str = main_str[1:max_width-label_width-3] * "..."
            conn_str = conn_str[1:max_width-label_width-3] * "   "
        end

        push!(lines, label * main_str)
        if q < N
            push!(lines, " " ^ label_width * conn_str)
        end
    end

    return join(lines, "\n")
end

function show_circuit(io::IO, result::Union{CompilationResult, DecompositionResult}; max_width::Int=80)
    print(io, show_circuit(result; max_width=max_width))
end

# ==============================================================================
# Additional utility functions
# ==============================================================================

"""
    circuit_stats(result) -> NamedTuple

Get detailed statistics about the compiled circuit.
"""
function circuit_stats(result::CompilationResult)
    n_rz = 15 * result.n_2q_gates

    return (
        n_qubits = result.n_qubits,
        circuit_depth = result.circuit_depth,
        n_2q_gates = result.n_2q_gates,
        n_rz_rotations = n_rz,
        n_t_gates = result.n_t_gates,
        n_clifford_gates = result.n_clifford_gates,
        fidelity = result.fidelity,
        synthesis_error = result.approximation_error
    )
end

"""
    save_circuit(filename, result; format=:qasm)

Save circuit to a file in specified format (:qasm, :qiskit, :txt).
"""
function save_circuit(filename::String, result::CompilationResult; format::Symbol=:qasm)
    content = if format == :qasm
        to_openqasm(result)
    elseif format == :qiskit
        to_qiskit(result)
    elseif format == :txt
        """
# TenSynth Compilation Result
# Qubits: $(result.n_qubits)
# Fidelity: $(result.fidelity)
# Depth: $(result.circuit_depth)
# 2-qubit gates: $(result.n_2q_gates)
# T-gates: $(result.n_t_gates)
# Method: $(result.method)

$(show_circuit(result))
"""
    else
        throw(ArgumentError("Unknown format: $format. Use :qasm, :qiskit, or :txt"))
    end

    open(filename, "w") do f
        write(f, content)
    end

    return nothing
end
