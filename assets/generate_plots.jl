#!/usr/bin/env julia
# Generate showcase plots for the TenSynth README.
# Run from the TenSynth package root:
#   julia --project=. assets/generate_plots.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TenSynth
using TenSynth.Core
using TenSynth.Core: to_matrix
using TenSynth.MPS
using TenSynth.MPS: synthesize, synthesize_su4
using TenSynth.iMPS
using TenSynth.iMPS: apply_circuit!, infidelity, generate_training_states
using TenSynth.iMPS: compile_unitary, UnitaryCompilationConfig
using TenSynth.iMPS: is_two_qubit
using TenSynth.Hamiltonians
using TenSynth.Hamiltonians: trotterize_tfim

using ITensors
using ITensorMPS

using LinearAlgebra
using Random
using Plots
gr()

const ASSETS_DIR = @__DIR__
const SIZE = (650, 400)

# ── Helpers ──────────────────────────────────────────────────────────────────

"""
Compute average infidelity between two iMPS circuits applied to product states.
"""
function compute_avg_infidelity(circuit, ref_circuit, eval_states, bond_config)
    total = 0.0
    for psi_init in eval_states
        psi_test = deepcopy(psi_init)
        psi_ref = deepcopy(psi_init)
        apply_circuit!(psi_test, circuit, bond_config)
        apply_circuit!(psi_ref, ref_circuit, bond_config)
        total += infidelity(psi_test, psi_ref)
    end
    return total / length(eval_states)
end

"""
Count T gates in an iMPS ParameterizedCircuit via synthesis estimation.
"""
function count_t_gates_imps(circuit, epsilon)
    total = 0
    for gate in circuit.gates
        if is_two_qubit(gate)
            U = to_matrix(gate)
            _, n_t, _ = synthesize_su4(U, epsilon)
            total += n_t
        else
            # Single-qubit: 3 RZ rotations
            t_per_rz = ceil(Int, 3 * log2(1 / epsilon))
            total += 3 * t_per_rz
        end
    end
    return total
end

# ── Plot 1: MPS TFIM Ground State — Fidelity vs Layers ──────────────────────

function plot_mps_fidelity_vs_layers()
    Random.seed!(42)
    N = 67
    h_field = 1.001
    chi_dmrg = 64
    layers = [1, 2, 3, 4]

    # Step 1: DMRG via ITensors
    println("  Running DMRG (N=$N, h=$h_field, chi=$chi_dmrg)...")
    sites = siteinds("S=1/2", N)
    os = OpSum()
    for j in 1:(N - 1)
        os += -1.0, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
        os += -h_field, "Sx", j
    end
    H_it = ITensorMPS.MPO(os, sites)
    psi0 = random_mps(sites; linkdims=chi_dmrg)
    sweeps = Sweeps(10)
    setmaxdim!(sweeps, chi_dmrg)
    setcutoff!(sweeps, 1e-10)
    energy, psi = dmrg(H_it, psi0, sweeps; outputlevel=0)
    println("  DMRG energy: $(round(energy, digits=6))")

    # Step 2: Convert to TenSynth FiniteMPS
    mps_target = TenSynth.MPS.from_itensors(psi)

    # Step 3: Decompose at increasing layers
    fidelities = Float64[]
    decomp_results = []
    for d in layers
        println("  Decomposing at depth $d...")
        Random.seed!(100 + d)
        result = decompose(mps_target;
            method=:iterative,
            max_layers=d,
            n_sweeps=50,
            target_fidelity=1.0,
            verbose=false
        )
        push!(fidelities, result.fidelity)
        push!(decomp_results, result)
        println("    fidelity = $(round(result.fidelity, digits=6))")
    end

    # Step 4: Plot
    p = plot(layers, fidelities,
        xlabel="Circuit layers", ylabel="Fidelity",
        title="TFIM Ground State Compilation (N=$N, h=$h_field)",
        marker=:circle, linewidth=2.5, markersize=7,
        color=:steelblue, legend=false,
        size=SIZE, titlefontsize=12, guidefontsize=11,
        margin=5Plots.mm)

    outpath = joinpath(ASSETS_DIR, "mps_fidelity_vs_layers.png")
    savefig(p, outpath)
    println("  Saved: $outpath")
    return fidelities, decomp_results
end

# ── Plot 2: MPS TFIM Ground State — Fidelity vs T Gates ─────────────────────

function plot_mps_fidelity_vs_tgates(fidelities, decomp_results)
    epsilon = 1e-2
    layers = [1, 2, 3, 4]
    t_gate_counts = Int[]

    for (i, result) in enumerate(decomp_results)
        println("  Synthesizing depth $(layers[i]) circuit (epsilon=$epsilon)...")
        synth_result = synthesize(result.circuit, result.circuit_inds;
            epsilon=epsilon,
            remove_redundancy=true,
            verbose=false
        )
        push!(t_gate_counts, synth_result.n_t_gates)
        println("    T gates = $(synth_result.n_t_gates)")
    end

    p = plot(t_gate_counts, fidelities,
        xlabel="T gates", ylabel="Fidelity",
        title="TFIM Ground State: Fidelity vs T-Gate Cost",
        marker=:circle, linewidth=2.5, markersize=7,
        color=:coral, legend=false,
        size=SIZE, titlefontsize=12, guidefontsize=11,
        margin=5Plots.mm)

    for (i, (t, f)) in enumerate(zip(t_gate_counts, fidelities))
        annotate!(p, t, f + 0.005, text("d=$(layers[i])", 9, :center))
    end

    outpath = joinpath(ASSETS_DIR, "mps_fidelity_vs_tgates.png")
    savefig(p, outpath)
    println("  Saved: $outpath")
end

# ── Plot 3: iMPS Unitary Compilation — Error vs RZZ per Bond ────────────────

function plot_imps_error_vs_rzz()
    Random.seed!(789)
    unit_cell = 2
    J = 1.0
    h = 1.0
    t = 1.0
    n_eval = 16
    chi_sim = 32

    bond_config = BondConfig(chi_sim, 1e-10)

    # Step 1: Deep reference Trotter (100 steps)
    println("  Creating reference Trotter circuit (100 steps)...")
    ref_circuit = trotterize_tfim(unit_cell, t, J, h; order=:second, n_steps=100)

    # Step 2: Fixed evaluation product states
    Random.seed!(111)
    eval_states = generate_training_states(unit_cell, n_eval)

    # Step 3: Trotter error line (n_steps = 1..10)
    trotter_steps = 1:10
    trotter_errors = Float64[]
    trotter_circuits = []
    println("  Computing Trotter errors...")
    for n in trotter_steps
        circ = trotterize_tfim(unit_cell, t, J, h; order=:second, n_steps=n)
        err = compute_avg_infidelity(circ, ref_circuit, eval_states, bond_config)
        push!(trotter_errors, err)
        push!(trotter_circuits, circ)
        println("    n_steps=$n: error = $(round(err, sigdigits=3))")
    end

    # Step 4: Optimized circuits (depths 2, 3, 4)
    opt_depths = [2, 3, 4]
    opt_errors = Float64[]
    opt_circuits = []
    for d in opt_depths
        println("  Optimizing from Trotter depth $d...")
        trotter_init = trotterize_tfim(unit_cell, t, J, h; order=:second, n_steps=d)
        ansatz = deepcopy(trotter_init)

        config = UnitaryCompilationConfig(;
            n_train=8,
            n_test=n_eval,
            max_chi=chi_sim,
            max_iter=200,
            verbose=false
        )
        result = compile_unitary(ref_circuit, ansatz, unit_cell; config=config)

        err = compute_avg_infidelity(result.circuit, ref_circuit, eval_states, bond_config)
        push!(opt_errors, err)
        push!(opt_circuits, result.circuit)
        println("    optimized error = $(round(err, sigdigits=3))")
    end

    # Step 5: Plot (log-log)
    trotter_x = collect(Float64, trotter_steps)
    opt_x = collect(Float64, opt_depths)

    p = plot(trotter_x, trotter_errors,
        xlabel="RZZ gates per bond",
        ylabel="Unitary error (avg. infidelity)",
        title="iMPS Unitary Compilation: TFIM (h=$h, t=$t)",
        marker=:circle, linewidth=2.5, markersize=5,
        color=:steelblue, label="Trotter (2nd order)",
        xscale=:log10, yscale=:log10,
        size=SIZE, titlefontsize=12, guidefontsize=11,
        legendfontsize=9, legend=:topright,
        margin=5Plots.mm)

    scatter!(p, opt_x, opt_errors,
        marker=:star5, markersize=10, color=:crimson,
        label="Optimized (Trotter init)")

    outpath = joinpath(ASSETS_DIR, "imps_error_vs_rzz.png")
    savefig(p, outpath)
    println("  Saved: $outpath")
    return trotter_errors, trotter_circuits, opt_errors, opt_circuits, trotter_steps, opt_depths, unit_cell
end

# ── Plot 4: iMPS Unitary Compilation — Error vs T Gates ──────────────────────

function plot_imps_error_vs_tgates(trotter_errors, trotter_circuits,
                                    opt_errors, opt_circuits,
                                    trotter_steps, opt_depths, unit_cell)
    epsilon = 1e-2

    println("  Counting T gates for Trotter circuits...")
    trotter_t_gates = Float64[]
    for (i, circ) in enumerate(trotter_circuits)
        n_t = count_t_gates_imps(circ, epsilon)
        t_per_qubit = n_t / unit_cell
        push!(trotter_t_gates, t_per_qubit)
        println("    n_steps=$(trotter_steps[i]): T gates = $n_t ($(t_per_qubit)/qubit)")
    end

    println("  Counting T gates for optimized circuits...")
    opt_t_gates = Float64[]
    for (i, circ) in enumerate(opt_circuits)
        n_t = count_t_gates_imps(circ, epsilon)
        t_per_qubit = n_t / unit_cell
        push!(opt_t_gates, t_per_qubit)
        println("    depth=$(opt_depths[i]): T gates = $n_t ($(t_per_qubit)/qubit)")
    end

    p = plot(trotter_t_gates, trotter_errors,
        xlabel="T gates per qubit",
        ylabel="Unitary error (avg. infidelity)",
        title="iMPS: Error vs T-Gate Cost",
        marker=:circle, linewidth=2.5, markersize=5,
        color=:steelblue, label="Trotter",
        xscale=:log10, yscale=:log10,
        size=SIZE, titlefontsize=12, guidefontsize=11,
        legendfontsize=9, legend=:topright,
        margin=5Plots.mm)

    scatter!(p, opt_t_gates, opt_errors,
        marker=:star5, markersize=10, color=:crimson,
        label="Optimized")

    outpath = joinpath(ASSETS_DIR, "imps_error_vs_tgates.png")
    savefig(p, outpath)
    println("  Saved: $outpath")
end

# ── Main ─────────────────────────────────────────────────────────────────────

println("Generating TenSynth showcase plots...")
println()

println("Plot 1/4: MPS TFIM Ground State - Fidelity vs Layers")
fids, decomp_results = plot_mps_fidelity_vs_layers()
println()

println("Plot 2/4: MPS TFIM Ground State - Fidelity vs T Gates")
plot_mps_fidelity_vs_tgates(fids, decomp_results)
println()

println("Plot 3/4: iMPS Unitary Compilation - Error vs RZZ per Bond")
trotter_errs, trotter_circs, opt_errs, opt_circs, t_steps, o_depths, uc = plot_imps_error_vs_rzz()
println()

println("Plot 4/4: iMPS Unitary Compilation - Error vs T Gates per Qubit")
plot_imps_error_vs_tgates(trotter_errs, trotter_circs, opt_errs, opt_circs, t_steps, o_depths, uc)
println()

println("Done! All plots saved to $ASSETS_DIR")
