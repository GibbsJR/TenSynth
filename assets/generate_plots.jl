#!/usr/bin/env julia
# Generate showcase plots for the TenSynth README.
# Run from the TenSynth package root:
#   julia --project=. assets/generate_plots.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TenSynth
using TenSynth.Core
using TenSynth.MPS
using TenSynth.MPS: wMPS, bond_dimensions, entanglement_entropy

using LinearAlgebra
using Random
using Plots
gr()

const ASSETS_DIR = @__DIR__
const SIZE = (650, 400)

# ── Plot 1: Fidelity vs Circuit Depth ─────────────────────────────────────────

function plot_fidelity_vs_depth()
    Random.seed!(42)
    N = 6

    test_states = [
        ("W state",     wMPS(N)),
        ("Random χ=2",  randMPS(N, 2)),
        ("Random χ=4",  randMPS(N, 4)),
        ("Random χ=8",  randMPS(N, 8)),
    ]

    depths = 1:8
    fidelity_vs_depth = Dict{String, Vector{Float64}}()

    for (label, mps) in test_states
        fids = Float64[]
        for d in depths
            result = decompose(mps;
                method=:iterative,
                max_layers=d,
                n_sweeps=50,
                target_fidelity=1.0,
                verbose=false
            )
            push!(fids, result.fidelity)
        end
        fidelity_vs_depth[label] = fids
        println("  $label: depth 1 → $(round(fids[1], digits=4)), depth 8 → $(round(fids[end], digits=4))")
    end

    p = plot(title="Fidelity vs Circuit Depth (N=$N)",
             xlabel="Circuit depth (layers)", ylabel="Fidelity",
             legend=:bottomright, size=SIZE,
             titlefontsize=13, guidefontsize=11, legendfontsize=9,
             margin=5Plots.mm)

    colors = [:steelblue, :seagreen, :darkorange, :crimson]
    markers = [:circle, :diamond, :utriangle, :square]
    for (i, (label, _)) in enumerate(test_states)
        plot!(p, collect(depths), fidelity_vs_depth[label],
              label=label, marker=markers[i], linewidth=2.5,
              markersize=6, color=colors[i])
    end

    hline!(p, [0.99], linestyle=:dash, color=:gray, alpha=0.5, label="99% fidelity")

    outpath = joinpath(ASSETS_DIR, "fidelity_vs_depth.png")
    savefig(p, outpath)
    println("  Saved: $outpath")
    return fidelity_vs_depth, test_states
end

# ── Plot 2: Entropy Predicts Circuit Depth ────────────────────────────────────

function plot_entropy_vs_depth(fidelity_vs_depth, test_states)
    threshold = 0.99
    scatter_data = []

    for (label, mps) in test_states
        max_S = maximum(entanglement_entropy(mps, k) for k in 1:(length(mps.tensors)-1))
        fids = fidelity_vs_depth[label]
        depth_needed = findfirst(f -> f >= threshold, fids)
        if depth_needed === nothing
            depth_needed = length(fids)
        end
        push!(scatter_data, (label=label, max_entropy=max_S, depth=depth_needed))
    end

    max_entropies = [d.max_entropy for d in scatter_data]
    depths_needed = [d.depth for d in scatter_data]

    p = scatter(max_entropies, depths_needed,
        xlabel="Maximum entanglement entropy",
        ylabel="Layers to 99% fidelity",
        title="Entropy Predicts Circuit Depth",
        markersize=10, legend=false,
        size=(550, 400), color=:steelblue,
        titlefontsize=13, guidefontsize=11,
        margin=5Plots.mm)

    for d in scatter_data
        annotate!(p, d.max_entropy + 0.04, d.depth + 0.15,
                  text(d.label, 9, :left))
    end

    outpath = joinpath(ASSETS_DIR, "entropy_vs_depth.png")
    savefig(p, outpath)
    println("  Saved: $outpath")
end

# ── Plot 3: Method Comparison ─────────────────────────────────────────────────

function plot_method_comparison()
    Random.seed!(42)
    N = 6
    chi_values = [2, 3, 4, 6, 8]
    method_comparison = Dict{Symbol, Vector{Float64}}()

    for m in [:analytical, :iterative]
        fids = Float64[]
        for chi in chi_values
            Random.seed!(42)
            mps = randMPS(N, chi)
            result = decompose(mps;
                method=m,
                max_layers=6,
                n_sweeps=50,
                target_fidelity=1.0,
                verbose=false
            )
            push!(fids, result.fidelity)
        end
        method_comparison[m] = fids
        println("  $m: ", join(["χ=$c → $(round(f, digits=4))"
                               for (c, f) in zip(chi_values, fids)], ", "))
    end

    p = plot(title="Method Comparison vs Bond Dimension (N=$N, depth=6)",
             xlabel="Bond dimension χ", ylabel="Fidelity",
             legend=:bottomleft, size=SIZE,
             titlefontsize=13, guidefontsize=11, legendfontsize=9,
             margin=5Plots.mm)

    plot!(p, chi_values, method_comparison[:analytical],
          label="Analytical (SVD)", marker=:circle, linewidth=2.5,
          markersize=7, color=:steelblue)
    plot!(p, chi_values, method_comparison[:iterative],
          label="Iterative (optimized)", marker=:diamond, linewidth=2.5,
          markersize=7, color=:coral)

    hline!(p, [0.99], linestyle=:dash, color=:gray, alpha=0.5, label="99% fidelity")

    outpath = joinpath(ASSETS_DIR, "method_comparison.png")
    savefig(p, outpath)
    println("  Saved: $outpath")
end

# ── Main ──────────────────────────────────────────────────────────────────────

println("Generating TenSynth showcase plots...")
println()

println("Plot 1/3: Fidelity vs Circuit Depth")
fvd, ts = plot_fidelity_vs_depth()
println()

println("Plot 2/3: Entropy Predicts Circuit Depth")
plot_entropy_vs_depth(fvd, ts)
println()

println("Plot 3/3: Method Comparison")
plot_method_comparison()
println()

println("Done! All plots saved to $ASSETS_DIR")
