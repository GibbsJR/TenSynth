# Configuration structures for MPS optimization
# Adapted from MPS2Circuit/src/config.jl

"""
    HyperParams

Configuration struct for optimization and decomposition hyperparameters.

# Fields
- `N::Int`: Number of qubits (required)
- `max_chi::Int = 64`: Maximum bond dimension for MPS truncation
- `max_trunc_err::Float64 = 1e-10`: SVD truncation threshold
- `n_sweeps::Int = 100`: Number of optimization sweeps
- `n_layer_sweeps::Int = 5`: Number of sweeps per layer during optimization
- `slowdown::Float64 = 0.6`: Learning rate for polar_unitary updates (0=full update, 1=no update)
- `converge_threshold::Float64 = 1e-6`: Convergence threshold for fidelity improvement
- `print_every::Int = 10`: Print progress every N sweeps (0=silent)
"""
Base.@kwdef struct HyperParams
    N::Int
    max_chi::Int = 64
    max_trunc_err::Float64 = 1e-10
    n_sweeps::Int = 100
    n_layer_sweeps::Int = 5
    slowdown::Float64 = 0.6
    converge_threshold::Float64 = 1e-6
    print_every::Int = 10

    function HyperParams(N, max_chi, max_trunc_err, n_sweeps, n_layer_sweeps,
                         slowdown, converge_threshold, print_every)
        N > 0 || throw(ArgumentError("N must be positive, got $N"))
        max_chi > 0 || throw(ArgumentError("max_chi must be positive, got $max_chi"))
        max_trunc_err >= 0 || throw(ArgumentError("max_trunc_err must be non-negative, got $max_trunc_err"))
        n_sweeps > 0 || throw(ArgumentError("n_sweeps must be positive, got $n_sweeps"))
        n_layer_sweeps > 0 || throw(ArgumentError("n_layer_sweeps must be positive, got $n_layer_sweeps"))
        0 <= slowdown <= 1 || throw(ArgumentError("slowdown must be in [0, 1], got $slowdown"))
        converge_threshold >= 0 || throw(ArgumentError("converge_threshold must be non-negative, got $converge_threshold"))
        print_every >= 0 || throw(ArgumentError("print_every must be non-negative, got $print_every"))
        new(N, max_chi, max_trunc_err, n_sweeps, n_layer_sweeps,
            slowdown, converge_threshold, print_every)
    end
end

function Base.show(io::IO, hp::HyperParams)
    print(io, "HyperParams(N=$(hp.N), max_chi=$(hp.max_chi), n_sweeps=$(hp.n_sweeps), slowdown=$(hp.slowdown))")
end

function Base.show(io::IO, ::MIME"text/plain", hp::HyperParams)
    println(io, "HyperParams:")
    println(io, "  N               = $(hp.N)")
    println(io, "  max_chi         = $(hp.max_chi)")
    println(io, "  max_trunc_err   = $(hp.max_trunc_err)")
    println(io, "  n_sweeps        = $(hp.n_sweeps)")
    println(io, "  n_layer_sweeps  = $(hp.n_layer_sweeps)")
    println(io, "  slowdown        = $(hp.slowdown)")
    println(io, "  converge_threshold = $(hp.converge_threshold)")
    print(io, "  print_every     = $(hp.print_every)")
end
