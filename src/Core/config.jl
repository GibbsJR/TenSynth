# Shared configuration types

struct BondConfig
    max_chi::Int
    max_trunc_err::Float64

    function BondConfig(max_chi::Int, max_trunc_err::Float64)
        max_chi > 0 || throw(ArgumentError("max_chi must be positive"))
        max_trunc_err >= 0 || throw(ArgumentError("max_trunc_err must be non-negative"))
        new(max_chi, max_trunc_err)
    end
end

BondConfig(max_chi::Int) = BondConfig(max_chi, 1e-10)

struct OptimConfig
    algorithm::Symbol
    max_iterations::Int
    convergence_threshold::Float64
    gradient_delta::Float64
    arnoldi_tol::Float64
    lbfgs_memory::Int

    function OptimConfig(
        algorithm::Symbol,
        max_iterations::Int,
        convergence_threshold::Float64,
        gradient_delta::Float64,
        arnoldi_tol::Float64,
        lbfgs_memory::Int
    )
        algorithm in (:LBFGS, :Adam, :GD) || throw(ArgumentError("Unknown algorithm: $algorithm"))
        max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
        convergence_threshold > 0 || throw(ArgumentError("convergence_threshold must be positive"))
        gradient_delta > 0 || throw(ArgumentError("gradient_delta must be positive"))
        arnoldi_tol > 0 || throw(ArgumentError("arnoldi_tol must be positive"))
        lbfgs_memory > 0 || throw(ArgumentError("lbfgs_memory must be positive"))
        new(algorithm, max_iterations, convergence_threshold, gradient_delta, arnoldi_tol, lbfgs_memory)
    end
end

function OptimConfig(;
    algorithm::Symbol = :LBFGS,
    max_iterations::Int = 10000,
    convergence_threshold::Float64 = 1e-6,
    gradient_delta::Float64 = 1e-6,
    arnoldi_tol::Float64 = 1e-8,
    lbfgs_memory::Int = 32
)
    OptimConfig(algorithm, max_iterations, convergence_threshold, gradient_delta, arnoldi_tol, lbfgs_memory)
end
