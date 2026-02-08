# Optimization routines for circuit compilation
# Adapted from iMPS2Circuit/src/Algorithms/Optimization.jl

using LinearAlgebra
using Optim
using LineSearches
using Random

# ============================================================================
# Compilation Problem Structure
# ============================================================================

mutable struct CompilationProblem{T<:Number}
    circuit::ParameterizedCircuit
    psi_inits_train::Vector{iMPSType{T}}
    psi_targets_train::Vector{iMPSType{T}}
    psi_inits_test::Vector{iMPSType{T}}
    psi_targets_test::Vector{iMPSType{T}}
    config::BondConfig
    delta::Float64
    tol::Float64

    # Optimization state
    best_params::Vector{Float64}
    best_train_cost::Float64
    best_test_cost::Float64
    iteration::Int

    # History
    train_history::Vector{Float64}
    test_history::Vector{Float64}
end

function CompilationProblem(circuit::ParameterizedCircuit, psi_inits::Vector{iMPSType{T}},
                            psi_targets::Vector{iMPSType{T}}, config::BondConfig;
                            test_fraction::Float64=0.25,
                            delta::Float64=1e-6, tol::Float64=1e-10) where T
    n = length(psi_inits)
    if n != length(psi_targets)
        throw(ArgumentError("Number of initial and target states must match"))
    end

    n_test = max(1, round(Int, n * test_fraction))
    n_train = n - n_test

    indices = randperm(n)
    train_idx = indices[1:n_train]
    test_idx = indices[n_train+1:end]

    psi_inits_train = psi_inits[train_idx]
    psi_targets_train = psi_targets[train_idx]
    psi_inits_test = psi_inits[test_idx]
    psi_targets_test = psi_targets[test_idx]

    best_params = copy(get_params(circuit))

    return CompilationProblem{T}(
        circuit,
        psi_inits_train, psi_targets_train,
        psi_inits_test, psi_targets_test,
        config, delta, tol,
        best_params, Inf, Inf, 0,
        Float64[], Float64[]
    )
end

function CompilationProblem(circuit::ParameterizedCircuit,
                            psi_inits_train::Vector{iMPSType{T}}, psi_targets_train::Vector{iMPSType{T}},
                            psi_inits_test::Vector{iMPSType{T}}, psi_targets_test::Vector{iMPSType{T}},
                            config::BondConfig;
                            delta::Float64=1e-6, tol::Float64=1e-10) where T
    best_params = copy(get_params(circuit))

    return CompilationProblem{T}(
        circuit,
        psi_inits_train, psi_targets_train,
        psi_inits_test, psi_targets_test,
        config, delta, tol,
        best_params, Inf, Inf, 0,
        Float64[], Float64[]
    )
end

# ============================================================================
# Cost and Gradient Functions for Optim.jl
# ============================================================================

function _cost_function(params::Vector{Float64}, problem::CompilationProblem)
    set_params!(problem.circuit, params)
    return compute_cost_batch(problem.circuit, problem.psi_inits_train,
                              problem.psi_targets_train, problem.config;
                              tol=problem.tol)
end

function _gradient_function!(grad::Vector{Float64}, params::Vector{Float64},
                             problem::CompilationProblem)
    set_params!(problem.circuit, params)
    compute_gradient_batch!(grad, problem.circuit, problem.psi_inits_train,
                            problem.psi_targets_train, problem.config;
                            method=:backward, delta=problem.delta, tol=problem.tol)
    return grad
end

function _fg_function!(F, G, params::Vector{Float64}, problem::CompilationProblem)
    set_params!(problem.circuit, params)

    if G !== nothing
        compute_gradient_batch!(G, problem.circuit, problem.psi_inits_train,
                                problem.psi_targets_train, problem.config;
                                method=:backward, delta=problem.delta, tol=problem.tol)
    end

    if F !== nothing
        return compute_cost_batch(problem.circuit, problem.psi_inits_train,
                                  problem.psi_targets_train, problem.config;
                                  tol=problem.tol)
    end
end

# ============================================================================
# Optimization Result
# ============================================================================

struct OptimizationResult
    converged::Bool
    final_train_cost::Float64
    final_test_cost::Float64
    iterations::Int
    params::Vector{Float64}
    train_history::Vector{Float64}
    test_history::Vector{Float64}
end

# ============================================================================
# Optim.jl-based optimization
# ============================================================================

"""
    optimize!(problem::CompilationProblem; method::Symbol=:lbfgs, max_iter::Int=100, ...) -> OptimizationResult
"""
function optimize!(problem::CompilationProblem;
                   method::Symbol=:lbfgs,
                   max_iter::Int=100,
                   g_tol::Float64=1e-6,
                   f_tol::Float64=1e-8,
                   show_trace::Bool=true,
                   callback::Union{Nothing, Function}=nothing)

    x0 = copy(get_params(problem.circuit))

    empty!(problem.train_history)
    empty!(problem.test_history)

    has_test = !isempty(problem.psi_inits_test)

    function internal_callback(state)
        problem.iteration = state.iteration

        train_cost = state.value
        push!(problem.train_history, train_cost)

        if has_test
            test_cost = compute_cost_batch(problem.circuit, problem.psi_inits_test,
                                           problem.psi_targets_test, problem.config;
                                           tol=problem.tol)
            push!(problem.test_history, test_cost)
        else
            test_cost = train_cost
        end

        if test_cost < problem.best_test_cost
            problem.best_test_cost = test_cost
            problem.best_train_cost = train_cost
            problem.best_params = copy(get_params(problem.circuit))
        end

        if callback !== nothing
            callback(state.iteration, train_cost, test_cost)
        end

        return false
    end

    if method == :lbfgs
        optimizer = LBFGS(linesearch=BackTracking(order=3))
    elseif method == :gd
        optimizer = GradientDescent(linesearch=BackTracking())
    elseif method == :cg
        optimizer = ConjugateGradient()
    else
        throw(ArgumentError("Unknown method: $method. Use :lbfgs, :gd, or :cg"))
    end

    fg!(F, G, x) = _fg_function!(F, G, x, problem)

    options = Optim.Options(
        iterations=max_iter,
        g_tol=g_tol,
        f_reltol=f_tol,
        show_trace=show_trace,
        show_every=1,
        callback=internal_callback
    )

    result = Optim.optimize(Optim.only_fg!(fg!), x0, optimizer, options)

    set_params!(problem.circuit, problem.best_params)

    return OptimizationResult(
        Optim.converged(result),
        problem.best_train_cost,
        problem.best_test_cost,
        problem.iteration,
        copy(problem.best_params),
        copy(problem.train_history),
        copy(problem.test_history)
    )
end

# ============================================================================
# Simple Gradient Descent
# ============================================================================

function optimize_gd!(problem::CompilationProblem;
                      lr::Float64=0.1,
                      max_iter::Int=100,
                      tol::Float64=1e-6,
                      verbose::Bool=true)

    params = copy(get_params(problem.circuit))
    grad = zeros(Float64, length(params))

    empty!(problem.train_history)
    empty!(problem.test_history)

    has_test = !isempty(problem.psi_inits_test)
    prev_cost = Inf
    converged = false

    for iter in 1:max_iter
        problem.iteration = iter

        set_params!(problem.circuit, params)
        compute_gradient_batch!(grad, problem.circuit, problem.psi_inits_train,
                                problem.psi_targets_train, problem.config;
                                method=:backward, delta=problem.delta, tol=problem.tol)

        params .-= lr .* grad

        set_params!(problem.circuit, params)
        train_cost = compute_cost_batch(problem.circuit, problem.psi_inits_train,
                                        problem.psi_targets_train, problem.config;
                                        tol=problem.tol)
        if has_test
            test_cost = compute_cost_batch(problem.circuit, problem.psi_inits_test,
                                           problem.psi_targets_test, problem.config;
                                           tol=problem.tol)
            push!(problem.test_history, test_cost)
        else
            test_cost = train_cost
        end

        push!(problem.train_history, train_cost)

        if test_cost < problem.best_test_cost
            problem.best_test_cost = test_cost
            problem.best_train_cost = train_cost
            problem.best_params = copy(params)
        end

        if verbose && iter % 10 == 1
            if has_test
                @info "Iteration $iter: train=$train_cost, test=$test_cost"
            else
                @info "Iteration $iter: cost=$train_cost"
            end
        end

        if abs(prev_cost - train_cost) / max(abs(prev_cost), 1e-10) < tol
            converged = true
            if verbose
                @info "Converged at iteration $iter"
            end
            break
        end

        prev_cost = train_cost
    end

    set_params!(problem.circuit, problem.best_params)

    return OptimizationResult(
        converged,
        problem.best_train_cost,
        problem.best_test_cost,
        problem.iteration,
        copy(problem.best_params),
        copy(problem.train_history),
        copy(problem.test_history)
    )
end

# ============================================================================
# Adam Optimizer
# ============================================================================

function optimize_adam!(problem::CompilationProblem;
                        lr::Float64=0.01,
                        beta1::Float64=0.9,
                        beta2::Float64=0.999,
                        eps::Float64=1e-8,
                        max_iter::Int=100,
                        tol::Float64=1e-6,
                        verbose::Bool=true)

    params = copy(get_params(problem.circuit))
    n = length(params)
    grad = zeros(Float64, n)
    m = zeros(Float64, n)
    v = zeros(Float64, n)

    empty!(problem.train_history)
    empty!(problem.test_history)

    has_test = !isempty(problem.psi_inits_test)
    prev_cost = Inf
    converged = false

    for iter in 1:max_iter
        problem.iteration = iter

        set_params!(problem.circuit, params)
        compute_gradient_batch!(grad, problem.circuit, problem.psi_inits_train,
                                problem.psi_targets_train, problem.config;
                                method=:backward, delta=problem.delta, tol=problem.tol)

        m .= beta1 .* m .+ (1 - beta1) .* grad
        v .= beta2 .* v .+ (1 - beta2) .* grad.^2

        m_hat = m ./ (1 - beta1^iter)
        v_hat = v ./ (1 - beta2^iter)

        params .-= lr .* m_hat ./ (sqrt.(v_hat) .+ eps)

        set_params!(problem.circuit, params)
        train_cost = compute_cost_batch(problem.circuit, problem.psi_inits_train,
                                        problem.psi_targets_train, problem.config;
                                        tol=problem.tol)
        if has_test
            test_cost = compute_cost_batch(problem.circuit, problem.psi_inits_test,
                                           problem.psi_targets_test, problem.config;
                                           tol=problem.tol)
            push!(problem.test_history, test_cost)
        else
            test_cost = train_cost
        end

        push!(problem.train_history, train_cost)

        if test_cost < problem.best_test_cost
            problem.best_test_cost = test_cost
            problem.best_train_cost = train_cost
            problem.best_params = copy(params)
        end

        if verbose && iter % 10 == 1
            if has_test
                @info "Iteration $iter: train=$train_cost, test=$test_cost"
            else
                @info "Iteration $iter: cost=$train_cost"
            end
        end

        if abs(prev_cost - train_cost) / max(abs(prev_cost), 1e-10) < tol
            converged = true
            if verbose
                @info "Converged at iteration $iter"
            end
            break
        end

        prev_cost = train_cost
    end

    set_params!(problem.circuit, problem.best_params)

    return OptimizationResult(
        converged,
        problem.best_train_cost,
        problem.best_test_cost,
        problem.iteration,
        copy(problem.best_params),
        copy(problem.train_history),
        copy(problem.test_history)
    )
end
