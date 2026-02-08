# TenSynth.jl

A Julia package for tensor network circuit synthesis — compiling quantum states and operators into quantum circuits using MPS, MPO, and infinite MPS representations.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/GibbsJR/TenSynth.git")
```

## Modules

- **Core** — Shared types, gate constants, parameterizations, and linear algebra utilities
- **MPS** — Finite MPS state-to-circuit compilation (variational optimization, analytical decomposition, layer addition)
- **MPO** — MPO unitary-to-circuit compilation (HST cost optimization)
- **iMPS** — Infinite MPS ground state and unitary compilation (iTEBD, Trotter circuits)
- **Hamiltonians** — Hamiltonian definitions (TFIM, Heisenberg, XY) and Trotterization

### PyCall Extension

Optional Clifford+T synthesis via `GridSynth` and `TraSynth` (requires PyCall and corresponding Python packages).

## Usage

```julia
using TenSynth

# MPS state compilation
using TenSynth.MPS

# MPO unitary compilation
using TenSynth.MPO

# Infinite MPS
using TenSynth.iMPS

# Hamiltonians and Trotter decomposition
using TenSynth.Hamiltonians
```

See the [examples/](examples/) directory for Jupyter notebooks demonstrating each module.

## Tests

```julia
using Pkg
Pkg.test("TenSynth")
```

## Requirements

Julia 1.10+
