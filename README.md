# TenSynth.jl

**Compile quantum states and unitaries into quantum circuits using tensor networks.**

- **Variational and analytical** circuit synthesis from Matrix Product States (MPS), Matrix Product Operators (MPO), and infinite MPS (iMPS)
- **Targets both NISQ and fault-tolerant hardware** — native parameterized gates or Clifford+T decomposition
- **Bridge classical and quantum simulation** — tensor networks efficiently capture quantum systems up to moderate entanglement; TenSynth maps these into compact circuits ready to upload to quantum hardware, extending simulations into high-entanglement regimes beyond the reach of classical methods

```mermaid
graph TD
    A["TN calculation produces target state or unitary expressed as MPS/MPO/iMPS"]
    A --> B["TenSynth learns efficient circuit representation,<br/>prioritising low CNOT (NISQ) or T (Fault-Tolerant) gate count"]
    B --> C["Circuit ready to upload to QC for high-entanglement quantum simulation"]
    style B fill:#4a90d9,color:#fff
```

## Showcase

**Compiling the near-critical TFIM ground state (67 qubits) into quantum circuits.**

<p align="center">
  <img src="assets/mps_fidelity_vs_layers.png" width="420" alt="MPS fidelity vs circuit layers"/>
  &nbsp;&nbsp;
  <img src="assets/mps_fidelity_vs_tgates.png" width="420" alt="MPS fidelity vs T-gate count"/>
</p>

<p align="center">
  <em>Left:</em> Fidelity of the compiled TFIM ground state (h=1.001, near the quantum critical point) improves with circuit depth.
  <em>Right:</em> The same data plotted against total T-gate count after Clifford+T synthesis.
</p>

**Optimizing translational-invariant unitaries beyond Trotter.**

<p align="center">
  <img src="assets/imps_error_vs_rzz.png" width="420" alt="iMPS error vs RZZ gates per bond"/>
  &nbsp;&nbsp;
  <img src="assets/imps_error_vs_tgates.png" width="420" alt="iMPS error vs T gates per qubit"/>
</p>

<p align="center">
  <em>Left:</em> Variational optimization (red, initialized from Trotter) achieves lower unitary error than plain second-order Trotter circuits (blue) at the same circuit depth.
  <em>Right:</em> After Clifford+T synthesis, the T-gate advantage of optimized circuits persists.
</p>

## Quick Start

```julia
using TenSynth
using TenSynth.MPS

# Create a random 6-qubit MPS with bond dimension 4
mps = randMPS(6, 4)

# Compile to a quantum circuit
result = decompose(mps; method=:iterative, max_layers=5)

println("Fidelity: ", round(result.fidelity, digits=6))
println("Gates:    ", result.n_gates, " two-qubit + single-qubit layers")

# Export to OpenQASM
qasm = to_qasm(result)
```

## Features

- **MPS state compilation** — analytical (SVD), iterative (variational), and layer-addition protocols
- **MPO unitary compilation** — Hilbert–Schmidt test cost optimization for time-evolution operators
- **Infinite MPS** — ground state preparation and unitary compilation via iTEBD
- **Hamiltonians** — TFIM, Heisenberg, XY, XXZ with 1st/2nd/4th-order Trotter decomposition
- **Circuit topologies** — staircase and brickwork layouts with configurable depth
- **Clifford+T synthesis** — optional GridSynth / Trasyn integration via PyCall

## Modules

| Module | Description |
|--------|-------------|
| `TenSynth.Core` | Gate constants, parameterizations, linear algebra utilities |
| `TenSynth.MPS` | Finite MPS → circuit (decompose, compile, optimize) |
| `TenSynth.MPO` | MPO unitary → circuit (HST cost optimization) |
| `TenSynth.iMPS` | Infinite MPS ground states & unitary compilation |
| `TenSynth.Hamiltonians` | Spin chain Hamiltonians and Trotterization |

## Example Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 1 | [Getting Started](examples/01_getting_started.ipynb) | MPS basics, three decomposition methods, QASM export |
| 2 | [Entanglement & Depth](examples/02_entanglement_and_circuit_depth.ipynb) | Why entanglement determines circuit depth |
| 3 | [DMRG Ground States](examples/03_dmrg_ground_states_to_circuits.ipynb) | TFIM ground states across the phase transition |
| 4 | [MPO Compilation](examples/04_mpo_unitary_compilation.ipynb) | Time-evolution operator compilation |
| 5 | [Hamiltonians & Trotter](examples/05_hamiltonians_and_trotter.ipynb) | Four spin models, Trotter error scaling |
| 6 | [iMPS Ground States](examples/06_imps_ground_states.ipynb) | iTEBD convergence, infinite state preparation |
| 7 | [iMPS Unitaries](examples/07_imps_unitary_compilation.ipynb) | Infinite unitary compilation with train/test splits |
| 8 | [Clifford+T Synthesis](examples/08_clifford_t_synthesis.ipynb) | Fault-tolerant T-gate resource counting |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/GibbsJR/TenSynth.git")
```

Requires **Julia 1.10+**.

## Tests

```julia
using Pkg
Pkg.test("TenSynth")
```

Runs 476 tests across all modules (Core, MPS, MPO, iMPS, Hamiltonians, integration).
