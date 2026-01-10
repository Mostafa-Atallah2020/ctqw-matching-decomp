Code repository for the paper:

> **Simulating Quantum Walk Hamiltonians without Pauli Decomposition**
>
> Mostafa Atallah¹, Alvin Gonzales², Daniel Dilley², Igor Gaidai¹, Zain H. Saleem², Rebekah Herrman¹
>
> ¹ University of Tennessee Knoxville, ² Argonne National Laboratory

## Overview

This project compares two methods for implementing CTQW Hamiltonians on quantum computers:
- **Matching Decomposition**: Groups edges into matchings for parallel execution
- **Pauli Decomposition**: Decomposes the Hamiltonian into Pauli strings

## Usage

### Generate Erdos-Renyi Graphs

```bash
cd analysis
python generate_erdos_renyi_graphs.py --vertices 4 8 16 32 64 128 -n 100 -p 0.01 0.02 0.03 --seed 42
```

### Run CX Count Comparison

```bash
python cx_count_erdos_renyi_compute.py graphs/erdos_renyi/*.g6 -o outputs/cx_scaling/erdos_renyi
```

### Generate Plots

```bash
python cx_count_erdos_renyi_plot.py outputs/cx_scaling/erdos_renyi/p0_01
```

## Project Structure

```
dyn-CTQW/
├── src/
│   ├── core/
│   │   ├── multi_edge_graph.py          # MultiEdgeGraph class
│   │   └── decompositions/
│   │       ├── matching.py              # Matching decomposition
│   │       └── pauli.py                 # Pauli decomposition
│   └── utils/
│       ├── circuit/                     # Circuit utilities
│       │   ├── exact_evolution.py       # Exact evolution operators
│       │   ├── gate_ops.py              # Gate operations
│       │   └── space_reduction.py       # Space reduction techniques
│       └── graph/                       # Graph utilities
│           ├── drawer.py                # Graph visualization
│           ├── g6_utils.py              # g6 format utilities
│           └── properties.py            # Graph property calculations
├── analysis/
│   ├── generate_erdos_renyi_graphs.py   # Erdos-Renyi graph generator
│   ├── cx_count_erdos_renyi_compute.py  # CX count computation
│   ├── cx_count_erdos_renyi_plot.py     # CX scaling plots
│   ├── trotterization_error_compute.py  # Trotter error analysis
│   ├── trotterization_error_plot.py     # Trotter error plots
│   ├── graphs/                          # Generated graphs (.g6)
│   └── outputs/                         # Results storage
├── examples/
│   ├── CTQW trotterization.ipynb        # Trotterization examples
│   └── matching and pauli decompositions.ipynb
├── test/                                # Unit tests
└── data/graphs/                         # Graph datasets in g6 format
```

## Citation

```bibtex
@article{atallah2025simulating,
  title={Simulating Quantum Walk Hamiltonians without Pauli Decomposition},
  author={Atallah, Mostafa and Gonzales, Alvin and Dilley, Daniel and Gaidai, Igor and Saleem, Zain H. and Herrman, Rebekah},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

For questions or issues, please contact: matalla3@vols.utk.edu