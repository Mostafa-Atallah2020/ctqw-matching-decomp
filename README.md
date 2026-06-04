Code repository for the paper:

> **A matching decomposition algorithm for simulating quantum walk Hamiltonians**
>
> Mostafa Atallah¹ʼ³, Alvin Gonzales², Daniel Dilley², Igor Gaidai¹ʼ⁴, Zain H. Saleem², Rebekah Herrman¹
>
> ¹ University of Tennessee Knoxville, USA · ² Argonne National Laboratory, Lemont, IL, USA · ³ Cairo University, Giza, Egypt · ⁴ University of Tennessee Chattanooga, USA

## Overview

This project compares two methods for implementing CTQW Hamiltonians on quantum computers:
- **Matching Decomposition**: Groups edges into matchings for parallel execution
- **Pauli Decomposition**: Decomposes the Hamiltonian into Pauli strings

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/Mostafa-Atallah2020/ctqw-matching-decomp.git
```

Or clone and install in editable mode (recommended for development):

```bash
git clone https://github.com/Mostafa-Atallah2020/ctqw-matching-decomp.git
cd ctqw-matching-decomp
pip install -e ".[dev]"        # omit [dev] to skip test/lint tools
```

Once installed, the package is importable from anywhere:

```python
from ctqw_matching_decomp import MultiEdgeGraph, MatchingDecomposition, PauliDecomposition
```

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
ctqw-matching-decomp/
├── ctqw_matching_decomp/                   # Installable Python package
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
@article{atallah2025matching,
  title={A matching decomposition algorithm for simulating quantum walk Hamiltonians},
  author={Atallah, Mostafa and Gonzales, Alvin and Dilley, Daniel and Gaidai, Igor and Saleem, Zain H. and Herrman, Rebekah},
  journal={arXiv preprint arXiv:2601.11418},
  year={2025}
}
```

## Contact

For questions or issues, please contact: matalla3@vols.utk.edu