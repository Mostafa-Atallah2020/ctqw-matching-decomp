#!/usr/bin/env python3
"""
Operator Difference Norm Analysis Script

This script analyzes operator norm differences between quantum walk implementations:
1. Trotterized quantum walk using Matching decomposition vs exact CTQW
2. Trotterized quantum walk using Pauli decomposition vs exact CTQW  
"""

import argparse
import itertools
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from scipy.linalg import expm

# Handle different directory structures
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

# If analysis/, src/, and data/ are at the same level (structure: base/{analysis, src, data})
project_root = parent_dir
src_path = os.path.join(parent_dir, 'src')

sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

try:
    from src.graphs import Graph, PowerOf2EdgeGraph
except ImportError:
    # Fallback if src.graphs doesn't work
    try:
        from graphs import Graph, PowerOf2EdgeGraph
    except ImportError as e:
        print(f"Error importing graph classes: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Project root: {project_root}")
        print(f"Src path: {src_path}")
        print(f"Python path: {sys.path}")
        print("Please ensure that src/graphs.py exists and all dependencies are available.")
        print("You may need to install missing dependencies or fix import paths in src/graphs.py")
        sys.exit(1)


def generate_test_graphs(n_vertices, n_graphs=100, seed=None):
    """Generate test graphs and save them to files."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    graphs = []
    while len(graphs) < n_graphs:
        g = nx.gnm_random_graph(n_vertices, random.randint(n_vertices, n_vertices*(n_vertices-1)//2))
        if nx.is_connected(g) and all(nx.is_isomorphic(g, h) == False for h in graphs):
            graphs.append(g)
    return graphs


def save_graphs(graphs, n_vertices, n_graphs, output_dir, verbose=True):
    """Save graphs in graph6 format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{n_graphs}graph_random_{n_vertices}c.g6"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        for graph in graphs:
            f.write(nx.to_graph6_bytes(graph, header=False).decode('ascii'))
    
    if verbose:
        print(f"Saved {len(graphs)} graphs to {filepath}")
    return filepath


def unitary_to_pauli(U, tolerance=1e-10):
    """Convert a unitary matrix to Pauli decomposition with optional truncation"""
    # Handle both numpy arrays and Operator objects
    if hasattr(U, 'data'):
        U_matrix = U.data
    else:
        U_matrix = U
    
    n = int(np.log2(U_matrix.shape[0]))
    dim = 2**n
    pauli_strings = []
    coeffs = []
    for pauli_string in [''.join(p) for p in itertools.product('IXYZ', repeat=n)]:
        P = Pauli(pauli_string)
        P_op = Operator(P).data
        coeff = np.trace(P_op.conj().T @ U_matrix) / dim
        if not np.isclose(coeff, 0, atol=tolerance):
            pauli_strings.append(pauli_string)
            coeffs.append(coeff)
    return SparsePauliOp(pauli_strings, coeffs)


def trotterized_quantum_walk(graph, time, n_steps):
    """Original trotterized quantum walk using subgraph decomposition"""
    edge_set = set(graph.edges())
    power_of_2_graph = PowerOf2EdgeGraph(edge_set)
    subgraphs = power_of_2_graph.subgraphs
    delta_t = time / n_steps
    
    n_qubits = power_of_2_graph.n_qubits
    n_states = 2**n_qubits
    time_evo_op = np.eye(n_states, dtype=complex)
    
    for _ in range(n_steps):
        for subgraph in subgraphs:
            # Create a mapping from subgraph nodes to indices
            node_to_index = {node: i for i, node in enumerate(subgraph.nodes())}
            subgraph_size = len(node_to_index)
            
            # Create the adjacency matrix for the subgraph
            adj_matrix = np.zeros((subgraph_size, subgraph_size), dtype=complex)
            for u, v in subgraph.edges():
                i, j = node_to_index[u], node_to_index[v]
                adj_matrix[i, j] = adj_matrix[j, i] = 1
            
            # Compute the unitary for the subgraph
            subgraph_unitary = expm(-1j * adj_matrix * delta_t)
            
            # Create the full unitary by embedding the subgraph unitary
            full_unitary = np.eye(n_states, dtype=complex)
            for i, node_i in enumerate(subgraph.nodes()):
                for j, node_j in enumerate(subgraph.nodes()):
                    full_unitary[node_i, node_j] = subgraph_unitary[i, j]
            
            time_evo_op = np.dot(full_unitary, time_evo_op)
    
    return Operator(time_evo_op)


def trotterized_pauli_decomp(graph, time, n_steps):
    """Trotterized quantum walk using first-order Trotter decomposition of Pauli terms"""
    static_graph = Graph(set(graph.edges()))
    n_qubits = static_graph.n_qubits
    n_states = 2**n_qubits
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    
    # Pad adjacency matrix if necessary
    if adj_matrix.shape[0] < n_states:
        padded_adj = np.zeros((n_states, n_states), dtype=complex)
        padded_adj[:adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix
        adj_matrix = padded_adj
    
    # Decompose adjacency matrix into individual Pauli terms
    pauli_decomp = unitary_to_pauli(adj_matrix)
    
    delta_t = time / n_steps
    time_evo_op = np.eye(n_states, dtype=complex)
    
    # Apply first-order Trotter decomposition: exp(sum_i P_i) ≈ prod_i exp(P_i)
    for _ in range(n_steps):
        step_evolution = np.eye(n_states, dtype=complex)
        
        # Apply each Pauli term separately (first-order Trotter)
        for pauli_string, coeff in zip(pauli_decomp.paulis, pauli_decomp.coeffs):
            P_matrix = Operator(pauli_string).data
            single_pauli_evolution = expm(-1j * coeff * P_matrix * delta_t)
            step_evolution = np.dot(single_pauli_evolution, step_evolution)
        
        time_evo_op = np.dot(step_evolution, time_evo_op)
    
    return Operator(time_evo_op)


def original_ctqw(graph, time):
    """Original continuous-time quantum walk"""
    static_graph = Graph(set(graph.edges()))
    n_qubits = static_graph.n_qubits
    n_states = 2**n_qubits
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    time_evo_op = expm(-1j * adj_matrix * time)
    
    # Pad the time evolution operator if necessary
    if time_evo_op.shape[0] < n_states:
        padded_op = np.eye(n_states, dtype=complex)
        padded_op[:time_evo_op.shape[0], :time_evo_op.shape[1]] = time_evo_op
        time_evo_op = padded_op
    
    return Operator(time_evo_op)


def compare_all_walks(graph, time, n_steps):
    """Compare trotterized and pauli methods vs original"""
    trotterized_op = trotterized_quantum_walk(graph, time, n_steps)
    trotterized_pauli_op = trotterized_pauli_decomp(graph, time, n_steps)
    original_op = original_ctqw(graph, time)
    
    # Calculate differences vs original only
    qw_vs_original = np.linalg.norm((trotterized_op - original_op).data, ord=2)
    pauli_vs_original = np.linalg.norm((trotterized_pauli_op - original_op).data, ord=2)
    
    return {
        'qw_vs_original': qw_vs_original,
        'pauli_vs_original': pauli_vs_original
    }


def plot_results(graphs, times, n_steps_list, title, save_dir, verbose=True):
    """Plot results comparing trotterized and pauli methods vs original"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'legend.title_fontsize': 16,
        'figure.figsize': (18, 6),
        'figure.dpi': 300,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.grid': True,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 7,
        'ytick.major.size': 7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
    })
    
    # Calculate results for both comparisons
    results = {
        t: {
            N: {
                'qw_vs_original': [],
                'pauli_vs_original': []
            } for N in n_steps_list
        } for t in times
    }
    
    if verbose:
        print(f"Analyzing {len(graphs)} graphs for {title}...")
    for i, graph in enumerate(graphs):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(graphs)} graphs")
        for t in times:
            for N in n_steps_list:
                comparison_results = compare_all_walks(graph, t, N)
                for key, value in comparison_results.items():
                    results[t][N][key].append(value)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Comparison types and their labels
    comparisons = [
        ('qw_vs_original', 'Matchings CTQW vs exact CTQW', axes[0]),
        ('pauli_vs_original', 'Pauli Decomposition vs exact CTQW', axes[1])
    ]
    
    for comp_key, comp_label, ax in comparisons:
        for i, t in enumerate(times):
            N_values = sorted(n_steps_list)
            mean_diff = [np.mean(results[t][N][comp_key]) for N in N_values]
            std_diff = [np.std(results[t][N][comp_key]) for N in N_values]
            
            # Use standard deviation for error bars
            yerr_low = []
            yerr_high = []
            for m, s in zip(mean_diff, std_diff):
                yerr_high.append(s)
                lower_limit = max(s, m - m*0.9)
                yerr_low.append(lower_limit)
            
            ax.errorbar(N_values, mean_diff, 
                       yerr=[yerr_low, yerr_high],
                       fmt='-o',
                       capsize=6,
                       capthick=2,
                       label=f't = {t}',
                       color=colors[i],
                       elinewidth=2,
                       markeredgewidth=2)
        
        ax.set_yscale('log')
        ax.set_xlabel('Number of Trotter Steps (N)')
        ax.set_ylabel('Operator Norm Difference')
        ax.set_title(comp_label)
        ax.grid(True, which='both', linestyle=':', alpha=0.2)
    
    # Create a single legend outside the plot area
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
              loc='center right', 
              bbox_to_anchor=(0.98, 0.5),
              frameon=True, 
              fancybox=True, 
              shadow=True)
    
    # Add overall title
    fig.suptitle(title, fontsize=24, y=1.02)
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)
    
    # Save the plot
    filename = title.lower().replace(' ', '_').replace('-', '_') + '.pdf'
    filepath = save_dir / filename
    fig.savefig(filepath, 
                format='pdf', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1)
    
    if verbose:
        print(f"Plot saved as: {filepath}")
    plt.close(fig)  # Close to free memory
    return results


def print_summary_statistics(results_dict, verbose=True):
    """Print summary statistics for all results"""
    if not verbose:
        return
        
    print("\nSummary Statistics:")
    print("==================")
    
    for graph_size, results in results_dict.items():
        print(f"\n{graph_size} graphs:")
        times = sorted(results.keys())
        for t in times:
            for N in [10, 50]:  # Show results for two representative step counts
                if N in results[t]:
                    trotter_orig = np.mean(results[t][N]['qw_vs_original'])
                    pauli_orig = np.mean(results[t][N]['pauli_vs_original'])
                    print(f"  t={t}, N={N}: Trotter-Original={trotter_orig:.2e}, Pauli-Original={pauli_orig:.2e}")


def main():
    # Set default directories based on structure: base/{analysis, src, data}
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)
    default_graphs_dir = os.path.join(parent_dir, 'data', 'graphs')
    default_plots_dir = os.path.join(parent_dir, 'data', 'plots')

    parser = argparse.ArgumentParser(description='Analyze operator norm differences between quantum walk implementations')
    
    # Graph generation parameters
    parser.add_argument('--vertex-sizes', nargs='+', type=int, default=[8, 16, 32],
                        help='List of vertex sizes for graphs (default: 8 16 32)')
    parser.add_argument('--n-graphs', type=int, default=100,
                        help='Number of graphs to generate per vertex size (default: 100)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Analysis parameters
    parser.add_argument('--times', nargs='+', type=float, default=[0.1, 0.5, 1.0],
                        help='Time values for quantum walk (default: 0.1 0.5 1.0)')
    parser.add_argument('--n-steps', nargs='+', type=int, default=[5, 10, 20, 50, 100],
                        help='Number of Trotter steps (default: 5 10 20 50 100)')
    
    # Output directories
    parser.add_argument('--graphs-dir', type=str, default=default_graphs_dir,
                        help=f'Directory to save generated graphs (default: {default_graphs_dir})')
    parser.add_argument('--plots-dir', type=str, default=default_plots_dir,
                        help=f'Directory to save plots (default: {default_plots_dir})')
    
    # Control options
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip graph generation (use existing graphs)')
    parser.add_argument('--verbose', '-v', choices=['true', 'false'], default='true',
                        help='Verbose output (default: true)')
    
    args = parser.parse_args()
    
    # Convert verbose string to boolean
    args.verbose = args.verbose.lower() == 'true'
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Vertex sizes: {args.vertex_sizes}")
        print(f"  Number of graphs per size: {args.n_graphs}")
        print(f"  Times: {args.times}")
        print(f"  Trotter steps: {args.n_steps}")
        print(f"  Seed: {args.seed}")
        print(f"  Graphs directory: {args.graphs_dir}")
        print(f"  Plots directory: {args.plots_dir}")
    
    all_results = {}
    
    for n_vertices in args.vertex_sizes:
        print(f"\n{'='*50}")
        print(f"Processing {n_vertices}-vertex graphs")
        print(f"{'='*50}")
        
        if not args.skip_generation:
            if args.verbose:
                print(f"Generating {args.n_graphs} random connected graphs with {n_vertices} vertices...")
            graphs = generate_test_graphs(n_vertices, args.n_graphs, args.seed)
            save_graphs(graphs, n_vertices, args.n_graphs, args.graphs_dir, args.verbose)
        else:
            # Load existing graphs if needed
            graphs_file = Path(args.graphs_dir) / f"{args.n_graphs}graph_random_{n_vertices}c.g6"
            if graphs_file.exists():
                if args.verbose:
                    print(f"Loading graphs from {graphs_file}")
                graphs = list(nx.read_graph6(graphs_file))
            else:
                if args.verbose:
                    print(f"Graph file {graphs_file} not found, generating new graphs...")
                graphs = generate_test_graphs(n_vertices, args.n_graphs, args.seed)
                save_graphs(graphs, n_vertices, args.n_graphs, args.graphs_dir, args.verbose)
        
        if args.verbose:
            print(f"Running analysis on {len(graphs)} graphs...")
        title = f'{n_vertices}-Vertex Graphs'
        results = plot_results(graphs, args.times, args.n_steps, title, args.plots_dir, args.verbose)
        all_results[f"{n_vertices}-vertex"] = results
    
    # Print overall summary
    print_summary_statistics(all_results, args.verbose)
    if args.verbose:
        print(f"\nOperator difference norm analysis complete! Plots saved to {args.plots_dir}")


if __name__ == "__main__":
    main()