#!/usr/bin/env python3
"""
Operator Difference Norm Analysis Script

This script analyzes operator difference norms between quantum walk implementations:
1. Trotterized quantum walk using First-Fit Algorithm decomposition vs exact CTQW
2. Trotterized quantum walk using Structure-Aware Algorithm decomposition vs exact CTQW  
3. Trotterized quantum walk using Pauli decomposition vs exact CTQW

Loads graphs from ../data/graphs directory.
"""

import argparse
import itertools
import os
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
    from src.graphs import IntersectingEdgesGraph
except ImportError:
    # Fallback if src.graphs doesn't work
    try:
        from graphs import IntersectingEdgesGraph
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


def load_graphs(graphs_dir, n_vertices, n_graphs, verbose=True):
    """Load graphs from graph6 format files."""
    graphs_dir = Path(graphs_dir)
    filename = f"{n_graphs}graph_random_{n_vertices}c.g6"
    filepath = graphs_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    if verbose:
        print(f"Loading graphs from {filepath}")
    
    graphs = list(nx.read_graph6(filepath))
    
    if verbose:
        print(f"Loaded {len(graphs)} graphs with {n_vertices} vertices")
    
    return graphs


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


def trotterized_quantum_walk(graph, time, n_steps, matching_algorithm='first-fit'):
    """Trotterized quantum walk using subgraph decomposition with matching algorithms
    
    Args:
        graph: NetworkX graph
        time: Evolution time
        n_steps: Number of Trotter steps
        matching_algorithm: 'first-fit' or 'structure-aware'
    """
    if matching_algorithm not in ['first-fit', 'structure-aware']:
        raise ValueError(f"Invalid matching algorithm: {matching_algorithm}. "
                       f"Valid options are: 'first-fit', 'structure-aware'")
    
    # Map new names to internal names
    algorithm_mapping = {
        'first-fit': 'greedy',
        'structure-aware': 'parallel'
    }
    internal_algorithm = algorithm_mapping[matching_algorithm]
    
    # Get integer edges from NetworkX graph
    int_edges = set(graph.edges())
    
    # Determine number of qubits needed to represent all nodes
    max_node = max(max(edge) for edge in int_edges) if int_edges else 0
    n_qubits = max_node.bit_length()  # Number of bits needed to represent max_node
    if n_qubits == 0:  # Handle case where max_node is 0
        n_qubits = 1
    
    # Convert integer edges to binary string edges for StaticGraph
    binary_edges = set()
    for u, v in int_edges:
        # Convert integers to binary strings with proper zero-padding
        u_binary = format(u, f'0{n_qubits}b')
        v_binary = format(v, f'0{n_qubits}b')
        binary_edges.add((u_binary, v_binary))
    
    # Create IntersectingEdgesGraph with binary string edges
    decomp_graph = IntersectingEdgesGraph(binary_edges, matchings=internal_algorithm)
    subgraphs = decomp_graph.subgraphs
    delta_t = time / n_steps
    
    n_states = 2**n_qubits
    time_evo_op = np.eye(n_states, dtype=complex)
    
    for _ in range(n_steps):
        for subgraph in subgraphs:
            try:
                # Get integer edges from the subgraph for matrix operations
                if hasattr(subgraph, '_StaticGraph__int_edges'):
                    subgraph_edges = list(subgraph._StaticGraph__int_edges)
                elif hasattr(subgraph, 'edges'):
                    # Convert from string edges to int edges if needed
                    subgraph_edges = []
                    for edge in subgraph.edges:
                        if isinstance(edge[0], str):
                            u = int(edge[0], 2)
                            v = int(edge[1], 2)
                            subgraph_edges.append((u, v))
                        else:
                            subgraph_edges.append(edge)
                else:
                    continue
                
                # Get nodes
                if hasattr(subgraph, 'nodes'):
                    if isinstance(subgraph.nodes, set):
                        subgraph_nodes = list(subgraph.nodes)
                    else:
                        subgraph_nodes = subgraph.nodes
                else:
                    # Infer nodes from edges
                    if subgraph_edges:
                        all_nodes = set()
                        for u, v in subgraph_edges:
                            all_nodes.add(u)
                            all_nodes.add(v)
                        subgraph_nodes = list(all_nodes)
                    else:
                        continue
                
            except Exception as e:
                print(f"Warning: Could not process subgraph: {e}")
                continue
            
            if not subgraph_edges:
                continue
                
            # Create a mapping from subgraph nodes to indices
            node_to_index = {node: i for i, node in enumerate(subgraph_nodes)}
            subgraph_size = len(node_to_index)
            
            # Create the adjacency matrix for the subgraph
            adj_matrix = np.zeros((subgraph_size, subgraph_size), dtype=complex)
            for u, v in subgraph_edges:
                if u in node_to_index and v in node_to_index:
                    i, j = node_to_index[u], node_to_index[v]
                    adj_matrix[i, j] = adj_matrix[j, i] = 1
            
            # Compute the unitary for the subgraph
            subgraph_unitary = expm(-1j * adj_matrix * delta_t)
            
            # Create the full unitary by embedding the subgraph unitary
            full_unitary = np.eye(n_states, dtype=complex)
            for i, node_i in enumerate(subgraph_nodes):
                for j, node_j in enumerate(subgraph_nodes):
                    if node_i < n_states and node_j < n_states:
                        full_unitary[node_i, node_j] = subgraph_unitary[i, j]
            
            time_evo_op = np.dot(full_unitary, time_evo_op)
    
    return Operator(time_evo_op)


def trotterized_pauli_decomp(graph, time, n_steps):
    """Trotterized quantum walk using first-order Trotter decomposition of Pauli terms"""
    # Get integer edges from NetworkX graph
    int_edges = set(graph.edges())
    
    # Determine number of qubits needed to represent all nodes
    max_node = max(max(edge) for edge in int_edges) if int_edges else 0
    n_qubits = max_node.bit_length()  # Number of bits needed to represent max_node
    if n_qubits == 0:  # Handle case where max_node is 0
        n_qubits = 1
    
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
    # Get integer edges from NetworkX graph
    int_edges = set(graph.edges())
    
    # Determine number of qubits needed to represent all nodes
    max_node = max(max(edge) for edge in int_edges) if int_edges else 0
    n_qubits = max_node.bit_length()  # Number of bits needed to represent max_node
    if n_qubits == 0:  # Handle case where max_node is 0
        n_qubits = 1
    
    n_states = 2**n_qubits
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    time_evo_op = expm(-1j * adj_matrix * time)
    
    # Pad the time evolution operator if necessary
    if time_evo_op.shape[0] < n_states:
        padded_op = np.eye(n_states, dtype=complex)
        padded_op[:time_evo_op.shape[0], :time_evo_op.shape[1]] = time_evo_op
        time_evo_op = padded_op
    
    return Operator(time_evo_op)


def compare_all_walks(graph, time, n_steps, matching_algorithm='first-fit'):
    """Compare trotterized and pauli methods vs original
    
    Args:
        graph: NetworkX graph
        time: Evolution time
        n_steps: Number of Trotter steps
        matching_algorithm: 'first-fit' or 'structure-aware'
    """
    trotterized_op = trotterized_quantum_walk(graph, time, n_steps, matching_algorithm)
    trotterized_pauli_op = trotterized_pauli_decomp(graph, time, n_steps)
    original_op = original_ctqw(graph, time)
    
    # Calculate differences vs original only
    qw_vs_original = np.linalg.norm((trotterized_op - original_op).data, ord=2)
    pauli_vs_original = np.linalg.norm((trotterized_pauli_op - original_op).data, ord=2)
    
    return {
        'qw_vs_original': qw_vs_original,
        'pauli_vs_original': pauli_vs_original
    }


def plot_results(graphs, times, n_steps_list, title, save_dir, matching_algorithm='first-fit', verbose=True):
    """Plot results comparing trotterized and pauli methods vs original
    
    Args:
        graphs: List of NetworkX graphs
        times: List of evolution times
        n_steps_list: List of Trotter step counts
        title: Plot title
        save_dir: Directory to save plots
        matching_algorithm: 'first-fit' or 'structure-aware'
        verbose: Whether to print progress
    """
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
    
    # Map algorithm names to display names
    algorithm_display_names = {
        'first-fit': 'First-Fit Algorithm',
        'structure-aware': 'Structure-Aware Algorithm'
    }
    display_name = algorithm_display_names.get(matching_algorithm, matching_algorithm)
    
    if verbose:
        print(f"Analyzing {len(graphs)} graphs for {title} (matching: {display_name})...")
    for i, graph in enumerate(graphs):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(graphs)} graphs")
        for t in times:
            for N in n_steps_list:
                comparison_results = compare_all_walks(graph, t, N, matching_algorithm)
                for key, value in comparison_results.items():
                    results[t][N][key].append(value)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Update comparison labels based on matching algorithm
    trotter_label = f'{display_name} CTQW vs exact CTQW'
    
    # Comparison types and their labels
    comparisons = [
        ('qw_vs_original', trotter_label, axes[0]),
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
        ax.set_ylabel('Operator Difference Norm')
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
    
    # Add overall title with matching algorithm info
    full_title = f"{title} ({display_name})"
    fig.suptitle(full_title, fontsize=24, y=1.02)
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)
    
    # Save the plot with matching algorithm in filename
    filename = title.lower().replace(' ', '_').replace('-', '_')
    filename += f"_{matching_algorithm.replace('-', '_')}.pdf"
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
    # Set default directories based on script location in ./analysis
    script_dir = os.path.dirname(__file__)
    default_graphs_dir = os.path.join(script_dir, '..', 'data', 'graphs')
    default_plots_dir = os.path.join(script_dir, '..', 'data', 'plots')

    parser = argparse.ArgumentParser(description='Analyze operator difference norms between quantum walk implementations')
    
    # Graph loading parameters
    parser.add_argument('--vertex-sizes', nargs='+', type=int, default=[8, 16, 32],
                        help='List of vertex sizes for graphs to analyze (default: 8 16 32)')
    parser.add_argument('--n-graphs', type=int, default=100,
                        help='Number of graphs to load per vertex size (default: 100)')
    
    # Analysis parameters
    parser.add_argument('--times', nargs='+', type=float, default=[0.1, 0.5, 1.0],
                        help='Time values for quantum walk (default: 0.1 0.5 1.0)')
    parser.add_argument('--n-steps', nargs='+', type=int, default=[5, 10, 20, 50, 100],
                        help='Number of Trotter steps (default: 5 10 20 50 100)')
    parser.add_argument('--matching-algorithm', choices=['first-fit', 'structure-aware'], default='first-fit',
                        help='Matching algorithm for Trotterized quantum walk (default: first-fit)')
    
    # Input/Output directories
    parser.add_argument('--graphs-dir', type=str, default=default_graphs_dir,
                        help=f'Directory to load graphs from (default: {default_graphs_dir})')
    parser.add_argument('--plots-dir', type=str, default=default_plots_dir,
                        help=f'Directory to save plots (default: {default_plots_dir})')
    
    # Control options
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Map algorithm names to display names
    algorithm_display_names = {
        'first-fit': 'First-Fit Algorithm',
        'structure-aware': 'Structure-Aware Algorithm'
    }
    
    if args.verbose:
        display_name = algorithm_display_names.get(args.matching_algorithm, args.matching_algorithm)
        print(f"Analysis Configuration:")
        print(f"  Vertex sizes: {args.vertex_sizes}")
        print(f"  Number of graphs per size: {args.n_graphs}")
        print(f"  Times: {args.times}")
        print(f"  Trotter steps: {args.n_steps}")
        print(f"  Matching algorithm: {display_name}")
        print(f"  Graphs directory: {args.graphs_dir}")
        print(f"  Plots directory: {args.plots_dir}")
    
    all_results = {}
    
    for n_vertices in args.vertex_sizes:
        print(f"\n{'='*50}")
        print(f"Processing {n_vertices}-vertex graphs with {algorithm_display_names.get(args.matching_algorithm, args.matching_algorithm)}")
        print(f"{'='*50}")
        
        try:
            # Load graphs
            graphs = load_graphs(args.graphs_dir, n_vertices, args.n_graphs, args.verbose)
            
            if args.verbose:
                print(f"Running analysis on {len(graphs)} graphs...")
            
            title = f'{n_vertices}-Vertex Graphs'
            results = plot_results(graphs, args.times, args.n_steps, title, args.plots_dir, 
                                 matching_algorithm=args.matching_algorithm, verbose=args.verbose)
            all_results[f"{n_vertices}-vertex"] = results
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please run generate_graphs.py first to create the graph files.")
            continue
        except Exception as e:
            print(f"Error processing {n_vertices}-vertex graphs: {e}")
            continue
    
    if all_results:
        # Print overall summary
        print_summary_statistics(all_results, args.verbose)
        if args.verbose:
            print(f"\nOperator difference norm analysis complete! Plots saved to {args.plots_dir}")
    else:
        print("\nNo graphs were successfully analyzed. Please check that graph files exist.")


if __name__ == "__main__":
    main()