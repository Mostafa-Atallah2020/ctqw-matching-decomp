#!/usr/bin/env python3
"""
Operator Difference Norm Analysis Script

This script analyzes operator difference norms between quantum walk implementations:
1. Trotterized quantum walk using First-Fit Greedy decomposition vs exact CTQW
2. Trotterized quantum walk using Hamming Distance Greedy decomposition vs exact CTQW  
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


def trotterized_quantum_walk(graph, time, n_steps, matching_algorithm='first-fit-greedy'):
    """Trotterized quantum walk using subgraph decomposition with matching algorithms
    
    Args:
        graph: NetworkX graph
        time: Evolution time
        n_steps: Number of Trotter steps
        matching_algorithm: 'first-fit-greedy' or 'hamming-distance-greedy'
    """
    if matching_algorithm not in ['first-fit-greedy', 'hamming-distance-greedy']:
        raise ValueError(f"Invalid matching algorithm: {matching_algorithm}. "
                       f"Valid options are: 'first-fit-greedy', 'hamming-distance-greedy'")
    
    # Map new names to internal names
    algorithm_mapping = {
        'first-fit-greedy': 'greedy',
        'hamming-distance-greedy': 'parallel'
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


def analyze_graphs(graphs, times, n_steps_list, verbose=True):
    """Analyze graphs with all three methods and return results
    
    Args:
        graphs: List of NetworkX graphs
        times: List of evolution times
        n_steps_list: List of Trotter step counts
        verbose: Whether to print progress
        
    Returns:
        dict: Results for all three methods
    """
    results = {
        'first-fit-greedy': {
            t: {N: [] for N in n_steps_list} for t in times
        },
        'hamming-distance-greedy': {
            t: {N: [] for N in n_steps_list} for t in times
        },
        'pauli-decomposition': {
            t: {N: [] for N in n_steps_list} for t in times
        }
    }
    
    if verbose:
        print(f"Analyzing {len(graphs)} graphs...")
    
    for i, graph in enumerate(graphs):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(graphs)} graphs")
        
        for t in times:
            for N in n_steps_list:
                # Get exact CTQW for comparison
                original_op = original_ctqw(graph, t)
                
                # First-Fit Greedy
                first_fit_op = trotterized_quantum_walk(graph, t, N, 'first-fit-greedy')
                first_fit_diff = np.linalg.norm((first_fit_op - original_op).data, ord=2)
                results['first-fit-greedy'][t][N].append(first_fit_diff)
                
                # Hamming Distance Greedy
                hamming_op = trotterized_quantum_walk(graph, t, N, 'hamming-distance-greedy')
                hamming_diff = np.linalg.norm((hamming_op - original_op).data, ord=2)
                results['hamming-distance-greedy'][t][N].append(hamming_diff)
                
                # Pauli Decomposition
                pauli_op = trotterized_pauli_decomp(graph, t, N)
                pauli_diff = np.linalg.norm((pauli_op - original_op).data, ord=2)
                results['pauli-decomposition'][t][N].append(pauli_diff)
    
    return results


def create_comprehensive_plot(all_results, vertex_sizes, times, n_steps_list, save_dir, verbose=True):
    """Create a 3x3 grid plot comparing all methods across all vertex sizes
    
    Args:
        all_results: Dictionary of results for all vertex sizes
        vertex_sizes: List of vertex sizes
        times: List of evolution times  
        n_steps_list: List of Trotter step counts
        save_dir: Directory to save plots
        verbose: Whether to print progress
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting parameters
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.figsize': (18, 12),
        'figure.dpi': 300,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })
    
    # Create 3x3 grid: rows=vertex sizes, columns=methods
    fig, axes = plt.subplots(len(vertex_sizes), 3, figsize=(18, 6*len(vertex_sizes)))
    if len(vertex_sizes) == 1:
        axes = axes.reshape(1, 3)
    
    # Method information
    methods = [
        ('first-fit-greedy', 'First-Fit Greedy vs Exact CTQW'),
        ('hamming-distance-greedy', 'Hamming Distance Greedy vs Exact CTQW'),
        ('pauli-decomposition', 'Pauli Decomposition vs Exact CTQW')
    ]
    
    # Color palette for different times
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot each combination
    for row, n_vertices in enumerate(vertex_sizes):
        vertex_key = f"{n_vertices}-vertex"
        
        if vertex_key not in all_results:
            continue
            
        vertex_results = all_results[vertex_key]
        
        for col, (method_key, method_label) in enumerate(methods):
            ax = axes[row, col]
            
            # Plot each time value
            for t_idx, t in enumerate(times):
                N_values = sorted(n_steps_list)
                mean_values = []
                std_values = []
                
                for N in N_values:
                    if method_key in vertex_results and t in vertex_results[method_key] and N in vertex_results[method_key][t]:
                        data = vertex_results[method_key][t][N]
                        if data:  # Check if data exists
                            mean_values.append(np.mean(data))
                            std_values.append(np.std(data))
                        else:
                            mean_values.append(np.nan)
                            std_values.append(np.nan)
                    else:
                        mean_values.append(np.nan)
                        std_values.append(np.nan)
                
                # Plot with error bars
                ax.errorbar(N_values, mean_values,
                           yerr=std_values,
                           fmt='-o',
                           capsize=4,
                           capthick=1.5,
                           label=f't = {t}',
                           color=colors[t_idx],
                           linewidth=2,
                           markersize=6)
            
            # Format subplot
            ax.set_yscale('log')
            ax.set_xlabel('Number of Trotter Steps (N)')
            # ax.set_ylabel('Operator Difference Norm')
            ax.grid(True, which='both', linestyle=':', alpha=0.3)
            
            # Set title
            if row == 0:
                # Method name at top
                ax.set_title(method_label, fontsize=16, fontweight='bold')
            
            # Add vertex size label on left
            if col == 0:
                ax.text(-0.5, 0.5, f'{n_vertices} Vertices', 
                       transform=ax.transAxes, rotation=90, 
                       verticalalignment='center', horizontalalignment='center',
                       fontsize=16, fontweight='bold')
            
            # Add legend only to top-right subplot
            if row == 0 and col == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add overall title
    fig.suptitle('Operator Difference Norms', fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, left=0.08, right=0.92)
    
    # Save the plot
    filename = 'comprehensive_operator_difference_analysis.pdf'
    filepath = save_dir / filename
    fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    
    if verbose:
        print(f"Comprehensive plot saved as: {filepath}")
    
    plt.close(fig)
    return filepath


def print_summary_statistics(all_results, verbose=True):
    """Print summary statistics for all results"""
    if not verbose:
        return
        
    print("\nSummary Statistics:")
    print("=" * 80)
    
    for vertex_key, vertex_results in all_results.items():
        print(f"\n{vertex_key.upper()} GRAPHS:")
        print("-" * 50)
        
        methods = [
            ('first-fit-greedy', 'First-Fit Greedy'),
            ('hamming-distance-greedy', 'Hamming Distance Greedy'),
            ('pauli-decomposition', 'Pauli Decomposition')
        ]
        
        # Show statistics for representative time and step values
        for t in [0.1, 1.0]:  # Representative times
            if any(t in vertex_results.get(method, {}) for method, _ in methods):
                print(f"\n  Time t = {t}:")
                for N in [10, 50]:  # Representative step counts
                    print(f"    Trotter Steps N = {N}:")
                    for method_key, method_name in methods:
                        if (method_key in vertex_results and 
                            t in vertex_results[method_key] and 
                            N in vertex_results[method_key][t] and
                            vertex_results[method_key][t][N]):
                            
                            data = vertex_results[method_key][t][N]
                            mean_val = np.mean(data)
                            std_val = np.std(data)
                            print(f"      {method_name:>22}: {mean_val:.2e} ± {std_val:.2e}")
                        else:
                            print(f"      {method_name:>22}: No data")


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
    
    # Input/Output directories
    parser.add_argument('--graphs-dir', type=str, default=default_graphs_dir,
                        help=f'Directory to load graphs from (default: {default_graphs_dir})')
    parser.add_argument('--plots-dir', type=str, default=default_plots_dir,
                        help=f'Directory to save plots (default: {default_plots_dir})')
    
    # Control options
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Comprehensive Quantum Walk Analysis Configuration:")
        print(f"  Vertex sizes: {args.vertex_sizes}")
        print(f"  Number of graphs per size: {args.n_graphs}")
        print(f"  Times: {args.times}")
        print(f"  Trotter steps: {args.n_steps}")
        print(f"  Graphs directory: {args.graphs_dir}")
        print(f"  Plots directory: {args.plots_dir}")
    
    all_results = {}
    
    # Process each vertex size
    for n_vertices in args.vertex_sizes:
        print(f"\n{'='*70}")
        print(f"Processing {n_vertices}-vertex graphs")
        print(f"{'='*70}")
        
        try:
            # Load graphs
            graphs = load_graphs(args.graphs_dir, n_vertices, args.n_graphs, args.verbose)
            
            # Analyze with all three methods
            results = analyze_graphs(graphs, args.times, args.n_steps, args.verbose)
            all_results[f"{n_vertices}-vertex"] = results
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please run generate_graphs.py first to create the graph files.")
            continue
        except Exception as e:
            print(f"Error processing {n_vertices}-vertex graphs: {e}")
            continue
    
    if all_results:
        # Create comprehensive plot
        if args.verbose:
            print(f"\n{'='*70}")
            print("Creating comprehensive analysis plot...")
            print(f"{'='*70}")
        
        create_comprehensive_plot(all_results, args.vertex_sizes, args.times, 
                                args.n_steps, args.plots_dir, args.verbose)
        
        # Print summary statistics
        print_summary_statistics(all_results, args.verbose)
        
        if args.verbose:
            print(f"\nComprehensive operator difference norm analysis complete!")
            print(f"Results saved to {args.plots_dir}")
    else:
        print("\nNo graphs were successfully analyzed. Please check that graph files exist.")


if __name__ == "__main__":
    main()