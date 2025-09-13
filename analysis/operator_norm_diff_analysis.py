#!/usr/bin/env python3
"""
Operator Norm Difference Analysis Between Matching/Pauli Decompositions and Exact CTQW

Compares both Matching and Pauli decompositions against exact CTQW for graphs in G6 files.
Analyzes operator difference norms (2-norm) between:
1. Trotterized matching decomposition vs exact CTQW
2. Trotterized Pauli decomposition vs exact CTQW

Usage: python operator_norm_diff_analysis.py <g6_file> [options]
Examples:
  python operator_norm_diff_analysis.py graphs.g6
  python operator_norm_diff_analysis.py graphs.g6 -m 1 -M 20 -s 2 -t 0.1 0.5 1.0
  python operator_norm_diff_analysis.py graphs.g6 --min_steps 5 --max_steps 100 --step_inc 10 --time_values 0.01 0.1 1.0
  python operator_norm_diff_analysis.py graphs.g6 -m 1 -M 50 -s 5 -t 0.1 1.0 -o custom_output
"""

import sys
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import re
from collections import defaultdict
import pandas as pd
from scipy.linalg import expm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Custom imports - adjust paths as needed
try:
    from src.graphs import StaticGraph, IntersectingEdgesGraph, MultiEdgeGraph
    from src.misc import graph_matchings_parallel
except ImportError:
    try:
        # Try alternative path structure
        sys.path.append("./../")
        from src.graphs import StaticGraph, IntersectingEdgesGraph, MultiEdgeGraph
        from src.misc import graph_matchings_parallel
    except ImportError:
        print("Error: Could not import custom graph classes.")
        print("Please ensure src.graphs and src.misc modules are available.")
        print("Current working directory:", Path.cwd())
        print("Script location:", Path(__file__).parent)
        print("Python path includes:")
        for p in sys.path:
            print(f"  {p}")
        sys.exit(1)

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate

def parse_g6_filename(filename):
    """Extract metadata from G6 filename."""
    path = Path(filename)
    stem = path.stem
    
    # Try multiple patterns for different filename formats
    patterns = [
        r'(\d+)graph_([A-Za-z_]+)_(\d+)([a-z]*)',  # 100graph_bipartite_16c
        r'(\d+)graph_([A-Za-z]+)_(\d+)([a-z]*)',   # Standard format  
        r'(\d+)graph([A-Za-z_]+)(\d+)([a-z]*)',    # Without underscore after graph
        r'(\d+)graph_([A-Za-z]+)(\d+)([a-z]*)',    # Alternative format
    ]
    
    for pattern in patterns:
        match = re.match(pattern, stem)
        if match:
            groups = match.groups()
            return {
                'n_graphs': int(groups[0]),
                'type': groups[1],
                'vertices': int(groups[2]),
                'suffix': groups[3] if len(groups) >= 4 and groups[3] else None,
                'filename': path.name
            }
    
    # If no pattern matches, try to extract just the numbers and words
    numbers = re.findall(r'\d+', stem)
    words = re.findall(r'[A-Za-z_]+', stem)
    
    result = {'filename': path.name}
    
    # Extract numbers: first is usually graph count, last is usually vertex count
    if len(numbers) >= 2:
        result['n_graphs'] = int(numbers[0])
        result['vertices'] = int(numbers[-1])
    elif len(numbers) == 1:
        # Could be either graph count or vertex count, check context
        if 'graph' in stem.lower():
            result['n_graphs'] = int(numbers[0])
        else:
            result['vertices'] = int(numbers[0])
    
    # Extract graph type from words (skip 'graph' itself)
    if words:
        graph_words = [w for w in words if w.lower() != 'graph']
        if graph_words:
            result['type'] = '_'.join(graph_words)
        else:
            result['type'] = 'Unknown'
    else:
        result['type'] = 'Unknown'
    
    return result

def g6_to_edge_set(g6_string):
    """Convert G6 string to edge set with binary vertex labels."""
    try:
        G = nx.from_graph6_bytes(g6_string.encode())
        n_vertices = len(G.nodes())
        
        if n_vertices == 0:
            return set()
        
        # Calculate bits needed for binary representation
        n_bits = max(1, int(np.ceil(np.log2(max(2, n_vertices)))))
        
        # Create vertex mapping to binary strings
        vertex_map = {vertex: format(i, f'0{n_bits}b') 
                     for i, vertex in enumerate(sorted(G.nodes()))}
        
        # Convert edges to binary format
        edge_set = set()
        for u, v in G.edges():
            u_bin, v_bin = vertex_map[u], vertex_map[v]
            edge_set.add((u_bin, v_bin))
            edge_set.add((v_bin, u_bin))  # Make undirected
        
        return edge_set
        
    except Exception as e:
        print(f"Error parsing G6 string: {e}")
        return None

def load_all_graphs(filename):
    """Load all graphs from G6 file."""
    metadata = parse_g6_filename(filename)
    graphs = []
    
    print(f"Loading graphs from {filename}")
    print(f"Expected: {metadata}")
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                edges = g6_to_edge_set(line)
                if edges is not None:
                    graphs.append(edges)
                else:
                    print(f"Warning: Failed to parse line {line_num + 1}")
    
    print(f"Successfully loaded {len(graphs)} graphs")
    return graphs, metadata

def analyze_graph_properties(edges):
    """Analyze basic properties of the graph."""
    if not edges:
        return {"n_vertices": 0, "n_edges": 0, "is_empty": True}
    
    vertices = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)
    
    n_vertices = len(vertices)
    n_edges = len(edges) // 2  # Bidirectional edges
    max_edges = n_vertices * (n_vertices - 1) // 2
    
    degrees = defaultdict(int)
    for u, v in edges:
        degrees[u] += 1
    
    return {
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "vertices": sorted(vertices),
        "density": n_edges / max_edges if max_edges > 0 else 0,
        "degrees": dict(degrees),
        "min_degree": min(degrees.values()) if degrees else 0,
        "max_degree": max(degrees.values()) if degrees else 0,
        "avg_degree": np.mean(list(degrees.values())) if degrees else 0
    }

def create_exact_ctqw_operator(edges, time):
    """Create exact CTQW operator using matrix exponentiation."""
    # Get relabeled graph for consistent vertex labeling
    matchings = graph_matchings_parallel(edges)
    relabeled_edges = set()
    for m in matchings:
        relabeled_edges = relabeled_edges.union(m)
    
    relabeled_G = StaticGraph(relabeled_edges)
    H = relabeled_G.get_adj_mat()
    
    # Exact time evolution: U(t) = exp(-iHt)
    exact_op = expm(-1j * H * time)
    return Operator(exact_op)

def create_matching_circuit(edges, n_steps, total_time):
    """Create quantum circuit using matching decomposition with proper Trotterization."""
    # Use relabeled graph for consistency
    matchings = graph_matchings_parallel(edges)
    relabeled_edges = set()
    for m in matchings:
        relabeled_edges = relabeled_edges.union(m)
    
    relabeled_G = StaticGraph(relabeled_edges)
    intersecting_G = IntersectingEdgesGraph(edges, matchings='parallel')
    
    full_qc = QuantumCircuit(relabeled_G.n_qubits)
    
    # Time per Trotter step
    dt = total_time / n_steps
    
    # Apply n_steps of Trotterized evolution
    for step in range(n_steps):
        # For each Trotter step, apply all subgraphs sequentially
        for subgraph in intersecting_G.subgraphs:
            G = MultiEdgeGraph(subgraph.edges)
            # Set rotation angle for this time slice
            G.rot_angle = dt
            sub_qc = G.get_qc(simplified=True)
            full_qc = full_qc.compose(sub_qc)
    
    return full_qc

def create_pauli_circuit(edges, n_steps, total_time):
    """Create quantum circuit using Pauli decomposition with proper Trotterization."""
    # Get relabeled graph for consistent vertex labeling
    matchings = graph_matchings_parallel(edges)
    relabeled_edges = set()
    for m in matchings:
        relabeled_edges = relabeled_edges.union(m)
    
    relabeled_G = StaticGraph(relabeled_edges)
    H = relabeled_G.get_adj_mat()
    n = relabeled_G.n_qubits
    
    # Decompose Hamiltonian into Pauli basis
    pauli_strings = []
    coeffs = []
    
    for pauli_string in ["".join(p) for p in itertools.product("IXYZ", repeat=n)]:
        P = Pauli(pauli_string)
        P_op = Operator(P).data
        coeff = np.trace(P_op.conj().T @ H) / (2**n)
        
        real_coeff = float(np.real(coeff))
        if not np.isclose(real_coeff, 0, atol=1e-10):
            pauli_strings.append(pauli_string)
            coeffs.append(real_coeff)
    
    # Create Trotterized Pauli evolution circuit
    pauli_qc = QuantumCircuit(n)
    
    if coeffs:
        # Time per Trotter step
        dt = total_time / n_steps
        
        # First-order Trotter decomposition
        for step in range(n_steps):
            # Apply each Pauli term separately (first-order Trotter)
            for pauli_string, coeff in zip(pauli_strings, coeffs):
                single_pauli_op = SparsePauliOp([pauli_string], [coeff])
                evo_gate = PauliEvolutionGate(single_pauli_op, time=dt)
                pauli_qc.append(evo_gate, range(n))
    
    # Decompose to get actual gates
    decomposed_qc = pauli_qc.decompose().decompose().decompose()
    return decomposed_qc

def compute_operator_differences(edges, trotter_steps_list, time_values):
    """Compute 2-norm differences between matching/Pauli and exact CTQW."""
    results = {
        'trotter_steps': trotter_steps_list,
        'time_values': time_values,
        'matching_differences': {},
        'pauli_differences': {},
        'properties': analyze_graph_properties(edges)
    }
    
    for time_val in time_values:
        # Get exact CTQW reference (computed once per time_val)
        try:
            exact_op = create_exact_ctqw_operator(edges, time_val)
        except Exception as e:
            print(f"Error creating exact CTQW for time {time_val}: {e}")
            results['matching_differences'][time_val] = [np.nan] * len(trotter_steps_list)
            results['pauli_differences'][time_val] = [np.nan] * len(trotter_steps_list)
            continue
        
        matching_diffs = []
        pauli_diffs = []
        
        for n_steps in trotter_steps_list:
            # Matching decomposition comparison
            try:
                matching_qc = create_matching_circuit(edges, n_steps, time_val)
                matching_op = Operator(matching_qc)
                
                # Calculate 2-norm difference
                diff = matching_op - exact_op
                two_norm = np.linalg.norm(diff.data, ord=2)
                matching_diffs.append(two_norm)
                
            except Exception as e:
                print(f"Error for matching, time {time_val}, steps {n_steps}: {e}")
                matching_diffs.append(np.nan)
            
            # Pauli decomposition comparison
            try:
                pauli_qc = create_pauli_circuit(edges, n_steps, time_val)
                pauli_op = Operator(pauli_qc)
                
                # Calculate 2-norm difference
                diff = pauli_op - exact_op
                two_norm = np.linalg.norm(diff.data, ord=2)
                pauli_diffs.append(two_norm)
                
            except Exception as e:
                print(f"Error for Pauli, time {time_val}, steps {n_steps}: {e}")
                pauli_diffs.append(np.nan)
        
        results['matching_differences'][time_val] = matching_diffs
        results['pauli_differences'][time_val] = pauli_diffs
    
    return results

def process_all_graphs(graphs, trotter_steps_list, time_values, expected_vertices=None):
    """Process all graphs and compute operator differences."""
    all_results = []
    valid_graphs = 0
    skipped_wrong_size = 0
    skipped_no_edges = 0
    skipped_not_power_of_2 = 0
    
    for i, edges in enumerate(graphs):
        print(f"Processing graph {i+1}/{len(graphs)}")
        
        props = analyze_graph_properties(edges)
        n_vertices = props['n_vertices']
        
        # Check if vertex count is a power of 2
        if n_vertices == 0 or (n_vertices & (n_vertices - 1)) != 0:
            print(f"  Skipping: {n_vertices} vertices (not a power of 2)")
            skipped_not_power_of_2 += 1
            continue
        
        # If we know the expected vertex count from filename, check it matches
        if expected_vertices is not None and n_vertices != expected_vertices:
            print(f"  Skipping: {n_vertices} vertices (expected {expected_vertices} from filename)")
            skipped_wrong_size += 1
            continue
            
        if props['n_edges'] == 0:
            print(f"  Skipping: {n_vertices} vertices, {props['n_edges']} edges (no edges)")
            skipped_no_edges += 1
            continue
        
        # Calculate number of qubits needed
        n_qubits = int(np.log2(n_vertices))
        print(f"  Graph: {n_vertices} vertices ({n_qubits} qubits), {props['n_edges']} edges, density={props['density']:.3f}")
        
        # Warn for large quantum circuits but don't skip
        if n_qubits > 8:
            print(f"    WARNING: {n_qubits} qubits will create large operators (2^{n_qubits} = {2**n_qubits} dimensions)")
        
        try:
            results = compute_operator_differences(edges, trotter_steps_list, time_values)
            results['graph_index'] = i
            results['n_qubits'] = n_qubits
            all_results.append(results)
            valid_graphs += 1
            
        except Exception as e:
            print(f"  Error processing graph {i}: {e}")
            continue
    
    print(f"\nProcessing summary:")
    print(f"  Successfully processed: {valid_graphs}/{len(graphs)} graphs")
    if skipped_wrong_size > 0:
        print(f"  Skipped (wrong vertex count): {skipped_wrong_size}")
    if skipped_not_power_of_2 > 0:
        print(f"  Skipped (not power of 2): {skipped_not_power_of_2}")
    if skipped_no_edges > 0:
        print(f"  Skipped (no edges): {skipped_no_edges}")
    
    return all_results

def create_convergence_plots(all_results, metadata, output_dir):
    """Create convergence analysis plots with improved color scheme."""
    if not all_results:
        print("No results to plot")
        return
    
    time_values = all_results[0]['time_values']
    trotter_steps = all_results[0]['trotter_steps']
    
    # Create single convergence plot with both methods
    plt.figure(figsize=(12, 8))
    
    # Define colors and line styles
    matching_color = '#1f77b4'  # Blue for matching
    pauli_color = '#ff7f0e'     # Orange for Pauli
    
    # Define line styles for different time values
    line_styles = ['-', '--', '-.', ':']
    if len(time_values) > len(line_styles):
        # Extend line styles if needed
        line_styles = (line_styles * ((len(time_values) // len(line_styles)) + 1))[:len(time_values)]
    
    for time_idx, time_val in enumerate(time_values):
        line_style = line_styles[time_idx]
        
        # Calculate statistics for matching decomposition
        matching_means = []
        matching_stds = []
        pauli_means = []
        pauli_stds = []
        
        for step_idx in range(len(trotter_steps)):
            # Matching statistics
            matching_step_diffs = []
            pauli_step_diffs = []
            
            for result in all_results:
                # Matching differences
                if step_idx < len(result['matching_differences'][time_val]):
                    diff = result['matching_differences'][time_val][step_idx]
                    if not np.isnan(diff):
                        matching_step_diffs.append(diff)
                
                # Pauli differences
                if step_idx < len(result['pauli_differences'][time_val]):
                    diff = result['pauli_differences'][time_val][step_idx]
                    if not np.isnan(diff):
                        pauli_step_diffs.append(diff)
            
            # Calculate means and stds
            if matching_step_diffs:
                matching_means.append(np.mean(matching_step_diffs))
                matching_stds.append(np.std(matching_step_diffs))
            else:
                matching_means.append(np.nan)
                matching_stds.append(np.nan)
            
            if pauli_step_diffs:
                pauli_means.append(np.mean(pauli_step_diffs))
                pauli_stds.append(np.std(pauli_step_diffs))
            else:
                pauli_means.append(np.nan)
                pauli_stds.append(np.nan)
        
        # Filter out NaN values for plotting
        valid_indices = [i for i, (m_m, m_s, p_m, p_s) in enumerate(zip(matching_means, matching_stds, pauli_means, pauli_stds)) 
                        if not np.isnan(m_m) and not np.isnan(m_s) and not np.isnan(p_m) and not np.isnan(p_s)]
        
        if valid_indices:
            valid_steps = [trotter_steps[i] for i in valid_indices]
            valid_matching_means = [matching_means[i] for i in valid_indices]
            valid_matching_stds = [matching_stds[i] for i in valid_indices]
            valid_pauli_means = [pauli_means[i] for i in valid_indices]
            valid_pauli_stds = [pauli_stds[i] for i in valid_indices]
            
            # Plot matching decomposition with consistent color and varied line style
            plt.errorbar(valid_steps, valid_matching_means, yerr=valid_matching_stds, 
                        marker='o', linewidth=2, capsize=5, capthick=2,
                        label=f'Matching t={time_val}', color=matching_color, 
                        linestyle=line_style, markersize=6)
            
            # Plot Pauli decomposition with consistent color and varied line style
            plt.errorbar(valid_steps, valid_pauli_means, yerr=valid_pauli_stds, 
                        marker='s', linewidth=2, capsize=5, capthick=2,
                        label=f'Pauli t={time_val}', color=pauli_color, 
                        linestyle=line_style, markersize=6)
    
    plt.xlabel('Trotter Steps', fontsize=12)
    plt.ylabel('2-norm Difference from Exact CTQW', fontsize=12)
    plt.title(f'Convergence to Exact CTQW: Matching vs Pauli Decompositions\n{metadata["type"]} graphs, {metadata["vertices"]} vertices', 
              fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_dir / f'{metadata["type"]}_{metadata["vertices"]}v_ctqw_convergence.pdf'
    plt.savefig(plot_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {plot_file}")
    
    # Also create separate plots for each method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Matching decomposition plot
    for time_idx, time_val in enumerate(time_values):
        line_style = line_styles[time_idx]
        
        # Calculate matching statistics
        matching_means = []
        matching_stds = []
        
        for step_idx in range(len(trotter_steps)):
            matching_step_diffs = []
            
            for result in all_results:
                if step_idx < len(result['matching_differences'][time_val]):
                    diff = result['matching_differences'][time_val][step_idx]
                    if not np.isnan(diff):
                        matching_step_diffs.append(diff)
            
            if matching_step_diffs:
                matching_means.append(np.mean(matching_step_diffs))
                matching_stds.append(np.std(matching_step_diffs))
            else:
                matching_means.append(np.nan)
                matching_stds.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [i for i, (m_m, m_s) in enumerate(zip(matching_means, matching_stds)) 
                        if not np.isnan(m_m) and not np.isnan(m_s)]
        
        if valid_indices:
            valid_steps = [trotter_steps[i] for i in valid_indices]
            valid_matching_means = [matching_means[i] for i in valid_indices]
            valid_matching_stds = [matching_stds[i] for i in valid_indices]
            
            ax1.errorbar(valid_steps, valid_matching_means, yerr=valid_matching_stds, 
                        marker='o', linewidth=2, capsize=5, capthick=2,
                        label=f't={time_val}', color=matching_color, 
                        linestyle=line_style, markersize=6)
    
    ax1.set_xlabel('Trotter Steps', fontsize=12)
    ax1.set_ylabel('2-norm Difference from Exact CTQW', fontsize=12)
    ax1.set_title('Matching Decomposition Convergence', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pauli decomposition plot
    for time_idx, time_val in enumerate(time_values):
        line_style = line_styles[time_idx]
        
        # Calculate Pauli statistics
        pauli_means = []
        pauli_stds = []
        
        for step_idx in range(len(trotter_steps)):
            pauli_step_diffs = []
            
            for result in all_results:
                if step_idx < len(result['pauli_differences'][time_val]):
                    diff = result['pauli_differences'][time_val][step_idx]
                    if not np.isnan(diff):
                        pauli_step_diffs.append(diff)
            
            if pauli_step_diffs:
                pauli_means.append(np.mean(pauli_step_diffs))
                pauli_stds.append(np.std(pauli_step_diffs))
            else:
                pauli_means.append(np.nan)
                pauli_stds.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [i for i, (p_m, p_s) in enumerate(zip(pauli_means, pauli_stds)) 
                        if not np.isnan(p_m) and not np.isnan(p_s)]
        
        if valid_indices:
            valid_steps = [trotter_steps[i] for i in valid_indices]
            valid_pauli_means = [pauli_means[i] for i in valid_indices]
            valid_pauli_stds = [pauli_stds[i] for i in valid_indices]
            
            ax2.errorbar(valid_steps, valid_pauli_means, yerr=valid_pauli_stds, 
                        marker='s', linewidth=2, capsize=5, capthick=2,
                        label=f't={time_val}', color=pauli_color, 
                        linestyle=line_style, markersize=6)
    
    ax2.set_xlabel('Trotter Steps', fontsize=12)
    ax2.set_ylabel('2-norm Difference from Exact CTQW', fontsize=12)
    ax2.set_title('Pauli Decomposition Convergence', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    separate_plot_file = output_dir / f'{metadata["type"]}_{metadata["vertices"]}v_ctqw_separate_convergence.pdf'
    plt.savefig(separate_plot_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Separate convergence plots saved to {separate_plot_file}")

def create_summary_statistics(all_results, metadata, output_dir):
    """Create summary statistics and save to files."""
    if not all_results:
        return
    
    time_values = all_results[0]['time_values']
    trotter_steps = all_results[0]['trotter_steps']
    
    # Compile statistics for both methods
    stats_data = []
    
    for time_val in time_values:
        # Final convergence values for both methods
        matching_final_values = []
        pauli_final_values = []
        
        for result in all_results:
            # Matching final values
            matching_differences = result['matching_differences'][time_val]
            if matching_differences and not np.isnan(matching_differences[-1]):
                matching_final_values.append(matching_differences[-1])
            
            # Pauli final values
            pauli_differences = result['pauli_differences'][time_val]
            if pauli_differences and not np.isnan(pauli_differences[-1]):
                pauli_final_values.append(pauli_differences[-1])
        
        if matching_final_values and pauli_final_values:
            stats_data.append({
                'time': time_val,
                'n_graphs': len(all_results),
                'matching_mean_final': np.mean(matching_final_values),
                'matching_std_final': np.std(matching_final_values),
                'matching_median_final': np.median(matching_final_values),
                'pauli_mean_final': np.mean(pauli_final_values),
                'pauli_std_final': np.std(pauli_final_values),
                'pauli_median_final': np.median(pauli_final_values),
                'matching_min': np.min(matching_final_values),
                'matching_max': np.max(matching_final_values),
                'pauli_min': np.min(pauli_final_values),
                'pauli_max': np.max(pauli_final_values)
            })
    
    # Save as CSV
    if stats_data:
        df = pd.DataFrame(stats_data)
        csv_file = output_dir / f'{metadata["type"]}_{metadata["vertices"]}v_ctqw_comparison_statistics.csv'
        df.to_csv(csv_file, index=False)
        print(f"Statistics saved to {csv_file}")
    
    # Save detailed results
    results_file = output_dir / f'{metadata["type"]}_{metadata["vertices"]}v_ctqw_detailed_results.npz'
    
    # Prepare data for saving
    save_data = {
        'metadata': metadata,
        'trotter_steps': trotter_steps,
        'time_values': time_values,
        'n_graphs_processed': len(all_results)
    }
    
    # Add difference arrays for each time value and method
    for time_val in time_values:
        matching_differences_array = []
        pauli_differences_array = []
        for result in all_results:
            matching_differences_array.append(result['matching_differences'][time_val])
            pauli_differences_array.append(result['pauli_differences'][time_val])
        save_data[f'matching_differences_time_{time_val}'] = matching_differences_array
        save_data[f'pauli_differences_time_{time_val}'] = pauli_differences_array
    
    # Add graph properties
    properties_list = [result['properties'] for result in all_results]
    save_data['graph_properties'] = properties_list
    
    np.savez(results_file, **save_data)
    print(f"Detailed results saved to {results_file}")
    
    # Create text summary
    summary_file = output_dir / f'{metadata["type"]}_{metadata["vertices"]}v_ctqw_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Operator Norm Difference Analysis Summary\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Dataset: {metadata['filename']}\n")
        f.write(f"Graph type: {metadata['type']}\n")
        f.write(f"Vertices: {metadata['vertices']}\n")
        f.write(f"Expected graphs: {metadata.get('n_graphs', 'Unknown')}\n")
        f.write(f"Processed graphs: {len(all_results)}\n\n")
        
        f.write(f"Analysis parameters:\n")
        f.write(f"  Trotter steps: {min(trotter_steps)} to {max(trotter_steps)} (steps: {trotter_steps})\n")
        f.write(f"  Time values: {time_values}\n\n")
        
        f.write(f"Methods compared:\n")
        f.write(f"  1. Matching decomposition vs Exact CTQW\n")
        f.write(f"  2. Pauli decomposition vs Exact CTQW\n\n")
        
        if stats_data:
            f.write(f"Summary statistics (final Trotter step convergence):\n")
            for stat in stats_data:
                f.write(f"  Time {stat['time']}:\n")
                f.write(f"    Matching vs CTQW:\n")
                f.write(f"      Mean: {stat['matching_mean_final']:.2e} ± {stat['matching_std_final']:.2e}\n")
                f.write(f"      Median: {stat['matching_median_final']:.2e}\n")
                f.write(f"      Range: [{stat['matching_min']:.2e}, {stat['matching_max']:.2e}]\n")
                f.write(f"    Pauli vs CTQW:\n")
                f.write(f"      Mean: {stat['pauli_mean_final']:.2e} ± {stat['pauli_std_final']:.2e}\n")
                f.write(f"      Median: {stat['pauli_median_final']:.2e}\n")
                f.write(f"      Range: [{stat['pauli_min']:.2e}, {stat['pauli_max']:.2e}]\n\n")
    
    print(f"Summary report saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze operator norm differences between matching/Pauli decompositions and exact CTQW',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s graphs.g6
  %(prog)s graphs.g6 -m 1 -M 20 -s 2 -t 0.1 0.5 1.0
  %(prog)s graphs.g6 --min_steps 5 --max_steps 100 --step_inc 10 --time_values 0.01 0.1 1.0
  %(prog)s graphs.g6 -m 1 -M 50 -s 5 -t 0.1 1.0 -o custom_output
        """
    )
    
    # Positional argument
    parser.add_argument('g6_file', help='Path to G6 file')
    
    # Trotter steps arguments with shortcuts
    parser.add_argument('--min_steps', '-m', type=int, default=1,
                       help='Minimum number of Trotter steps (default: 1)')
    parser.add_argument('--max_steps', '-M', type=int, default=20,
                       help='Maximum number of Trotter steps (default: 20)')
    parser.add_argument('--step_inc', '-s', type=int, default=1,
                       help='Increment for Trotter steps (default: 1)')
    
    # Other arguments
    parser.add_argument('-t', '--time_values', nargs='+', type=float, default=[0.1, 0.5, 1.0], 
                       help='Time values to test (default: 0.1 0.5 1.0)')
    parser.add_argument('-o', '--output_dir', default=None, 
                       help='Output directory (default: analysis/outputs/plots/[auto-generated])')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_steps < 1:
        parser.error("--min_steps must be at least 1")
    if args.max_steps < args.min_steps:
        parser.error("--max_steps must be greater than or equal to --min_steps")
    if args.step_inc < 1:
        parser.error("--step_inc must be at least 1")
    
    # Generate Trotter steps list
    trotter_steps_list = list(range(args.min_steps, args.max_steps + 1, args.step_inc))
    
    # Load all graphs
    graphs, metadata = load_all_graphs(args.g6_file)
    
    if not graphs:
        print("No graphs found in file")
        return
    
    # Get expected vertex count from filename
    expected_vertices = metadata.get('vertices')
    if expected_vertices and isinstance(expected_vertices, int):
        n_qubits = int(np.log2(expected_vertices)) if expected_vertices > 0 else 0
        print(f"Expected vertex count from filename: {expected_vertices} ({n_qubits} qubits)")
        
        # Warn about computational complexity for large systems
        if n_qubits > 8:
            print(f"WARNING: {n_qubits}-qubit systems will be computationally intensive!")
            print(f"Each operator will be {2**n_qubits} x {2**n_qubits} = {(2**n_qubits)**2:,} elements")
            memory_estimate = (2**n_qubits)**2 * 16 / (1024**3)  # 16 bytes per complex number, convert to GB
            print(f"Estimated memory per operator: {memory_estimate:.1f} GB")
            if n_qubits > 10:
                print("Consider using fewer Trotter steps for very large systems.")
    else:
        expected_vertices = None
        print("Could not determine expected vertex count from filename")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        graph_type = metadata.get('type', 'Unknown')
        vertices = metadata.get('vertices', 'Unknown')
        # Create output directory in analysis/outputs/plots
        base_output_dir = Path("analysis") / "outputs" / "plots"
        base_output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = base_output_dir / f"{graph_type}_{vertices}v_ctqw_comparison"
    
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    print(f"\nAnalysis parameters:")
    print(f"  Trotter steps: {args.min_steps} to {args.max_steps} (increment: {args.step_inc})")
    print(f"  Generated steps: {trotter_steps_list}")
    print(f"  Time values: {args.time_values}")
    print(f"  Comparison reference: Exact CTQW")
    print(f"  Methods: Matching decomposition, Pauli decomposition")
    
    # Process all graphs
    all_results = process_all_graphs(graphs, trotter_steps_list, args.time_values, expected_vertices)
    
    if not all_results:
        print("No graphs were successfully processed")
        if expected_vertices:
            vertex_check = expected_vertices > 0 and (expected_vertices & (expected_vertices - 1)) == 0
            if not vertex_check:
                print(f"Note: Expected vertex count {expected_vertices} is not a power of 2")
        return
    
    # Create analysis plots and statistics
    print("\nCreating analysis plots and statistics...")
    create_convergence_plots(all_results, metadata, output_dir)
    create_summary_statistics(all_results, metadata, output_dir)
    
    print(f"\nAnalysis complete!")
    print(f"Processed {len(all_results)} graphs successfully")
    print(f"Both matching and Pauli decompositions compared against exact CTQW")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()