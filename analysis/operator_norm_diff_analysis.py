#!/usr/bin/env python3
"""
Fixed Operator 2-norm Difference Between Matching and Pauli Decompositions for G6 Files

Compares Matching and Pauli decompositions with proper sign conventions across all graphs in a G6 file.
Uses only 2-norm (spectral norm) for operator differences.
Fixed sign convention issues that were causing large artificial differences.

Usage: python operator_norm_diff_analysis.py <g6_file> [options]
Examples:
  python operator_norm_diff_analysis.py graphs.g6
  python operator_norm_diff_analysis.py graphs.g6 -N 15 -t 0.1 0.5 1.0 2.0
  python operator_norm_diff_analysis.py graphs.g6 --max_trotter_steps 20 --time_values 0.01 0.1 1.0
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

def create_matching_circuit(edges, n_steps, time_step):
    """Create quantum circuit using matching decomposition with Trotterization."""
    # Use relabeled graph for consistency (same as Pauli method)
    matchings = graph_matchings_parallel(edges)
    relabeled_edges = set()
    for m in matchings:
        relabeled_edges = relabeled_edges.union(m)
    
    relabeled_G = StaticGraph(relabeled_edges)
    intersecting_G = IntersectingEdgesGraph(edges, matchings='parallel')
    
    full_qc = QuantumCircuit(relabeled_G.n_qubits)
    
    for step in range(n_steps):
        for subgraph in intersecting_G.subgraphs:
            G = MultiEdgeGraph(subgraph.edges)
            # Fixed: Use consistent positive time evolution
            G.rot_angle = time_step / n_steps
            sub_qc = G.get_qc(simplified=True)
            full_qc = full_qc.compose(sub_qc)
    
    return full_qc

def create_pauli_circuit(edges, time_step):
    """Create quantum circuit using Pauli decomposition (exact evolution)."""
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
    
    # Create exact Pauli evolution circuit (no Trotterization for reference)
    pauli_qc = QuantumCircuit(n)
    
    if coeffs:
        pauli_op = SparsePauliOp(pauli_strings, coeffs)
        # Fixed: Use consistent positive time evolution
        evo_gate = PauliEvolutionGate(pauli_op, time=time_step)
        pauli_qc.append(evo_gate, range(n))
    
    # Decompose to get actual gates
    decomposed_qc = pauli_qc.decompose().decompose().decompose()
    return decomposed_qc

def compute_operator_difference(edges, trotter_steps_list, time_values):
    """Compute 2-norm differences between matching and Pauli operators."""
    results = {
        'trotter_steps': trotter_steps_list,
        'time_values': time_values,
        'differences': {},
        'properties': analyze_graph_properties(edges)
    }
    
    for time_val in time_values:
        differences = []
        
        # Get exact Pauli reference (computed once per time_val)
        try:
            pauli_qc = create_pauli_circuit(edges, time_val)
            pauli_op = Operator(pauli_qc)
        except Exception as e:
            print(f"Error creating Pauli circuit for time {time_val}: {e}")
            results['differences'][time_val] = [np.nan] * len(trotter_steps_list)
            continue
        
        for n_steps in trotter_steps_list:
            try:
                # Create matching circuit with n_steps Trotter steps
                matching_qc = create_matching_circuit(edges, n_steps, time_val)
                matching_op = Operator(matching_qc)
                
                # Calculate 2-norm difference (spectral norm)
                diff = matching_op - pauli_op
                two_norm = np.linalg.norm(diff.data, ord=2)
                differences.append(two_norm)
                
            except Exception as e:
                print(f"Error for time {time_val}, steps {n_steps}: {e}")
                differences.append(np.nan)
        
        results['differences'][time_val] = differences
    
    return results

def generate_trotter_steps(max_steps=None, min_steps=None, increment=None, steps_list=None):
    """Generate Trotter steps list with appropriate increments or custom list."""
    
    # If custom list is provided, use it directly
    if steps_list is not None:
        return sorted(list(set(steps_list)))  # Remove duplicates and sort
    
    # If min/max/increment are provided, use them
    if min_steps is not None and max_steps is not None and increment is not None:
        return list(range(min_steps, max_steps + 1, increment))
    
    # Default intelligent increments based on max_steps
    if max_steps is None:
        max_steps = 10  # Default fallback
    
    if max_steps <= 20:
        # For small max_steps, use increment of 1
        return list(range(1, max_steps + 1))
    elif max_steps <= 50:
        # For medium max_steps, use increment of 5
        steps = list(range(5, max_steps + 1, 5))
        if 1 not in steps:
            steps = [1] + steps
        return sorted(steps)
    else:
        # For large max_steps, use increment of 10
        steps = list(range(10, max_steps + 1, 10))
        if 1 not in steps:
            steps = [1] + steps
        if 5 not in steps and max_steps > 5:
            steps = [1, 5] + [s for s in steps if s > 5]
        return sorted(steps)

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
            results = compute_operator_difference(edges, trotter_steps_list, time_values)
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
    """Create convergence analysis plot with all times in one plot."""
    if not all_results:
        print("No results to plot")
        return
    
    time_values = all_results[0]['time_values']
    trotter_steps = all_results[0]['trotter_steps']
    
    # Create single convergence plot
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(time_values)))
    
    for color, time_val in zip(colors, time_values):
        # Calculate mean and std for each Trotter step
        means = []
        stds = []
        
        for step_idx in range(len(trotter_steps)):
            step_diffs = []
            for result in all_results:
                if step_idx < len(result['differences'][time_val]):
                    diff = result['differences'][time_val][step_idx]
                    if not np.isnan(diff):
                        step_diffs.append(diff)
            
            if step_diffs:
                means.append(np.mean(step_diffs))
                stds.append(np.std(step_diffs))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        # Filter out NaN values for plotting
        valid_indices = [i for i, (m, s) in enumerate(zip(means, stds)) 
                        if not np.isnan(m) and not np.isnan(s)]
        
        if valid_indices:
            valid_steps = [trotter_steps[i] for i in valid_indices]
            valid_means = [means[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices]
            
            # Plot mean with error bars
            plt.errorbar(valid_steps, valid_means, yerr=valid_stds, 
                        marker='o', linewidth=2, capsize=5, capthick=2,
                        label=f'Time = {time_val}', color=color)
    
    plt.xlabel('Trotter Steps', fontsize=12)
    plt.ylabel('2-norm Difference', fontsize=12)
    plt.title(f'2-norm Difference Between Matching and Pauli Decompositions\n{metadata["type"]} graphs, {metadata["vertices"]} vertices', 
              fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_dir / f'{metadata["type"]}_{metadata["vertices"]}v_convergence.pdf'
    plt.savefig(plot_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {plot_file}")

def create_summary_statistics(all_results, metadata, output_dir):
    """Create summary statistics and save to files."""
    if not all_results:
        return
    
    time_values = all_results[0]['time_values']
    trotter_steps = all_results[0]['trotter_steps']
    
    # Compile statistics
    stats_data = []
    
    for time_val in time_values:
        # Final convergence values
        final_values = []
        for result in all_results:
            differences = result['differences'][time_val]
            if differences and not np.isnan(differences[-1]):
                final_values.append(differences[-1])
        
        if final_values:
            stats_data.append({
                'time': time_val,
                'n_graphs': len(final_values),
                'mean_final_diff': np.mean(final_values),
                'std_final_diff': np.std(final_values),
                'median_final_diff': np.median(final_values),
                'min_final_diff': np.min(final_values),
                'max_final_diff': np.max(final_values)
            })
    
    # Save as CSV
    if stats_data:
        df = pd.DataFrame(stats_data)
        csv_file = output_dir / f'{metadata["type"]}_{metadata["vertices"]}v_statistics.csv'
        df.to_csv(csv_file, index=False)
        print(f"Statistics saved to {csv_file}")
    
    # Save detailed results
    results_file = output_dir / f'{metadata["type"]}_{metadata["vertices"]}v_detailed_results.npz'
    
    # Prepare data for saving
    save_data = {
        'metadata': metadata,
        'trotter_steps': trotter_steps,
        'time_values': time_values,
        'n_graphs_processed': len(all_results)
    }
    
    # Add difference arrays for each time value
    for time_val in time_values:
        differences_array = []
        for result in all_results:
            differences_array.append(result['differences'][time_val])
        save_data[f'differences_time_{time_val}'] = differences_array
    
    # Add graph properties
    properties_list = [result['properties'] for result in all_results]
    save_data['graph_properties'] = properties_list
    
    np.savez(results_file, **save_data)
    print(f"Detailed results saved to {results_file}")
    
    # Create text summary
    summary_file = output_dir / f'{metadata["type"]}_{metadata["vertices"]}v_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Operator 2-norm Difference Between Matching and Pauli Decompositions Summary\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Dataset: {metadata['filename']}\n")
        f.write(f"Graph type: {metadata['type']}\n")
        f.write(f"Vertices: {metadata['vertices']}\n")
        f.write(f"Expected graphs: {metadata.get('n_graphs', 'Unknown')}\n")
        f.write(f"Processed graphs: {len(all_results)}\n\n")
        
        f.write(f"Analysis parameters:\n")
        f.write(f"  Trotter steps: {min(trotter_steps)} to {max(trotter_steps)} (steps: {trotter_steps})\n")
        f.write(f"  Time values: {time_values}\n\n")
        
        if stats_data:
            f.write(f"Summary statistics:\n")
            for stat in stats_data:
                f.write(f"  Time {stat['time']}:\n")
                f.write(f"    Mean final 2-norm difference: {stat['mean_final_diff']:.2e}\n")
                f.write(f"    Std final 2-norm difference: {stat['std_final_diff']:.2e}\n")
                f.write(f"    Median final 2-norm difference: {stat['median_final_diff']:.2e}\n")
                f.write(f"    Range: [{stat['min_final_diff']:.2e}, {stat['max_final_diff']:.2e}]\n\n")
    
    print(f"Summary report saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze 2-norm difference between matching and Pauli decompositions for all graphs in G6 file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s graphs.g6
  %(prog)s graphs.g6 -N 15 -t 0.1 0.5 1.0 2.0
  %(prog)s graphs.g6 --max_trotter_steps 100 --time_values 0.01 0.1 1.0
  %(prog)s graphs.g6 --trotter_steps 10 20 30 40 50 -t 0.1 1.0
  %(prog)s graphs.g6 -m 5 -M 50 -i 10 -t 0.1
  %(prog)s graphs.g6 -N 80 -t 0.1 1.0 -o custom_output
        """
    )
    
    # Positional argument
    parser.add_argument('g6_file', help='Path to G6 file')
    
    # Trotter steps arguments (mutually exclusive groups)
    trotter_group = parser.add_mutually_exclusive_group()
    trotter_group.add_argument('-N', '--max_trotter_steps', type=int, default=10, 
                              help='Maximum number of Trotter steps with intelligent increments (default: 10)')
    trotter_group.add_argument('--trotter_steps', nargs='+', type=int, 
                              help='Custom list of Trotter steps (e.g., --trotter_steps 10 20 30 40)')
    
    # Min/max/increment for Trotter steps (only valid when used together)
    parser.add_argument('--min_trotter', '-m', type=int, 
                       help='Minimum Trotter steps (use with --max_trotter and --trotter_inc)')
    parser.add_argument('--max_trotter', '-M', type=int, 
                       help='Maximum Trotter steps (use with --min_trotter and --trotter_inc)')
    parser.add_argument('--trotter_inc', '-i', type=int, 
                       help='Trotter steps increment (use with --min_trotter and --max_trotter)')
    
    # Other arguments
    parser.add_argument('-t', '--time_values', nargs='+', type=float, default=[0.1, 0.5, 1.0], 
                       help='Time values to test (default: 0.1 0.5 1.0)')
    parser.add_argument('-o', '--output_dir', default=None, 
                       help='Output directory (default: analysis/outputs/plots/[auto-generated])')
    
    args = parser.parse_args()
    
    # Validate min/max/increment arguments
    min_max_inc_args = [args.min_trotter, args.max_trotter, args.trotter_inc]
    min_max_inc_provided = sum(x is not None for x in min_max_inc_args)
    
    if min_max_inc_provided > 0 and min_max_inc_provided < 3:
        parser.error("--min_trotter, --max_trotter, and --trotter_inc must all be provided together")
    
    if args.trotter_steps and min_max_inc_provided > 0:
        parser.error("Cannot use --trotter_steps with --min_trotter/--max_trotter/--trotter_inc")
    
    if args.max_trotter_steps != 10 and min_max_inc_provided > 0:  # 10 is the default
        parser.error("Cannot use -N/--max_trotter_steps with --min_trotter/--max_trotter/--trotter_inc")
    
    if args.max_trotter_steps != 10 and args.trotter_steps:  # 10 is the default
        parser.error("Cannot use -N/--max_trotter_steps with --trotter_steps")
    
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
        output_dir = base_output_dir / f"{graph_type}_{vertices}v"
    
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Generate Trotter steps list based on arguments
    if args.trotter_steps:
        trotter_steps_list = generate_trotter_steps(steps_list=args.trotter_steps)
        print(f"Using custom Trotter steps: {trotter_steps_list}")
    elif min_max_inc_provided == 3:
        trotter_steps_list = generate_trotter_steps(
            min_steps=args.min_trotter, 
            max_steps=args.max_trotter, 
            increment=args.trotter_inc
        )
        print(f"Using min/max/increment - Steps: {args.min_trotter} to {args.max_trotter} by {args.trotter_inc}")
        print(f"Generated steps: {trotter_steps_list}")
    else:
        trotter_steps_list = generate_trotter_steps(max_steps=args.max_trotter_steps)
        print(f"Using intelligent increments up to {args.max_trotter_steps}")
        print(f"Generated steps: {trotter_steps_list}")
    
    print(f"\nAnalysis parameters:")
    print(f"  Trotter steps: {trotter_steps_list}")
    print(f"  Time values: {args.time_values}")
    
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
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()