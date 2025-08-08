#!/usr/bin/env python3
"""
CTQW Analysis: Matching vs Pauli Decomposition with Degree-Based Heuristic Hypercube Labeling
Standalone script for processing large quantum graph datasets with degree-based vertex labeling.

Implementation of the degree-based heuristic algorithm from pseudocode:
- Start with vertex M having maximum degree Δ(G)
- Assign labels to minimize Hamming distance while maximizing edges with Hamming distance 1
- Use bit-flip tracking to maintain consistency across neighborhoods
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add the current directory to Python path for imports
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

import networkx as nx
import numpy as np

# Import from quantum_walk_utils
from quantum_walk_utils import *


def calculate_graph_properties(graph: nx.Graph) -> dict:
    """Calculate various graph properties for analysis"""
    properties = {}
    
    # Basic properties
    properties["edge_count"] = len(graph.edges())
    properties["edge_density"] = nx.density(graph)
    properties["is_bipartite"] = nx.is_bipartite(graph)
    
    # Connected graph properties
    if nx.is_connected(graph):
        properties["diameter"] = nx.diameter(graph)
    else:
        properties["diameter"] = None
        
    # Clique number (maximum clique size)
    try:
        properties["clique_number"] = len(max(nx.find_cliques(graph), key=len))
    except:
        properties["clique_number"] = 1
    
    # Degree properties
    degrees = [graph.degree(node) for node in graph.nodes()]
    properties["max_degree"] = max(degrees) if degrees else 0
    properties["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0
    
    # Clustering coefficient
    properties["avg_clustering"] = nx.average_clustering(graph)
    
    return properties


class HypercubeHeuristicAnalyzer:
    def __init__(self, n_qubits: int, delta_t: float, matchings: str, logger: Logger):
        self.base_analyzer = BaseAnalyzer(n_qubits, delta_t, matchings)
        self.logger = logger

    def analyze_graph(self, graph: nx.Graph, n_steps: int = 1) -> dict:
        start_time = time.time()

        # Get original and hypercube edges
        original_edges = GraphProcessor.graph_to_bitstring(graph)
        hypercube_edges = GraphProcessor.graph_to_bitstring_hypercube(graph)
        
        # Calculate Hamming costs
        original_hamming_cost = sum(
            DegreeBasedHypercubeLabeler.hamming_distance(u, v) 
            for u, v in original_edges
        )
        hypercube_hamming_cost = sum(
            DegreeBasedHypercubeLabeler.hamming_distance(u, v) 
            for u, v in hypercube_edges
        )
        hamming_improvement = original_hamming_cost - hypercube_hamming_cost
        
        # Get hypercube labeling quality statistics
        hypercube_labeling = HeuristicHypercubeLabeler.heuristic_hypercube_labeling(graph)
        quality_stats = DegreeBasedHypercubeLabeler.analyze_labeling_quality(graph, hypercube_labeling)
        
        # Debug: Compare labeling methods
        original_labeling = {node: format(node, f"0{max(1, len(bin(len(graph.nodes()) - 1)) - 2)}b") for node in graph.nodes()}
        original_quality = DegreeBasedHypercubeLabeler.analyze_labeling_quality(graph, original_labeling)
        
        self.logger.log(f"Original Hamming cost: {original_hamming_cost} (avg: {original_quality.get('avg_cost', 0):.2f})")
        self.logger.log(f"Degree-based heuristic Hamming cost: {hypercube_hamming_cost} (avg: {quality_stats.get('avg_cost', 0):.2f})")
        self.logger.log(f"Hamming improvement: {hamming_improvement}")
        self.logger.log(f"Original edges with dist 1: {original_quality.get('edges_with_dist_1', 0)} ({original_quality.get('edges_with_dist_1_ratio', 0):.3f})")
        self.logger.log(f"Degree-based edges with dist 1: {quality_stats.get('edges_with_dist_1', 0)} ({quality_stats.get('edges_with_dist_1_ratio', 0):.3f})")
        self.logger.log(f"Hamming distance distribution: {quality_stats.get('hamming_dist_distribution', {})}")

        # Analyze with original labeling
        self.logger.log("=== ORIGINAL LABELING ANALYSIS ===")
        
        self.logger.log("Computing Original Matching Dynamic walk")
        original_matching_metrics = self.base_analyzer.analyze_matching(original_edges, n_steps)
        if not original_matching_metrics:
            self.logger.log("Failed to get original matching metrics")
            return None
        self.logger.log_metrics(
            "Original Matching",
            original_matching_metrics.cx_count,
            original_matching_metrics.u3_count,
            original_matching_metrics.depth,
        )

        self.logger.log("Computing Original Pauli decomposition")
        original_pauli_metrics = self.base_analyzer.analyze_pauli(original_edges)
        if not original_pauli_metrics:
            self.logger.log("Failed to get original Pauli metrics")
            return None
        self.logger.log_metrics(
            "Original Pauli", 
            original_pauli_metrics.cx_count, 
            original_pauli_metrics.u3_count, 
            original_pauli_metrics.depth
        )

        # Analyze with degree-based heuristic labeling (ONLY MATCHING)
        self.logger.log("=== DEGREE-BASED HEURISTIC LABELING ANALYSIS (MATCHING ONLY) ===")
        
        self.logger.log("Computing Degree-based Heuristic Matching Dynamic walk")
        hypercube_matching_metrics = self.base_analyzer.analyze_matching(hypercube_edges, n_steps)
        if not hypercube_matching_metrics:
            self.logger.log("Failed to get degree-based heuristic matching metrics")
            return None
        self.logger.log_metrics(
            "Degree-based Heuristic Matching",
            hypercube_matching_metrics.cx_count,
            hypercube_matching_metrics.u3_count,
            hypercube_matching_metrics.depth,
        )

        # Calculate differences for original labeling (Matching vs Pauli)
        original_cx_diff = original_matching_metrics.cx_count - original_pauli_metrics.cx_count
        original_u3_diff = original_matching_metrics.u3_count - original_pauli_metrics.u3_count

        if original_cx_diff < 0:
            original_category = "win"
        elif original_cx_diff > 0:
            original_category = "lose"
        else:
            original_category = "draw"

        # Calculate main comparison: Degree-based Heuristic Matching vs Original Pauli
        hypercube_vs_pauli_cx_diff = hypercube_matching_metrics.cx_count - original_pauli_metrics.cx_count
        hypercube_vs_pauli_u3_diff = hypercube_matching_metrics.u3_count - original_pauli_metrics.u3_count

        if hypercube_vs_pauli_cx_diff < 0:
            hypercube_vs_pauli_category = "win"
        elif hypercube_vs_pauli_cx_diff > 0:
            hypercube_vs_pauli_category = "lose"
        else:
            hypercube_vs_pauli_category = "draw"

        # Calculate improvements from degree-based heuristic labeling
        matching_cx_improvement = original_matching_metrics.cx_count - hypercube_matching_metrics.cx_count
        matching_u3_improvement = original_matching_metrics.u3_count - hypercube_matching_metrics.u3_count
        matching_depth_improvement = original_matching_metrics.depth - hypercube_matching_metrics.depth

        # Calculate graph properties
        graph_properties = calculate_graph_properties(graph)

        analysis_time = time.time() - start_time
        self.logger.log(f"Analysis completed in {analysis_time:.2f}s")
        self.logger.log(f"Original Matching vs Pauli category: {original_category}")
        self.logger.log(f"Degree-based Heuristic Matching vs Original Pauli category: {hypercube_vs_pauli_category}")
        self.logger.log(f"Matching CX improvement from degree-based heuristic: {matching_cx_improvement}")
        
        # Additional debug info
        if matching_cx_improvement < 0:
            self.logger.log(f"WARNING: Degree-based heuristic labeling made CX count WORSE by {-matching_cx_improvement}")
            self.logger.log(f"Graph properties: edges={len(graph.edges())}, max_degree={graph_properties.get('max_degree', 0)}")

        return {
            # Primary category (based on original matching vs pauli for consistency)
            "category": original_category,
            
            # Original labeling results
            "original_matching_cx": original_matching_metrics.cx_count,
            "original_matching_u3": original_matching_metrics.u3_count,
            "original_matching_depth": original_matching_metrics.depth,
            "original_pauli_cx": original_pauli_metrics.cx_count,
            "original_pauli_u3": original_pauli_metrics.u3_count,
            "original_pauli_depth": original_pauli_metrics.depth,
            "original_cx_diff": original_cx_diff,
            "original_u3_diff": original_u3_diff,
            "original_category": original_category,
            
            # Degree-based heuristic labeling results (matching only)
            "degree_based_matching_cx": hypercube_matching_metrics.cx_count,
            "degree_based_matching_u3": hypercube_matching_metrics.u3_count,
            "degree_based_matching_depth": hypercube_matching_metrics.depth,
            
            # Degree-based Heuristic Matching vs Original Pauli comparison
            "degree_based_vs_pauli_cx_diff": hypercube_vs_pauli_cx_diff,
            "degree_based_vs_pauli_u3_diff": hypercube_vs_pauli_u3_diff,
            "degree_based_vs_pauli_category": hypercube_vs_pauli_category,
            
            # Labeling comparison
            "original_hamming_cost": original_hamming_cost,
            "degree_based_hamming_cost": hypercube_hamming_cost,
            "hamming_improvement": hamming_improvement,
            
            # Circuit improvements from degree-based heuristic labeling
            "matching_cx_improvement": matching_cx_improvement,
            "matching_u3_improvement": matching_u3_improvement,
            "matching_depth_improvement": matching_depth_improvement,
            
            # Quality metrics for degree-based heuristic labeling
            "edges_with_hamming_dist_1": quality_stats.get('edges_with_dist_1', 0),
            "edges_with_dist_1_ratio": quality_stats.get('edges_with_dist_1_ratio', 0),
            "avg_edge_hamming_cost": quality_stats.get('avg_cost', 0),
            "max_edge_hamming_dist": quality_stats.get('max_hamming_dist', 0),
            
            # Quality comparison metrics
            "original_edges_with_dist_1_ratio": original_quality.get('edges_with_dist_1_ratio', 0),
            "original_avg_edge_hamming_cost": original_quality.get('avg_cost', 0),
            
            # Meta information
            "analysis_time": analysis_time,
            "graph_properties": graph_properties,
        }


def save_checkpoint(results, categories, checkpoint_file):
    """Save progress so we can resume if interrupted"""
    try:
        checkpoint_data = {
            "results": results,
            "categories": categories,
            "processed_count": len(results),
            "timestamp": time.time(),
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"✓ Saved checkpoint: {len(results)} graphs processed")
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


def load_checkpoint(checkpoint_file):
    """Load previous progress if available"""
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
                print(f"✓ Found checkpoint: resuming from {data['processed_count']} graphs")
                return data
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
    return None


def setup_paths(script_dir):
    """Setup and validate all necessary paths"""
    # Data directory - default to parent/data/graphs
    data_dir = script_dir.parent / "data" / "graphs"

    # Output directory - default to script_dir/outputs
    output_dir = script_dir / "outputs" / "degree_based_heuristic_analysis"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, output_dir


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CTQW Analysis: Matching vs Pauli Decomposition using Degree-Based Heuristic Hypercube Labeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument(
        "--n-vertices", "-v",
        type=int,
        default=8,
        help="Number of vertices (determines number of qubits as log2(n_vertices))"
    )
    
    parser.add_argument(
        "--delta-t", "-dt",
        type=float,
        default=0.1,
        help="Time step for quantum walk evolution"
    )
    
    parser.add_argument(
        "--n-graphs", "-n",
        type=int,
        default=100,
        help="Number of graphs to process (overridden to 11117 when using BM graphs)"
    )
    
    parser.add_argument(
        "--graph-type", "-t",
        type=str,
        default="random",
        choices=["BM", "random"],
        help="Type of graphs to process"
    )
    
    parser.add_argument(
        "--matchings", "-m",
        type=str,
        default="greedy",
        choices=["greedy", "parallel"],
        help="Matching algorithm to use"
    )
    
    # Processing options
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N graphs"
    )
    
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Show progress every N graphs"
    )
    
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1,
        help="Number of Trotter steps"
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Calculate n_qubits from n_vertices and validate it's a power of 2
    import math
    if args.n_vertices <= 0:
        print(f"❌ Error: n_vertices must be positive, got {args.n_vertices}")
        return 1
    
    # Check if n_vertices is a power of 2
    if (args.n_vertices & (args.n_vertices - 1)) != 0:
        print(f"❌ Error: n_vertices must be a power of 2 (2, 4, 8, 16, 32, ...), got {args.n_vertices}")
        print(f"💡 Valid values: {[2**i for i in range(1, 8)]}")
        return 1
    
    n_qubits = int(math.log2(args.n_vertices))
    
    # Override n_graphs to 11117 for BM graphs
    if args.graph_type == "BM":
        args.n_graphs = 11117
    
    print("=" * 80)
    print("🚀 CTQW ANALYSIS - MATCHING VS PAULI WITH DEGREE-BASED HEURISTIC HYPERCUBE LABELING")
    print("=" * 80)

    # Calculate derived values
    n_vertices = args.n_vertices

    # Setup paths
    script_dir = Path(__file__).parent.absolute()
    data_dir, output_dir = setup_paths(script_dir)

    # Determine input file automatically from arguments
    graph_file = data_dir / f"{args.n_graphs}graph_{args.graph_type}_{n_vertices}c.g6"

    print(f"📁 Script location: {script_dir}")
    print(f"📊 Input file: {graph_file}")
    print(f"💾 Output directory: {output_dir}")
    print(f"📈 Processing {args.n_graphs} {args.graph_type} graphs with {n_qubits} qubits ({n_vertices} vertices)")
    print(f"🔧 Using {args.matchings} matching algorithm")
    print(f"🎯 Using degree-based heuristic hypercube embedding labeling")
    print(f"⏱️  Delta t: {args.delta_t}")

    # Verify input file exists
    if not graph_file.exists():
        print(f"❌ Error: Input file not found: {graph_file}")
        print(f"💡 Tip: Check --n-graphs, --graph-type, and --n-vertices arguments")
        print(f"📁 Expected data directory: {data_dir}")
        return 1

    # Setup directories and logging
    try:
        dirs = setup_directories(str(output_dir))
        log_filename = f"degree_based_heuristic_analysis_{args.matchings}_{args.graph_type}_{n_vertices}c.log"
        logger = Logger(os.path.join(dirs["logs"], log_filename))
        checkpoint_file = output_dir / f"checkpoint_degree_based_{args.matchings}_{args.graph_type}_{n_vertices}c.json"
    except Exception as e:
        print(f"❌ Error setting up directories: {e}")
        return 1

    # Log initial parameters
    logger.log(f"\n\nStarting degree-based heuristic hypercube labeling analysis with:")
    logger.log(f"Script location: {script_dir}")
    logger.log(f"Number of qubits: {n_qubits}")
    logger.log(f"Number of vertices: {n_vertices}")
    logger.log(f"Delta t: {args.delta_t}")
    logger.log(f"Matching algorithm: {args.matchings}")
    logger.log(f"Graph type: {args.graph_type}")
    logger.log(f"Input file: {graph_file}")
    logger.log(f"Output directory: {output_dir}")
    logger.log(f"Checkpoint interval: {args.checkpoint_interval}")
    logger.log(f"Using degree-based heuristic hypercube embedding labeling")

    try:
        # Parse graph info and setup analyzer
        graph_info = GraphProcessor.parse_graph_filename(str(graph_file))
        logger.log(f"Parsed graph info: {graph_info}")

        analyzer = HypercubeHeuristicAnalyzer(n_qubits, args.delta_t, args.matchings, logger)
        results_manager = ResultsManager(dirs["results"], graph_info)
        plot_manager = PlotManager(dirs["plots"])

        # Try to load checkpoint
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            results = checkpoint_data["results"]
            categories = checkpoint_data["categories"]
            start_index = len(results)
            print(f"🔄 Resuming from graph {start_index}")
        else:
            results = []
            categories = {"win": 0, "lose": 0, "draw": 0}
            start_index = 0
            print(f"🆕 Starting fresh analysis")

        original_graphs = []
        # Initialize property collections by category
        properties_by_category = {
            "win": defaultdict(list),
            "lose": defaultdict(list), 
            "draw": defaultdict(list)
        }

        # Initialize labeling improvement statistics
        labeling_stats = {
            "total_hamming_improvement": 0,
            "total_matching_cx_improvement": 0,
            "total_matching_u3_improvement": 0,
            "total_degree_based_vs_pauli_wins": 0,  # Degree-based heuristic matching better than original Pauli
            "total_edges_with_dist_1": 0,
            "total_edges_with_dist_1_ratio": 0,
            "graphs_with_improvements": 0,
            "graphs_processed": 0
        }

        # Count total graphs
        with open(graph_file, "r") as f:
            total_lines = sum(1 for _ in f)
        logger.log(f"Found {total_lines} graphs in file")
        print(f"📈 Found {total_lines} total graphs")

        # Process graphs
        start_time = time.time()
        print(f"⏱️  Analysis started at {time.strftime('%H:%M:%S')}")
        print("-" * 80)

        with open(graph_file, "r") as f:
            for i, line in enumerate(f):
                # Skip to checkpoint position
                if i < start_index:
                    continue

                # Progress updates
                if i % args.progress_interval == 0:
                    progress = i / total_lines * 100
                    elapsed = time.time() - start_time
                    if i > start_index:
                        speed = (i - start_index) / (elapsed / 60)  # graphs per minute
                        eta_hours = (total_lines - i) / speed / 60 if speed > 0 else 0
                        
                        # Calculate current average analysis time
                        current_avg_time = ""
                        if results:
                            recent_times = [r["analysis_time"] for r in results[-10:] if "analysis_time" in r]
                            if recent_times:
                                avg_time = sum(recent_times) / len(recent_times)
                                current_avg_time = f" | Avg: {avg_time:.1f}s/graph"
                        
                        # Show degree-based heuristic improvements
                        avg_hamming_improvement = labeling_stats["total_hamming_improvement"] / max(1, labeling_stats["graphs_processed"])
                        avg_cx_improvement = labeling_stats["total_matching_cx_improvement"] / max(1, labeling_stats["graphs_processed"])
                        degree_based_win_rate = (labeling_stats["total_degree_based_vs_pauli_wins"] / max(1, labeling_stats["graphs_processed"])) * 100
                        avg_dist_1_ratio = labeling_stats["total_edges_with_dist_1_ratio"] / max(1, labeling_stats["graphs_processed"])
                        
                        print(
                            f"📊 Progress: {progress:.1f}% ({i}/{total_lines}) | "
                            f"Speed: {speed:.1f}/min | ETA: {eta_hours:.1f}h{current_avg_time} | "
                            f"Results: W{categories['win']} L{categories['lose']} D{categories['draw']} | "
                            f"Hamming↓{avg_hamming_improvement:.1f} CX↓{avg_cx_improvement:.1f} DvP:{degree_based_win_rate:.0f}% Dist1:{avg_dist_1_ratio:.2f}"
                        )

                logger.log(f"\nProcessing graph {i}/{total_lines}")
                try:
                    line = line.strip()
                    if not line:
                        logger.log(f"Empty line at {i}, skipping")
                        continue

                    graph = nx.from_graph6_bytes(line.encode())
                    logger.log(
                        f"Graph {i} has {len(graph.nodes())} nodes and {len(graph.edges())} edges"
                    )

                    result = analyzer.analyze_graph(graph, args.n_steps)
                    if result:
                        categories[result["category"]] += 1
                        results.append({"index": i, **result})
                        original_graphs.append(graph)
                        
                        # Update labeling statistics
                        labeling_stats["total_hamming_improvement"] += result["hamming_improvement"]
                        labeling_stats["total_matching_cx_improvement"] += result["matching_cx_improvement"]
                        labeling_stats["total_matching_u3_improvement"] += result["matching_u3_improvement"]
                        labeling_stats["total_edges_with_dist_1"] += result.get("edges_with_hamming_dist_1", 0)
                        labeling_stats["total_edges_with_dist_1_ratio"] += result.get("edges_with_dist_1_ratio", 0)
                        
                        if result["degree_based_vs_pauli_category"] == "win":
                            labeling_stats["total_degree_based_vs_pauli_wins"] += 1
                        labeling_stats["graphs_processed"] += 1
                        
                        if (result["hamming_improvement"] > 0 or 
                            result["matching_cx_improvement"] > 0):
                            labeling_stats["graphs_with_improvements"] += 1
                        
                        # Collect graph properties by category
                        category = result["category"]
                        graph_props = result["graph_properties"]
                        for prop_name, prop_value in graph_props.items():
                            if prop_value is not None and isinstance(prop_value, (int, float)):
                                properties_by_category[category][prop_name].append(prop_value)
                        
                        logger.log(
                            f"Successfully processed graph {i} - Category: {result['category']} - Time: {result['analysis_time']:.2f}s"
                        )
                        logger.log(
                            f"Improvements: Hamming: {result['hamming_improvement']}, "
                            f"Matching CX: {result['matching_cx_improvement']}, "
                            f"Degree-based vs Pauli: {result['degree_based_vs_pauli_category']}, "
                            f"Edges with dist 1 ratio: {result.get('edges_with_dist_1_ratio', 0):.3f}"
                        )
                    else:
                        logger.log(f"Failed to analyze graph {i}")

                    # Save checkpoint every N graphs
                    if i % args.checkpoint_interval == 0 and i > start_index:
                        save_checkpoint(results, categories, checkpoint_file)

                    # Clean up memory every 20 graphs
                    if i % 20 == 0:
                        gc.collect()

                except KeyboardInterrupt:
                    print(f"\n⏹️  Interrupted by user at graph {i}")
                    logger.log(f"Interrupted by user at graph {i}")
                    save_checkpoint(results, categories, checkpoint_file)
                    return 0
                except Exception as e:
                    logger.log(f"Error processing graph {i}: {str(e)}")
                    print(f"⚠️  Error at graph {i}: {str(e)}")
                    continue

        # Final Results
        print("\n" + "=" * 80)
        print("🏁 DEGREE-BASED HEURISTIC HYPERCUBE LABELING ANALYSIS COMPLETE")
        print("=" * 80)

        total_processed = sum(categories.values())
        success_rate = (total_processed / total_lines) * 100 if total_lines > 0 else 0

        logger.log("\nFinal Results Summary:")
        logger.log(f"Total graphs in file: {total_lines}")
        logger.log(f"Successfully processed: {total_processed}")
        logger.log(f"Matching algorithm used: {args.matchings}")
        logger.log(f"Using degree-based heuristic hypercube embedding labeling")
        logger.log_final_stats(categories, total_processed)

        # Log degree-based heuristic labeling improvements
        if labeling_stats["graphs_processed"] > 0:
            avg_hamming_improvement = labeling_stats["total_hamming_improvement"] / labeling_stats["graphs_processed"]
            avg_matching_cx_improvement = labeling_stats["total_matching_cx_improvement"] / labeling_stats["graphs_processed"]
            avg_matching_u3_improvement = labeling_stats["total_matching_u3_improvement"] / labeling_stats["graphs_processed"]
            degree_based_vs_pauli_win_rate = (labeling_stats["total_degree_based_vs_pauli_wins"] / labeling_stats["graphs_processed"]) * 100
            avg_edges_with_dist_1 = labeling_stats["total_edges_with_dist_1"] / labeling_stats["graphs_processed"]
            avg_edges_with_dist_1_ratio = labeling_stats["total_edges_with_dist_1_ratio"] / labeling_stats["graphs_processed"]
            
            improvement_rate = (labeling_stats["graphs_with_improvements"] / labeling_stats["graphs_processed"]) * 100
            
            logger.log("\nDegree-Based Heuristic Hypercube Labeling Improvement Summary:")
            logger.log(f"Average Hamming cost improvement: {avg_hamming_improvement:.2f}")
            logger.log(f"Average Matching CX improvement: {avg_matching_cx_improvement:.2f}")
            logger.log(f"Average Matching U3 improvement: {avg_matching_u3_improvement:.2f}")
            logger.log(f"Degree-based Heuristic Matching vs Original Pauli win rate: {degree_based_vs_pauli_win_rate:.1f}%")
            logger.log(f"Average edges with Hamming distance 1: {avg_edges_with_dist_1:.2f}")
            logger.log(f"Average ratio of edges with Hamming distance 1: {avg_edges_with_dist_1_ratio:.3f}")
            logger.log(f"Graphs with any improvement: {labeling_stats['graphs_with_improvements']} ({improvement_rate:.1f}%)")

        # Calculate overall timing statistics
        if results:
            analysis_times = [r["analysis_time"] for r in results if "analysis_time" in r]
            if analysis_times:
                avg_time_per_graph = sum(analysis_times) / len(analysis_times)
                min_time = min(analysis_times)
                max_time = max(analysis_times)
                logger.log(f"Average time per graph: {avg_time_per_graph:.2f}s")
                logger.log(f"Min time per graph: {min_time:.2f}s")
                logger.log(f"Max time per graph: {max_time:.2f}s")
                logger.log(f"Total analysis time: {sum(analysis_times):.2f}s")

        # Log detailed graph properties by category
        logger.log("\nDetailed Graph Properties by Category:\n")
        
        for category in ["win", "lose", "draw"]:
            if category in properties_by_category and properties_by_category[category]:
                props = properties_by_category[category]
                if len(props.get("edge_count", [])) > 0:  # Only log if we have data
                    logger.log(f"\n{category.upper()} Graphs Statistics:")
                    logger.log(f"Total graphs processed: {len(props['edge_count'])}")
                    
                    # Edge properties
                    if props.get("edge_count"):
                        edge_counts = props["edge_count"]
                        logger.log(f"Edge count - Min: {min(edge_counts)}, "
                                  f"Max: {max(edge_counts)}, "
                                  f"Mean: {np.mean(edge_counts):.2f}")
                    
                    if props.get("edge_density"):
                        edge_densities = props["edge_density"]
                        logger.log(f"Edge density - Min: {min(edge_densities):.4f}, "
                                  f"Max: {max(edge_densities):.4f}, "
                                  f"Mean: {np.mean(edge_densities):.4f}")
                    
                    # Degree properties
                    if props.get("max_degree"):
                        max_degrees = props["max_degree"]
                        logger.log(f"Max degree - Min: {min(max_degrees)}, "
                                  f"Max: {max(max_degrees)}, "
                                  f"Mean: {np.mean(max_degrees):.2f}")
                    
                    # Bipartite classification
                    if props.get("is_bipartite"):
                        bipartite_count = sum(props["is_bipartite"])
                        non_bipartite_count = len(props["is_bipartite"]) - bipartite_count
                        logger.log(f"Bipartite graphs: {bipartite_count}, "
                                  f"Non-bipartite graphs: {non_bipartite_count}")

        print(f"📊 Total graphs: {total_lines}")
        print(f"✅ Successfully processed: {total_processed} ({success_rate:.1f}%)")
        print(f"🔧 Matching algorithm: {args.matchings}")
        print(f"🎯 Using degree-based heuristic hypercube embedding labeling")
        print(f"🏆 Results breakdown:")
        print(f"   🟢 Win (Matching better):  {categories['win']}")
        print(f"   🔴 Lose (Pauli better):    {categories['lose']}")
        print(f"   🟡 Draw (Equal):           {categories['draw']}")

        # Display labeling improvement statistics
        if labeling_stats["graphs_processed"] > 0:
            avg_hamming_improvement = labeling_stats["total_hamming_improvement"] / labeling_stats["graphs_processed"]
            avg_matching_cx_improvement = labeling_stats["total_matching_cx_improvement"] / labeling_stats["graphs_processed"]
            degree_based_vs_pauli_win_rate = (labeling_stats["total_degree_based_vs_pauli_wins"] / labeling_stats["graphs_processed"]) * 100
            avg_edges_with_dist_1_ratio = labeling_stats["total_edges_with_dist_1_ratio"] / labeling_stats["graphs_processed"]
            improvement_rate = (labeling_stats["graphs_with_improvements"] / labeling_stats["graphs_processed"]) * 100
            
            print(f"🎯 Degree-Based Heuristic Hypercube Labeling Improvements:")
            print(f"   📉 Average Hamming reduction: {avg_hamming_improvement:.2f}")
            print(f"   🚀 Average Matching CX reduction: {avg_matching_cx_improvement:.2f}")
            print(f"   🏆 Degree-based Heuristic Matching vs Original Pauli win rate: {degree_based_vs_pauli_win_rate:.1f}%")
            print(f"   🎯 Average ratio of edges with Hamming distance 1: {avg_edges_with_dist_1_ratio:.3f}")
            print(f"   ✨ Graphs with improvements: {labeling_stats['graphs_with_improvements']} ({improvement_rate:.1f}%)")

        # Display timing statistics
        if results:
            analysis_times = [r["analysis_time"] for r in results if "analysis_time" in r]
            if analysis_times:
                avg_time_per_graph = sum(analysis_times) / len(analysis_times)
                min_time = min(analysis_times)
                max_time = max(analysis_times)
                total_analysis_time = sum(analysis_times)
                print(f"⏱️  Timing Statistics:")
                print(f"   📊 Average time per graph: {avg_time_per_graph:.2f}s")
                print(f"   ⚡ Fastest graph: {min_time:.2f}s")
                print(f"   🐌 Slowest graph: {max_time:.2f}s")
                print(f"   🕐 Total analysis time: {total_analysis_time:.2f}s ({total_analysis_time/60:.1f}m)")

        if total_processed > 0:
            # Separate results by category
            win_results = [r for r in results if r["category"] == "win"]
            lose_results = [r for r in results if r["category"] == "lose"]
            draw_results = [r for r in results if r["category"] == "draw"]

            # Calculate category averages including degree-based heuristic improvements
            def calc_category_avgs(cat_results, prefix):
                if not cat_results:
                    return {}
                
                # All metrics to average
                metrics = [
                    "original_matching_cx", "original_matching_u3", "original_matching_depth",
                    "original_pauli_cx", "original_pauli_u3", "original_pauli_depth",
                    "degree_based_matching_cx", "degree_based_matching_u3", "degree_based_matching_depth", 
                    "original_hamming_cost", "degree_based_hamming_cost", "hamming_improvement",
                    "matching_cx_improvement", "matching_u3_improvement", "matching_depth_improvement",
                    "degree_based_vs_pauli_cx_diff", "degree_based_vs_pauli_u3_diff",
                    "edges_with_hamming_dist_1", "edges_with_dist_1_ratio", "avg_edge_hamming_cost",
                    "max_edge_hamming_dist", "original_edges_with_dist_1_ratio", "original_avg_edge_hamming_cost",
                    "analysis_time"
                ]
                    
                totals = defaultdict(float)
                counts = defaultdict(int)
                
                for r in cat_results:
                    # Regular metrics
                    for metric in metrics:
                        if metric in r and isinstance(r[metric], (int, float)):
                            totals[metric] += r[metric]
                            counts[metric] += 1
                    
                    # Graph properties (handle different types appropriately)
                    if "graph_properties" in r:
                        graph_props = r["graph_properties"]
                        for prop_name, prop_value in graph_props.items():
                            if prop_value is not None and isinstance(prop_value, (int, float)):
                                totals[f"graph_{prop_name}"] += prop_value
                                counts[f"graph_{prop_name}"] += 1
                
                # Calculate averages
                result = {}
                for metric in totals:
                    if counts[metric] > 0:
                        result[f"{prefix}_{metric}"] = totals[metric] / counts[metric]
                
                return result

            avg_stats = {}
            avg_stats.update(calc_category_avgs(win_results, "win"))
            avg_stats.update(calc_category_avgs(lose_results, "lose"))
            avg_stats.update(calc_category_avgs(draw_results, "draw"))

            # Add configuration, timing stats, and labeling improvements to avg_stats
            timing_stats = {}
            if results:
                analysis_times = [r["analysis_time"] for r in results if "analysis_time" in r]
                if analysis_times:
                    timing_stats = {
                        "timing_avg_per_graph": sum(analysis_times) / len(analysis_times),
                        "timing_min_per_graph": min(analysis_times),
                        "timing_max_per_graph": max(analysis_times),
                        "timing_total_analysis": sum(analysis_times)
                    }
            
            # Add labeling improvement statistics
            if labeling_stats["graphs_processed"] > 0:
                labeling_improvement_stats = {
                    "labeling_avg_hamming_improvement": labeling_stats["total_hamming_improvement"] / labeling_stats["graphs_processed"],
                    "labeling_avg_matching_cx_improvement": labeling_stats["total_matching_cx_improvement"] / labeling_stats["graphs_processed"],
                    "labeling_avg_matching_u3_improvement": labeling_stats["total_matching_u3_improvement"] / labeling_stats["graphs_processed"],
                    "labeling_degree_based_vs_pauli_win_rate": (labeling_stats["total_degree_based_vs_pauli_wins"] / labeling_stats["graphs_processed"]) * 100,
                    "labeling_avg_edges_with_dist_1": labeling_stats["total_edges_with_dist_1"] / labeling_stats["graphs_processed"],
                    "labeling_avg_edges_with_dist_1_ratio": labeling_stats["total_edges_with_dist_1_ratio"] / labeling_stats["graphs_processed"],
                    "labeling_improvement_rate": (labeling_stats["graphs_with_improvements"] / labeling_stats["graphs_processed"]) * 100
                }
                avg_stats.update(labeling_improvement_stats)
            
            avg_stats.update({
                "config_n_qubits": n_qubits,
                "config_n_vertices": n_vertices,
                "config_delta_t": args.delta_t,
                "config_matchings": args.matchings,
                "config_graph_type": args.graph_type,
                "config_n_steps": args.n_steps,
                "config_using_degree_based_heuristic_hypercube_labeling": True
            })
            avg_stats.update(timing_stats)

            # Log averages and timing
            logger.log("\nAverage Metrics by Category (with Degree-Based Heuristic Hypercube Labeling):")
            for metric, value in avg_stats.items():
                if isinstance(value, (int, float)):
                    logger.log(f"{metric}: {value:.2f}")
                else:
                    logger.log(f"{metric}: {value}")
            
            logger.log("=" * 70 + "\n" + "=" * 70 + "\n")
            print(f"\n💾 Saving results...")
            
            # Save results with timing and labeling improvement information
            summary_with_enhancements = {**categories}
            if timing_stats:
                summary_with_enhancements.update(timing_stats)
            summary_with_enhancements.update(labeling_stats)
            
            results_manager.save_results(summary_with_enhancements, f"summary_degree_based_{args.matchings}")
            results_manager.save_results({"detailed_results": results}, f"detailed_degree_based_{args.matchings}")
            results_manager.save_results(avg_stats, f"averages_degree_based_{args.matchings}")

            # Save categorized graphs
            results_manager.save_categorized_graphs(results, original_graphs)

            # Create visualization
            title = f"Degree-Based Heuristic Hypercube Labeling: Matching ({args.matchings}) vs Pauli Results ({n_qubits} qubits)"
            pie_chart = plot_manager.create_pie_chart(categories, title)
            chart_filename = f'pie_chart_degree_based_{args.matchings}_{graph_info["size"]}_{graph_info["vertices"]}c.png'
            plot_manager.save_plot(pie_chart, chart_filename)

            # Clean up checkpoint
            if checkpoint_file.exists():
                checkpoint_file.unlink()

            print(f"✅ Results saved to: {output_dir}")
            print(f"📈 Chart saved as: {chart_filename}")

        else:
            logger.log("No results to save - all graphs failed processing")
            print("❌ No results to save - all graphs failed processing")

        total_time = time.time() - start_time
        print(f"\n⏱️  Total runtime: {total_time/3600:.1f} hours")
        print(f"🏁 Analysis completed at {time.strftime('%H:%M:%S')}")

        return 0

    except Exception as e:
        logger.log(f"Critical error in main execution: {str(e)}")
        print(f"❌ Critical error: {str(e)}")
        # Save whatever we have so far
        if "results" in locals() and "categories" in locals():
            save_checkpoint(results, categories, checkpoint_file)
        raise


if __name__ == "__main__":
    """Entry point when run as standalone script"""
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n⏹️  Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)