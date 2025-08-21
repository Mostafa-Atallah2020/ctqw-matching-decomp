#!/usr/bin/env python3
"""
CTQW Analysis: Matching vs Pauli Decomposition
Standalone script for processing large quantum graph datasets.
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
    
    # Automorphism group size estimation (simplified)
    try:
        # This is a rough estimate - exact calculation is computationally expensive
        properties["estimated_group_size"] = estimate_group_size(graph)
    except:
        properties["estimated_group_size"] = 1
        
    # Orbit count estimation (simplified)
    try:
        properties["estimated_orbit_count"] = estimate_orbit_count(graph)
    except:
        properties["estimated_orbit_count"] = len(graph.nodes())
    
    return properties


def estimate_group_size(graph: nx.Graph) -> int:
    """Rough estimation of automorphism group size"""
    # Simple heuristic based on symmetry indicators
    n = len(graph.nodes())
    if n <= 1:
        return 1
    
    # Check for some common symmetric structures
    if nx.is_regular(graph):
        degree = list(graph.degree())[0][1]
        if degree == n - 1:  # Complete graph
            return math.factorial(n)
        elif degree == 0:  # Empty graph
            return math.factorial(n)
        elif degree == 1:  # Matching or path-like
            # Rough estimate for matching-like structures
            return 2 ** (n // 2)
    
    # Default conservative estimate
    return max(1, n // 4)


def estimate_orbit_count(graph: nx.Graph) -> int:
    """Rough estimation of number of orbits under automorphism group"""
    # Group nodes by degree sequence and other simple invariants
    degree_sequence = sorted([graph.degree(node) for node in graph.nodes()])
    unique_degrees = len(set(degree_sequence))
    
    # Very rough heuristic
    return min(len(graph.nodes()), max(1, unique_degrees))


class MatchingVsPauliAnalyzer:
    def __init__(self, n_qubits: int, delta_t: float, matchings: str, logger: Logger):
        self.analyzer = BaseAnalyzer(n_qubits, delta_t, matchings)
        self.logger = logger

    def analyze_graph(self, edges: set, graph: nx.Graph, n_steps: int = 1) -> dict:
        start_time = time.time()

        self.logger.log("Computing Matchings Dynamic walk")
        matching_metrics = self.analyzer.analyze_matching(edges, n_steps)
        if not matching_metrics:
            self.logger.log("Failed to get matching metrics")
            return None
        self.logger.log_metrics(
            "Matchings",
            matching_metrics.cx_count,
            matching_metrics.u3_count,
            matching_metrics.depth,
        )

        self.logger.log("Computing Pauli decomposition")
        pauli_metrics = self.analyzer.analyze_pauli(edges)
        if not pauli_metrics:
            self.logger.log("Failed to get Pauli metrics")
            return None
        self.logger.log_metrics(
            "Pauli", pauli_metrics.cx_count, pauli_metrics.u3_count, pauli_metrics.depth
        )

        cx_diff = matching_metrics.cx_count - pauli_metrics.cx_count
        u3_diff = matching_metrics.u3_count - pauli_metrics.u3_count

        if cx_diff < 0:
            category = "win"
        elif cx_diff > 0:
            category = "lose"
        else:
            category = "draw"

        # Calculate graph properties
        graph_properties = calculate_graph_properties(graph)

        analysis_time = time.time() - start_time
        self.logger.log(f"Analysis completed in {analysis_time:.2f}s - Category: {category}")

        return {
            "category": category,
            "matching_cx": matching_metrics.cx_count,
            "matching_u3": matching_metrics.u3_count,
            "pauli_cx": pauli_metrics.cx_count,
            "pauli_u3": pauli_metrics.u3_count,
            "cx_diff": cx_diff,
            "u3_diff": u3_diff,
            "matching_depth": matching_metrics.depth,
            "pauli_depth": pauli_metrics.depth,
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
    output_dir = script_dir / "outputs" / "matching_vs_pauli"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, output_dir


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CTQW Analysis: Matching vs Pauli Decomposition",
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
        default="parallel",
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
    
    print("=" * 70)
    print("🚀 CTQW ANALYSIS - MATCHING VS PAULI DECOMPOSITION")
    print("=" * 70)

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
        log_filename = f"analysis_{args.matchings}_{args.graph_type}_{n_vertices}c.log"
        logger = Logger(os.path.join(dirs["logs"], log_filename))
        checkpoint_file = output_dir / f"checkpoint_{args.matchings}_{args.graph_type}_{n_vertices}c.json"
    except Exception as e:
        print(f"❌ Error setting up directories: {e}")
        return 1

    # Log initial parameters
    logger.log(f"\n\nStarting analysis with:")
    logger.log(f"Script location: {script_dir}")
    logger.log(f"Number of qubits: {n_qubits}")
    logger.log(f"Number of vertices: {n_vertices}")
    logger.log(f"Delta t: {args.delta_t}")
    logger.log(f"Matching algorithm: {args.matchings}")
    logger.log(f"Graph type: {args.graph_type}")
    logger.log(f"Input file: {graph_file}")
    logger.log(f"Output directory: {output_dir}")
    logger.log(f"Checkpoint interval: {args.checkpoint_interval}")

    try:
        # Parse graph info and setup analyzer
        graph_info = GraphProcessor.parse_graph_filename(str(graph_file))
        logger.log(f"Parsed graph info: {graph_info}")

        analyzer = MatchingVsPauliAnalyzer(n_qubits, args.delta_t, args.matchings, logger)
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

        # Count total graphs
        with open(graph_file, "r") as f:
            total_lines = sum(1 for _ in f)
        logger.log(f"Found {total_lines} graphs in file")
        print(f"📈 Found {total_lines} total graphs")

        # Process graphs
        start_time = time.time()
        print(f"⏱️  Analysis started at {time.strftime('%H:%M:%S')}")
        print("-" * 70)

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
                        
                        print(
                            f"📊 Progress: {progress:.1f}% ({i}/{total_lines}) | "
                            f"Speed: {speed:.1f}/min | ETA: {eta_hours:.1f}h{current_avg_time} | "
                            f"Results: W{categories['win']} L{categories['lose']} D{categories['draw']}"
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

                    edges = GraphProcessor.graph_to_bitstring(graph)
                    logger.log(f"Converted to {len(edges)} bitstring edges")

                    result = analyzer.analyze_graph(edges, graph, args.n_steps)
                    if result:
                        categories[result["category"]] += 1
                        results.append({"index": i, **result})
                        original_graphs.append(graph)
                        
                        # Collect graph properties by category
                        category = result["category"]
                        graph_props = result["graph_properties"]
                        for prop_name, prop_value in graph_props.items():
                            if prop_value is not None:
                                properties_by_category[category][prop_name].append(prop_value)
                        
                        logger.log(
                            f"Successfully processed graph {i} - Category: {result['category']} - Time: {result['analysis_time']:.2f}s"
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
        print("\n" + "=" * 70)
        print("🏁 ANALYSIS COMPLETE")
        print("=" * 70)

        total_processed = sum(categories.values())
        success_rate = (total_processed / total_lines) * 100 if total_lines > 0 else 0

        logger.log("\nFinal Results Summary:")
        logger.log(f"Total graphs in file: {total_lines}")
        logger.log(f"Successfully processed: {total_processed}")
        logger.log(f"Matching algorithm used: {args.matchings}")
        logger.log_final_stats(categories, total_processed)

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
                    
                    # Bipartite classification
                    if props.get("is_bipartite"):
                        bipartite_count = sum(props["is_bipartite"])
                        non_bipartite_count = len(props["is_bipartite"]) - bipartite_count
                        logger.log(f"Bipartite graphs: {bipartite_count}, "
                                  f"Non-bipartite graphs: {non_bipartite_count}")
                    
                    # Diameter (only for connected graphs)
                    diameters = [d for d in props.get("diameter", []) if d is not None]
                    if diameters:
                        logger.log(f"Diameter (connected graphs) - Min: {min(diameters)}, "
                                  f"Max: {max(diameters)}, "
                                  f"Mean: {np.mean(diameters):.2f}")
                    
                    # Clique number
                    if props.get("clique_number"):
                        clique_numbers = props["clique_number"]
                        logger.log(f"Clique number - Min: {min(clique_numbers)}, "
                                  f"Max: {max(clique_numbers)}, "
                                  f"Mean: {np.mean(clique_numbers):.2f}")
                    
                    # Degree properties
                    if props.get("max_degree"):
                        max_degrees = props["max_degree"]
                        logger.log(f"Maximum degree - Min: {min(max_degrees)}, "
                                  f"Max: {max(max_degrees)}, "
                                  f"Mean: {np.mean(max_degrees):.2f}")
                    
                    if props.get("avg_degree"):
                        avg_degrees = props["avg_degree"]
                        logger.log(f"Average degree - Min: {min(avg_degrees):.2f}, "
                                  f"Max: {max(avg_degrees):.2f}, "
                                  f"Mean: {np.mean(avg_degrees):.2f}")
                    
                    # Clustering coefficient
                    if props.get("avg_clustering"):
                        clustering_coeffs = props["avg_clustering"]
                        logger.log(f"Clustering coefficient - Min: {min(clustering_coeffs):.4f}, "
                                  f"Max: {max(clustering_coeffs):.4f}, "
                                  f"Mean: {np.mean(clustering_coeffs):.4f}")
                    
                    # Group size estimation
                    if props.get("estimated_group_size"):
                        group_sizes = props["estimated_group_size"]
                        logger.log(f"Estimated group size - Min: {min(group_sizes)}, "
                                  f"Max: {max(group_sizes)}, "
                                  f"Mean: {np.mean(group_sizes):.2f}")
                    
                    # Orbit count estimation
                    if props.get("estimated_orbit_count"):
                        orbit_counts = props["estimated_orbit_count"]
                        logger.log(f"Estimated orbit count - Min: {min(orbit_counts)}, "
                                  f"Max: {max(orbit_counts)}, "
                                  f"Mean: {np.mean(orbit_counts):.2f}")

        print(f"📊 Total graphs: {total_lines}")
        print(f"✅ Successfully processed: {total_processed} ({success_rate:.1f}%)")
        print(f"🔧 Matching algorithm: {args.matchings}")
        print(f"🏆 Results breakdown:")
        print(f"   🟢 Win (Matching better):  {categories['win']}")
        print(f"   🔴 Lose (Pauli better):    {categories['lose']}")
        print(f"   🟡 Draw (Equal):           {categories['draw']}")

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

            # Calculate category averages including graph properties
            def calc_category_avgs(cat_results, prefix):
                if not cat_results:
                    return {}
                metrics = [
                    "matching_cx",
                    "matching_u3",
                    "pauli_cx",
                    "pauli_u3",
                    "matching_depth",
                    "pauli_depth",
                ]
                if cat_results and "analysis_time" in cat_results[0]:
                    metrics.append("analysis_time")
                    
                totals = defaultdict(float)
                counts = defaultdict(int)
                
                for r in cat_results:
                    # Regular metrics
                    for metric in metrics:
                        if metric in r:
                            totals[metric] += r[metric]
                            counts[metric] += 1
                    
                    # Graph properties
                    if "graph_properties" in r:
                        graph_props = r["graph_properties"]
                        for prop_name, prop_value in graph_props.items():
                            if prop_value is not None:
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

            # Add configuration and timing stats to avg_stats
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
            
            avg_stats.update({
                "config_n_qubits": n_qubits,
                "config_n_vertices": n_vertices,
                "config_delta_t": args.delta_t,
                "config_matchings": args.matchings,
                "config_graph_type": args.graph_type,
                "config_n_steps": args.n_steps
            })
            avg_stats.update(timing_stats)

            # Log averages and timing
            logger.log("\nAverage Gate Counts by Category:")
            for metric, value in avg_stats.items():
                if isinstance(value, (int, float)):
                    logger.log(f"{metric}: {value:.2f}")
                else:
                    logger.log(f"{metric}: {value}")
                    # End log with two solid lines
            
            logger.log("=" * 70 + "\n" + "=" * 70 + "\n")
            print(f"\n💾 Saving results...")
            # Save results with timing information
            summary_with_timing = {**categories}
            if timing_stats:
                summary_with_timing.update(timing_stats)
            
            results_manager.save_results(summary_with_timing, f"summary_{args.matchings}")
            results_manager.save_results({"detailed_results": results}, f"detailed_{args.matchings}")
            results_manager.save_results(avg_stats, f"averages_{args.matchings}")

            # Save categorized graphs
            results_manager.save_categorized_graphs(results, original_graphs)

            # Create visualization
            title = f"Matching ({args.matchings}) vs Pauli Results ({n_qubits} qubits)"
            pie_chart = plot_manager.create_pie_chart(categories, title)
            chart_filename = f'pie_chart_{args.matchings}_{graph_info["size"]}_{graph_info["vertices"]}c.png'
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