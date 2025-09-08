#!/usr/bin/env python3
"""
Log Parser for CTQW Analysis Results
Parses multiple runs from log files and calculates statistics (mean ± std)
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CTQWLogParser:
    """Parser for CTQW analysis log files"""
    
    def __init__(self):
        # Key metrics to extract and calculate statistics for
        self.key_metrics = [
            'win_percentage', 'lose_percentage', 'draw_percentage',
            'win_count', 'lose_count', 'draw_count', 'total_processed',
            'timing_avg_per_graph', 'timing_min_per_graph', 'timing_max_per_graph', 'timing_total_analysis',
            
            # Win category averages
            'win_matching_cx', 'win_matching_u3', 'win_pauli_cx', 'win_pauli_u3',
            'win_matching_depth', 'win_pauli_depth', 'win_analysis_time',
            'win_graph_edge_count', 'win_graph_edge_density', 'win_graph_diameter',
            'win_graph_clique_number', 'win_graph_max_degree', 'win_graph_avg_degree',
            'win_graph_avg_clustering', 'win_graph_estimated_group_size', 'win_graph_estimated_orbit_count',
            
            # Win graph structure statistics
            'win_graph_bipartite_count', 'win_graph_nonbipartite_count',
            
            # Lose category averages
            'lose_matching_cx', 'lose_matching_u3', 'lose_pauli_cx', 'lose_pauli_u3',
            'lose_matching_depth', 'lose_pauli_depth', 'lose_analysis_time',
            'lose_graph_edge_count', 'lose_graph_edge_density', 'lose_graph_diameter',
            'lose_graph_clique_number', 'lose_graph_max_degree', 'lose_graph_avg_degree',
            'lose_graph_avg_clustering', 'lose_graph_estimated_group_size', 'lose_graph_estimated_orbit_count',
            
            # Lose graph structure statistics
            'lose_graph_bipartite_count', 'lose_graph_nonbipartite_count',
            
            # Draw category averages
            'draw_matching_cx', 'draw_matching_u3', 'draw_pauli_cx', 'draw_pauli_u3',
            'draw_matching_depth', 'draw_pauli_depth', 'draw_analysis_time',
            'draw_graph_edge_count', 'draw_graph_edge_density', 'draw_graph_diameter',
            'draw_graph_clique_number', 'draw_graph_max_degree', 'draw_graph_avg_degree',
            'draw_graph_avg_clustering', 'draw_graph_estimated_group_size', 'draw_graph_estimated_orbit_count',
            
            # Draw graph structure statistics
            'draw_graph_bipartite_count', 'draw_graph_nonbipartite_count',
        ]
        
        # Configuration parameters to extract (these should be consistent across runs)
        self.config_params = [
            'config_n_qubits', 'config_n_vertices', 'config_delta_t', 
            'config_matchings', 'config_graph_type', 'config_n_steps'
        ]

    def find_run_boundaries(self, log_content: str) -> List[Tuple[int, int]]:
        """Find the start and end positions of each run in the log file"""
        lines = log_content.split('\n')
        
        # Find all "Final Results Summary:" lines
        start_indices = []
        for i, line in enumerate(lines):
            if "Final Results Summary:" in line.strip():
                start_indices.append(i)
        
        # Find all "timing_total_analysis:" lines (these mark the end of runs)
        end_indices = []
        for i, line in enumerate(lines):
            if "timing_total_analysis:" in line.strip():
                end_indices.append(i)
        
        print(f"Found {len(start_indices)} start markers and {len(end_indices)} end markers")
        if start_indices:
            print(f"Start lines: {start_indices}")
        if end_indices:
            print(f"End lines: {end_indices}")
        
        # Create run boundaries by pairing starts with ends
        runs = []
        for i in range(min(len(start_indices), len(end_indices))):
            runs.append((start_indices[i], end_indices[i]))
            print(f"Run {i+1}: lines {start_indices[i]} to {end_indices[i]}")
            
        return runs

    def extract_config_from_run_start(self, lines: List[str], run_index: int) -> Dict:
        """Extract configuration from the beginning of a run"""
        config_data = {}
        
        # Find the corresponding "Starting analysis with:" line
        start_markers = []
        for i, line in enumerate(lines):
            if "Starting analysis with:" in line:
                start_markers.append(i)
        
        if run_index < len(start_markers):
            config_start = start_markers[run_index]
            # Look at the next ~20 lines for config data
            config_end = min(config_start + 20, len(lines))
            
            for i in range(config_start, config_end):
                line = lines[i].strip()
                
                if "Number of qubits:" in line:
                    match = re.search(r'Number of qubits:\s*(\d+)', line)
                    if match:
                        config_data['config_n_qubits'] = float(match.group(1))
                        
                elif "Number of vertices:" in line:
                    match = re.search(r'Number of vertices:\s*(\d+)', line)
                    if match:
                        config_data['config_n_vertices'] = float(match.group(1))
                        
                elif "Delta t:" in line:
                    match = re.search(r'Delta t:\s*([0-9.]+)', line)
                    if match:
                        config_data['config_delta_t'] = float(match.group(1))
                        
                elif "Matching algorithm:" in line:
                    match = re.search(r'Matching algorithm:\s*([a-zA-Z_]+)', line)
                    if match:
                        config_data['config_matchings'] = match.group(1)
                        
                elif "Graph type:" in line:
                    match = re.search(r'Graph type:\s*([a-zA-Z_]+)', line)
                    if match:
                        config_data['config_graph_type'] = match.group(1)
        
        return config_data

    def parse_graph_statistics_section(self, lines: List[str], start_idx: int, category: str) -> Dict:
        """Parse a graph statistics section (WIN/LOSE/DRAW)"""
        stats = {}
        
        # Look for the section header
        section_found = False
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Remove timestamp prefix if present
            if line.startswith('[2025-'):
                bracket_end = line.find('] ')
                if bracket_end != -1:
                    line = line[bracket_end + 2:].strip()
            
            # Check if we found the right section
            if f"{category} Graphs Statistics:" in line:
                section_found = True
                i += 1
                continue
            
            # If we haven't found the section yet, keep looking
            if not section_found:
                i += 1
                continue
            
            # Stop if we hit another section or empty line after statistics
            if (("Graphs Statistics:" in line and category not in line) or 
                (line == "" and i > start_idx + 10)):
                break
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Parse specific statistics
            if "Total graphs processed:" in line:
                match = re.search(r'Total graphs processed:\s*(\d+)', line)
                if match:
                    stats[f'{category.lower()}_graph_total_processed'] = int(match.group(1))
                    
            elif "Bipartite graphs:" in line and "Non-bipartite graphs:" in line:
                # Parse: "Bipartite graphs: 11, Non-bipartite graphs: 0"
                bipartite_match = re.search(r'Bipartite graphs:\s*(\d+)', line)
                nonbipartite_match = re.search(r'Non-bipartite graphs:\s*(\d+)', line)
                
                if bipartite_match:
                    stats[f'{category.lower()}_graph_bipartite_count'] = int(bipartite_match.group(1))
                if nonbipartite_match:
                    stats[f'{category.lower()}_graph_nonbipartite_count'] = int(nonbipartite_match.group(1))
            
            # Parse mean values from existing patterns
            elif "Mean:" in line:
                if "Edge count" in line:
                    match = re.search(r'Mean:\s*([0-9.]+)', line)
                    if match:
                        stats[f'{category.lower()}_graph_edge_count'] = float(match.group(1))
                        
                elif "Edge density" in line:
                    match = re.search(r'Mean:\s*([0-9.]+)', line)
                    if match:
                        stats[f'{category.lower()}_graph_edge_density'] = float(match.group(1))
                        
                elif "Diameter" in line:
                    match = re.search(r'Mean:\s*([0-9.]+)', line)
                    if match:
                        stats[f'{category.lower()}_graph_diameter'] = float(match.group(1))
                        
                elif "Clique number" in line:
                    match = re.search(r'Mean:\s*([0-9.]+)', line)
                    if match:
                        stats[f'{category.lower()}_graph_clique_number'] = float(match.group(1))
                        
                elif "Maximum degree" in line:
                    match = re.search(r'Mean:\s*([0-9.]+)', line)
                    if match:
                        stats[f'{category.lower()}_graph_max_degree'] = float(match.group(1))
                        
                elif "Average degree" in line:
                    match = re.search(r'Mean:\s*([0-9.]+)', line)
                    if match:
                        stats[f'{category.lower()}_graph_avg_degree'] = float(match.group(1))
                        
                elif "Clustering coefficient" in line:
                    match = re.search(r'Mean:\s*([0-9.]+)', line)
                    if match:
                        stats[f'{category.lower()}_graph_avg_clustering'] = float(match.group(1))
                        
                elif "Estimated group size" in line:
                    match = re.search(r'Mean:\s*([0-9.]+)', line)
                    if match:
                        stats[f'{category.lower()}_graph_estimated_group_size'] = float(match.group(1))
                        
                elif "Estimated orbit count" in line:
                    match = re.search(r'Mean:\s*([0-9.]+)', line)
                    if match:
                        stats[f'{category.lower()}_graph_estimated_orbit_count'] = float(match.group(1))
            
            i += 1
        
        return stats

    def parse_run(self, run_content: str) -> Optional[Dict]:
        """Parse a single run and extract all relevant metrics"""
        data = {}
        lines = run_content.split('\n')
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # Remove timestamp prefix if present: [2025-09-07 02:11:54]
            if line.startswith('[2025-'):
                # Find the closing bracket and extract everything after it
                bracket_end = line.find('] ')
                if bracket_end != -1:
                    line = line[bracket_end + 2:].strip()
            
            # Skip empty lines or lines that are just equals signs
            if not line or line.startswith('=') or "Final Results Summary" in line:
                continue
            
            # Extract basic results (win/lose/draw counts and percentages)
            if line.startswith("win:") and "(" in line and "%" in line:
                match = re.search(r'win:\s*(\d+)\s*\(([0-9.]+)%\)', line)
                if match:
                    data['win_count'] = int(match.group(1))
                    data['win_percentage'] = float(match.group(2))
                    
            elif line.startswith("lose:") and "(" in line and "%" in line:
                match = re.search(r'lose:\s*(\d+)\s*\(([0-9.]+)%\)', line)
                if match:
                    data['lose_count'] = int(match.group(1))
                    data['lose_percentage'] = float(match.group(2))
                    
            elif line.startswith("draw:") and "(" in line and "%" in line:
                match = re.search(r'draw:\s*(\d+)\s*\(([0-9.]+)%\)', line)
                if match:
                    data['draw_count'] = int(match.group(1))
                    data['draw_percentage'] = float(match.group(2))
            
            # Total processed
            elif "Successfully processed:" in line:
                match = re.search(r'Successfully processed:\s*(\d+)', line)
                if match:
                    data['total_processed'] = int(match.group(1))
            
            # Timing information
            elif "Average time per graph:" in line:
                match = re.search(r'Average time per graph:\s*([0-9.]+)s', line)
                if match:
                    data['timing_avg_per_graph'] = float(match.group(1))
                    
            elif "Min time per graph:" in line:
                match = re.search(r'Min time per graph:\s*([0-9.]+)s', line)
                if match:
                    data['timing_min_per_graph'] = float(match.group(1))
                    
            elif "Max time per graph:" in line:
                match = re.search(r'Max time per graph:\s*([0-9.]+)s', line)
                if match:
                    data['timing_max_per_graph'] = float(match.group(1))
                    
            elif "Total analysis time:" in line:
                match = re.search(r'Total analysis time:\s*([0-9.]+)s', line)
                if match:
                    data['timing_total_analysis'] = float(match.group(1))
            
            # Parse graph statistics sections
            elif "WIN Graphs Statistics:" in line:
                win_stats = self.parse_graph_statistics_section(lines, i, "WIN")
                data.update(win_stats)
                
            elif "LOSE Graphs Statistics:" in line:
                lose_stats = self.parse_graph_statistics_section(lines, i, "LOSE")
                data.update(lose_stats)
                
            elif "DRAW Graphs Statistics:" in line:
                draw_stats = self.parse_graph_statistics_section(lines, i, "DRAW")
                data.update(draw_stats)
            
            # Category averages - look for exact metric patterns
            else:
                # Check for any of our key metrics
                for metric in self.key_metrics:
                    if metric.endswith('_percentage') or metric.endswith('_count') or metric.startswith('timing_'):
                        continue  # Already handled above
                        
                    # Look for pattern like "win_matching_cx: 105.00"
                    if line.startswith(f"{metric}:"):
                        match = re.search(f'^{metric}:\\s*([0-9.]+)', line)
                        if match:
                            data[metric] = float(match.group(1))
                            break
        
        # Ensure all key metrics are present with default values
        for metric in self.key_metrics:
            if metric not in data:
                data[metric] = 0.0
        
        # Print what we found
        print(f"  Extracted {len(data)} metrics")
        
        # Return None if we didn't extract meaningful data
        if not data or 'total_processed' not in data or data['total_processed'] == 0:
            print(f"  Missing total_processed, found keys: {list(data.keys())}")
            return None
            
        return data

    def parse_log_file(self, log_file: str, start_run: int = 1, end_run: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """Parse log file and extract data from runs start_run to end_run"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading log file: {e}")
            return [], {}
        
        lines = content.split('\n')
        
        # Find run boundaries
        run_boundaries = self.find_run_boundaries(content)
        total_runs = len(run_boundaries)
        
        print(f"Found {total_runs} runs in log file")
        
        if total_runs == 0:
            print("No runs found in log file")
            print("Looking for patterns in first 20 lines:")
            for i, line in enumerate(lines[:20]):
                if "Final Results Summary" in line or "=======" in line:
                    print(f"  Line {i}: {repr(line)}")
            return [], {}
        
        # Adjust end_run if not specified
        if end_run is None:
            end_run = total_runs
        
        # Validate run range
        if start_run < 1 or start_run > total_runs:
            print(f"Invalid start_run {start_run}. Must be between 1 and {total_runs}")
            return [], {}
            
        if end_run < start_run or end_run > total_runs:
            print(f"Invalid end_run {end_run}. Must be between {start_run} and {total_runs}")
            return [], {}
        
        print(f"Parsing runs {start_run} to {end_run}")
        
        # Extract data from specified runs
        runs_data = []
        config_data = {}
        
        for i in range(start_run - 1, end_run):
            start_line, end_line = run_boundaries[i]
            run_content = '\n'.join(lines[start_line:end_line + 1])
            
            # Parse the run data
            run_data = self.parse_run(run_content)
            
            # Add config data
            if run_data:
                config_for_run = self.extract_config_from_run_start(lines, i)
                run_data.update(config_for_run)
                
                # Add sequential run number (1, 2, 3, etc.)
                run_data['run_num'] = len(runs_data) + 1
                
                runs_data.append(run_data)
                
                # Extract config data from first successful run
                if not config_data:
                    for param in self.config_params:
                        if param in run_data:
                            config_data[param] = run_data[param]
                
                print(f"✓ Parsed run {i + 1}: {run_data.get('total_processed', 'N/A')} graphs processed")
            else:
                print(f"✗ Failed to parse run {i + 1}")
                # show some content
                print(f"  Content preview: {run_content[:200]}...")
        
        return runs_data, config_data

    def calculate_statistics(self, runs_data: List[Dict]) -> Dict:
        """Calculate mean ± std for all metrics across runs"""
        if not runs_data:
            return {}
        
        stats = {}
        
        # Calculate statistics for each metric
        for metric in self.key_metrics:
            values = []
            for run in runs_data:
                if metric in run and run[metric] is not None:
                    values.append(run[metric])
                else:
                    values.append(0.0)  # Use 0.0 as default instead of None
            
            if values:
                mean_val = statistics.mean(values)
                if len(values) > 1:
                    std_val = statistics.stdev(values)
                else:
                    std_val = 0.0
                
                stats[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'count': len(values),
                    'values': values
                }
        
        return stats

    def get_output_folder_name(self, log_file_path: str) -> str:
        """Generate output folder name from log file name"""
        log_path = Path(log_file_path)
        log_name = log_path.stem  # Get filename without extension
        
        # Remove common prefixes/suffixes to get a clean folder name
        folder_name = log_name
        if folder_name.startswith('analysis_'):
            folder_name = folder_name[9:]  # Remove 'analysis_' prefix
        if folder_name.endswith('_log'):
            folder_name = folder_name[:-4]  # Remove '_log' suffix
        
        return folder_name

    def save_results(self, stats: Dict, config_data: Dict, runs_data: List[Dict], 
                    output_dir: str, log_file_path: str, prefix: str = "ctqw_stats"):
        """Save results in multiple formats"""
        # Create subfolder based on log file name
        folder_name = self.get_output_folder_name(log_file_path)
        output_path = Path(output_dir) / folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define metrics to exclude from summary
        exclude_from_summary = {
            'win_count', 'lose_count', 'draw_count', 'total_processed',
            'timing_min_per_graph', 'timing_max_per_graph',
            'win_analysis_time', 'lose_analysis_time', 'draw_analysis_time'
        }
        
        # 1. Summary CSV with mean ± std (filtered metrics)
        summary_file = output_path / f"{prefix}_summary.csv"
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Mean', 'Std', 'Values'])
            
            for metric, stat_data in stats.items():
                if metric not in exclude_from_summary:
                    mean_str = f"{stat_data['mean']:.3f}"
                    std_str = f"{stat_data['std']:.3f}"
                    values_str = ', '.join([f"{v:.3f}" for v in stat_data['values']])
                    
                    writer.writerow([
                        metric, 
                        mean_str, 
                        std_str, 
                        values_str
                    ])
        
        # 2. Configuration file
        config_file = output_path / f"{prefix}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # 3. Raw data CSV with explicit run_num handling
        if runs_data:
            raw_file = output_path / f"{prefix}_raw_data.csv"
            
            # Get all unique keys from all runs
            all_keys = set()
            for run in runs_data:
                all_keys.update(run.keys())
            
            # Ensure run_num is in the keys
            all_keys.add('run_num')
            
            # Create ordered column list: run_num first, then config, then others
            columns = ['run_num']
            
            # Add config columns
            config_columns = sorted([k for k in all_keys if k.startswith('config_')])
            columns.extend(config_columns)
            
            # Add all other columns except run_num and config columns
            other_columns = sorted([k for k in all_keys if k != 'run_num' and not k.startswith('config_')])
            columns.extend(other_columns)
            
            # Write CSV with explicit handling
            with open(raw_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)  # Write header
                
                for i, run in enumerate(runs_data):
                    row = []
                    for col in columns:
                        if col == 'run_num':
                            row.append(i + 1)  # Always use sequential numbering
                        elif col in run:
                            row.append(run[col])
                        elif col in self.key_metrics:
                            row.append(0.0)  # Default for metrics
                        else:
                            row.append('')  # Empty for missing values
                    
                    writer.writerow(row)
        
        # 4. Detailed JSON with all data
        detailed_file = output_path / f"{prefix}_detailed.json"
        detailed_data = {
            'config': config_data,
            'statistics': stats,
            'runs_data': runs_data,
            'summary': {
                'total_runs': len(runs_data),
                'metrics_count': len(stats),
                'log_file': log_file_path,
                'output_folder': folder_name
            }
        }
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        print(f"\n📊 Results saved to {output_path}:")
        print(f"  • {summary_file.name} - Summary statistics (mean ± std)")
        print(f"  • {config_file.name} - Configuration parameters")
        print(f"  • {raw_file.name} - Raw data from all runs")
        print(f"  • {detailed_file.name} - Complete dataset with statistics")

    def print_summary(self, stats: Dict, config_data: Dict, runs_count: int):
        """Print a summary of the results"""
        print("\n" + "="*80)
        print("📈 CTQW ANALYSIS STATISTICS SUMMARY")
        print("="*80)
        
        if config_data:
            print(f"Configuration:")
            for param, value in config_data.items():
                print(f"  {param}: {value}")
        
        print(f"\nRuns analyzed: {runs_count}")
        print(f"Metrics calculated: {len(stats)}")
        
        # Print key results
        key_results = [
            'win_percentage', 'lose_percentage', 'draw_percentage',
            'timing_avg_per_graph'
        ]
        
        print(f"\n🎯 Key Results (Mean ± Std):")
        for metric in key_results:
            if metric in stats:
                stat = stats[metric]
                print(f"  {metric}: {stat['mean']:.3f} ± {stat['std']:.3f}")
        
        # Print win/lose/draw averages comparison
        categories = ['win', 'lose', 'draw']
        
        print(f"\n🔄 Gate Count Comparison (CX gates):")
        for category in categories:
            matching_metric = f"{category}_matching_cx"
            pauli_metric = f"{category}_pauli_cx"
            
            if matching_metric in stats and pauli_metric in stats:
                matching_stat = stats[matching_metric]
                pauli_stat = stats[pauli_metric]
                diff = matching_stat['mean'] - pauli_stat['mean']
                
                print(f"  {category.upper()}:")
                print(f"    Matching: {matching_stat['mean']:.2f} ± {matching_stat['std']:.2f}")
                print(f"    Pauli:    {pauli_stat['mean']:.2f} ± {pauli_stat['std']:.2f}")
                print(f"    Diff:     {diff:+.2f} (Matching - Pauli)")
        
        # Print bipartite graph statistics if available
        print(f"\n🔗 Graph Structure Statistics:")
        for category in categories:
            bipartite_metric = f"{category}_graph_bipartite_count"
            nonbipartite_metric = f"{category}_graph_nonbipartite_count"
            bipartite_pct_metric = f"{category}_graph_bipartite_percentage"
            nonbipartite_pct_metric = f"{category}_graph_nonbipartite_percentage"
            
            if bipartite_metric in stats and nonbipartite_metric in stats:
                bipartite_stat = stats[bipartite_metric]
                nonbipartite_stat = stats[nonbipartite_metric]
                
                print(f"  {category.upper()} graphs:")
                print(f"    Bipartite:     {bipartite_stat['mean']:.1f} ± {bipartite_stat['std']:.1f}")
                print(f"    Non-bipartite: {nonbipartite_stat['mean']:.1f} ± {nonbipartite_stat['std']:.1f}")
                
                # Print percentages if available
                if bipartite_pct_metric in stats and nonbipartite_pct_metric in stats:
                    bipartite_pct_stat = stats[bipartite_pct_metric]
                    nonbipartite_pct_stat = stats[nonbipartite_pct_metric]
                    print(f"    Bipartite %:     {bipartite_pct_stat['mean']:.1f}% ± {bipartite_pct_stat['std']:.1f}%")
                    print(f"    Non-bipartite %: {nonbipartite_pct_stat['mean']:.1f}% ± {nonbipartite_pct_stat['std']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Parse CTQW analysis log files and calculate statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("log_file", help="Path to the log file to parse")
    parser.add_argument("--start-run", "-s", type=int, default=1, 
                       help="Starting run number (1-indexed)")
    parser.add_argument("--end-run", "-e", type=int, default=None,
                       help="Ending run number (1-indexed). If not specified, processes all runs from start-run")
    parser.add_argument("--output-dir", "-o", type=str, default="./analysis/outputs/matching_vs_pauli/stats",
                       help="Output directory for results")
    parser.add_argument("--prefix", "-p", type=str, default="ctqw_stats",
                       help="Prefix for output files")
    
    args = parser.parse_args()
    
    # Validate log file exists
    if not Path(args.log_file).exists():
        print(f"❌ Error: Log file not found: {args.log_file}")
        return 1
    
    print(f"🔍 Parsing log file: {args.log_file}")
    
    # Initialize parser
    parser_obj = CTQWLogParser()
    
    # Parse log file
    runs_data, config_data = parser_obj.parse_log_file(
        args.log_file, args.start_run, args.end_run
    )
    
    if not runs_data:
        print("❌ No valid run data found")
        return 1
    
    # Calculate statistics
    stats = parser_obj.calculate_statistics(runs_data)
    
    # Print summary
    parser_obj.print_summary(stats, config_data, len(runs_data))
    
    # Save results
    parser_obj.save_results(stats, config_data, runs_data, args.output_dir, args.log_file, args.prefix)
    
    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)