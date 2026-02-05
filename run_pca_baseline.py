#!/usr/bin/env python
"""
Standalone script to evaluate PCA baseline on BC-CSP datasets.

Usage:
    python run_pca_baseline.py --dataset data/bccsp/bccsp50_validation_seed1234.pkl \
                                --val_size 100 \
                                --radius 0.15 \
                                --output results/pca_results.pkl

This script:
1. Loads BC-CSP instances from a pickle file
2. Runs PCA algorithm on each instance
3. Computes covered packets and tour costs
4. Saves results in the same format as eval.py
"""

import argparse
import pickle
import numpy as np
import time
import os
from tqdm import tqdm
from datetime import timedelta

# Import PCA implementation
from problems.bccsp.pca_baseline import pca_solve, pca_cost


def load_bccsp_dataset(filepath, offset=0, num_samples=None):
    """
    Load BC-CSP dataset from pickle file.
    
    Expected format: list of tuples (loc, packets, max_length, radius)
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    if num_samples is None:
        num_samples = len(data) - offset
    
    return data[offset:offset + num_samples]


def evaluate_pca_on_dataset(dataset, print_progress=True):
    """
    Run PCA on all instances in the dataset.
    
    Args:
        dataset: List of BC-CSP instances
        print_progress: Whether to show progress bar
    
    Returns:
        results: List of (cost, tour, duration) tuples
        stats: Dictionary with summary statistics
    """
    results = []
    costs = []
    covered_packets_list = []
    tour_lengths = []  # Number of nodes visited
    tour_distances = []  # Actual distance traveled
    durations = []
    
    iterator = tqdm(dataset, disable=not print_progress) if print_progress else dataset
    
    for inst in iterator:
        t0 = time.time()
        
        # Parse instance
        if isinstance(inst, tuple):
            loc, packets, max_length, radius = inst
            loc = np.asarray(loc, dtype=np.float64)
            packets = np.asarray(packets, dtype=np.float64)
        elif isinstance(inst, dict):
            loc = np.asarray(inst["loc"], dtype=np.float64)
            packets = np.asarray(inst["packets"], dtype=np.float64)
            max_length = float(inst["max_length"])
            radius = float(inst["radius"])
        else:
            raise ValueError(f"Unsupported instance format: {type(inst)}")
        
        # Run PCA
        tour, covered_packets, tour_distance = pca_solve(
            loc, packets, max_length, radius,
            mu=1.0,
            print_enable=False
        )
        
        # Cost is negative covered packets (we minimize)
        cost = -covered_packets
        
        duration = time.time() - t0
        
        # Store results
        results.append((cost, tour, duration))
        costs.append(cost)
        covered_packets_list.append(covered_packets)
        tour_lengths.append(len(tour))  # Nodes visited
        tour_distances.append(tour_distance)  # Distance traveled
        durations.append(duration)
    
    # Compute statistics
    stats = {
        "avg_packets": np.mean(covered_packets_list),
        "std_packets": np.std(covered_packets_list),
        "stderr_packets": np.std(covered_packets_list) / np.sqrt(len(covered_packets_list)),
        "avg_distance": np.mean(tour_distances),
        "std_distance": np.std(tour_distances),
        "stderr_distance": np.std(tour_distances) / np.sqrt(len(tour_distances)),
        "avg_nodes": np.mean(tour_lengths),
        "std_nodes": np.std(tour_lengths),
        "stderr_nodes": np.std(tour_lengths) / np.sqrt(len(tour_lengths)),
        "avg_duration": np.mean(durations),
        "std_duration": np.std(durations),
        "total_duration": np.sum(durations),
    }
    
    return results, stats


def save_results(results, output_path):
    """
    Save results in format compatible with eval.py output.
    
    Format: (results_list, parallelism)
    where results_list is [(cost, tour, duration), ...]
    """
    parallelism = 1  # PCA is run serially
    with open(output_path, "wb") as f:
        pickle.dump((results, parallelism), f)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run PCA baseline on BC-CSP datasets")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to BC-CSP dataset pickle file")
    parser.add_argument("--val_size", type=int, default=100,
                        help="Number of instances to evaluate")
    parser.add_argument("--offset", type=int, default=0,
                        help="Offset in dataset to start from")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results (default: auto-generate)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory for results")
    parser.add_argument("--radius", type=float, default=0.15,
                        help="Coverage radius (for display only; read from dataset)")
    parser.add_argument("--no_progress_bar", action="store_true",
                        help="Disable progress bar")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (for reproducibility)")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    print("=" * 80)
    print("PCA Baseline Evaluation for BC-CSP")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Instances: {args.val_size} (offset: {args.offset})")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_bccsp_dataset(args.dataset, args.offset, args.val_size)
    print(f"Loaded {len(dataset)} instances")
    
    # Show sample instance info
    if len(dataset) > 0:
        sample = dataset[0]
        if isinstance(sample, tuple):
            loc, packets, max_length, radius = sample
        else:
            loc = sample["loc"]
            packets = sample["packets"]
            max_length = sample["max_length"]
            radius = sample["radius"]
        
        print(f"Instance info: N={len(loc)}, radius={radius}, budget={max_length}")
        print(f"Packet range: [{np.min(packets):.0f}, {np.max(packets):.0f}]")
    print()
    
    # Run PCA
    print("Running PCA baseline...")
    start_time = time.time()
    results, stats = evaluate_pca_on_dataset(dataset, print_progress=not args.no_progress_bar)
    total_time = time.time() - start_time
    
    # Print statistics
    print("\n" + "=" * 80)
    print("PCA Baseline Results")
    print("=" * 80)
    print(f"Average packets collected: {stats['avg_packets']:.2f} ± {2 * stats['stderr_packets']:.2f}")
    print(f"Average distance traveled: {stats['avg_distance']:.2f} ± {2 * stats['stderr_distance']:.2f}")
    print(f"Average nodes visited: {stats['avg_nodes']:.2f} ± {2 * stats['stderr_nodes']:.2f}")
    print(f"Average serial duration: {stats['avg_duration']:.4f} ± {2 * stats['std_duration'] / np.sqrt(len(dataset)):.4f} s")
    print(f"Total duration: {timedelta(seconds=int(total_time))}")
    print("=" * 80)
    print()
    
    # Save results
    if args.output is None:
        # Auto-generate output path
        dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
        results_dir = os.path.join(args.results_dir, "bccsp", dataset_basename)
        os.makedirs(results_dir, exist_ok=True)
        
        output_path = os.path.join(
            results_dir,
            f"{dataset_basename}-pca-radius{args.radius:.3f}-"
            f"{args.offset}-{args.offset + len(results)}.pkl"
        )
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_results(results, output_path)
    print(f"\nDone! Results saved to: {output_path}")


if __name__ == "__main__":
    main()