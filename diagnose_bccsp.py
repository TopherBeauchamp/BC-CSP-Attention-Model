#!/usr/bin/env python
"""
Comprehensive diagnostic: compare Gurobi ILP, PCA, and attention model
on the exact same instances to find why Gurobi underperforms.

Usage:
    python diagnose_bccsp2.py --dataset data\bccsp\bccsp20_bccsp_20_budget3p24_seed1234.pkl \
        --model outputs/bccsp_20/bccsp20_mixedbudget_correct_20260213T051123/epoch-99.pt \
        --num_instances 3 --timeout 120
"""

import argparse
import pickle
import numpy as np
import torch
import time
from utils import load_model, move_to
from torch.utils.data import DataLoader
from problems.bccsp.pca_baseline import pca_solve, compute_covered_packets as pca_compute_covered
from problems.bccsp.gurobi_bccsp import solve_bccsp_gurobi, compute_covered_packets as gurobi_compute_covered, compute_tour_distance


def compute_coverage_detail(tour_sensor_indices, loc, packets, radius):
    """Compute coverage with full detail."""
    N = len(loc)
    covered = np.zeros(N, dtype=bool)
    covered_by = {}  # sensor_k -> which visited node covers it

    for idx in tour_sensor_indices:
        for k in range(N):
            if not covered[k]:
                d = np.linalg.norm(loc[idx] - loc[k])
                if d <= radius:
                    covered[k] = True
                    covered_by[k] = (idx, d)

    total = float(np.sum(packets[covered]))
    return total, covered, covered_by


def feed_tour_to_gurobi_verifier(tour_sensor_indices, loc, packets, radius, max_length):
    """Verify a tour using independent computation (matching Gurobi's logic)."""
    depot = np.zeros(2)

    # Distance
    dist = 0.0
    current = depot.copy()
    for idx in tour_sensor_indices:
        dist += np.linalg.norm(loc[idx] - current)
        current = loc[idx].copy()
    dist += np.linalg.norm(current - depot)

    # Coverage (using radius + 1e-10 like Gurobi)
    N = len(loc)
    covered_gurobi = np.zeros(N, dtype=bool)
    for idx in tour_sensor_indices:
        for k in range(N):
            if np.linalg.norm(loc[idx] - loc[k]) <= radius + 1e-10:
                covered_gurobi[k] = True
    packets_gurobi = float(np.sum(packets[covered_gurobi]))

    # Coverage (using exact radius like model)
    covered_exact = np.zeros(N, dtype=bool)
    for idx in tour_sensor_indices:
        for k in range(N):
            if np.linalg.norm(loc[idx] - loc[k]) <= radius:
                covered_exact[k] = True
    packets_exact = float(np.sum(packets[covered_exact]))

    return dist, packets_gurobi, packets_exact, dist <= max_length + 1e-6


def print_problem_structure(loc, packets, radius):
    """Print the coverage graph structure."""
    N = len(loc)
    depot = np.zeros(2)

    print(f"\n  Node details (depot at origin):")
    print(f"  {'Node':>4} {'X':>8} {'Y':>8} {'Packets':>8} {'Dist2Depot':>10} {'Covers':>30}")

    for i in range(N):
        d2d = np.linalg.norm(loc[i] - depot)
        covers = []
        for j in range(N):
            if np.linalg.norm(loc[i] - loc[j]) <= radius:
                covers.append(j)
        covers_str = str(covers)
        print(f"  {i:>4} {loc[i][0]:>8.4f} {loc[i][1]:>8.4f} {packets[i]:>8.0f} {d2d:>10.4f} {covers_str:>30}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num_instances", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--verbose_gurobi", action="store_true")
    args = parser.parse_args()

    # Load raw dataset
    with open(args.dataset, "rb") as f:
        raw_data = pickle.load(f)

    # Get model tours if model provided
    model_tours = {}
    if args.model:
        print(f"Loading model from {args.model}...")
        model, _ = load_model(args.model)
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        model.to(device)
        model.eval()
        model.set_decode_type("greedy")

        dataset = model.problem.make_dataset(
            filename=args.dataset, num_samples=args.num_instances, offset=args.offset
        )
        dataloader = DataLoader(dataset, batch_size=args.num_instances)

        for batch in dataloader:
            batch = move_to(batch, device)
            with torch.no_grad():
                sequences, costs = model.sample_many(batch, batch_rep=1, iter_rep=1)
            sequences = sequences.cpu().numpy()
            costs = costs.cpu().numpy()

            for i in range(args.num_instances):
                seq = sequences[i]
                action_tour = seq[seq > -1].tolist()
                sensor_tour = [int(a) - 1 for a in action_tour if int(a) > 0]
                model_tours[i] = {
                    "action_tour": action_tour,
                    "sensor_tour": sensor_tour,
                    "cost": float(costs[i])
                }
            break

    print("\n" + "=" * 100)
    print("COMPREHENSIVE BC-CSP DIAGNOSTIC")
    print("=" * 100)

    for idx in range(args.num_instances):
        inst = raw_data[args.offset + idx]
        if isinstance(inst, tuple):
            loc, packets, max_length, radius = inst
            loc = np.asarray(loc, dtype=np.float64)
            packets = np.asarray(packets, dtype=np.float64)
            max_length = float(max_length)
            radius = float(radius)
        else:
            loc = np.asarray(inst["loc"], dtype=np.float64)
            packets = np.asarray(inst["packets"], dtype=np.float64)
            max_length = float(inst["max_length"])
            radius = float(inst["radius"])

        N = len(loc)
        depot = np.zeros(2)
        total_possible_packets = float(np.sum(packets))

        print(f"\n{'━' * 100}")
        print(f"INSTANCE {idx}: N={N}, radius={radius}, budget={max_length}")
        print(f"Total possible packets (all sensors): {total_possible_packets:.0f}")
        print(f"{'━' * 100}")

        # Print problem structure
        print_problem_structure(loc, packets, radius)

        # ─── Run PCA ───
        print(f"\n  ┌─ PCA Baseline ─────────────────────────────────────────")
        t0 = time.time()
        pca_tour, pca_covered, pca_dist = pca_solve(loc, packets, max_length, radius, mu=1.0)
        pca_time = time.time() - t0

        pca_dist_verify, pca_pkt_gurobi, pca_pkt_exact, pca_feasible = \
            feed_tour_to_gurobi_verifier(pca_tour, loc, packets, radius, max_length)

        print(f"  │ Tour: {pca_tour}")
        print(f"  │ Nodes visited: {len(pca_tour)}")
        print(f"  │ Distance: {pca_dist_verify:.6f} / {max_length}")
        print(f"  │ Feasible: {pca_feasible}")
        print(f"  │ Packets (exact radius):    {pca_pkt_exact:.0f}")
        print(f"  │ Packets (radius+1e-10):    {pca_pkt_gurobi:.0f}")
        print(f"  │ Time: {pca_time:.4f}s")
        print(f"  └────────────────────────────────────────────────────────")

        # ─── Run Gurobi ───
        print(f"\n  ┌─ Gurobi ILP (timeout={args.timeout}s) ──────────────────────")
        try:
            t0 = time.time()
            gurobi_obj, gurobi_tour, gurobi_solve_time = solve_bccsp_gurobi(
                depot, loc, packets, max_length, radius,
                timeout=args.timeout,
                verbose=args.verbose_gurobi
            )
            gurobi_time = time.time() - t0

            gurobi_dist_verify, gurobi_pkt_gurobi, gurobi_pkt_exact, gurobi_feasible = \
                feed_tour_to_gurobi_verifier(gurobi_tour, loc, packets, radius, max_length)

            print(f"  │ Tour: {gurobi_tour}")
            print(f"  │ Nodes visited: {len(gurobi_tour)}")
            print(f"  │ Distance: {gurobi_dist_verify:.6f} / {max_length}")
            print(f"  │ Feasible: {gurobi_feasible}")
            print(f"  │ Gurobi objective (reported): {gurobi_obj:.0f}")
            print(f"  │ Packets (exact radius):      {gurobi_pkt_exact:.0f}")
            print(f"  │ Packets (radius+1e-10):      {gurobi_pkt_gurobi:.0f}")
            print(f"  │ Solve time: {gurobi_solve_time:.4f}s")

            # Check if Gurobi's reported obj matches verification
            if abs(gurobi_obj - gurobi_pkt_gurobi) > 0.5:
                print(f"  │ *** MISMATCH: Gurobi reports {gurobi_obj:.0f} but verification gives {gurobi_pkt_gurobi:.0f} ***")

        except Exception as e:
            print(f"  │ ERROR: {e}")
            gurobi_tour = []
            gurobi_pkt_exact = 0
            gurobi_pkt_gurobi = 0
            gurobi_dist_verify = 0
        print(f"  └────────────────────────────────────────────────────────")

        # ─── Model ───
        if idx in model_tours:
            mt = model_tours[idx]
            print(f"\n  ┌─ Attention Model ────────────────────────────────────")

            model_dist, model_pkt_gurobi, model_pkt_exact, model_feasible = \
                feed_tour_to_gurobi_verifier(mt["sensor_tour"], loc, packets, radius, max_length)

            print(f"  │ Action tour: {mt['action_tour']}")
            print(f"  │ Sensor tour: {mt['sensor_tour']}")
            print(f"  │ Nodes visited: {len(mt['sensor_tour'])}")
            print(f"  │ Distance: {model_dist:.6f} / {max_length}")
            print(f"  │ Feasible: {model_feasible}")
            print(f"  │ get_costs (neg packets):      {mt['cost']:.0f}")
            print(f"  │ Packets (exact radius):        {model_pkt_exact:.0f}")
            print(f"  │ Packets (radius+1e-10):        {model_pkt_gurobi:.0f}")
            print(f"  └────────────────────────────────────────────────────────")

            # ─── Feed MODEL's tour to Gurobi as warm start comparison ───
            if model_feasible and model_pkt_exact > gurobi_pkt_exact:
                print(f"\n  *** MODEL BEATS GUROBI: {model_pkt_exact:.0f} > {gurobi_pkt_exact:.0f} ***")
                print(f"  This means Gurobi's ILP formulation is NOT finding the optimal solution!")

                # Show what the model visits that Gurobi doesn't
                model_set = set(mt["sensor_tour"])
                gurobi_set = set(gurobi_tour)
                print(f"  Model visits but Gurobi doesn't: {model_set - gurobi_set}")
                print(f"  Gurobi visits but model doesn't: {gurobi_set - model_set}")

    print(f"\n{'=' * 100}")
    print("DONE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()