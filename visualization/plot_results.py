#!/usr/bin/env python
# Fix OpenMP conflict between PyTorch and Gurobi (must be before any imports)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Generate RSN visual comparison figure for Greedy (PCA), NCO (Attention), and Optimal (Gurobi)
on the same BC-CSP instance.

Runs all three algorithms, then produces a side-by-side plot with coverage radius circles.

Coordinates in [0,1] are scaled to physical units (e.g., 1000m x 1000m) for the figure.

Usage:
    # With all three algorithms:
    python plot_rsn_visual.py \
        --dataset data/bccsp/bccsp20_bccsp_20_budget3p24_seed1234.pkl \
        --model outputs/bccsp_20/.../epoch-99.pt \
        --instance_idx 0 --gurobi_timeout 120

    # Without model (just Greedy + Optimal):
    python plot_rsn_visual.py \
        --dataset data/bccsp/bccsp20_bccsp_20_budget3p24_seed1234.pkl \
        --instance_idx 0 --gurobi_timeout 120

    # With precomputed tours (skip solver runs):
    python plot_rsn_visual.py \
        --dataset data/bccsp/bccsp20_bccsp_20_budget3p24_seed1234.pkl \
        --instance_idx 0 \
        --tour_greedy "3,7,12,15,8,1" \
        --tour_nco "3,12,7,15,1" \
        --tour_optimal "3,12,15,8"

    # Demo with random data:
    python plot_rsn_visual.py --demo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import pickle
import sys
import time

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
})

# ─── Solver imports (graceful fallback) ───
try:
    from problems.bccsp.pca_baseline import pca_solve
except ImportError:
    try:
        from pca_baseline import pca_solve
    except ImportError:
        pca_solve = None

try:
    from problems.bccsp.gurobi_bccsp import solve_bccsp_gurobi
except ImportError:
    try:
        from gurobi_bccsp import solve_bccsp_gurobi
    except ImportError:
        solve_bccsp_gurobi = None

try:
    from utils import load_model, move_to
    import torch
    from torch.utils.data import DataLoader
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False


# =====================================================================
# Core computation
# =====================================================================

def compute_tour_metrics(tour_indices, loc, packets, radius):
    """Compute data collected and distance for a tour."""
    depot = np.array([0.0, 0.0])
    N = len(loc)

    dist = 0.0
    current = depot.copy()
    for idx in tour_indices:
        dist += np.linalg.norm(loc[idx] - current)
        current = loc[idx].copy()
    if len(tour_indices) > 0:
        dist += np.linalg.norm(current - depot)

    covered = np.zeros(N, dtype=bool)
    for idx in tour_indices:
        for k in range(N):
            if np.linalg.norm(loc[idx] - loc[k]) <= radius + 1e-10:
                covered[k] = True
    data_collected = float(np.sum(packets[covered]))
    return data_collected, dist


def run_greedy(loc, packets, max_length, radius):
    """Run PCA/Greedy baseline."""
    if pca_solve is None:
        raise RuntimeError("pca_baseline not importable. Place pca_baseline.py in the working dir or problems/bccsp/.")
    t0 = time.time()
    tour, covered_pkts, tour_dist = pca_solve(loc, packets, max_length, radius, mu=1.0)
    elapsed = time.time() - t0
    data_collected, dist = compute_tour_metrics(tour, loc, packets, radius)
    print(f"  Greedy: {len(tour)} nodes, {data_collected:.0f} pkts, dist={dist:.4f}, time={elapsed:.3f}s")
    return tour


def run_optimal(loc, packets, max_length, radius, timeout=120):
    """Run Gurobi ILP."""
    if solve_bccsp_gurobi is None:
        raise RuntimeError("gurobi_bccsp not importable. Place gurobi_bccsp.py in the working dir or problems/bccsp/.")
    depot = np.zeros(2)
    t0 = time.time()
    obj_val, tour, solve_time = solve_bccsp_gurobi(
        depot, loc, packets, max_length, radius, timeout=timeout
    )
    elapsed = time.time() - t0
    data_collected, dist = compute_tour_metrics(tour, loc, packets, radius)
    print(f"  Optimal: {len(tour)} nodes, {data_collected:.0f} pkts, dist={dist:.4f}, time={elapsed:.3f}s")
    return tour


def run_nco(model_path, dataset_path, instance_idx, no_cuda=False):
    """Run NCO (attention model)."""
    if not HAS_MODEL:
        raise RuntimeError("Cannot load model: utils/load_model not available.")

    model, _ = load_model(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu")
    model.to(device)
    model.eval()
    model.set_decode_type("greedy")

    dataset = model.problem.make_dataset(
        filename=dataset_path, num_samples=1, offset=instance_idx
    )
    dataloader = DataLoader(dataset, batch_size=1)

    for batch in dataloader:
        batch = move_to(batch, device)
        t0 = time.time()
        with torch.no_grad():
            sequences, costs = model.sample_many(batch, batch_rep=1, iter_rep=1)
        elapsed = time.time() - t0
        seq = sequences[0].cpu().numpy()
        action_tour = seq[seq > -1].tolist()
        sensor_tour = [int(a) - 1 for a in action_tour if int(a) > 0]
        print(f"  NCO: {len(sensor_tour)} nodes, cost={float(costs[0]):.0f}, time={elapsed:.3f}s")
        return sensor_tour

    return []


# =====================================================================
# Plotting
# =====================================================================

def plot_rsn_comparison(
    loc,
    packets,
    radius,
    max_length,
    algo_results,
    filename="rsn_comparison.png",
    area_size=1000,
    scale_factor=1.0,
    caption=None,
):
    """
    Plot side-by-side RSN tour visualizations.

    loc/radius/distances are in original [0,1] space.
    scale_factor converts to physical units for display.
    """
    n_algos = len(algo_results)
    depot = np.array([0.0, 0.0])

    # Scale for display
    disp_loc = loc * scale_factor
    disp_radius = radius * scale_factor
    disp_depot = depot * scale_factor

    fig, axes = plt.subplots(1, n_algos, figsize=(4.2 * n_algos, 4.8))
    if n_algos == 1:
        axes = [axes]

    for ax_idx, (ax, result) in enumerate(zip(axes, algo_results)):
        tour = result['tour']
        name = result['name']
        data_collected = result['data_collected']
        distance = result['distance'] * scale_factor  # scale distance for display

        # --- Coverage radius circles around visited nodes ---
        for idx in tour:
            circle = plt.Circle(
                (disp_loc[idx, 0], disp_loc[idx, 1]), disp_radius,
                facecolor='none', edgecolor='#388e3c',
                linewidth=1.0, linestyle='--', alpha=0.5, zorder=1,
            )
            ax.add_patch(circle)

        # --- All sensor nodes as black dots ---
        ax.plot(disp_loc[:, 0], disp_loc[:, 1], 'ko', markersize=5, zorder=5)

        # --- Depot as diamond ---
        ax.plot(disp_depot[0], disp_depot[1], 'kD', markersize=7, zorder=6)

        # --- Tour as green lines ---
        if len(tour) > 0:
            path_x = [disp_depot[0]]
            path_y = [disp_depot[1]]
            for idx in tour:
                path_x.append(disp_loc[idx, 0])
                path_y.append(disp_loc[idx, 1])
            path_x.append(disp_depot[0])
            path_y.append(disp_depot[1])

            ax.plot(path_x, path_y, 'g-', linewidth=1.8, zorder=3, alpha=0.85)

            for idx in tour:
                ax.plot(disp_loc[idx, 0], disp_loc[idx, 1], 'go', markersize=6,
                        zorder=5, markeredgecolor='black', markeredgewidth=0.5)

        # --- Labels: index + (packets) directly above each node ---
        label_gap = area_size * 0.018  # gap between node dot and bottom of label
        label_nudge_x = area_size * 0.015  # slight rightward shift
        for i in range(len(loc)):
            # Node index (bold black) — placed above the node
            ax.text(
                disp_loc[i, 0] + label_nudge_x, disp_loc[i, 1] + label_gap + area_size * 0.016,
                f"{i}",
                fontsize=8, fontweight='bold', color='black',
                ha='center', va='bottom', zorder=7,
            )
            # Packet count (red) — placed between index and node
            ax.text(
                disp_loc[i, 0] + label_nudge_x, disp_loc[i, 1] + label_gap,
                f"({int(packets[i])})",
                fontsize=6.5, color='red',
                ha='center', va='bottom', zorder=7,
            )

        # --- Axis styling ---
        ax.set_xlim(-area_size * 0.08, area_size * 1.10)
        ax.set_ylim(-area_size * 0.08, area_size * 1.10)
        ax.set_aspect('equal')

        subtitle_letter = chr(ord('a') + ax_idx)
        ax.set_title(
            f"({subtitle_letter}) {name}.\n"
            f"Data Collected: {int(data_collected)}  |  Distance Travelled: {int(distance)}m",
            fontsize=11,
            linespacing=1.5,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


# =====================================================================
# Main
# =====================================================================

def load_instance(dataset_path, instance_idx):
    """Load a single instance from a pickle dataset."""
    with open(dataset_path, "rb") as f:
        raw_data = pickle.load(f)

    inst = raw_data[instance_idx]
    if isinstance(inst, tuple):
        loc, packets, max_length, radius = inst
    else:
        loc = inst["loc"]
        packets = inst["packets"]
        max_length = inst["max_length"]
        radius = inst["radius"]

    loc = np.asarray(loc, dtype=np.float64)
    packets = np.asarray(packets, dtype=np.float64)
    max_length = float(max_length)
    radius = float(radius)
    return loc, packets, max_length, radius


def demo():
    """Generate a demo figure with random data."""
    np.random.seed(42)
    N = 20

    loc = np.random.rand(N, 2)  # [0,1] space
    packets = np.random.randint(10, 100, size=N).astype(float)
    radius = 0.15
    max_length = 1.8  # in [0,1] space

    scale = 1000.0
    area_size = 1000

    # Fake tours for demo
    tours = {
        'Optimal': [3, 7, 12, 15, 8, 1],
        'Greedy': [3, 12, 7, 15, 1],
        'NCO': [3, 12, 15, 8],
    }

    algo_results = []
    for name, tour in tours.items():
        dc, dist = compute_tour_metrics(tour, loc, packets, radius)
        algo_results.append({
            'name': name, 'tour': tour,
            'data_collected': dc, 'distance': dist,
        })

    plot_rsn_comparison(
        loc, packets, radius, max_length,
        algo_results,
        filename="rsn_comparison_demo.png",
        area_size=area_size,
        scale_factor=scale,
        caption=(
            f"Fig. 3.  Visual comparison of a {area_size}m x {area_size}m RSN "
            f"with {N} nodes. "
            f"\u03B5 = 50Wh with a maximum distance of {int(max_length * scale)}m."
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate RSN visual comparison for Greedy/NCO/Optimal."
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to .pkl dataset file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to NCO .pt model file")
    parser.add_argument("--instance_idx", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gurobi_timeout", type=float, default=120)
    parser.add_argument("--output", type=str, default="rsn_comparison.png")
    parser.add_argument("--demo", action="store_true")

    # Manual tour overrides (comma-separated 0-indexed sensor indices)
    parser.add_argument("--tour_greedy", type=str, default=None)
    parser.add_argument("--tour_nco", type=str, default=None)
    parser.add_argument("--tour_optimal", type=str, default=None)

    # Display scaling
    parser.add_argument("--scale", type=float, default=1000.0,
                        help="Scale factor from [0,1] to physical units (default 1000 = 1000m)")
    parser.add_argument("--area_size", type=float, default=None,
                        help="Display area size in meters (default: auto from scale)")

    # Caption overrides
    parser.add_argument("--battery_wh", type=float, default=90,
                        help="Battery power in Wh for caption (default 90)")
    parser.add_argument("--caption", type=str, default=None,
                        help="Full custom caption (overrides auto-generated)")

    args = parser.parse_args()

    if args.demo or args.dataset is None:
        demo()
        return

    # Load instance
    loc, packets, max_length, radius = load_instance(args.dataset, args.instance_idx)
    N = len(loc)
    scale = args.scale
    area_size = args.area_size if args.area_size else scale

    print(f"Instance {args.instance_idx}: N={N}, radius={radius:.4f}, budget={max_length:.4f}")
    print(f"Display scale: {scale}x  ({radius:.4f} -> {radius*scale:.0f}m)")

    def parse_tour(s):
        if s is None:
            return None
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    # ─── Get tours ───
    algo_results = []

    # Greedy
    greedy_tour = parse_tour(args.tour_greedy)
    if greedy_tour is None:
        print("Running Greedy (PCA)...")
        try:
            greedy_tour = run_greedy(loc, packets, max_length, radius)
        except Exception as e:
            print(f"  Greedy failed: {e}")
            greedy_tour = None

    if greedy_tour is not None:
        dc, dist = compute_tour_metrics(greedy_tour, loc, packets, radius)
        algo_results.append({
            'name': 'Greedy', 'tour': greedy_tour,
            'data_collected': dc, 'distance': dist,
        })

    # NCO
    nco_tour = parse_tour(args.tour_nco)
    if nco_tour is None and args.model is not None:
        print("Running NCO (Attention model)...")
        try:
            nco_tour = run_nco(args.model, args.dataset, args.instance_idx, args.no_cuda)
        except Exception as e:
            print(f"  NCO failed: {e}")
            nco_tour = None

    if nco_tour is not None:
        dc, dist = compute_tour_metrics(nco_tour, loc, packets, radius)
        algo_results.append({
            'name': 'NCO', 'tour': nco_tour,
            'data_collected': dc, 'distance': dist,
        })

    # Optimal
    optimal_tour = parse_tour(args.tour_optimal)
    if optimal_tour is None:
        print(f"Running Optimal (Gurobi, timeout={args.gurobi_timeout}s)...")
        try:
            optimal_tour = run_optimal(loc, packets, max_length, radius, args.gurobi_timeout)
        except Exception as e:
            print(f"  Optimal failed: {e}")
            optimal_tour = None

    if optimal_tour is not None:
        dc, dist = compute_tour_metrics(optimal_tour, loc, packets, radius)
        algo_results.append({
            'name': 'Optimal', 'tour': optimal_tour,
            'data_collected': dc, 'distance': dist,
        })

    if not algo_results:
        print("No algorithms produced results. Exiting.")
        return

    # ─── Caption ───
    if args.caption:
        caption = args.caption
    else:
        caption = (
            f"Visual comparison of a {int(area_size)}m x {int(area_size)}m RSN "
            f"with {N} nodes. "
            f"\u03B5 = {int(args.battery_wh)}Wh with a maximum distance of "
            f"{int(max_length * scale)}m."
        )

    # ─── Plot ───
    plot_rsn_comparison(
        loc, packets, radius, max_length,
        algo_results,
        filename=args.output,
        area_size=area_size,
        scale_factor=scale,
        caption=caption,
    )


if __name__ == "__main__":
    main()