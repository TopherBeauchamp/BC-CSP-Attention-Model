import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from utils import load_model, move_to
from problems.bccsp.problem_bccsp import BCCSPDataset
from problems.bccsp.problem_bccsp import BCCSP


def render_bccsp_solution(dataset_item, tour, save_path=None, show=True, title=None):
    """
    Visualize BC-CSP solution with depot, sensors, coverage circles, and tour path.
    
    dataset_item: dict with 'loc', 'depot', 'packets', 'radius', 'max_length'
    tour: 1D tensor of visited nodes (including depot=0, sensors=1..N)
    """
    loc = dataset_item['loc']  # (N, 2)
    depot = dataset_item['depot']  # (2,)
    packets = dataset_item['packets']  # (N,)
    radius = dataset_item['radius'].item() if torch.is_tensor(dataset_item['radius']) else dataset_item['radius']
    max_length = dataset_item['max_length'].item() if torch.is_tensor(dataset_item['max_length']) else dataset_item['max_length']
    
    if torch.is_tensor(loc):
        loc = loc.detach().cpu().numpy()
    if torch.is_tensor(depot):
        depot = depot.detach().cpu().numpy()
    if torch.is_tensor(packets):
        packets = packets.detach().cpu().numpy()
    if torch.is_tensor(tour):
        tour = tour.detach().cpu().numpy()
    
    # Filter out -1 padding and convert to list
    tour = tour[tour >= 0].tolist()
    
    N = loc.shape[0]
    
    # Calculate coverage: which sensors are covered?
    covered = np.zeros(N, dtype=bool)
    visited_sensors = []
    
    for node_idx in tour:
        if node_idx == 0:  # depot
            continue
        sensor_idx = node_idx - 1  # Convert from action space (1..N) to sensor index (0..N-1)
        visited_sensors.append(sensor_idx)
        
        # Mark all sensors within radius as covered
        center = loc[sensor_idx]
        distances = np.sqrt(((loc - center) ** 2).sum(axis=1))
        covered |= (distances <= radius)
    
    # Calculate tour length
    coords_with_depot = np.vstack([depot[None, :], loc])  # (N+1, 2)

    # For visualization/length, the tour implicitly starts at depot even if the first action isn't 0
    display_tour = tour
    if len(display_tour) > 0 and display_tour[0] != 0:
        display_tour = [0] + display_tour

    tour_coords = coords_with_depot[display_tour]

    if len(tour_coords) > 1:
        tour_length = np.sum(np.sqrt(np.sum((tour_coords[1:] - tour_coords[:-1])**2, axis=1)))
    else:
        tour_length = 0.0

    
    # Calculate total collected packets
    total_packets = np.sum(packets[covered])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    
    # Draw coverage circles for visited sensors
    for sensor_idx in visited_sensors:
        circle = Circle(loc[sensor_idx], radius, fill=False, 
                       edgecolor='lightblue', linewidth=2, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    
    # Draw tour path
    if len(tour_coords) > 1:
        ax.plot(tour_coords[:, 0], tour_coords[:, 1],
                'k-', linewidth=2, alpha=0.6, label='Tour path')

        # Draw direction arrows along the path
        for i in range(len(tour_coords) - 1):
            x0, y0 = tour_coords[i]
            x1, y1 = tour_coords[i + 1]

            # Midpoint of the segment
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)

            dx = x1 - x0
            dy = y1 - y0

            ax.annotate(
                "",
                xy=(xm + 0.15 * dx, ym + 0.15 * dy),
                xytext=(xm - 0.15 * dx, ym - 0.15 * dy),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    lw=1.5,
                    alpha=0.8
                ),
                zorder=9
            )

        # Close the tour back to depot
        if tour[-1] != 0:
            ax.plot([tour_coords[-1, 0], depot[0]], 
                   [tour_coords[-1, 1], depot[1]], 
                   'k--', linewidth=2, alpha=0.6)
    
    # Draw depot
    ax.scatter(depot[0], depot[1], s=400, c='red', marker='s', 
              edgecolors='black', linewidths=2, label='Depot', zorder=10)
    
    # Draw sensors: covered (green) vs uncovered (gray)
    covered_idx = np.where(covered)[0]
    uncovered_idx = np.where(~covered)[0]
    
    if len(uncovered_idx) > 0:
        ax.scatter(loc[uncovered_idx, 0], loc[uncovered_idx, 1], 
                  s=100, c='lightgray', edgecolors='black', linewidths=1, 
                  label='Uncovered sensors', zorder=5)
    
    if len(covered_idx) > 0:
        # Size by packet count
        sizes = 50 + packets[covered_idx] * 3
        scatter = ax.scatter(loc[covered_idx, 0], loc[covered_idx, 1], 
                           s=sizes, c=packets[covered_idx], cmap='Greens',
                           edgecolors='black', linewidths=1, 
                           label='Covered sensors', zorder=5)
        plt.colorbar(scatter, ax=ax, label='Packets')
    
    # Mark visited sensors with stars
    if len(visited_sensors) > 0:
        visited_locs = loc[visited_sensors]
        ax.scatter(visited_locs[:, 0], visited_locs[:, 1], 
                  s=200, marker='*', c='gold', edgecolors='black', 
                  linewidths=1, label='Visited', zorder=8)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if title is None:
        title = (f"BC-CSP Solution\n"
                f"Collected: {total_packets:.0f} packets | "
                f"Tour length: {tour_length:.3f}/{max_length:.3f} | "
                f"Visited: {len(visited_sensors)}/{N} sensors")
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize BC-CSP solutions")
    parser.add_argument("--model", required=True, help="Path to trained model (epoch-*.pt)")
    parser.add_argument("--graph_size", type=int, default=20, help="Number of sensors")
    parser.add_argument("--n", type=int, default=3, help="Number of instances to visualize")
    parser.add_argument("--decode", choices=["greedy", "sampling"], default="greedy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--radius", type=float, default=0.15, help="Coverage radius")
    parser.add_argument("--max_length", type=float, default=None, help="Budget constraint")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, _ = load_model(args.model)
    model.to(device)
    model.eval()
    model.set_decode_type(args.decode)
    
    # Create dataset
    torch.manual_seed(args.seed)
    ds = BCCSPDataset(size=args.graph_size, num_samples=args.n, 
                     radius=args.radius, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.n)
    
    batch = next(iter(dl))
    batch = move_to(batch, device)
    
    # Run model
    print(f"Generating solutions with {args.decode} decoding...")
    with torch.no_grad():
        cost, ll, tours = model(batch, return_pi=True)
    
    problem = BCCSP()

    with torch.no_grad():
        cost2, _ = problem.get_costs(batch, tours)

    diff = (cost2 - cost).abs().max().item()
    print(f"\n[OFFICIAL COST CHECK] max|problem.get_costs - model_cost| = {diff:.8f}")

    print("\n=== PRINTING TOURS ===")

    for i in range(tours.size(0)):
        tour_raw = tours[i].cpu()
        tour_valid = tour_raw[tour_raw >= 0]

        print(f"\nInstance {i}")
        print("Raw tour tensor:", tour_raw.tolist())
        print("Valid tour (no -1):", tour_valid.tolist())

        if len(tour_valid) > 0:
            print("Last action:", int(tour_valid[-1].item()))

        print(f"\nAverage collected packets: {-cost.mean().item():.2f}")
        print(f"Solutions range: {-cost.max().item():.2f} to {-cost.min().item():.2f} packets\n")
        
    # Visualize
    for i in range(min(args.n, 3)):
        dataset_item = {
            'loc': batch['loc'][i].cpu(),
            'depot': batch['depot'][i].cpu(),
            'packets': batch['packets'][i].cpu(),
            'radius': batch['radius'][i].cpu() if batch['radius'].dim() > 0 else batch['radius'],
            'max_length': batch['max_length'][i].cpu() if batch['max_length'].dim() > 0 else batch['max_length']
        }
        
        save_path = f"bccsp_solution_{i}.png"
        render_bccsp_solution(
            dataset_item, 
            tours[i].cpu(), 
            save_path=save_path,
            show=False,
            title=f"Instance {i} | Collected: {-cost[i].item():.0f} packets"
        )
    
    print(f"Saved visualizations: bccsp_solution_0.png, bccsp_solution_1.png, bccsp_solution_2.png")