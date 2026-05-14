#!/usr/bin/env python
"""
Generate aggregate BC-CSP performance figures from overallResults JSON files.

Produces two output files:
  1. Two-panel bar chart: avg data packets collected and avg nodes visited vs battery budget
  2. Timing comparison table figure (LaTeX booktabs style)

NCO timing note: eval.py records one wall-clock time for the entire batch and assigns that
same value to every instance in the batch. For the 100-instance runs (one batch of 100),
avg_serial_duration_s is the full-batch time, so we divide by num_instances to get a
fair per-instance comparison against the per-instance PCA/Gurobi timings.

Usage:
    python visualization/plot_results.py
    python visualization/plot_results.py --results_dir results/overallResults
    python visualization/plot_results.py --instance_count 100 --output_dir images
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
})

# Normalized budget distance → battery label (Wh)
# Derived from the physical drone model: 36 m/Wh, 1 unit = 1000 m
BUDGET_TO_WH = {1.8: 50, 2.52: 70, 3.24: 90, 3.96: 110}

ALGO_ORDER  = ['pca', 'nco', 'gurobi']
ALGO_LABELS = {'pca': 'Greedy', 'nco': 'NCO', 'gurobi': 'Gurobi (30s, 5% MILP Gap)'}
ALGO_COLORS = {'pca': '#1f77b4', 'nco': '#2ca02c', 'gurobi': '#d62728'}


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def _parse_budget(text):
    """Extract budget float from a string that contains 'budget<value>' (p = decimal point)."""
    m = re.search(r'budget([\d]+[p.][\d]+|[\d]+)', text)
    return float(m.group(1).replace('p', '.')) if m else None


def load_algo_results(results_root, algo, instance_count):
    """Return {budget_float: record_dict} for one algorithm / instance-count folder."""
    folder = os.path.join(results_root, algo, f"{instance_count}_Instances")
    if not os.path.isdir(folder):
        return {}
    out = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(folder, fname)) as fh:
            record = json.load(fh)
        budget = _parse_budget(record.get('dataset', '')) or _parse_budget(fname)
        if budget is None:
            continue
        # Keep the record with more instances if duplicates exist for a budget level
        if budget not in out or record['num_instances'] > out[budget]['num_instances']:
            out[budget] = record
    return out


# ──────────────────────────────────────────────
# Timing helper
# ──────────────────────────────────────────────

def per_instance_ms(record, algo):
    """Return (avg_ms, stderr_ms) per instance.

    NCO records the entire batch duration for every instance in the batch, so
    avg_serial_duration_s equals the full-batch wall time. Dividing by num_instances
    recovers the effective per-instance time for a fair comparison.
    """
    avg_s = record['avg_serial_duration_s']
    std_s = record['stderr_serial_duration_s']
    if algo == 'nco':
        n = record['num_instances']
        avg_s /= n
        std_s /= n
    return avg_s * 1000.0, std_s * 1000.0


# ──────────────────────────────────────────────
# Figure 1: packets + nodes bar chart
# ──────────────────────────────────────────────

def plot_packets_and_nodes(all_results, budgets, output_path):
    wh_labels = [str(BUDGET_TO_WH.get(b, f"{b:.2f}")) for b in budgets]
    n_b = len(budgets)
    n_a = len(ALGO_ORDER)

    bar_w = 0.22
    x = np.arange(n_b)
    offsets = np.linspace(-(n_a - 1) / 2, (n_a - 1) / 2, n_a) * bar_w

    fig, (ax_pkt, ax_nd) = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric, stderr_key, ylabel in [
        (ax_pkt, 'avg_packets_collected', 'stderr_packets_collected', 'Data Packets'),
        (ax_nd,  'avg_nodes_visited',     'stderr_nodes_visited',     'Nodes Visited'),
    ]:
        for i, algo in enumerate(ALGO_ORDER):
            if algo not in all_results:
                continue
            vals = np.array([all_results[algo].get(b, {}).get(metric, np.nan) for b in budgets])
            errs = np.array([all_results[algo].get(b, {}).get(stderr_key, np.nan) for b in budgets])
            ax.bar(
                x + offsets[i], vals, bar_w,
                label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo],
                yerr=errs,
                capsize=3,
                error_kw={'linewidth': 0.8},
                zorder=3,
            )

        ax.set_xlabel('Battery Power (Wh)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(wh_labels)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend()
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ──────────────────────────────────────────────
# Figure 2: timing table
# ──────────────────────────────────────────────

def plot_timing_table(all_results, budgets, output_path):
    wh_labels = [str(BUDGET_TO_WH.get(b, f"{b:.2f}")) for b in budgets]
    headers = ['Battery (Wh)'] + [ALGO_LABELS[a] for a in ALGO_ORDER]
    rows = []
    for b, wh in zip(budgets, wh_labels):
        row = [wh]
        for algo in ALGO_ORDER:
            if algo in all_results and b in all_results[algo]:
                avg_ms, std_ms = per_instance_ms(all_results[algo][b], algo)
                row.append(f"{avg_ms:.1f} ± {std_ms:.1f}")
            else:
                row.append('—')
        rows.append(row)

    n_rows = len(rows)
    n_cols = len(headers)

    fig_h = 0.9 + n_rows * 0.45
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Table 1: Average Time per Instance (ms)', fontsize=12, pad=14)

    # Column layout
    col_starts = [0.08, 0.28, 0.50, 0.72]
    col_widths  = [0.20, 0.22, 0.22, 0.22]
    col_cx = [s + w / 2 for s, w in zip(col_starts, col_widths)]
    x_left, x_right = 0.08, 0.94

    # Vertical spacing
    usable_h = 0.78
    row_h = usable_h / (n_rows + 1)
    y_top = 0.88
    header_y = y_top - row_h * 0.5
    data_ys = [y_top - row_h * (i + 1.5) for i in range(n_rows)]

    def hline(y, lw):
        ax.plot([x_left, x_right], [y, y], color='black', linewidth=lw,
                transform=ax.transData, clip_on=False)

    # Top rule
    hline(y_top, 1.5)

    # Header
    for j, h in enumerate(headers):
        ax.text(col_cx[j], header_y, h, ha='center', va='center',
                fontsize=11, fontweight='bold')

    # Mid rule (below header)
    hline(y_top - row_h, 0.8)

    # Data rows
    for ry, row in zip(data_ys, rows):
        for j, val in enumerate(row):
            ax.text(col_cx[j], ry, val, ha='center', va='center', fontsize=11)

    # Bottom rule
    hline(data_ys[-1] - row_h * 0.5, 1.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate aggregate BC-CSP performance figures from overallResults JSON files."
    )
    parser.add_argument('--results_dir', default='results/overallResults',
                        help="Root directory containing nco/, pca/, gurobi/ subfolders")
    parser.add_argument('--instance_count', type=int, default=100,
                        help="Instance-count folder to load (100, 1000, 10000)")
    parser.add_argument('--output_dir', default='images',
                        help="Directory to write output figures")
    parser.add_argument('--chart_name', default='Packet_Nodes_Graph.png',
                        help="Filename for the packets/nodes bar chart")
    parser.add_argument('--table_name', default='Time_Table.png',
                        help="Filename for the timing table figure")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for algo in ALGO_ORDER:
        data = load_algo_results(args.results_dir, algo, args.instance_count)
        if data:
            all_results[algo] = data
            wh = [BUDGET_TO_WH.get(b, '?') for b in sorted(data)]
            print(f"  {algo:6s}: {len(data)} budget levels -> {wh} Wh")
        else:
            print(f"  {algo:6s}: no data found in "
                  f"{args.results_dir}/{algo}/{args.instance_count}_Instances/")

    if not all_results:
        print("No data loaded. Check --results_dir.")
        return

    all_budgets = sorted(set().union(*[set(v) for v in all_results.values()]))
    print(f"Budget levels: {all_budgets}")

    plot_packets_and_nodes(
        all_results, all_budgets,
        os.path.join(args.output_dir, args.chart_name),
    )
    plot_timing_table(
        all_results, all_budgets,
        os.path.join(args.output_dir, args.table_name),
    )


if __name__ == '__main__':
    main()
