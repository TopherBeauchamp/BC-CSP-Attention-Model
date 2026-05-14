#!/usr/bin/env python
"""
Compute research-grade statistics comparing NCO, Gurobi (ILP), and PCA baseline
for the BC-CSP (Budget-Constrained Covering Salesman Problem).

Statistics produced:
  1.  NCO % of Gurobi optimal (packets collected) — per budget, range, average
  2.  NCO optimality gap (100 - #1)
  3.  NCO % improvement over PCA greedy baseline — per budget, range, average
  4.  PCA % of Gurobi optimal (baseline quality floor)
  5.  Fraction of the PCA→Gurobi gap closed by NCO
  6.  Runtime reduction: NCO vs Gurobi ILP — per budget, range, average
  7.  Speedup factor: Gurobi / NCO
  8.  Compute efficiency: packets per second for each algorithm
  9.  Budget scaling: packet yield growth from lowest to highest battery
  10. Result consistency: relative uncertainty (stderr/mean) per algorithm

Output: JSON + Markdown report in results/overallResults/statistics/

Usage:
    python tools/compute_statistics.py
    python tools/compute_statistics.py --instance_count 100
    python tools/compute_statistics.py --results_dir results/overallResults --output_dir results/overallResults/statistics
"""

import os
import re
import json
import argparse
from datetime import datetime

BUDGET_TO_WH = {1.8: 50, 2.52: 70, 3.24: 90, 3.96: 110}
ALGO_ORDER = ["pca", "nco", "gurobi"]
ALGO_LABELS = {"pca": "Greedy (PCA)", "nco": "NCO", "gurobi": "Gurobi (ILP)"}


# ─────────────────────────────────────────────
# Data loading  (identical logic to plot_results.py)
# ─────────────────────────────────────────────

def _parse_budget(text):
    m = re.search(r"budget([\d]+[p.][\d]+|[\d]+)", text)
    return float(m.group(1).replace("p", ".")) if m else None


def load_algo_results(results_root, algo, instance_count):
    """Return {budget_float: record_dict} for one algorithm / instance-count folder."""
    folder = os.path.join(results_root, algo, f"{instance_count}_Instances")
    if not os.path.isdir(folder):
        return {}
    out = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(folder, fname)) as fh:
            record = json.load(fh)
        budget = _parse_budget(record.get("dataset", "")) or _parse_budget(fname)
        if budget is None:
            continue
        if budget not in out or record["num_instances"] > out[budget]["num_instances"]:
            out[budget] = record
    return out


def per_instance_s(record, algo):
    """Return (avg_s, stderr_s) per instance.

    NCO stores the full-batch wall time in avg_serial_duration_s, so dividing
    by num_instances gives the effective per-instance cost.
    """
    avg_s = record["avg_serial_duration_s"]
    std_s = record["stderr_serial_duration_s"]
    if algo == "nco":
        n = record["num_instances"]
        avg_s /= n
        std_s /= n
    return avg_s, std_s


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def safe_pct(numerator, denominator):
    if denominator == 0 or denominator is None or numerator is None:
        return None
    return (numerator / denominator) * 100.0


def _r(value, decimals=3):
    return round(value, decimals) if value is not None else None


def summarize(values, decimals=2):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": None, "max": None, "mean": None}
    return {
        "min":  round(min(vals), decimals),
        "max":  round(max(vals), decimals),
        "mean": round(sum(vals) / len(vals), decimals),
    }


# ─────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────

def compute_stats(all_results, all_budgets, instance_count):
    stats = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "instance_count": instance_count,
            "budget_levels_wh":         [BUDGET_TO_WH.get(b, b) for b in all_budgets],
            "budget_levels_normalized": all_budgets,
            "algorithms":               ALGO_LABELS,
            "note_nco_timing": (
                "NCO avg_serial_duration_s is the full-batch wall time; "
                "divided by num_instances here for a fair per-instance comparison."
            ),
        }
    }

    # Accumulation lists for cross-budget summaries
    nco_pct_of_gurobi_list   = []
    optimality_gap_list      = []
    nco_pct_over_pca_list    = []
    pca_pct_of_gurobi_list   = []
    gap_closed_list          = []
    runtime_reduction_list   = []
    speedup_list             = []
    nco_efficiency_list      = []
    gurobi_efficiency_list   = []
    pca_efficiency_list      = []

    per_budget = {}

    for b in all_budgets:
        wh    = BUDGET_TO_WH.get(b, f"{b:.2f}")
        key   = f"{wh}Wh"
        entry = {"budget_normalized": b, "battery_wh": wh}

        # ── Raw metrics ────────────────────────────────────────────────
        g_rec = all_results.get("gurobi", {}).get(b)
        n_rec = all_results.get("nco",    {}).get(b)
        p_rec = all_results.get("pca",    {}).get(b)

        g_pkts = g_rec["avg_packets_collected"] if g_rec else None
        n_pkts = n_rec["avg_packets_collected"] if n_rec else None
        p_pkts = p_rec["avg_packets_collected"] if p_rec else None

        entry["avg_packets_collected"] = {
            "gurobi": _r(g_pkts, 3),
            "nco":    _r(n_pkts, 3),
            "pca":    _r(p_pkts, 3),
        }
        entry["stderr_packets_collected"] = {
            "gurobi": _r(g_rec["stderr_packets_collected"], 4) if g_rec else None,
            "nco":    _r(n_rec["stderr_packets_collected"], 4) if n_rec else None,
            "pca":    _r(p_rec["stderr_packets_collected"], 4) if p_rec else None,
        }

        # ── Stat 1: NCO % of Gurobi optimal ───────────────────────────
        pct_nco_g = safe_pct(n_pkts, g_pkts)
        entry["nco_pct_of_gurobi_optimal"] = _r(pct_nco_g)
        if pct_nco_g is not None:
            nco_pct_of_gurobi_list.append(pct_nco_g)

        # ── Stat 2: Optimality gap ─────────────────────────────────────
        opt_gap = (100 - pct_nco_g) if pct_nco_g is not None else None
        entry["nco_optimality_gap_pct"] = _r(opt_gap)
        if opt_gap is not None:
            optimality_gap_list.append(opt_gap)

        # ── Stat 3: NCO % improvement over PCA ────────────────────────
        pct_nco_p = safe_pct(n_pkts - p_pkts, p_pkts) if (n_pkts and p_pkts) else None
        entry["nco_pct_improvement_over_pca"] = _r(pct_nco_p)
        if pct_nco_p is not None:
            nco_pct_over_pca_list.append(pct_nco_p)

        # ── Stat 4: PCA % of Gurobi optimal ───────────────────────────
        pct_pca_g = safe_pct(p_pkts, g_pkts)
        entry["pca_pct_of_gurobi_optimal"] = _r(pct_pca_g)
        if pct_pca_g is not None:
            pca_pct_of_gurobi_list.append(pct_pca_g)

        # ── Stat 5: Fraction of PCA→Gurobi gap closed by NCO ──────────
        # = (NCO - PCA) / (Gurobi - PCA)
        if g_pkts and n_pkts and p_pkts and (g_pkts - p_pkts) != 0:
            gap_closed = safe_pct(n_pkts - p_pkts, g_pkts - p_pkts)
            entry["nco_pct_of_pca_to_gurobi_gap_closed"] = _r(gap_closed)
            gap_closed_list.append(gap_closed)
        else:
            entry["nco_pct_of_pca_to_gurobi_gap_closed"] = None

        # ── Absolute packet differences ────────────────────────────────
        entry["nco_absolute_gap_to_gurobi_pkts"] = _r(g_pkts - n_pkts, 2) if (g_pkts and n_pkts) else None
        entry["nco_absolute_gain_over_pca_pkts"]  = _r(n_pkts - p_pkts, 2) if (n_pkts and p_pkts) else None
        entry["pca_absolute_gap_to_gurobi_pkts"]  = _r(g_pkts - p_pkts, 2) if (g_pkts and p_pkts) else None

        # ── Timing (per instance) ──────────────────────────────────────
        g_s = n_s = p_s = None
        if g_rec:
            g_s, _ = per_instance_s(g_rec, "gurobi")
            entry["gurobi_time_per_instance_ms"] = _r(g_s * 1000, 3)
        if n_rec:
            n_s, _ = per_instance_s(n_rec, "nco")
            entry["nco_time_per_instance_ms"] = _r(n_s * 1000, 4)
        if p_rec:
            p_s, _ = per_instance_s(p_rec, "pca")
            entry["pca_time_per_instance_ms"] = _r(p_s * 1000, 4)

        # ── Stat 6: Runtime reduction NCO vs Gurobi ───────────────────
        if g_s and n_s:
            rt_red = safe_pct(g_s - n_s, g_s)
            entry["runtime_reduction_nco_vs_gurobi_pct"] = _r(rt_red, 2)
            if rt_red is not None:
                runtime_reduction_list.append(rt_red)

        # ── Stat 7: Speedup factor ─────────────────────────────────────
        if g_s and n_s and n_s > 0:
            speedup = g_s / n_s
            entry["speedup_factor_nco_vs_gurobi"] = _r(speedup, 1)
            speedup_list.append(speedup)

        # ── Stat 8: Compute efficiency (packets per second) ───────────
        if n_pkts and n_s and n_s > 0:
            eff_n = n_pkts / n_s
            entry["nco_efficiency_pkts_per_s"] = _r(eff_n, 1)
            nco_efficiency_list.append(eff_n)
        if g_pkts and g_s and g_s > 0:
            eff_g = g_pkts / g_s
            entry["gurobi_efficiency_pkts_per_s"] = _r(eff_g, 1)
            gurobi_efficiency_list.append(eff_g)
        if p_pkts and p_s and p_s > 0:
            eff_p = p_pkts / p_s
            entry["pca_efficiency_pkts_per_s"] = _r(eff_p, 1)
            pca_efficiency_list.append(eff_p)

        per_budget[key] = entry

    stats["per_budget"] = per_budget

    # ── Summary (range + mean across budget levels) ───────────────────
    stats["summary"] = {
        "nco_pct_of_gurobi_optimal": {
            **summarize(nco_pct_of_gurobi_list),
            "description": "NCO packets collected as % of Gurobi (ILP) optimal",
        },
        "nco_optimality_gap_pct": {
            **summarize(optimality_gap_list),
            "description": "Percentage points below Gurobi optimal (100 - nco_pct_of_gurobi_optimal)",
        },
        "nco_pct_improvement_over_pca": {
            **summarize(nco_pct_over_pca_list),
            "description": "NCO percentage improvement over PCA greedy baseline",
        },
        "pca_pct_of_gurobi_optimal": {
            **summarize(pca_pct_of_gurobi_list),
            "description": "PCA (greedy) packets collected as % of Gurobi (ILP) optimal",
        },
        "nco_pct_of_pca_to_gurobi_gap_closed": {
            **summarize(gap_closed_list),
            "description": "Fraction of the PCA-to-Gurobi optimality gap closed by NCO (%)",
        },
        "runtime_reduction_nco_vs_gurobi_pct": {
            **summarize(runtime_reduction_list),
            "description": "NCO runtime reduction versus Gurobi ILP (%)",
        },
        "speedup_factor_nco_vs_gurobi": {
            **summarize(speedup_list, decimals=1),
            "description": "How many times faster NCO is than Gurobi ILP",
        },
        "nco_compute_efficiency_pkts_per_s": {
            **summarize(nco_efficiency_list, decimals=0),
            "description": "NCO packets collected per second of compute",
        },
        "gurobi_compute_efficiency_pkts_per_s": {
            **summarize(gurobi_efficiency_list, decimals=0),
            "description": "Gurobi ILP packets collected per second of compute",
        },
        "pca_compute_efficiency_pkts_per_s": {
            **summarize(pca_efficiency_list, decimals=0),
            "description": "PCA greedy packets collected per second of compute",
        },
    }

    # ── Stat 9: Budget scaling analysis ───────────────────────────────
    if len(all_budgets) >= 2:
        lo_b, hi_b = all_budgets[0], all_budgets[-1]
        lo_wh = BUDGET_TO_WH.get(lo_b, lo_b)
        hi_wh = BUDGET_TO_WH.get(hi_b, hi_b)
        scaling = {}
        for algo in ALGO_ORDER:
            lo_rec = all_results.get(algo, {}).get(lo_b)
            hi_rec = all_results.get(algo, {}).get(hi_b)
            if lo_rec and hi_rec:
                lo_p = lo_rec["avg_packets_collected"]
                hi_p = hi_rec["avg_packets_collected"]
                abs_inc = hi_p - lo_p
                pct_inc = safe_pct(abs_inc, lo_p)
                per_wh  = abs_inc / (hi_wh - lo_wh) if hi_wh != lo_wh else None
                scaling[ALGO_LABELS[algo]] = {
                    f"packets_at_{lo_wh}Wh":       _r(lo_p, 2),
                    f"packets_at_{hi_wh}Wh":       _r(hi_p, 2),
                    "absolute_packet_increase":     _r(abs_inc, 2),
                    "pct_packet_increase":          _r(pct_inc, 2),
                    "avg_packets_gained_per_wh":    _r(per_wh, 2),
                }
        stats["budget_scaling_analysis"] = {
            "from_wh":       lo_wh,
            "to_wh":         hi_wh,
            "description":   f"Packet yield change from {lo_wh}Wh to {hi_wh}Wh battery",
            "by_algorithm":  scaling,
        }

    # ── Stat 10: Result consistency ───────────────────────────────────
    consistency = {}
    for algo in ALGO_ORDER:
        cv_vals = []
        for b in all_budgets:
            rec = all_results.get(algo, {}).get(b)
            if rec and rec["avg_packets_collected"] > 0:
                rel_se = rec["stderr_packets_collected"] / rec["avg_packets_collected"] * 100
                cv_vals.append(rel_se)
        if cv_vals:
            consistency[ALGO_LABELS[algo]] = {
                "mean_relative_stderr_pct": _r(sum(cv_vals) / len(cv_vals), 3),
                "description": "Mean (stderr / mean) × 100 across budgets; lower = more consistent",
            }
    stats["result_consistency"] = consistency

    return stats


# ─────────────────────────────────────────────
# Markdown report generation
# ─────────────────────────────────────────────

def _fmt(value, fmt=".2f", suffix=""):
    if value is None:
        return "—"
    return f"{value:{fmt}}{suffix}"


def generate_markdown(stats, all_budgets, all_results):
    md = []
    meta = stats["metadata"]
    s    = stats["summary"]

    md.append("# BC-CSP Performance Statistics Report")
    md.append("")
    md.append(f"**Generated:** {meta['generated_at']}")
    md.append(f"**Instance count:** {meta['instance_count']}")
    md.append(f"**Budget levels:** {', '.join(str(w) + ' Wh' for w in meta['budget_levels_wh'])}")
    md.append("")
    md.append("> NCO timing note: `avg_serial_duration_s` stores the full-batch wall time.")
    md.append("> Values are divided by `num_instances` for a fair per-instance comparison.")
    md.append("")

    # ── Table 1: Raw packets ──────────────────────────────────────────
    md.append("## Table 1 — Average Packets Collected (mean ± stderr)")
    md.append("")
    md.append("| Battery (Wh) | Gurobi (ILP) | NCO | Greedy (PCA) |")
    md.append("|:---:|---:|---:|---:|")
    for b in all_budgets:
        wh  = BUDGET_TO_WH.get(b, b)
        key = f"{wh}Wh"
        e   = stats["per_budget"][key]
        pkts = e["avg_packets_collected"]
        se   = e["stderr_packets_collected"]
        g_s  = f"{pkts['gurobi']:.2f} ± {se['gurobi']:.2f}" if pkts["gurobi"] else "—"
        n_s  = f"{pkts['nco']:.2f} ± {se['nco']:.2f}"       if pkts["nco"]    else "—"
        p_s  = f"{pkts['pca']:.2f} ± {se['pca']:.2f}"       if pkts["pca"]    else "—"
        md.append(f"| **{wh} Wh** | {g_s} | {n_s} | {p_s} |")
    md.append("")

    # ── Table 2: NCO vs Gurobi ────────────────────────────────────────
    md.append("## Table 2 — NCO Solution Quality Relative to Gurobi (ILP) Optimal")
    md.append("")
    md.append("| Battery (Wh) | NCO % of Optimal | Optimality Gap | Gap Closed vs PCA | Abs. Gap (pkts) |")
    md.append("|:---:|---:|---:|---:|---:|")
    for b in all_budgets:
        wh  = BUDGET_TO_WH.get(b, b)
        e   = stats["per_budget"][f"{wh}Wh"]
        md.append(
            f"| **{wh} Wh** "
            f"| {_fmt(e.get('nco_pct_of_gurobi_optimal'), '.2f', '%')} "
            f"| {_fmt(e.get('nco_optimality_gap_pct'), '.2f', '%')} "
            f"| {_fmt(e.get('nco_pct_of_pca_to_gurobi_gap_closed'), '.1f', '%')} "
            f"| {_fmt(e.get('nco_absolute_gap_to_gurobi_pkts'), '.2f')} |"
        )
    nco_g = s["nco_pct_of_gurobi_optimal"]
    opt_g = s["nco_optimality_gap_pct"]
    gc    = s["nco_pct_of_pca_to_gurobi_gap_closed"]
    md.append("")
    md.append(
        f"**Summary:** NCO achieves **{_fmt(nco_g['min'], '.2f')}%–{_fmt(nco_g['max'], '.2f')}%** "
        f"of Gurobi optimal (mean **{_fmt(nco_g['mean'], '.2f')}%**); "
        f"optimality gap {_fmt(opt_g['min'], '.2f')}%–{_fmt(opt_g['max'], '.2f')}% "
        f"(mean {_fmt(opt_g['mean'], '.2f')}%). "
        f"NCO closes **{_fmt(gc['min'], '.1f')}%–{_fmt(gc['max'], '.1f')}%** "
        f"of the PCA→Gurobi gap (mean {_fmt(gc['mean'], '.1f')}%)."
    )
    md.append("")

    # ── Table 3: NCO vs PCA ───────────────────────────────────────────
    md.append("## Table 3 — NCO Improvement Over PCA Greedy Baseline")
    md.append("")
    md.append("| Battery (Wh) | NCO % Improvement | Abs. Gain (pkts) | PCA % of Optimal | PCA Abs. Gap (pkts) |")
    md.append("|:---:|---:|---:|---:|---:|")
    for b in all_budgets:
        wh = BUDGET_TO_WH.get(b, b)
        e  = stats["per_budget"][f"{wh}Wh"]
        pct  = e.get("nco_pct_improvement_over_pca")
        gain = e.get("nco_absolute_gain_over_pca_pkts")
        pp   = e.get("pca_pct_of_gurobi_optimal")
        pg   = e.get("pca_absolute_gap_to_gurobi_pkts")
        md.append(
            f"| **{wh} Wh** "
            f"| {'+' if (pct or 0) >= 0 else ''}{_fmt(pct, '.2f', '%')} "
            f"| {'+' if (gain or 0) >= 0 else ''}{_fmt(gain, '.2f')} "
            f"| {_fmt(pp, '.2f', '%')} "
            f"| {_fmt(pg, '.2f')} |"
        )
    nco_p   = s["nco_pct_improvement_over_pca"]
    pca_opt = s["pca_pct_of_gurobi_optimal"]
    md.append("")
    md.append(
        f"**Summary:** NCO improves over PCA by **{_fmt(nco_p['min'], '.2f')}%–{_fmt(nco_p['max'], '.2f')}%** "
        f"(mean **{_fmt(nco_p['mean'], '.2f')}%**). "
        f"PCA achieves {_fmt(pca_opt['min'], '.2f')}%–{_fmt(pca_opt['max'], '.2f')}% of optimal "
        f"(mean {_fmt(pca_opt['mean'], '.2f')}%)."
    )
    md.append("")

    # ── Table 4: Runtime ──────────────────────────────────────────────
    md.append("## Table 4 — Runtime Comparison (per instance)")
    md.append("")
    md.append("| Battery (Wh) | Gurobi (ms) | NCO (ms) | PCA (ms) | Runtime Reduction | Speedup |")
    md.append("|:---:|---:|---:|---:|---:|---:|")
    for b in all_budgets:
        wh = BUDGET_TO_WH.get(b, b)
        e  = stats["per_budget"][f"{wh}Wh"]
        gt  = e.get("gurobi_time_per_instance_ms")
        nt  = e.get("nco_time_per_instance_ms")
        pt  = e.get("pca_time_per_instance_ms")
        rr  = e.get("runtime_reduction_nco_vs_gurobi_pct")
        sp  = e.get("speedup_factor_nco_vs_gurobi")
        md.append(
            f"| **{wh} Wh** "
            f"| {_fmt(gt, '.1f')} "
            f"| {_fmt(nt, '.4f')} "
            f"| {_fmt(pt, '.4f')} "
            f"| {_fmt(rr, '.1f', '%')} "
            f"| {_fmt(sp, '.0f', '×')} |"
        )
    rt  = s["runtime_reduction_nco_vs_gurobi_pct"]
    sf  = s["speedup_factor_nco_vs_gurobi"]
    md.append("")
    md.append(
        f"**Summary:** NCO is **{_fmt(sf['min'], '.0f')}×–{_fmt(sf['max'], '.0f')}×** faster than Gurobi "
        f"(mean **{_fmt(sf['mean'], '.0f')}×**); "
        f"runtime reduced by {_fmt(rt['min'], '.1f')}%–{_fmt(rt['max'], '.1f')}% "
        f"(mean {_fmt(rt['mean'], '.1f')}%)."
    )
    md.append("")

    # ── Table 5: Compute efficiency ───────────────────────────────────
    md.append("## Table 5 — Compute Efficiency (Packets per Second)")
    md.append("")
    md.append("| Battery (Wh) | Gurobi (pkts/s) | NCO (pkts/s) | PCA (pkts/s) | NCO/Gurobi Efficiency Ratio |")
    md.append("|:---:|---:|---:|---:|---:|")
    for b in all_budgets:
        wh = BUDGET_TO_WH.get(b, b)
        e  = stats["per_budget"][f"{wh}Wh"]
        ge = e.get("gurobi_efficiency_pkts_per_s")
        ne = e.get("nco_efficiency_pkts_per_s")
        pe = e.get("pca_efficiency_pkts_per_s")
        ratio = (ne / ge) if (ne and ge and ge > 0) else None
        md.append(
            f"| **{wh} Wh** "
            f"| {_fmt(ge, '.1f')} "
            f"| {_fmt(ne, '.1f')} "
            f"| {_fmt(pe, '.1f')} "
            f"| {_fmt(ratio, '.0f', '×')} |"
        )
    md.append("")

    # ── Table 6: Budget scaling ───────────────────────────────────────
    if "budget_scaling_analysis" in stats:
        sc = stats["budget_scaling_analysis"]
        lo_wh, hi_wh = sc["from_wh"], sc["to_wh"]
        md.append(f"## Table 6 — Packet Yield Scaling ({lo_wh} Wh → {hi_wh} Wh)")
        md.append("")
        md.append(f"| Algorithm | Pkts @ {lo_wh} Wh | Pkts @ {hi_wh} Wh | Abs. Increase | % Increase | Pkts/Wh |")
        md.append("|:---|---:|---:|---:|---:|---:|")
        for algo_label, v in sc["by_algorithm"].items():
            lo_k = f"packets_at_{lo_wh}Wh"
            hi_k = f"packets_at_{hi_wh}Wh"
            md.append(
                f"| {algo_label} "
                f"| {_fmt(v.get(lo_k), '.1f')} "
                f"| {_fmt(v.get(hi_k), '.1f')} "
                f"| +{_fmt(v.get('absolute_packet_increase'), '.1f')} "
                f"| +{_fmt(v.get('pct_packet_increase'), '.1f', '%')} "
                f"| {_fmt(v.get('avg_packets_gained_per_wh'), '.2f')} |"
            )
        md.append("")

    # ── Table 7: Consistency ──────────────────────────────────────────
    if "result_consistency" in stats:
        md.append("## Table 7 — Result Consistency (Relative Standard Error)")
        md.append("")
        md.append("| Algorithm | Mean (stderr / mean) × 100% |")
        md.append("|:---|---:|")
        for algo_label, v in stats["result_consistency"].items():
            md.append(f"| {algo_label} | {_fmt(v.get('mean_relative_stderr_pct'), '.3f', '%')} |")
        md.append("")
        md.append("Lower relative stderr indicates more consistent solution quality across instances.")
        md.append("")

    # ── Key takeaways ─────────────────────────────────────────────────
    md.append("## Key Takeaways for Research Paper")
    md.append("")
    nco_g_mean = nco_g["mean"]
    nco_p_mean = nco_p["mean"]
    gc_mean    = gc["mean"]
    sf_mean    = sf["mean"]
    md.append(
        f"1. **Solution quality:** NCO achieves a mean of **{_fmt(nco_g_mean, '.2f')}%** of the Gurobi "
        f"ILP optimum across all battery budgets, "
        f"and improves over the PCA greedy baseline by a mean of **{_fmt(nco_p_mean, '.2f')}%**."
    )
    md.append(
        f"2. **Gap closure:** NCO closes **{_fmt(gc_mean, '.1f')}%** of the quality gap between the "
        f"greedy baseline and the ILP optimum on average."
    )
    md.append(
        f"3. **Computational efficiency:** NCO runs **{_fmt(sf_mean, '.0f')}×** faster than Gurobi per "
        f"instance on average, enabling real-time or large-scale deployment where ILP is intractable."
    )
    if "budget_scaling_analysis" in stats:
        nco_sc = stats["budget_scaling_analysis"]["by_algorithm"].get("NCO", {})
        md.append(
            f"4. **Budget scaling:** NCO yields a "
            f"{_fmt(nco_sc.get('pct_packet_increase'), '.1f')}% increase in packets collected "
            f"from {lo_wh} Wh to {hi_wh} Wh "
            f"({_fmt(nco_sc.get('avg_packets_gained_per_wh'), '.2f')} pkts/Wh)."
        )
    md.append("")

    return "\n".join(md)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute BC-CSP research statistics comparing NCO, Gurobi, and PCA."
    )
    parser.add_argument("--results_dir",   default="results/overallResults",
                        help="Root dir containing nco/, pca/, gurobi/ subfolders")
    parser.add_argument("--instance_count", type=int, default=100,
                        help="Instance-count folder to use for three-way comparison (default: 100)")
    parser.add_argument("--output_dir",    default="results/overallResults/statistics",
                        help="Directory to write output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading results from '{args.results_dir}' ({args.instance_count} instances)...")
    all_results = {}
    for algo in ALGO_ORDER:
        data = load_algo_results(args.results_dir, algo, args.instance_count)
        if data:
            all_results[algo] = data
            wh_list = [BUDGET_TO_WH.get(b, b) for b in sorted(data)]
            print(f"  {algo:7s}: {len(data)} budget levels — {wh_list} Wh")
        else:
            print(f"  {algo:7s}: no data in {args.results_dir}/{algo}/{args.instance_count}_Instances/")

    if not all_results:
        print("No data loaded. Check --results_dir.")
        return

    all_budgets = sorted(set().union(*[set(v) for v in all_results.values()]))
    print(f"  Budgets:  {all_budgets} (normalized)\n")

    stats = compute_stats(all_results, all_budgets, args.instance_count)

    # Save JSON
    json_name = f"statistics_{args.instance_count}_instances.json"
    json_path = os.path.join(args.output_dir, json_name)
    with open(json_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved JSON:     {json_path}")

    # Save Markdown
    md_name = f"statistics_{args.instance_count}_instances.md"
    md_path = os.path.join(args.output_dir, md_name)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(generate_markdown(stats, all_budgets, all_results))
    print(f"Saved Markdown: {md_path}")

    # Print summary to console
    s = stats["summary"]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for stat_key, label in [
        ("nco_pct_of_gurobi_optimal",         "NCO % of Gurobi optimal"),
        ("nco_optimality_gap_pct",             "NCO optimality gap"),
        ("nco_pct_improvement_over_pca",       "NCO improvement over PCA"),
        ("pca_pct_of_gurobi_optimal",          "PCA % of Gurobi optimal"),
        ("nco_pct_of_pca_to_gurobi_gap_closed","Gap closed by NCO (PCA->Gurobi)"),
        ("runtime_reduction_nco_vs_gurobi_pct","Runtime reduction vs Gurobi"),
        ("speedup_factor_nco_vs_gurobi",       "Speedup factor vs Gurobi"),
    ]:
        v = s.get(stat_key, {})
        lo, hi, mn = v.get("min"), v.get("max"), v.get("mean")
        if mn is not None:
            print(f"  {label}:")
            print(f"    range {lo} – {hi}   mean {mn}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
