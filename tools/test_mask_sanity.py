#!/usr/bin/env python
"""
Mask sanity check for BC-CSP.

Rolls out a random policy on N random instances (no model required) and
asserts at every step that no unmasked non-depot action has zero marginal
coverage gain.

Usage:
    python test_mask_sanity.py
    python test_mask_sanity.py --num_instances 50 --n 20 --radius 0.15 --seed 0
"""

import argparse
import torch
from problems.bccsp.problem_bccsp import generate_instance,  BCCSP
from problems.bccsp.state_bccsp import StateBCCSP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def marginal_packets_batch(state):
    """
    Returns marginal packet gain for every action in every batch row.
    Shape: (B, N+1). Column 0 depot is always 0.
    """
    ids = state.ids.squeeze(-1)
    loc = state.loc[ids]
    packets = state.packets[ids]
    radius = state.radius[ids]
    covered = state.covered_[:, 0, :]

    B, N, _ = loc.size()

    distances = (loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1)
    in_range = distances <= radius[:, None, None]
    uncovered = ~covered

    marginal = (in_range & uncovered[:, None, :]).float() * packets[:, None, :]
    marginal = marginal.sum(dim=-1)

    out = torch.zeros(B, N + 1, device=loc.device)
    out[:, 1:] = marginal
    return out

def marginal_packets(state, action_idx: int) -> float:
    """
    Marginal coverage packets gained by visiting sensor `action_idx` (1-indexed).
    Returns 0.0 for the depot (action_idx == 0).
    Assumes state has batch size 1.
    """
    if action_idx == 0:
        return 0.0

    sensor = action_idx - 1            # 0-indexed into loc / packets
    loc     = state.loc[0]             # (N, 2)
    packets = state.packets[0]         # (N,)
    radius  = state.radius[0]          # scalar
    covered = state.covered_[0, 0, :]  # (N,) bool

    center    = loc[sensor]
    dist      = (loc - center).norm(dim=-1)       # (N,)
    in_range  = dist <= radius                     # (N,) bool
    uncovered = ~covered                           # (N,) bool

    return (packets * (in_range & uncovered).float()).sum().item()

def rollout_batch(instances, visited_dtype=torch.uint8):
    batch = {
        k: torch.stack([inst[k] for inst in instances], dim=0)
        for k in instances[0].keys()
    }

    state = StateBCCSP.initialize(batch, visited_dtype=visited_dtype)
    actions = []
    violations = []

    while not state.all_finished():
        mask = state.get_mask()[:, 0, :]
        feasible = ~mask
        marginal = marginal_packets_batch(state)

        bad = feasible[:, 1:] & (marginal[:, 1:] <= 0)
        if bad.any():
            rows, cols = bad.nonzero(as_tuple=True)
            for r, c in zip(rows.tolist(), cols.tolist()):
                violations.append(
                    f"batch row {r}: action {c + 1} unmasked but marginal=0"
                )

        selected = []
        finished = state.get_finished()

        for b in range(mask.size(0)):
            if finished[b]:
                pick = torch.tensor(0, device=mask.device)
            else:
                choices = feasible[b].nonzero(as_tuple=True)[0]
                pick = choices[torch.randint(len(choices), (1,)).item()]
            selected.append(pick)

        selected = torch.stack(selected)
        actions.append(selected)
        state = state.update(selected)

    pi = torch.stack(actions, dim=1)

    cost, _ = BCCSP.get_costs(batch, pi)
    reward_from_problem = -cost
    reward_from_state = state.cur_total_covered.squeeze(-1)

    if not torch.allclose(reward_from_state, reward_from_problem, atol=1e-5):
        violations.append(
            f"reward mismatch: state={reward_from_state.tolist()} "
            f"problem={reward_from_problem.tolist()}"
        )

    return violations
# ---------------------------------------------------------------------------
# Single-instance rollout
# ---------------------------------------------------------------------------

def rollout(instance, instance_id: int):
    """
    Random-policy rollout for one instance.

    At each step:
      1. Compute the mask.
      2. Assert every unmasked non-depot action has marginal > 0.
      3. Pick a uniformly random unmasked action and step the state.

    Returns (violations, n_steps, n_sensor_steps).
    """
    batch = {k: v.unsqueeze(0) for k, v in instance.items()}   # B=1
    state = StateBCCSP.initialize(batch)

    violations = []
    n_steps = 0
    n_sensor_steps = 0

    while not state.all_finished():
        mask_flat = state.get_mask()[0, 0, :]   # (N+1,)  True = infeasible
        feasible  = (~mask_flat).nonzero(as_tuple=True)[0]  # feasible action indices

        # ── ASSERTION ──────────────────────────────────────────────────────
        for a in feasible.tolist():
            if a == 0:
                continue  # depot is always fine
            m = marginal_packets(state, a)
            if m <= 0.0:
                violations.append(
                    f"instance {instance_id} step {n_steps}: "
                    f"sensor {a} (0-idx {a-1}) is unmasked but marginal={m:.4f}  "
                    f"covered={state.covered_[0, 0, a-1].item()}"
                )
        # ───────────────────────────────────────────────────────────────────

        # Random feasible action
        pick = feasible[torch.randint(len(feasible), (1,)).item()]
        selected = pick.unsqueeze(0)   # (1,)

        if pick.item() > 0:
            n_sensor_steps += 1
        n_steps += 1

        state = state.update(selected)

    return violations, n_steps, n_sensor_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_instances", type=int, default=30)
    parser.add_argument("--n",             type=int,   default=20,   help="sensors per instance")
    parser.add_argument("--radius",        type=float, default=0.15)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"BC-CSP mask sanity check  "
          f"(instances={args.num_instances}, N={args.n}, "
          f"radius={args.radius}, seed={args.seed})")

    all_violations = []

    # ------------------------------------------------------------------
    # 1) Single-instance rollout (your original test)
    # ------------------------------------------------------------------
    total_steps = 0
    total_sensor_steps = 0

    for i in range(args.num_instances):
        instance = generate_instance(size=args.n, radius=args.radius)
        violations, n_steps, n_sensor = rollout(instance, i)

        all_violations.extend(violations)
        total_steps        += n_steps
        total_sensor_steps += n_sensor

    print(f"  single-instance rollout: {total_steps} steps | {total_sensor_steps} sensor selections")

    # ------------------------------------------------------------------
    # 2) Batch rollout tests (tests ids + vectorization)
    # ------------------------------------------------------------------
    batch_instances = [
        generate_instance(size=args.n, radius=args.radius)
        for _ in range(args.num_instances)
    ]

    v_uint8 = rollout_batch(batch_instances, visited_dtype=torch.uint8)
    print("  batch rollout check complete (uint8 visited mask)")

    v_int64 = rollout_batch(batch_instances, visited_dtype=torch.int64)
    print("  batch rollout check complete (int64 compressed mask)")

    all_violations.extend(v_uint8)
    all_violations.extend(v_int64)

    print()

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    if all_violations:
        print(f"FAILED  —  {len(all_violations)} violation(s):")
        for v in all_violations:
            print(f"  {v}")
    else:
        print("PASSED  —  mask, batch behavior, compressed mask, and reward agreement all look good.")
        print("Safe to begin training.")


if __name__ == "__main__":
    main()