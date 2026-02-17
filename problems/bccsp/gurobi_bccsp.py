#!/usr/bin/env python
"""
Gurobi ILP solver for Budget-Constrained Covering Salesman Problem (BC-CSP)

Optimized undirected formulation:
- Edges x[i,j] for i<j are binary for sensor-sensor, integer {0,1,2} for depot-sensor
- This allows out-and-back tours (depot→A→depot uses edge {0,A} twice)
- Arc pruning removes edges that can never appear in a feasible tour
- Lazy subtour elimination with correct DFS-based component detection

Decision variables:
- x[i,j]: edge {i,j} in tour (binary for i,j>0; integer 0..2 if i=0)
- y[i]: binary, 1 if node i is visited (i=1..N)
- z[k]: binary, 1 if sensor k is covered (k=0..N-1)
"""

from gurobipy import *
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)


def solve_bccsp_gurobi(
    depot: np.ndarray,
    loc: np.ndarray,
    packets: np.ndarray,
    max_length: float,
    radius: float,
    threads: int = 0,
    timeout: Optional[float] = None,
    gap: Optional[float] = None,
    verbose: bool = False
) -> Tuple[float, List[int], float]:
    """
    Solve BC-CSP using optimized Gurobi formulation.
    """
    N = len(loc)

    # Points: depot=0, sensors=1..N
    points = np.vstack([depot.reshape(1, 2), loc])  # (N+1, 2)
    n = N + 1

    # Pairwise distances
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            dist[i, j] = d
            dist[j, i] = d

    # Coverage: which nodes (1..N) can cover each sensor k (0..N-1)
    coverage = [[] for _ in range(N)]
    for k in range(N):
        for i in range(N):
            if np.linalg.norm(loc[i] - loc[k]) <= radius + 1e-10:
                coverage[k].append(i + 1)

    # ─── Arc pruning: edge (i,j) can only be in tour if ───
    # dist[0,i] + dist[i,j] + dist[j,0] <= max_length
    # (triangle inequality: must be able to reach i, go to j, return)
    feasible_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if dist[0, i] + dist[i, j] + dist[j, 0] <= max_length + 1e-6:
                feasible_edges.add((i, j))

    # =================================================================
    # Model
    # =================================================================
    m = Model("BCCSP")
    if not verbose:
        m.Params.outputFlag = 0

    # ----- Variables -----
    # x[i,j] for i<j: undirected edge (only feasible edges)
    x = {}
    for (i, j) in feasible_edges:
        if i == 0:
            # Depot edge: can be used 0, 1, or 2 times (for out-and-back)
            x[i, j] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f'x_{i}_{j}')
        else:
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

    # y[i]: node i visited, i=1..N
    y = {}
    for i in range(1, n):
        y[i] = m.addVar(vtype=GRB.BINARY, name=f'y_{i}')

    # z[k]: sensor k covered
    z = {}
    for k in range(N):
        z[k] = m.addVar(vtype=GRB.BINARY, name=f'z_{k}')

    m.update()

    # Helper to get x variable for any (i,j) pair
    def xvar(i, j):
        a, b = min(i, j), max(i, j)
        if (a, b) in x:
            return x[a, b]
        return None  # Edge pruned

    # ----- Objective -----
    m.setObjective(
        quicksum(packets[k] * z[k] for k in range(N)),
        GRB.MAXIMIZE
    )

    # ----- Constraints -----

    # 1. Degree constraints (undirected):
    #    For sensor i: sum of edges incident to i = 2 * y[i]
    for i in range(1, n):
        incident = []
        for j in range(n):
            if j != i:
                v = xvar(i, j)
                if v is not None:
                    incident.append(v)
        if incident:
            m.addConstr(quicksum(incident) == 2 * y[i], name=f'deg_{i}')
        else:
            # No feasible edges to this node - can't visit
            m.addConstr(y[i] == 0, name=f'unreachable_{i}')

    # Depot degree = 2 * tour_exists
    tour_exists = m.addVar(vtype=GRB.BINARY, name='tour_exists')
    depot_incident = []
    for j in range(1, n):
        v = xvar(0, j)
        if v is not None:
            depot_incident.append(v)

    if depot_incident:
        m.addConstr(quicksum(depot_incident) == 2 * tour_exists, name='depot_deg')

    # Link tour_exists to y
    for i in range(1, n):
        m.addConstr(tour_exists >= y[i], name=f'te_lb_{i}')
    m.addConstr(tour_exists <= quicksum(y[i] for i in range(1, n)), name='te_ub')

    # 2. Budget constraint
    m.addConstr(
        quicksum(x[i, j] * dist[i, j] for (i, j) in feasible_edges) <= max_length,
        name='budget'
    )

    # 3. Coverage
    for k in range(N):
        if len(coverage[k]) > 0:
            m.addConstr(z[k] <= quicksum(y[i] for i in coverage[k]), name=f'cov_{k}')
        else:
            m.addConstr(z[k] == 0, name=f'nocov_{k}')

    # 4. Strengthening: x[i,j] <= y[i] and x[i,j] <= y[j] for sensor-sensor edges
    for (i, j) in feasible_edges:
        if i > 0 and j > 0:
            m.addConstr(x[i, j] <= y[i], name=f'link_{i}_{j}_a')
            m.addConstr(x[i, j] <= y[j], name=f'link_{i}_{j}_b')
        elif i == 0:
            # depot-sensor: x[0,j] <= 2*y[j]
            m.addConstr(x[0, j] <= 2 * y[j], name=f'link_depot_{j}')

    # =================================================================
    # Lazy subtour elimination callback
    # =================================================================
    def find_connected_components(edge_list):
        """Find connected components using DFS."""
        adj = defaultdict(list)
        for i, j, count in edge_list:
            for _ in range(count):
                adj[i].append(j)
                adj[j].append(i)

        visited = set()
        components = []
        all_nodes = set()
        for i, j, _ in edge_list:
            all_nodes.add(i)
            all_nodes.add(j)

        for start in all_nodes:
            if start in visited:
                continue
            component = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for nb in adj[node]:
                    if nb not in visited:
                        stack.append(nb)
            components.append(component)

        return components

    def subtour_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get selected edges with multiplicities
            selected = []
            for (i, j) in feasible_edges:
                val = model.cbGetSolution(x[i, j])
                count = int(round(val))
                if count > 0:
                    selected.append((i, j, count))

            if not selected:
                return

            components = find_connected_components(selected)

            if len(components) <= 1:
                return

            # Add SEC for each subtour not containing depot
            for S in components:
                if 0 in S:
                    continue
                S_list = sorted(S)
                S_set = S
                # Sum of edges within S <= |S| - 1
                edges_in_S = []
                for (ei, ej) in feasible_edges:
                    if ei in S_set and ej in S_set:
                        edges_in_S.append(x[ei, ej])
                if edges_in_S:
                    model.cbLazy(quicksum(edges_in_S) <= len(S) - 1)

    # Solver params
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout is not None:
        m.Params.timeLimit = timeout
    if gap is not None:
        m.Params.mipGap = gap

    # Optimize
    m.optimize(subtour_callback)

    # =================================================================
    # Extract solution
    # =================================================================
    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        if m.SolCount == 0:
            raise RuntimeError("No feasible solution found")

        obj_val = m.objVal

        # Build edge usage map
        edge_usage = defaultdict(int)
        for (i, j) in feasible_edges:
            count = int(round(x[i, j].X))
            if count > 0:
                edge_usage[(i, j)] = count

        def use_edge(a, b):
            key = (min(a, b), max(a, b))
            if edge_usage[key] > 0:
                edge_usage[key] -= 1
                return True
            return False

        def available_neighbors(node):
            nbrs = []
            for j in range(n):
                if j != node:
                    key = (min(node, j), max(node, j))
                    if edge_usage[key] > 0:
                        nbrs.append(j)
            return nbrs

        # Follow tour from depot (Eulerian-style traversal)
        tour = []
        current = 0
        while True:
            nbrs = available_neighbors(current)
            if not nbrs:
                break
            # Prefer non-depot neighbors to avoid closing tour too early
            next_node = None
            for nb in nbrs:
                if nb != 0:
                    next_node = nb
                    break
            if next_node is None:
                next_node = nbrs[0]  # Must be depot

            use_edge(current, next_node)

            if next_node == 0:
                break
            tour.append(next_node - 1)  # 0-indexed sensor
            current = next_node

        return obj_val, tour, m.Runtime

    elif m.status == GRB.INFEASIBLE:
        return 0.0, [], 0.0
    else:
        raise RuntimeError(f"Optimization failed with status {m.status}")


def compute_tour_distance(tour: List[int], loc: np.ndarray, depot: np.ndarray) -> float:
    if len(tour) == 0:
        return 0.0
    dist = 0.0
    current = depot
    for idx in tour:
        dist += euclidean_distance(current, loc[idx])
        current = loc[idx]
    dist += euclidean_distance(current, depot)
    return dist


def compute_covered_packets(tour: List[int], loc: np.ndarray,
                            packets: np.ndarray, radius: float) -> float:
    N = len(loc)
    covered = np.zeros(N, dtype=bool)
    for visited_idx in tour:
        for k in range(N):
            if euclidean_distance(loc[visited_idx], loc[k]) <= radius + 1e-10:
                covered[k] = True
    return np.sum(packets[covered])


if __name__ == "__main__":
    np.random.seed(42)

    N = 10
    depot = np.array([0.0, 0.0])
    loc = np.random.uniform(0, 1, size=(N, 2))
    packets = np.random.randint(1, 101, size=N).astype(float)
    radius = 0.15
    max_length = 2.0

    print("=" * 80)
    print("Testing Gurobi ILP Solver for BC-CSP")
    print("=" * 80)
    print(f"Instance: {N} sensors, radius={radius}, budget={max_length}")
    print(f"Packets: {packets}")

    obj_val, tour, solve_time = solve_bccsp_gurobi(
        depot, loc, packets, max_length, radius, verbose=True
    )

    tour_dist = compute_tour_distance(tour, loc, depot)
    covered = compute_covered_packets(tour, loc, packets, radius)

    print(f"\nObjective: {obj_val:.2f}")
    print(f"Tour: {tour}")
    print(f"Distance: {tour_dist:.4f} / {max_length}")
    print(f"Covered packets: {covered:.2f}")
    print(f"Solve time: {solve_time:.4f} s")