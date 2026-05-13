"""
Prize-Based Covering Algorithm (PCA) for Budget-Constrained Covering Salesman Problem

Based on Algorithm 2 from the BC-CSP paper.
"""

import numpy as np
from typing import List, Tuple, Set


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)


def compute_1_covered_neighbors(loc: np.ndarray, radius: float) -> List[Set[int]]:
    """
    Compute N^1_i for each node i: nodes within coverage radius.
    
    Args:
        loc: (N, 2) array of sensor locations
        radius: coverage radius (Tr in the paper)
    
    Returns:
        List of sets, where N1[i] contains indices of nodes within radius of node i
    """
    N = len(loc)
    N1 = []
    
    for i in range(N):
        neighbors = set()
        for j in range(N):
            if i != j and euclidean_distance(loc[i], loc[j]) <= radius:
                neighbors.add(j)
        N1.append(neighbors)
    
    return N1


def compute_2_covered_neighbors(N1: List[Set[int]]) -> List[Set[int]]:
    """
    Compute N^2_i for each node i: nodes that are 2-hops away.
    
    N^2_i = union of N^1_k for all k in N^1_i, excluding nodes already in N^1_i and i itself.
    """
    N = len(N1)
    N2 = []
    
    for i in range(N):
        two_covered = set()
        for k in N1[i]:
            two_covered.update(N1[k])
        
        # Remove i itself and 1-covered nodes
        two_covered.discard(i)
        two_covered -= N1[i]
        N2.append(two_covered)
    
    return N2


def initialize_prizes(packets: np.ndarray, N1: List[Set[int]]) -> np.ndarray:
    """
    Initialize prizes for all nodes (lines 2-5 in Algorithm 2).
    
    p_i = sum of packets from node i and all its 1-covered neighbors.
    """
    N = len(packets)
    prizes = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        # Prize includes packets from i itself
        prize = packets[i]
        # Plus packets from all 1-covered neighbors
        for j in N1[i]:
            prize += packets[j]
        prizes[i] = prize
    
    return prizes


def compute_battery_feasible_nodes(
    loc: np.ndarray,
    current_pos: np.ndarray,
    depot_pos: np.ndarray,
    current_battery: float,
    unvisited: Set[int],
    mu: float
) -> Set[int]:
    """
    Compute F(i, B, U): battery-feasible nodes from current position.
    
    A node j is battery-feasible if:
    - j is unvisited (j in U)
    - Battery is sufficient: B >= c(i,j) * mu + c(j,s) * mu
    """
    feasible = set()
    
    for j in unvisited:
        dist_to_j = euclidean_distance(current_pos, loc[j])
        dist_j_to_depot = euclidean_distance(loc[j], depot_pos)
        
        energy_needed = (dist_to_j + dist_j_to_depot) * mu
        
        if current_battery >= energy_needed:
            feasible.add(j)
    
    return feasible


def compute_prize_cost_ratio(
    prizes: np.ndarray,
    loc: np.ndarray,
    current_pos: np.ndarray,
    feasible_nodes: Set[int],
    mu: float
) -> Tuple[int, float]:
    """
    Compute prize-cost ratio pcr(i, k) for all feasible nodes.
    
    pcr(i, k) = p_k / c(i, k)
    
    Returns:
        (best_node, best_ratio)
    """
    best_node = -1
    best_ratio = -np.inf
    
    for k in feasible_nodes:
        if prizes[k] <= 0:  # Skip nodes with no prize
            continue
            
        cost = euclidean_distance(current_pos, loc[k]) * mu
        if cost <= 0:
            ratio = np.inf
        else:
            ratio = prizes[k] / cost
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_node = k
    
    return best_node, best_ratio


def update_prizes_after_visit(
    j: int,
    packets: np.ndarray,
    prizes: np.ndarray,
    N1: List[Set[int]],
    N2: List[Set[int]]
) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
    """
    Update prizes after visiting node j (lines 11-31 in Algorithm 1).

    Returns:
        (updated_packets, updated_prizes, nodes_collected_from)
    """
    packets = packets.copy()
    prizes = prizes.copy()
    A_j = set()  # Nodes whose packets are collected when visiting j

    # Save original packet values BEFORE zeroing — needed for step 2.
    # The paper's "d_l" and "d_k" in lines 23-28 refer to the packet values
    # at the time of collection, not after zeroing.
    d_orig = packets.copy()

    # Step 1: Collect packets from j and its 1-covered neighbors (lines 12-20)
    if packets[j] != 0:
        packets[j] = 0
        A_j.add(j)

    for k in N1[j]:
        if packets[k] != 0:
            prizes[k] -= d_orig[k]
            packets[k] = 0
            A_j.add(k)

    prizes[j] = 0

    # Step 2: Update prizes of 1-covered neighbors and their neighbors (lines 21-31)
    for k in N1[j]:
        for l in N1[k]:
            if l in A_j:
                # Case 1 (lines 23-24): l's packets were collected this round.
                # l is in N1[j], so subtract l's original packets from k's prize.
                prizes[k] -= d_orig[l]
            else:
                # Case 2 (lines 26-28): l not collected this round.
                # If l is j's 2-covered node AND k was collected, update l's prize.
                if l in N2[j] and k in A_j:
                    prizes[l] -= d_orig[k]

    return packets, prizes, A_j


def compute_tour_cost(tour: List[int], loc: np.ndarray, depot: np.ndarray, mu: float) -> float:
    """Compute total energy cost of the tour."""
    if len(tour) == 0:
        return 0.0
    
    cost = 0.0
    current_pos = depot
    
    for node_idx in tour:
        cost += euclidean_distance(current_pos, loc[node_idx]) * mu
        current_pos = loc[node_idx]
    
    # Return to depot
    cost += euclidean_distance(current_pos, depot) * mu
    
    return cost


def compute_covered_packets(
    tour: List[int],
    loc: np.ndarray,
    packets: np.ndarray,
    radius: float
) -> float:
    """
    Compute total packets covered by the tour.
    
    A sensor is covered if it's within radius of any visited node.
    """
    N = len(loc)
    covered = np.zeros(N, dtype=bool)
    
    for visited_node in tour:
        # Check which nodes are covered by this visited node
        for i in range(N):
            if euclidean_distance(loc[visited_node], loc[i]) <= radius:
                covered[i] = True
    
    return np.sum(packets[covered])


def pca_solve(
    loc: np.ndarray,
    packets: np.ndarray,
    max_length: float,
    radius: float,
    mu: float = 1.0,
    depot: np.ndarray = None,
    print_enable: bool = False
) -> Tuple[List[int], float, float]:
    """
    Prize-Based Covering Algorithm (PCA) for BC-CSP.
    
    Args:
        loc: (N, 2) array of sensor locations
        packets: (N,) array of packet counts at each sensor
        max_length: Budget constraint (maximum tour length)
        radius: Coverage radius (Tr in the paper)
        mu: Energy consumption rate (default 1.0)
        depot: Depot location (default [0, 0])
        print_enable: Whether to print debug information
    
    Returns:
        (tour, total_covered_packets, tour_cost)
        tour: List of visited node indices (0-indexed)
        total_covered_packets: Total packets collected
        tour_cost: Energy cost of the tour
    """
    if depot is None:
        depot = np.array([0.0, 0.0])
    
    N = len(loc)
    
    # Line 1: Initialize
    R = []  # Tour (list of visited nodes)
    C_R = 0.0  # Energy cost
    P_R = 0.0  # Total prize collected
    U = set(range(N))  # Unvisited nodes
    B = max_length * mu  # Current battery (convert length to energy)
    i_pos = depot  # Current position
    
    packets_remaining = packets.copy().astype(np.float64)
    
    # Lines 2-5: Compute initial prizes
    N1 = compute_1_covered_neighbors(loc, radius)
    N2 = compute_2_covered_neighbors(N1)
    prizes = initialize_prizes(packets_remaining, N1)
    
    if print_enable:
        print(f"Initial prizes: {prizes}")
        print(f"Budget: {max_length}, Battery: {B}")
    
    iteration = 0
    # Line 6: Main loop - while there are battery-feasible nodes
    while True:
        # Line 7: Find battery-feasible nodes
        F = compute_battery_feasible_nodes(loc, i_pos, depot, B, U, mu)
        
        if len(F) == 0:
            break
        
        # Line 7: Select node with best prize-cost ratio
        j, pcr = compute_prize_cost_ratio(prizes, loc, i_pos, F, mu)
        
        if j == -1:  # No feasible node with positive prize
            break
        
        if print_enable:
            print(f"\nIteration {iteration}: Selecting node {j} with pcr={pcr:.4f}, prize={prizes[j]:.2f}")
        
        # Lines 8-10: Visit node j and update route info
        R.append(j)
        P_R += prizes[j]
        
        travel_cost = euclidean_distance(i_pos, loc[j]) * mu
        C_R += travel_cost
        B -= travel_cost
        
        U.remove(j)
        
        # Lines 11-31: Update prizes
        packets_remaining, prizes, A_j = update_prizes_after_visit(
            j, packets_remaining, prizes, N1, N2
        )
        
        # Line 32: Move to node j
        i_pos = loc[j]
        
        iteration += 1
    
    # Lines 34-35: Return to depot
    return_cost = euclidean_distance(i_pos, depot) * mu
    C_R += return_cost
    
    # Compute actual covered packets (for verification)
    total_covered = compute_covered_packets(R, loc, packets, radius)
    
    if print_enable:
        print(f"\nFinal tour: {R}")
        print(f"Tour cost: {C_R:.4f} / {max_length * mu:.4f}")
        print(f"Covered packets: {total_covered:.2f}")
        print(f"Prize collected: {P_R:.2f}")
    
    return R, total_covered, C_R / mu  # Return cost in terms of distance


def pca_tour_distance(loc: np.ndarray, tour: List[int], depot: np.ndarray = None) -> float:
    """
    Compute the total distance of a tour (for compatibility with eval.py).
    
    Args:
        loc: (N, 2) array of sensor locations
        tour: List of visited node indices
        depot: Depot location (default [0, 0])
    
    Returns:
        Total tour distance (not including energy coefficient mu)
    """
    if depot is None:
        depot = np.array([0.0, 0.0])
    
    if len(tour) == 0:
        return 0.0
    
    dist = 0.0
    current = depot
    
    for idx in tour:
        dist += euclidean_distance(current, loc[idx])
        current = loc[idx]
    
    # Return to depot
    dist += euclidean_distance(current, depot)
    
    return dist


def PCA(loc: np.ndarray, cover_range: int = 7, radius: float = 0.15,
        max_length: float = None, packets: np.ndarray = None,
        print_enable: bool = False) -> List[int]:
    """cover_range is ignored, kept for API compatibility."""
    if packets is None:
        # Generate random packets if not provided (for testing)
        packets = np.random.randint(1, 101, size=len(loc))
    
    if max_length is None:
        # Use default based on problem size
        n = len(loc)
        if n <= 20:
            max_length = 2.0
        elif n <= 50:
            max_length = 3.0
        else:
            max_length = 4.0
    
    tour, covered, cost = pca_solve(loc, packets, max_length, radius, 
                                     mu=1.0, print_enable=print_enable)
    return tour


def pca_cost(loc: np.ndarray, packets: np.ndarray, tour: List[int], radius: float) -> float:
    """
    Compute negative covered packets (cost to minimize = negative reward).
    
    This matches the cost computation in problem_bccsp.py.
    """
    covered_packets = compute_covered_packets(tour, loc, packets, radius)
    return -covered_packets  # Negative because we want to maximize coverage


if __name__ == "__main__":
    # Test the PCA algorithm
    np.random.seed(42)
    
    # Small test instance
    N = 10
    loc = np.random.uniform(0, 1, size=(N, 2))
    packets = np.random.randint(1, 101, size=N)
    radius = 0.15
    max_length = 2.0
    
    print("=" * 60)
    print("Testing PCA Algorithm")
    print("=" * 60)
    print(f"Instance: {N} sensors, radius={radius}, budget={max_length}")
    print(f"Packets: {packets}")
    
    tour, covered, cost = pca_solve(loc, packets, max_length, radius, print_enable=True)
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"Tour: {tour}")
    print(f"Tour length: {len(tour)} nodes")
    print(f"Tour cost: {cost:.4f}")
    print(f"Covered packets: {covered:.2f}")
    print(f"Objective (negative): {-covered:.2f}")