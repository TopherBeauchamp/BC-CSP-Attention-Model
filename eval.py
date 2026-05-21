import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta, datetime
import json
from utils.functions import parse_softmax_temperature
import pickle
try:
    # Preferred (repo layout)
    from problems.bccsp.pca_baseline import pca_solve
except Exception:
    # Fallback (standalone file)
    try:
        from pca_baseline import pca_solve  # type: ignore
    except Exception:
        pca_solve = None

try:
    # Preferred (repo layout)
    from problems.bccsp.gurobi_bccsp import solve_bccsp_gurobi
except Exception:
    # Fallback (standalone file)
    try:
        from gurobi_bccsp import solve_bccsp_gurobi  # type: ignore
    except Exception:
        solve_bccsp_gurobi = None

mp = torch.multiprocessing.get_context('spawn')

def normalize_bccsp_tour_to_sensor_indices(tour, N: int):
    """Accept either action-space tours (0=depot, 1..N=sensors) or sensor-index tours (0..N-1).
    Returns a clean list of sensor indices in [0, N-1].
    """
    if tour is None:
        return []
    t = [int(x) for x in tour if x is not None]
    t = [x for x in t if x >= 0]
    if len(t) == 0:
        return []
    # Heuristic: if any element equals N or if there are zeros with positives <= N, treat as action space
    is_action_space = (max(t) <= N) and (0 in t) and any(x > 0 for x in t)
    if max(t) == N:
        is_action_space = True
    if is_action_space:
        t = [x - 1 for x in t if x > 0]  # drop depot=0, shift
    # Now t should be sensor indices
    return [x for x in t if 0 <= x < N]


def sensor_indices_to_action_tour(sensor_tour):
    """Convert sensor indices (0..N-1) to action tour (0=depot, 1..N=sensors) with depot start/end."""
    st = [int(x) for x in sensor_tour if x is not None]
    st = [x for x in st if x >= 0]
    return [0] + [x + 1 for x in st] + [0]



def compute_bccsp_metrics(dataset, tours):
    """
    Compute BCCSP metrics from dataset and tours.

    tours may be either:
      - action-space: 0=depot, 1..N=sensors (model output)
      - sensor-space: 0..N-1 (baseline solvers)

    Returns:
        packets_collected: list of packets collected per instance
        distances: list of tour distances per instance
        nodes_visited: list of number of nodes visited per instance
    """
    packets_collected = []
    distances = []
    nodes_visited = []

    for inst, tour in zip(dataset, tours):
        # Parse instance
        if isinstance(inst, tuple):
            loc, packets, max_length, radius = inst
            loc = np.asarray(loc)
            packets = np.asarray(packets)
        elif isinstance(inst, dict):
            loc = np.asarray(inst["loc"])
            packets = np.asarray(inst["packets"])
            radius = float(inst.get("radius", 0.15))
        else:
            # Old CSP format
            loc = np.asarray(inst)
            packets = np.ones(len(loc)) * 50  # Default if not available
            radius = 0.15

        N = len(loc)
        sensor_tour = normalize_bccsp_tour_to_sensor_indices(tour, N)

        # Covered packets
        covered = np.zeros(N, dtype=bool)
        for visited_idx in sensor_tour:
            dists = np.linalg.norm(loc - loc[visited_idx], axis=1)
            covered |= (dists <= radius)
        total_packets = float(np.sum(packets[covered]))

        # Tour distance (depot at origin)
        depot = np.zeros(2)
        if len(sensor_tour) == 0:
            dist = 0.0
        else:
            tour_dist = 0.0
            current = depot
            for idx in sensor_tour:
                tour_dist += np.linalg.norm(loc[idx] - current)
                current = loc[idx]
            tour_dist += np.linalg.norm(current - depot)  # return to depot
            dist = float(tour_dist)

        packets_collected.append(total_packets)
        distances.append(dist)
        nodes_visited.append(len(sensor_tour))

    return packets_collected, distances, nodes_visited


def print_bccsp_results(packets_collected, distances, nodes_visited, durations, label="Results"):
    """Print formatted BCCSP results."""
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print(f"Average packets collected: {np.mean(packets_collected):.2f} ± {2 * np.std(packets_collected) / np.sqrt(len(packets_collected)):.2f}")
    print(f"Average distance traveled: {np.mean(distances):.2f} ± {2 * np.std(distances) / np.sqrt(len(distances)):.2f}")
    print(f"Average nodes visited: {np.mean(nodes_visited):.2f} ± {2 * np.std(nodes_visited) / np.sqrt(len(nodes_visited)):.2f}")
    print(f"Average serial duration: {np.mean(durations):.4f} ± {2 * np.std(durations) / np.sqrt(len(durations)):.4f} s")
    print("=" * 80 + "\n")


def _get_num_nodes(inst):
    """Extract node count from a dataset instance (tuple or dict format)."""
    if isinstance(inst, tuple):
        return len(inst[0])
    elif isinstance(inst, dict):
        return len(inst["loc"])
    return None


def save_overall_results_json(packets_collected, distances, nodes_visited, durations,
                              results_dir, subfolder, filename_stem, val_size,
                              num_nodes=None, total_duration_s=None, extra_fields=None):
    """Save a JSON summary of overall results to results/overallResults/{subfolder}/{val_size}_Instances_{num_nodes}_nodes/."""
    n = len(packets_collected)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_instances": n,
        "num_nodes": num_nodes,
        "avg_packets_collected": round(float(np.mean(packets_collected)), 4),
        "stderr_packets_collected": round(float(2 * np.std(packets_collected) / np.sqrt(n)), 4),
        "avg_distance_traveled": round(float(np.mean(distances)), 4),
        "stderr_distance_traveled": round(float(2 * np.std(distances) / np.sqrt(n)), 4),
        "avg_nodes_visited": round(float(np.mean(nodes_visited)), 4),
        "stderr_nodes_visited": round(float(2 * np.std(nodes_visited) / np.sqrt(n)), 4),
        "avg_serial_duration_s": round(float(np.mean(durations)), 4),
        "stderr_serial_duration_s": round(float(2 * np.std(durations) / np.sqrt(n)), 4),
        "total_duration_s": round(float(total_duration_s), 4) if total_duration_s is not None else None,
    }
    if extra_fields:
        summary.update(extra_fields)

    nodes_suffix = f"_{num_nodes}_nodes" if num_nodes is not None else ""
    out_dir = os.path.join(results_dir, "overallResults", subfolder, f"{val_size}_Instances{nodes_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filename_stem}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Overall results saved to {out_path}")


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    if opts.baseline is not None:
        assert opts.baseline.lower() in ("ls1", "pca", "gurobi"), "Baseline must be one of: ls1, pca, gurobi"

        # Load dataset directly (list of dicts or list of loc arrays)
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        data = data[opts.offset: opts.offset + opts.val_size]

        start_all = time.time()
        costs = []  # cost = negative covered packets (to minimize)
        tours = []  # store action-space tours for consistency with model output
        durations = []

        baseline = opts.baseline.lower()

        for inst in tqdm(data, disable=opts.no_progress_bar):
            t0 = time.time()

            # Parse instance (tuple or dict)
            if isinstance(inst, tuple):
                loc, packets, max_length, radius = inst
                loc = np.asarray(loc, dtype=np.float64)
                packets = np.asarray(packets, dtype=np.float64)
                max_length = float(max_length)
                radius = float(radius)
            elif isinstance(inst, dict):
                loc = np.asarray(inst["loc"], dtype=np.float64)
                packets = np.asarray(inst["packets"], dtype=np.float64)
                max_length = float(inst.get("max_length", inst.get("max_len", 0.0)))
                radius = float(inst.get("radius", opts.radius))
            else:
                raise ValueError(f"Unsupported instance format for baseline: {type(inst)}")

            if baseline == "pca":
                if pca_solve is None:
                    raise RuntimeError("PCA baseline requested but pca_solve is not available/importable")
                sensor_tour, covered_packets, _tour_distance = pca_solve(
                    loc, packets, max_length, radius, mu=1.0, print_enable=False
                )
                action_tour = sensor_indices_to_action_tour(sensor_tour)

            elif baseline == "gurobi":
                if solve_bccsp_gurobi is None:
                    raise RuntimeError("Gurobi baseline requested but solve_bccsp_gurobi is not available/importable")
                depot = np.zeros(2, dtype=np.float64)
                obj_val, sensor_tour, _solve_time = solve_bccsp_gurobi(
                    depot, loc, packets, max_length, radius,
                    threads=opts.gurobi_threads,
                    timeout=opts.gurobi_timeout,
                    gap=opts.gurobi_gap,
                    verbose=opts.gurobi_verbose
                )
                # obj_val is covered packets (maximization)
                covered_packets = obj_val
                action_tour = sensor_indices_to_action_tour(sensor_tour)

            else:
                raise ValueError(f"Unknown baseline: {baseline}")

            # Compute covered packets from the (action-space) tour using shared logic
            covered_packets_list, _dists, _nodes = compute_bccsp_metrics([inst], [action_tour])
            covered_packets = covered_packets_list[0]
            cost = -covered_packets

            durations.append(time.time() - t0)
            costs.append(cost)
            tours.append(action_tour)

        # Compute BCCSP metrics for printing
        packets_collected, distances, nodes_visited = compute_bccsp_metrics(data, tours)

        # Print results
        print_bccsp_results(packets_collected, distances, nodes_visited, durations,
                           label=f"Baseline ({baseline.upper()}) Results")
        total_duration = time.time() - start_all
        print(f"Total duration: {timedelta(seconds=int(total_duration))}")

        # Save overall JSON summary
        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
        num_nodes = _get_num_nodes(data[0]) if data else None
        save_overall_results_json(
            packets_collected, distances, nodes_visited, durations,
            results_dir=opts.results_dir,
            subfolder=baseline,
            filename_stem=f"{dataset_basename}-{baseline}-{opts.offset}-{opts.offset + len(costs)}",
            val_size=len(costs),
            num_nodes=num_nodes,
            total_duration_s=total_duration,
            extra_fields={"baseline": baseline, "dataset": dataset_basename,
                          "offset": opts.offset, "val_size": len(costs)},
        )

        # Save results in same format
        results = list(zip(costs, tours, durations))
        parallelism = 1
        results_dir = os.path.join(opts.results_dir, "bccsp", dataset_basename)
        os.makedirs(results_dir, exist_ok=True)
        out_file = os.path.join(
            results_dir,
            f"{dataset_basename}-{baseline}-"
            f"{opts.offset}-{opts.offset + len(costs)}{ext}"
        )
        save_dataset((results, parallelism), out_file)

        return costs, tours, durations

    model, _ = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative

    # Load dataset for metric computation
    with open(dataset_path, "rb") as f:
        full_data = pickle.load(f)
    data = full_data[opts.offset:opts.offset + len(tours)]
    
    # Compute BCCSP metrics
    packets_collected, distances, nodes_visited = compute_bccsp_metrics(data, tours)
    
    # Print results
    print_bccsp_results(packets_collected, distances, nodes_visited, durations,
                       label=f"Model Results ({opts.decode_strategy})")
    print(f"Average parallel duration: {np.mean(durations) / parallelism:.4f} s")
    print(f"Total duration: {timedelta(seconds=int(np.sum(durations) / parallelism))}")

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])

    # Save overall JSON summary
    num_nodes = _get_num_nodes(data[0]) if data else None
    save_overall_results_json(
        packets_collected, distances, nodes_visited, durations,
        results_dir=opts.results_dir,
        subfolder="nco",
        filename_stem=f"{dataset_basename}-{model_name}-{opts.decode_strategy}-{opts.offset}-{opts.offset + len(costs)}",
        val_size=len(costs),
        num_nodes=num_nodes,
        total_duration_s=float(np.sum(durations) / parallelism),
        extra_fields={"model": model_name, "decode_strategy": opts.decode_strategy,
                      "dataset": dataset_basename, "offset": opts.offset, "val_size": len(costs)},
    )
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            softmax_temp, opts.offset, opts.offset + len(costs), ext
        ))
    else:
        out_file = opts.o

    assert opts.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)

    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # fixed length
            elif model.problem.NAME in ("csp", "bccsp"):
                # CSP/BCCSP sequences are padded with -1 for unused steps
                seq = seq[seq > -1].tolist()
            else:
                assert False, "Unknown problem: {}".format(model.problem.NAME)

            # Note VRP only
            results.append((cost, seq, duration))

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    # parser.add_argument('--decode_type', type=str, default='greedy',
    #                     help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--baseline', type=str, default=None,
                        help="Run a baseline instead of the model (pca, gurobi)")
    parser.add_argument('--radius', type=float, default=0.15,
                        help="Radius for BCCSP baselines (PCA/Gurobi read radius from dataset)")
    parser.add_argument('--gurobi_threads', type=int, default=0, help='Gurobi threads (0=default)')
    parser.add_argument('--gurobi_timeout', type=float, default=None, help='Gurobi time limit in seconds')
    parser.add_argument('--gurobi_gap', type=float, default=None, help='Gurobi MIP gap (e.g. 0.01)')
    parser.add_argument('--gurobi_verbose', action='store_true', help='Enable Gurobi solver output')

    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    widths = opts.width if opts.width is not None else [0]

    for width in widths:
        for dataset_path in opts.datasets:
            eval_dataset(dataset_path, width, opts.softmax_temperature, opts)