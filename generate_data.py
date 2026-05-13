import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()



def generate_csp_data(dataset_size, graph_size, cover_range=7, radius=0.15):
    # loc: (dataset_size, graph_size, 2) as lists
    loc = np.random.uniform(size=(dataset_size, graph_size, 2)).tolist()

    # Match CSPDataset dict schema exactly:
    # {'loc': ..., 'cover_range': cover_range, 'radius': [[radius]]}
    # radius as [[r]] mirrors torch.tensor([[radius]]) in CSPDataset
    r = [[float(radius)]]
    return [
        {
            "loc": l,
            "cover_range": int(cover_range),
            "radius": r
        }
        for l in loc
    ]


def generate_bccsp_data(dataset_size, graph_size, radius=0.15, max_length=None, 
                        packets_low=1, packets_high=100):
    """
    Generate BCCSP (Budget-Constrained Covering Salesman Problem) data
    
    Returns a list of tuples: (loc, packets, max_length, radius)
    This format matches what BCCSPDataset expects in its pickle file loader.
    
    Args:
        dataset_size: Number of instances
        graph_size: Number of sensor nodes (N)
        radius: Coverage radius (default 0.15)
        max_length: Budget constraint (default based on graph_size)
        packets_low: Minimum packet count per sensor (default 1)
        packets_high: Maximum packet count per sensor (default 100)
    """
    # Default max_length based on graph_size (matching problem_bccsp.py defaults)
    if max_length is None:
        MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}
        max_length = MAX_LENGTHS.get(graph_size, 3.0)
    
    # Generate sensor locations (uniform [0,1]^2)
    loc = np.random.uniform(size=(dataset_size, graph_size, 2))
    
    # Generate packet counts (integer in [packets_low, packets_high])
    packets = np.random.randint(low=packets_low, high=packets_high + 1, 
                                size=(dataset_size, graph_size))
    
    # Return as list of tuples matching BCCSPDataset pickle format
    # (loc, packets, max_length, radius)
    return [
        (
            loc[i].tolist(),
            packets[i].tolist(),
            float(max_length),
            float(radius)
        )
        for i in range(dataset_size)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='all',
                        help="Problem: 'tsp', 'csp', 'bccsp' or 'all'")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    # NEW: CSP-specific args
    parser.add_argument('--cover_range', type=int, default=7,
                        help="CSP cover_range (kept for compatibility; radius-based CSP ignores it)")
    parser.add_argument('--radius', type=float, default=0.15,
                        help="CSP radius (coverage distance)")
    parser.add_argument('--max_length', type=float, default=None,
                        help="Budget constraint for BCCSP (default: auto based on graph_size)")
    parser.add_argument('--packets_low', type=int, default=1,
                        help="Minimum packet count per sensor for BCCSP (default 1)")
    parser.add_argument('--packets_high', type=int, default=100,
                        help="Maximum packet count per sensor for BCCSP (default 100)")

    opts = parser.parse_args()

    assert opts.filename is None or (opts.problem != 'all' and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': [None],
        'csp': [None],
        'bccsp': [None],
    }

    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        assert opts.problem in distributions_per_problem, f"Unknown problem: {opts.problem}"
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'tsp':
                    dataset = generate_tsp_data(opts.dataset_size, graph_size)
                elif problem == 'csp':
                    dataset = generate_csp_data(
                        opts.dataset_size,
                        graph_size,
                        cover_range=opts.cover_range,
                        radius=opts.radius
                    )
                elif problem == 'bccsp':
                    dataset = generate_bccsp_data(
                        opts.dataset_size,
                        graph_size,
                        radius=opts.radius,
                        max_length=opts.max_length,
                        packets_low=opts.packets_low,
                        packets_high=opts.packets_high
                    )
                else:
                    assert False, "Unknown problem: {}".format(problem)

                print(dataset[0])
                save_dataset(dataset, filename)