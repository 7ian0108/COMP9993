import argparse
import pprint as pp
import time
import warnings

import numpy as np
import tqdm
import tsplib95

try:
    from concorde.tsp import TSPSolver  
    HAS_CONCORDE = True
except ImportError:
    TSPSolver = None
    HAS_CONCORDE = False

import lkh  

warnings.filterwarnings("ignore")


def solve_tsp(nodes_coord: np.ndarray, num_nodes: int, opts) -> list:
    if opts.solver == "concorde":
        if not HAS_CONCORDE:
            raise RuntimeError("You selected solver=concorde but pyconcorde is not installed.")
        scale = 1e6
        solver = TSPSolver.from_data(
            nodes_coord[:, 0] * scale,
            nodes_coord[:, 1] * scale,
            norm="EUC_2D",
        )
        solution = solver.solve(verbose=False)
        tour = solution.tour  

    elif opts.solver == "lkh":
        scale = 1e6
        lkh_path = opts.lkh_path

        problem = tsplib95.models.StandardProblem()
        problem.name = 'TSP'
        problem.type = 'TSP'
        problem.dimension = num_nodes
        problem.edge_weight_type = 'EUC_2D'

        problem.node_coords = {n + 1: nodes_coord[n] * scale for n in range(num_nodes)}

        solution = lkh.solve(
            lkh_path,
            problem=problem,
            max_trials=opts.lkh_trails,
            runs=10
        )
       
        tour = [n - 1 for n in solution[0]]

    else:
        raise ValueError(f"Unknown solver: {opts.solver}")

    tour_sorted = np.sort(np.array(tour))
    assert (tour_sorted == np.arange(num_nodes)).all(), "Tour is not a permutation!"

    return tour


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=128000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--solver", type=str, default="lkh",
                        choices=["lkh", "concorde"])
    parser.add_argument("--lkh_trails", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--lkh_path",
        type=str,
        default=r"D:\COMP9991\GraphVae_improve\VAE_TSP\LKH-3.0.13\LKH-3.exe",
    )
    opts = parser.parse_args()

    assert opts.num_samples % opts.batch_size == 0, \
        "Number of samples must be divisible by batch size"

    np.random.seed(opts.seed)

    if opts.filename is None:
        opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_{opts.solver}.txt"

    pp.pprint(vars(opts))

    total_batches = opts.num_samples // opts.batch_size

    with open(opts.filename, "w") as f:
        start_time = time.time()

        pbar = tqdm.tqdm(range(total_batches), desc="Generating TSP data")
        for b_idx in pbar:
            num_nodes = np.random.randint(
                low=opts.min_nodes,
                high=opts.max_nodes + 1
            )
            assert opts.min_nodes <= num_nodes <= opts.max_nodes

        
            batch_nodes_coord = np.random.random(
                [opts.batch_size, num_nodes, 2]
            )

    
            tours = []
            for idx in range(opts.batch_size):
                nodes_coord = batch_nodes_coord[idx]
                tour = solve_tsp(nodes_coord, num_nodes, opts)
                tours.append(tour)

   
            for idx, tour in enumerate(tours):
           
                if (np.sort(tour) == np.arange(num_nodes)).all():
               
                    coords_flat = batch_nodes_coord[idx].reshape(-1)
                    coord_str = " ".join(str(v) for v in coords_flat)
                    f.write(coord_str)

      
                    f.write(" output ")


                    tour_1based = [node_idx + 1 for node_idx in tour]
                    tour_1based.append(tour_1based[0])
                    tour_str = " ".join(str(v) for v in tour_1based)
                    f.write(tour_str)

                    f.write("\n")

        end_time = time.time() - start_time

    print(f"Completed generation of {opts.num_samples} samples of "
          f"TSP{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time per instance: {end_time / opts.num_samples:.3f}s")

