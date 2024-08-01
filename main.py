import time
import argparse
import pickle
import os
from generator import Generator, BoundedGenerator
from network import ErodoRenyi
from solver import *
from scheduler import *
from utils import *

import wandb


# configuration
parser = argparse.ArgumentParser(description='distributed optimization')
parser.add_argument('--storing_filepath', type=str, help='storing_file_path')
parser.add_argument('--storing_filename', type=str, help='storing_file_name')
parser.add_argument('--data_dir', type=str, default="/export/home/a/ji151/distributed_stochastic/")
## data
parser.add_argument("-d", "--num_dimensions", type=int)
parser.add_argument("-s", "--sparsity", type=int)
parser.add_argument("--sigma", type=float, default=0.5)
## network
parser.add_argument("-m", "--num_nodes", default=1, type=int)
# parser.add_argument("-p", "--probability", default=1, type=float)
parser.add_argument("-rho", "--connectivity", default=0, type=float)
parser.add_argument("--radius", type=float, default=1.01)
## solver
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--solver")
parser.add_argument("--init_theta", type=float, default=0)
parser.add_argument("--gamma", type=float)
parser.add_argument("--lmda", type=float, default=4)
parser.add_argument("--num_iter", type=int, default=1000)
parser.add_argument("--scheduler", choices=("const","diminish", "diminish2", "multistage"), default="const")
parser.add_argument("-b", "--range", default=10, type=float)
parser.add_argument("--generator", choices=("gaussian", "uniform"), default="gaussian")
parser.add_argument("--first_stage_num_iter", type=float, default=10)
parser.add_argument("--num_lincon_stage", type=int, default=5)
parser.add_argument("--num_asyn_stage", type=int, default=5)
parser.add_argument("--factor", type=float, default=0.5)
## others
parser.add_argument("--seed", type=int, default=8989)
parser.add_argument("--no_wandb", action='store_true', default=False)
parser.add_argument("--sweep", action='store_true', default=False)
args = parser.parse_args()


def main():
    # register wandb
    if not args.no_wandb:
        wandb.init(project=f"" if not args.sweep else None,
                    entity="yaoji" if not args.sweep else None,
                    config=vars(args))
        wandb.run.log_code()
    if args.generator == "gaussian":
        generator = Generator(m=args.num_nodes, dimension=args.num_dimensions, sparsity=args.sparsity, noise_dev=args.sigma, seed=args.seed)
    else:
        generator = BoundedGenerator(m=args.num_nodes, dimension=args.num_dimensions, B=args.range, sparsity=args.sparsity, noise_dev=args.sigma, seed=args.seed)
    ground_truth = generator.theta
    ## processing network
    print(np.linalg.norm(generator.theta - np.ones_like(generator.theta))/np.sqrt(5000)/np.sqrt(0.5))
    # w = ErodoRenyi(m=args.num_nodes, rho=args.connectivity, p=args.probability, seed=args.seed).generate()
    w = ErodoRenyi(m=args.num_nodes, rho=args.connectivity, seed=args.seed).generate()

    # solver run
    solver = eval(args.solver)(generator, w, args)
    solver.fit()
    # finish_timer = time.time()
    # print("solver spend {} seconds".format(finish_timer-start_timer))
    # home_dir = args.data_dir
    # output_filepath = os.path.join(home_dir, args.storing_filepath)
    # output_filename = args.storing_filename
    # print(output_filepath + output_filename)
    # os.makedirs(output_filepath, exist_ok=True)
    # pickle.dump(outputs, open(output_filepath + output_filename, "wb"))


if __name__ == "__main__":
    main()
