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
parser.add_argument("-N", "--num_samples", type=int)
parser.add_argument("-d", "--num_dimensions", type=int)
parser.add_argument("-s", "--sparsity", type=int)
parser.add_argument("--sigma", type=float, default=0.5)

## network
parser.add_argument("-m", "--num_nodes", default=1, type=int)
parser.add_argument("-p", "--probability", default=1, type=float)
parser.add_argument("-rho", "--connectivity", default=0, type=float)
parser.add_argument("--radius_const", type=float, default=1.01)
## solver
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--solver", choices=("cta", "atc", "netlasso", "dual_average", "mirror_descent"))
parser.add_argument("--max_iter", type=int, default=100000)

parser.add_argument("--gamma", type=float)
parser.add_argument("--lambda_const", type=float, default=4)
parser.add_argument("--communication", type=int, default=1)
parser.add_argument("--computation", type=int, default=1)
parser.add_argument("--scheduler", choices=("const","diminish", "diminish2"), default="const")
parser.add_argument("-b", "--range", default=10, type=float)
parser.add_argument("--generator", choices=("gaussian", "uniform"), default="gaussian")
## others
parser.add_argument("--seed", type=int, default=8989)
parser.add_argument("--sweep",action='store_true', default=False)
args = parser.parse_args()


def main():
    # register wandb
    wandb.init(project=f"" if not args.sweep else None,
                   entity="yaoji" if not args.sweep else None,
                   config=vars(args))
    wandb.run.log_code()
    batch_size_generator = ConstBatchSizeGenerator(args.batch_size, args.max_iter)
    if args.generator == "gaussian":
        generator = Generator(batch_size_generator=batch_size_generator, m=args.num_nodes, dimension=args.num_dimensions, sparsity=args.sparsity, noise_dev=args.sigma, seed=args.seed)
    else:
        generator = BoundedGenerator(batch_size_generator=batch_size_generator, m=args.num_nodes, dimension=args.num_dimensions, B=args.range, sparsity=args.sparsity, noise_dev=args.sigma, seed=args.seed)
    ground_truth = generator.theta
    ## processing network
    proj_radius = np.linalg.norm(generator.theta, ord=1) * args.radius_const
    print(np.linalg.norm(generator.theta - np.ones_like(generator.theta))/np.sqrt(5000)/np.sqrt(0.5))
    w = ErodoRenyi(m=args.num_nodes, rho=args.connectivity, p=args.probability, seed=args.seed).generate()

    ## process stepsize
    if args.scheduler == "const":
        gamma = ConstScheduler(args)
    elif args.scheduler == "diminish":
        gamma = DiminishScheduler(args)
    elif args.scheduler == "diminish2":
        gamma = DiminishSchedulerSecond(proj_radius, args)
    # solver run
    if args.solver == 'cta':
        solver = CTA(generator, w, gamma, args)
    elif args.solver == 'atc':
        solver = ATC(generator, w, gamma, args)
    elif args.solver == 'dual_average':
        solver = DualAveraging(generator, w, gamma, args)
    elif args.solver == 'netlasso':
        solver = NetLasso(generator, w, gamma, args)
    elif args.solver == 'mirror_descent':
        solver = MirrorDescent(generator, w, gamma, args)
    else:
        raise NotImplementedError("solver mode currently only support centralized or distributed")
    # start_timer = time.time()
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
