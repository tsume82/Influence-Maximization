"""
Heuristics calculation given the spread function,
you can use single arguments or pass a json config file with all the arguments
"""

from functools import partial
import random
import time
import argparse
import json

import heuristics
from spread.monte_carlo import MonteCarlo_simulation as monte_carlo
from spread.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
from spread.two_hop import two_hop_spread
from utils import load_graph, dict2csv

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Heuristics computation')

	parser.add_argument('--k', type=int, default=10, help='seed set size')
	parser.add_argument('--g_type', default='CA-GrQc', choices=['wiki', 'amazon', 'epinions',
																'twitter', 'facebook', 'CA-GrQc'],
						help='graph type')
	parser.add_argument('--model', default="IC", choices=['IC', 'WC'], help='type of influence propagation model')
	parser.add_argument('--p', type=float, default=0.01, help='probability of influence spread in IC model')
	parser.add_argument('--no_simulations', type=int, default=100, help='number of simulations for spread calculation'
																		' when montecarlo mehtod is used')
	parser.add_argument('--heuristic', type=str, default="general_greedy", choices=["high_degree_nodes", "general_greedy",
																						  "CELF",
																					"single_discount_high_degree_nodes",
																					"generalized_degree_discount"])
	parser.add_argument('--random_seed', type=int, default=43)
	parser.add_argument('--out_dir', default=None,
						help='location of the output directory in case if outfile is preferred'
							 'to have default name')
	parser.add_argument('--out_name', default="", help='string that will be inserted in the out file names')
	parser.add_argument('--config_file', type=str, default=None, help="Input json file containing configurations parameters")
	args = parser.parse_args()

	# read the arguments from the json config file, is specified
	if args.config_file is not None:
		with open(args.config_file, "r") as f:
			in_params = json.load(f)

		ea_args = in_params["script_args"]

		args.__dict__.update(ea_args)

	G = load_graph(g_type=args.g_type)

	heuristic = getattr(heuristics, args.heuristic)
	heuristic = partial(heuristic, G=G, k=args.k)

	prng = random.Random(args.random_seed)

	if args.heuristic in ["general_greedy", "CELF", "generalized_degree_discount"]:
		heuristic = partial(heuristic, p=args.p)
		if args.heuristic in ["general_greedy", "CELF"]:
			heuristic = partial(heuristic, prng=prng, model=args.model, no_simulations=args.no_simulations)

	start = time.time()
	seed_set = heuristic()
	stop =time.time()
	exec_time = stop - start

	# save the spread value of the result
	spread = monte_carlo(G, seed_set, args.p, args.no_simulations, args.model, prng)

	print("Seed set: {}, spread: {} \nExec time: {}".format(seed_set, spread, exec_time))

	# save result
	if args.out_dir is None:
		out_dir = "."
	else:
		out_dir = args.out_dir
		import os

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

	log_file = out_dir + "/" + "log" + args.out_name

	out_dict = args.__dict__
	out_dict["exec_time"] = exec_time
	out_dict["influence_spread"] = spread
	out_dict["seed_set"] = seed_set

	dict2csv(args=out_dict, csv_name=log_file)

