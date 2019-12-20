import networkx as nx
from functools import partial
import random
import time
import argparse

import heuristics
from heuristics import general_greedy, CELF, high_degree_nodes, single_discount_high_degree_nodes, generalized_degree_discount
from evolutionary_algorithm import ea_influence_maximization
from spread.monte_carlo import MonteCarlo_simulation as monte_carlo
from spread.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
from utils import load_graph, dict2csv

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Evolutionary algorithm computation')

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

	args = parser.parse_args()

	G = load_graph(g_type=args.g_type)

	# G = nx.generators.barabasi_albert_graph(10, 2, seed=0)

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

