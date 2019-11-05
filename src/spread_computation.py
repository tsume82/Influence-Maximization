import networkx as nx
import argparse
import pandas as pd
from functools import partial
import random
import numpy as np

from utils import add_weights_WC, add_weights_IC, dict2csv
from load import read_graph
from spread.two_hop import two_hop_spread
from spread.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
from spread.monte_carlo import MonteCarlo_simulation as monte_carlo

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Spread functions computation')
	parser.add_argument('--k', type=int, default=5, help='seed set size')
	parser.add_argument('--p', type=float, default=0.2, help='probability of influence spread in IC model')
	parser.add_argument('--n', type=int, default=100, help='number of seed set extractions')
	parser.add_argument('--max_hop', type=int, default=2, help='number of simulations for spread calculation'
																		' when montecarlo mehtod is used')
	parser.add_argument('--model', default="IC", choices=['IC', 'WC'], help='type of influence propagation model')
	parser.add_argument('--g_nodes', type=int, default=1000, help='number of nodes in the graph')
	parser.add_argument('--g_new_edges', type=int, default=2, help='number of new edges in barabasi-albert graphs')
	parser.add_argument('--g_type', default='barabasi_albert', choices=['barabasi_albert', 'wiki', 'amazon',
																		'twitter', 'facebook', 'CA-GrQc'], help='graph type')
	parser.add_argument('--g_seed', type=int, default=0, help='random seed of the graph')
	parser.add_argument('--g_file', default=None, help='location of graph file')
	parser.add_argument('--random_seed', type=int, default=42, help='seed to initialize the pseudo-random number '
																	'generation')
	parser.add_argument('--no_simulations', type=int, default=100, help='number of simulations for spread calculation'
																		' when montecarlo mehtod is used')
	parser.add_argument('--out_dir', default=".", help='location of the output directory in case if outfile is preferred'
														'to have default name')
	args = parser.parse_args()

	# create / load graph
	if args.g_file is None:
		if args.g_type == "barabasi_albert":
			G = nx.generators.barabasi_albert_graph(args.g_nodes, args.g_new_edges, seed=args.g_seed)
		elif args.g_type == "wiki":
			G = read_graph("../graphs/wiki-Vote.txt", directed=True)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "amazon":
			G = read_graph("../graphs/amazon0302.txt", directed=True)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "twitter":
			G = read_graph("../graphs/twitter_combined.txt", directed=True)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "facebook":
			G = read_graph("../graphs/facebook_combined.txt", directed=False)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "CA-GrQc":
			G = read_graph("../graphs/CA-GrQc.txt", directed=True)
			args.g_nodes = len(G.nodes())
	# extract n seed sets

	seed_sets = []
	prng = random.Random(args.random_seed)
	for _ in range(args.n):
		seed_set = []
		for _ in range(args.k):
			seed_set.append(prng.randint(0, args.g_nodes-1))
		seed_sets.append(seed_set)

	monte_carlo_mh = partial(monte_carlo_max_hop, G=G, p=args.p, max_hop=args.max_hop, random_generator=prng,
							 no_simulations=args.no_simulations, model=args.model)
	monte_carlo_ = partial(monte_carlo, G=G, random_generator=prng, p=args.p, no_simulations=args.no_simulations, model=args.model)
	two_hop = partial(two_hop_spread, G=G, model=args.model, p=args.p)

	# output file
	outfile_name = args.out_dir + "/" + "results.csv"
	f = open(outfile_name, "w")

	out_cols = ["n{}".format(i) for i in range(args.k)]

	for col in ["monte_carlo", "monte_carlo_max_hop", "two_hop"]:
		out_cols.append(col)
	f.write(",".join(out_cols))

	G_nodes = np.array(G.nodes())

	import time

	# logging computation time
	comp_time_mc = np.zeros(len(seed_sets))
	comp_time_mh = np.zeros(len(seed_sets))
	comp_time_th = np.zeros(len(seed_sets))

	for i, S in enumerate(seed_sets):
		f.write("\n")
		A = G_nodes[S]
		now = time.time()
		mc, _ = monte_carlo_(A=A)
		comp_time_mc[i] = time.time() - now
		print("Monte carlo comp.time: {}".format(comp_time_mc[i]))
		now = time.time()
		mc_mh, _ = monte_carlo_mh(A=A)
		comp_time_mh[i] = time.time() - now
		print("Monte carlo max hop comp.time: {}".format(comp_time_mh[i]))
		now = time.time()
		th = two_hop(A=A)
		comp_time_th[i] = time.time() - now
		print("Two hop comp.time: {}".format(comp_time_th[i]))
		f.write(",".join(map(str, S)) + ",")
		f.write(",".join(map(str, [mc, mc_mh, th])))

	f.close()

	# calculate correlation among results
	df = pd.read_csv(outfile_name)
	mc_th_corr = df["monte_carlo"].corr(df["two_hop"], method='pearson')
	mc_mcmh_corr = df["monte_carlo"].corr(df["monte_carlo_max_hop"], method='pearson')
	std = df["monte_carlo"].std()
	print(mc_mcmh_corr)
	print(mc_th_corr)
	print(std)
	mc_min = df["monte_carlo"].min()
	mc_max = df["monte_carlo"].max()

	# write log file
	log_data = args.__dict__
	log_data["mc_th_corr"] = mc_th_corr
	log_data["mc_mcmh_corr"] = mc_mcmh_corr
	log_data["mc_std"] = std
	log_data["mc_min"] = mc_min
	log_data["mc_max"] = mc_max
	log_data["exec_time_mc_mean"] = comp_time_mc.mean()
	log_data["exec_time_mc_std"] = comp_time_mc.std()
	log_data["exec_time_mh_mean"] = comp_time_mh.mean()
	log_data["exec_time_mh_std"] = comp_time_mh.std()
	log_data["exec_time_th_mean"] = comp_time_th.mean()
	log_data["exec_time_th_std"] = comp_time_th.std()

	logfile_name = args.out_dir + "/" + "log.csv"
	dict2csv(log_data, logfile_name)
