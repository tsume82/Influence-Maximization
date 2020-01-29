"""
Creation of tiny datasets that represent the real world graphs via graph sampling
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations
import os
import networkx as nx

from spread.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
from spread.monte_carlo import MonteCarlo_simulation as monte_carlo
import SRW_RWF_ISRW as Graph_Sampling
from utils import load_graph


def plot_degree_distribution(G, title, save=False, out_name=None):
	"""
	plots degree distributions of the graph
	:param G: input graph
	:param title: plot title
	:param save: if true, saves the plot
	:param out_name: name of the out file
	:return:
	"""
	x_label = "degrees"
	if G.is_directed():
		degrees = np.array([G.out_degree(n) for n in G.nodes()])
		x_label = "out_" + x_label
	else:
		degrees = np.array([G.degree(n) for n in G.nodes()])
	h_out = {}
	for deg in np.unique(degrees):
		h_out[deg] = len(degrees[degrees == deg])
	degrees = np.array(list(h_out.keys()))
	counts = np.array(list(h_out.values()))
	plt.bar(degrees, counts)

	plt.xlabel(x_label)
	plt.ylabel("counts")
	plt.title(title)
	if save:
		if out_name is None:
			out_name = title + ".pdf"
		plt.savefig(out_name)
	plt.show()


if __name__=="__main__":
	# random walk graph sampling, see the link to the description in SRW_RWF_ISRW.py
	sampler = Graph_Sampling.SRW_RWF_ISRW()
	sampler.T = 150
	dataset_names = ["wiki", "amazon", "CA-GrQc"]
	models = ["IC", "WC"]
	nodes = 300
	seed = 0
	k = 3
	community = True
	spread_function = monte_carlo_max_hop
	spread_function_name = "monte_carlo_max_hop"
	# spread_function = monte_carlo
	# spread_function_name = "monte_carlo"

	for dataset_name in dataset_names:

		max_trials = 100
		G = load_graph(g_type=dataset_name)
		prng = random.Random(seed)
		if community:
			G_sampled1 = sampler.random_walk_sampling_with_fly_back(G, nodes/2, 0.15, prng)
			G_sampled2 = sampler.random_walk_sampling_with_fly_back(G, nodes/2, 0.15, prng)
			# compose two graphs together
			G_sampled = nx.compose(G_sampled1, G_sampled2)
			# while nodes in common keep sampling
			while len(G_sampled) < nodes and max_trials > 0:
				G_sampled1 = sampler.random_walk_sampling_with_fly_back(G, nodes / 2, 0.15, prng)
				G_sampled2 = sampler.random_walk_sampling_with_fly_back(G, nodes / 2, 0.15, prng)
				# compose two graphs together
				G_sampled = nx.compose(G_sampled1, G_sampled2)
				max_trials-=1
			# if no nodes overlap, link two subgraphs artificially
			if len(G_sampled) == nodes:
				G_sampled.add_edge(prng.choice(list(G_sampled1.nodes())), prng.choice(list(G_sampled2.nodes())))
			dataset_name += "_community"
		else:
			G_sampled = sampler.random_walk_sampling_with_fly_back(G, nodes, 0.15, prng)
		degree_dist_dir = "../experiments/datasets/degree_distributions/"
		# create dir if not exists
		if not os.path.exists(degree_dist_dir):
			os.makedirs(degree_dist_dir)
		plot_degree_distribution(G, title=dataset_name, save=True, out_name=degree_dist_dir + dataset_name + ".pdf")
		plot_degree_distribution(G_sampled, title="Tiny " + dataset_name, save=True, out_name=degree_dist_dir + "Tiny "+ \
								 dataset_name + ".pdf")
		# save the datasets
		datasets_dir = "../experiments/datasets/"
		nx.write_edgelist(G_sampled, datasets_dir+"Tiny_{}_{}nodes_seed{}.txt".format(dataset_name, nodes, seed), data=False)
		seed_sets = list(combinations(G_sampled.nodes, k))
		for model in models:
			spreads = {}
			for i, s in enumerate(seed_sets):
				print("{}/{}".format(i, len(seed_sets)))
				if spread_function_name == "monte_carlo":
					spreads[s] = spread_function(G_sampled, s, 0.01, 100, model, prng)
				else:
					spreads[s] = spread_function(G_sampled, s, 0.01, 100, model, 3, prng)
			ordered_dict = {k: v for k, v in sorted(spreads.items(), key=lambda item: item[1], reverse=True)}
			for i, key in enumerate(ordered_dict):
				ordered_dict[key] += (i,)
			# save the ground truth ranking
			out_dir = "../experiments/ground_truth/"
			with open(out_dir + "Tiny_{}_{}nodes_seed{}_{}_k{}_{}".format(dataset_name, nodes, seed, model, k, \
																		 spread_function_name ) + \
					  '.pickle', 'wb') as handle:
				pickle.dump(ordered_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# to read the pickle:
		# with open('Tiny_wiki_100nodes_seed0.pickle', 'rb') as handle:
		#     b = pickle.load(handle)
