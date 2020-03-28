import collections
import networkx as nx
import os
import numpy as np
import random
import argparse
import operator as op
from functools import reduce

import src.graph_sampling.SRW_RWF_ISRW as Graph_Sampling
from src.load import read_graph


def args2cmd(args, exec_name, hpc=False):
	"""
	outputs command string with arguments in args
	:param args: arguments dictionary
	:param exec_name: string with the name of python script
	:return: string with command
	"""
	if hpc:
		out = "python3 -m src." + exec_name.replace(".py", "")
	else:
		out = "python -m src." + exec_name.replace(".py", "")
	for k, v in args.items():
		out += " "
		out += "--{}={}".format(k, v)
	return out


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def config_file2cmd(config_file_name, out_dir, exec_name, hpc=False):
	"""
	outputs command string with arguments in args
	:param args: arguments dictionary
	:param exec_name: string with the name of python script
	:return: string with command
	"""
	if hpc:
		out = "python3 -m src." + exec_name.replace(".py", "")
	else:
		out = "python -m src." + exec_name.replace(".py", "")

	out += " --config_file={} --out_dir={}".format(config_file_name, out_dir)
	return out


def dict2csv(args, csv_name, delimiter=','):
	"""
	writes dictionary in csv format to the csv_name file
	:param args: in_dictionary
	:param csv_name: out file name
	:return:
	"""
	with open(csv_name, "w") as f:
		f.write(delimiter.join(args.keys()) + "\n")
		f.write(delimiter.join(str(x) for x in args.values()) + "\n")


class ReadOnlyWrapper(collections.Mapping):
	"""
	To make dicts read only (stackoverflow).

	"""

	def __init__(self, data):
		self._data = data

	def __getitem__(self, key):
		return self._data[key]

	def __len__(self):
		return len(self._data)

	def __iter__(self):
		return iter(self._data)

	def __str__(self):
		return str(self._data)

	def get_copy(self):
		return self._data.copy()


def make_dict_read_only(dict):
	"""
	Make a dictionary into a new read only dictionary.
	:param dict:
	:return:
	"""
	return ReadOnlyWrapper(dict)


def add_weights_IC(G, p):
	"""
	adds spread probabilities as edge weights to networkx graph under Independent Cascade model
	:param G:
	:param p:
	:return:
	"""
	weighted_edges = []
	for a in G.adjacency():
		u, V = a
		for v in V:
			weighted_edges.append((u, v, p))
	G_c = G.copy()
	G_c.add_weighted_edges_from(weighted_edges)
	return G_c


def degree_function(G):
	"""
	degree function for WC model probability calculation
	:param G:
	:return:
	"""
	if nx.is_directed(G):
		function = G.in_degree
	else:
		function = G.degree
	return function


def add_weights_WC(G):
	"""
	adds spread probabilities as edge weights to networkx graph under Weighted Cascade model
	:param G: directed networkx graph
	:return:
	"""
	if not G.is_directed():
		G = G.to_directed()
	my_degree_function = degree_function(G)
	weighted_edges = []
	for a in G.adjacency():
		u, V = a
		for v in V:
			p = 1.0 / my_degree_function(v)
			weighted_edges.append((u, v, p))
	G_c = G.copy()
	G_c.add_weighted_edges_from(weighted_edges)
	return G_c


def get_path_level(dir, subdir):
	level = subdir.replace(dir, "").count("/")
	return level


def traverse_level(dir, level):
	"""
	collects all directories paths and filenames of a certain level with respect to the input dir
	:param dir:
	:param level:
	:return:
	"""
	out = []
	for sub_dir_path, sub_dir_rel_path, files in os.walk(dir):
		lev = get_path_level(dir, sub_dir_path)
		if lev == level:
			out.append([sub_dir_path, sub_dir_rel_path, files])
	return out


def find_files(out_dir, file_contains=".emb"):
	"""
	finds all files (their absolute locations) containing in their names file_contains in the out_dir recursively
	returns list of files' paths
	"""
	all_files = []
	for sub_dir_path, sub_dir_rel_path, files in os.walk(out_dir):
		for f in files:
			if (file_contains in f):
				all_files.append(sub_dir_path + "/" + f)

	return all_files


def load_graph(g_file=None, g_type=None, g_nodes=None, g_new_edges=None, g_seed=None):
	"""
	loads the graph of type g_type, or creates a new one using g_nodes and g_new_edges info if the graph type is the
	barabasi_albert model, if g_file is not none the graph contained in g_file is loaded
	"""
	if g_file is not None:
		G = read_graph(g_file)
	else:
		datasets_dir = "datasets/"
		if g_type == "barabasi_albert":
			G = nx.generators.barabasi_albert_graph(g_nodes, g_new_edges, seed=g_seed)
		elif g_type == "wiki":
			G = read_graph(datasets_dir + "wiki-Vote.txt", directed=True)
		elif g_type == "amazon":
			G = read_graph(datasets_dir + "amazon0302.txt", directed=True)
		elif g_type == "twitter":
			G = read_graph(datasets_dir + "twitter_combined.txt", directed=True)
		elif g_type == "facebook":
			G = read_graph(datasets_dir + "facebook_combined.txt", directed=False)
		elif g_type == "CA-GrQc":
			G = read_graph(datasets_dir + "CA-GrQc.txt", directed=True)
		elif g_type == "epinions":
			G = read_graph(datasets_dir + "soc-Epinions1.txt", directed=True)
		elif g_type == "tiny_wiki":
			G = read_graph(datasets_dir + "Tiny_wiki_{}nodes_seed0.txt".format(g_nodes), directed=True)
		elif g_type == "tiny_amazon":
			G = read_graph(datasets_dir + "Tiny_amazon_{}nodes_seed0.txt".format(g_nodes), directed=True)
		elif g_type == "tiny_CA-GrQc":
			G = read_graph(datasets_dir + "Tiny_CA-GrQc_{}nodes_seed0.txt".format(g_nodes), directed=True)
		elif g_type == "tiny_wiki_community":
			G = read_graph(datasets_dir + "Tiny_wiki_community_{}nodes_seed0.txt".format(g_nodes), directed=True)
		elif g_type == "tiny_amazon_community":
			G = read_graph(datasets_dir + "Tiny_amazon_community_{}nodes_seed0.txt".format(g_nodes), directed=True)
		elif g_type == "tiny_CA-GrQc_community":
			G = read_graph(datasets_dir + "Tiny_CA-GrQc_community_{}nodes_seed0.txt".format(g_nodes), directed=True)
	return G


def get_rank_score(seed_set, dataset_name, model, k, spread_function="monte_carlo", g_nodes=100):
	"""
	returns the ranking of the seed_set among all the seed sets in the dataset, according to its spread function
	:param seed_set:
	:param dataset_name:
	:param model:
	:param k:
	:return: ranking position, number of all the possible sets
	"""
	ground_truth_name = dataset_name.replace("tiny", "Tiny")
	ground_truth_name += "_{}nodes_seed0_{}_k{}_{}.pickle".format(g_nodes, model, k, spread_function)
	ground_truth_name = "../experiments/ground_truth/" + ground_truth_name

	import pickle
	with open(ground_truth_name, 'rb') as handle:
		scores = pickle.load(handle)
	seed_set = tuple(seed_set)
	from itertools import permutations
	i = 0
	seed_set_perms = list(permutations(seed_set))
	while seed_set not in scores.keys():
		seed_set = seed_set_perms[i]
		i += 1

	# uncomment this if you want to display all the smaller scores
	# for k,v in scores.items():
	# 	if k == seed_set:
	# 		break
	# 	print("{}. seed set {} : {}".format(v[2], k, v))
	return scores[seed_set][2], len(scores)


def get_best_fitness(dataset_name, model, k, spread_function="monte_carlo", g_nodes=100):
	"""
	returns fitness of the best individual
	:param dataset_name:
	:param model:
	:param k:
	:param spread_function:
	:param g_nodes:
	:return:
	"""
	ground_truth_name = dataset_name.replace("tiny", "Tiny")
	ground_truth_name += "_{}nodes_seed0_{}_k{}_{}.pickle".format(g_nodes, model, k, spread_function)
	ground_truth_name = "../experiments/ground_truth/" + ground_truth_name

	import pickle
	with open(ground_truth_name, 'rb') as handle:
		scores = pickle.load(handle)
	best_result = scores[list(scores.keys())[0]]
	best_spread = best_result[0]
	return best_spread


def sample_graph(g_type=None, n=100, g_seed=0):
	"""
	samples a subgraph of g_type graph of dimension n
	:param g_type: name of the real world dataset to sample from
	:param n: number of nodes in the sampled graph
	:return: networkx graph
	"""
	G = load_graph(g_type)
	sampler = Graph_Sampling.SRW_RWF_ISRW()
	prng = random.Random(g_seed)
	G_sampled = sampler.random_walk_sampling_with_fly_back(G, n, 0.15, prng)
	return G_sampled


def random_nodes(G, n, prng):
	"""
	samples n random nodes from the graph G using prng as random generator
	"""
	graph_nodes = list(G.nodes)
	if n > len(graph_nodes):
		n = len(graph_nodes)
	nodes = prng.sample(graph_nodes, n)
	nodes = np.array(nodes)

	return nodes


def common_elements(lst1, lst2):
	"""
	returns number of unique elements in common between two lists
	:param lst1:
	:param lst2:
	:return:
	"""
	return len(set(lst1).intersection(lst2))


def diversity(population):
	"""
	returns the diversity of a given population, the diversity is computed as follows:
		1. for each individual: compute the percentage of common nodes with each other individual, calculate the average of
		these values
		2. compute the average similarity by calculating the average overlapping percentage of all the nodes ( calculated in step 1)
		3. compute the diversity as 1 - average similarity
	:param population:
	:return:
	"""

	indiv_mean_similarities = np.zeros(len(population))
	j = 0
	for individual in population:
		ind_similarity = np.zeros(len(population) - 1)
		i = 0
		k = len(individual.candidate)
		pop_copy = population.copy()
		pop_copy.remove(individual)
		for individual2 in pop_copy:
			ind_similarity[i] = common_elements(individual.candidate, individual2.candidate) / k
			i += 1
		indiv_mean_similarities[j] = ind_similarity.mean()
		j += 1

	return 1 - indiv_mean_similarities.mean()


def individuals_diversity(population):
	"""
	percentage of different individuals in a population
	"""
	pop = []
	for individual in population:
		pop.append(set(individual.candidate))

	return len(set(tuple(row) for row in pop)) / len(pop)


def ncr(n, r):
	"""
	number of combinations
	taken from stackoverflow
	:param n: population size
	:param r: sample size
	:return:
	"""
	r = min(r, n-r)
	numer = reduce(op.mul, range(n, n-r, -1), 1)
	denom = reduce(op.mul, range(1, r+1), 1)
	return numer / denom


def inverse_ncr(combinations, r):
	"""
	"inverse" ncr function, given r and ncr, returns n
	:param ncr:
	:param r:
	:return:
	"""
	n = 1
	ncr_n = ncr(n, r)
	while ncr_n < combinations:
		n += 1
		ncr_n = ncr(n, r)
	return n
