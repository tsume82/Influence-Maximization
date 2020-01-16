import collections
import networkx as nx
import os
import numpy as np

from load import read_graph


def args2cmd(args, exec_name, hpc=False):
	"""
	outputs command string with arguments in args
	:param args: arguments dictionary
	:param exec_name: string with the name of python script
	:return: string with command
	"""
	if hpc:
		out = "python3 " + exec_name
	else:
		out = "python " + exec_name
	for k, v in args.items():
		out += " "
		out += "--{}={}".format(k, v)
	return out


def config_file2cmd(config_file_name, exec_name, hpc=False):
	"""
	outputs command string with arguments in args
	:param args: arguments dictionary
	:param exec_name: string with the name of python script
	:return: string with command
	"""
	if hpc:
		out = "python3 " + exec_name
	else:
		out = "python " + exec_name

	out += "--config_file={}".format(config_file_name)
	return out


def dict2csv(args, csv_name):
	"""
	writes dictionary in csv format to the csv_name file
	:param args: in_dictionary
	:param csv_name: out file name
	:return:
	"""
	with open(csv_name, "w") as f:
		f.write(",".join(args.keys()) + "\n")
		f.write(",".join(str(x) for x in args.values()) + "\n")


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
		if g_type == "barabasi_albert":
			G = nx.generators.barabasi_albert_graph(g_nodes, g_new_edges, seed=g_seed)
		elif g_type == "wiki":
			G = read_graph("../experiments/datasets/wiki-Vote.txt", directed=True)
		elif g_type == "amazon":
			G = read_graph("../experiments/datasets/amazon0302.txt", directed=True)
		elif g_type == "twitter":
			G = read_graph("../experiments/datasets/twitter_combined.txt", directed=True)
		elif g_type == "facebook":
			G = read_graph("../experiments/datasets/facebook_combined.txt", directed=False)
		elif g_type == "CA-GrQc":
			G = read_graph("../experiments/datasets/CA-GrQc.txt", directed=True)
		elif g_type == "epinions":
			G = read_graph("../experiments/datasets/soc-Epinions1.txt", directed=True)

	return G


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
	reutrns the diversity of a given population
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
	percentage of different individuals in population
	"""
	pop = []
	for individual in population:
		pop.append(set(individual.candidate))

	return len(set(tuple(row) for row in pop)) / len(pop)

# G = nx.generators.random_graphs.barabasi_albert_graph(5, 3, seed=0)
# G_w = add_weights_WC(G)
# print(dict(G_w.adjacency()))
#
# print(G_w[0][4]["weight"])