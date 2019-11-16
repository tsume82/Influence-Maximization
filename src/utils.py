import collections
import networkx as nx
import os


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


# G = nx.generators.random_graphs.barabasi_albert_graph(5, 3, seed=0)
# G_w = add_weights_WC(G)
# print(dict(G_w.adjacency()))
#
# print(G_w[0][4]["weight"])