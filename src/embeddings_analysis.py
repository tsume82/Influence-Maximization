# here we study how the node2vec embeddings are correlated with the features we are interested in:
# out-degree, average path distance, "community-distance"

import networkx as nx
import argparse
import random
import numpy as np
from gensim.models import KeyedVectors
from itertools import combinations
import pandas as pd

from utils import load_graph, random_nodes


# read arguments
def read_arguments():

	parser = argparse.ArgumentParser(description='Calculation of node2vec embeddings')
	# parser.add_argument('--g_nodes', type=int, default=10, help='number of nodes in the graph')
	# parser.add_argument('--g_file', default=None, help='location of graph file')
	# parser.add_argument('--g_new_edges', type=int, default=2, help='number of new edges in barabasi-albert graphs')
	# parser.add_argument('--g_type', default='barabasi_albert', choices=['barabasi_albert', 'gaussian_random_partition',
	# 																	'wiki', 'amazon', 'epinions',
	# 																	'twitter', 'facebook', 'CA-GrQc'],
	# 					help='graph type')
	#
	# parser.add_argument('--g_seed', type=int, default=0, help='random seed of the graph')
	parser.add_argument('--random_seed', type=int, default=44)

	parser.add_argument('--N', type=int, default=100, help='number of random nodes to use for the correlations')
	parser.add_argument('--correlation_with', type=str, default="shortest_path", choices=["shortest_path", "out_degree"])

	parser.add_argument('--out_log_file', type=str, default="", help='correlation results will be appended in the out_log_file')

	parser.add_argument('--node2vec_file', type=str, default="embeddings.emb", help='file containing the node2vec embeddings of the'
																		'input graph')

	args = parser.parse_args()
	return args


def degree_function(G):
	if nx.is_directed(G):
		my_degree_function = G.in_degree
	else:
		my_degree_function = G.degree
	return my_degree_function


if __name__ == "__main__":

	args = read_arguments()

	# open out file
	df = pd.read_csv(args.out_log_file)

	# extract graph info from the log file
	if df["g_file"][0] != "None":
		args.g_file = df["g_file"][0]
	else:
		args.g_file = None
	args.g_nodes = int(df["g_nodes"][0])
	args.g_new_edges = int(df["g_new_edges"][0])
	args.g_type = df["g_type"][0]
	args.g_seed = int(df["g_seed"][0])

	G = load_graph(args.g_file, args.g_type, args.g_nodes, args.g_new_edges, args.g_seed)
	prng = random.Random(args.random_seed)

	# sample N random nodes
	nodes = random_nodes(G, args.N, prng)
	# we don't want repetitions to influence the results
	nodes = np.array(list(set(nodes.tolist())))

	# read node2vec embeddings
	model = KeyedVectors.load_word2vec_format(args.node2vec_file, binary=False)
	# print(model.distances(nodes[0].astype(str), nodes[2].astype(str)))

	# compute the distances between the embeddings
	pairs = combinations(nodes, 2)
	pairs = list(pairs)
	n_pairs = len(pairs)
	node2vec_distances = np.zeros(n_pairs)

	for i, pair in enumerate(pairs):
		node2vec_distances[i] = model.distance(str(pair[0]), str(pair[1]))

	# correlation with
	if args.correlation_with == "out_degree":
		# compute the out-degree of the nodes
		out_degrees = dict()
		degree = degree_function(G)
		for node in nodes:
			out_degrees[node] = degree(node)

		# compute the out-degree-distance among the nodes: difference among their degrees
		degree_distances = np.zeros(n_pairs)

		for i, pair in enumerate(pairs):
			degree_distances[i] = abs(out_degrees[pair[0]]-out_degrees[pair[1]])

		corr = np.corrcoef(degree_distances, node2vec_distances)[1][0]

		df["degree_node2vec_corr"] = corr

	elif args.correlation_with == "shortest_path":
		# compute the shortest path distance between nodes
		shortest_path_distances = np.zeros(n_pairs)

		for i, pair in enumerate(pairs):
			try:
				shortest_path_distances[i] = nx.shortest_path_length(G, source=pair[0], target=pair[1])
			except nx.NetworkXNoPath:
				shortest_path_distances[i] = -1

		# corr = np.corrcoef(degree_distances, node2vec_distances)[1][0]
		corr = np.corrcoef(shortest_path_distances[shortest_path_distances!=-1], node2vec_distances[shortest_path_distances!=-1])[1][0]

		df["shortest_path_node2vec_corr"] = corr

	# write the output
	df.to_csv(args.out_log_file)

	# TODO
	# compute the "community-distance" : if two nodes are in the same community their distance is 0 otherwise their distance
	# is the distance among communities, eheh we need communities distances here

	# compute the correlation between distances
