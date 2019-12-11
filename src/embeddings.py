import time
import networkx as nx
from node2vec import Node2Vec
import argparse
import random
import numpy as np

from load import read_graph
from utils import dict2csv
import torch


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Calculation of node2vec embeddings')

	parser.add_argument('--g_nodes', type=int, default=10, help='number of nodes in the graph')
	parser.add_argument('--g_new_edges', type=int, default=2, help='number of new edges in barabasi-albert graphs')
	parser.add_argument('--g_type', default='barabasi_albert', choices=['barabasi_albert', 'gaussian_random_partition',
																		'wiki', 'amazon', 'epinions',
																		'twitter', 'facebook', 'CA-GrQc'],
						help='graph type')
	parser.add_argument('--random_seed', type=int, default=44)

	# ---------------------------------------------------------------------------------------------------------------
	# taken from
	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk_length', type=int, default=80,
						help='Length of walk per source. Default is 80.')

	parser.add_argument('--num_walks', type=int, default=10,
						help='Number of walks per source. Default is 10.')

	parser.add_argument('--window_size', type=int, default=10,
						help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
						help='Number of epochs in SGD')

	# default changed to 4 because of the memory fitting problems
	parser.add_argument('--workers', type=int, default=4,
						help='Number of parallel workers. Default is 4.')

	parser.add_argument('--p', type=float, default=1,
						help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
						help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)
	# ---------------------------------------------------------------------------------------------------------------

	parser.add_argument('--min_count', type=int, default=1, help='?')
	parser.add_argument('--batch_words', type=int, default=4, help='?')

	parser.add_argument('--g_seed', type=int, default=0, help='random seed of the graph')
	parser.add_argument('--g_file', default=None, help='location of graph file')
	parser.add_argument('--out_file', default="embeddings", help='location of the output file containing the embeddings')
	parser.add_argument('--log_file', default="log", help='location of the log file containing info about the run')
	parser.add_argument('--out_dir', default=".",
						help='location of the output directory in case if outfile is preferred'
							 'to have default name')

	parser.add_argument('--out_name', default=None, help='string that will be inserted in the out file names')

	args = parser.parse_args()
	if args.g_file is not None:
		import load
		G = load.read_graph(args.g_file)
	else:
		if args.g_type == "barabasi_albert":
			G = nx.generators.barabasi_albert_graph(args.g_nodes, args.g_new_edges, seed=args.g_seed)
		elif args.g_type == "gaussian_random_partition":
			G = nx.gaussian_random_partition_graph(n=args.g_nodes, s=10, v=10, p_in=0.25, p_out=0.1, seed=args.g_seed)
		elif args.g_type == "wiki":
			G = read_graph("../experiments/datasets/wiki-Vote.txt", directed=True)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "amazon":
			G = read_graph("../experiments/datasets/amazon0302.txt", directed=True)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "twitter":
			G = read_graph("../experiments/datasets/twitter_combined.txt", directed=True)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "facebook":
			G = read_graph("../experiments/datasets/facebook_combined.txt", directed=False)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "CA-GrQc":
			G = read_graph("../experiments/datasets/CA-GrQc.txt", directed=True)
		elif args.g_type == "epinions":
			G = read_graph("../experiments/datasets/soc-Epinions1.txt", directed=True)
			args.g_nodes = len(G.nodes())

	args = parser.parse_args()

	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.random.manual_seed(args.random_seed)

	start = time.time()

	# Precompute probabilities and generate walks
	node2vec = Node2Vec(G, dimensions=args.dimensions, walk_length=args.walk_length, num_walks=args.num_walks,
						workers=args.workers, p=args.p, q=args.q)

	# Embed
	model = node2vec.fit(window=args.window_size, min_count=args.min_count, batch_words=args.batch_words, seed=args.random_seed,
						 iter=args.iter)
	# Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers`
	# are automatically passed (from the Node2Vec constructor)

	exec_time = time.time() - start
	print("exec time {}".format(exec_time))

	import os

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	# Save embeddings for later use
	model.wv.save_word2vec_format(args.out_dir + "/" + args.out_file + args.out_name + ".emb")

	# Save model for later use
	model.save(args.out_dir + "/" + args.out_file + args.out_name + ".model")

	out_dict = args.__dict__
	out_dict["exec_time"] = exec_time

	dict2csv(args=out_dict, csv_name=args.out_dir + "/" + args.log_file + args.out_name)

#################################################################
# FILES
# EMBEDDING_FILENAME = './embeddings.emb'
# EMBEDDING_MODEL_FILENAME = './embeddings.model'
#
# # Create a graph
# # graph = nx.fast_gnp_random_graph(n=100, p=0.5)
# # G = read_graph("../graphs/amazon0302.txt", directed=True)
# # graph = read_graph("../graphs/wiki-Vote.txt", directed=True)
# graph = read_graph("../graphs/soc-Epinions1.txt", directed=True)
# start = time.time()
# # Precompute probabilities and generate walks
# node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
#
# ## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# # Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
# #node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")
#
# # Embed
# model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
#
# exec_time = time.time() - start
# print("exec time {}".format(exec_time))
# # Look for most similar nodes
# model.wv.most_similar('2')  # Output node names are always strings
#
# # Save embeddings for later use
# model.wv.save_word2vec_format(EMBEDDING_FILENAME)
#
# # Save model for later use
# model.save(EMBEDDING_MODEL_FILENAME)
#
# with open("./log", "w") as f:
# 	f.write("Running time: {}".format(exec_time))

# open the graph
# graph = read_graph("../experiments/datasets/amazon0302.txt", directed=True)
# graph = read_graph("../experiments/datasets/soc-Epinions1.txt", directed=True)
#
# # open the file containing embeddings
# from gensim.models import KeyedVectors
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
#
# filename = 'embeddings_epinions.emb'
# model = KeyedVectors.load_word2vec_format(filename, binary=False)
# nodes = np.array(list(graph.nodes))
# # pick 100 randomly chosen nodes and compute their distances among each other, compute the correlation
# ref_node = np.random.choice(nodes)
# compare_with = np.random.choice(nodes, 100)
#
# # print(type(ref_node))
# # print(compare_with.astype(str))
#
# c2 = [str(i) for i in compare_with]
#
# # distances = model.distances('1234', ['1509'])
# distances = model.distances(str(ref_node), c2)
# print(distances)
#
# real_dist = []
# for c in compare_with:
# 	try:
# 		real_dist.append(nx.shortest_path_length(graph, source=ref_node, target=c))
# 	except nx.NetworkXNoPath:
# 		real_dist.append(100)
# print(real_dist)
# real_dist = np.array(real_dist)
#
# distances = distances[real_dist != 100]
# real_dist = real_dist[real_dist != 100]
#
# print(np.corrcoef(distances, real_dist))
# print(nx.shortest_path_length(graph, source=1234, target=1509))



# print(model.most_similar('1234'))
# v1 = model.get_vector('1234')
# v2 = model.get_vector('1509')
# distance = model.cosine_similarities(v1, [v2])
# distance2 = model.similarity('1234', '1509')
# print(distance)
# print(distance2)
# print(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1)))
#




# print("[('1509', 0.9002323150634766), ('878', 0.8163006901741028), ('3675', 0.7981219291687012), ('16480', 0.7655884623527527), ('21654', 0.7590134143829346), ('1811', 0.7559418678283691), ('2860', 0.7410027980804443), "
# 	  "('1975', 0.7081610560417175), ('434', 0.7053421139717102), ('14301', 0.7049664258956909)]")

# nodes = list(graph.nodes)
# nodes = graph
# # print(nodes[1234])
# # neighs = [n for n in graph.neighbors(nodes[1234])]
#
#
# neighs = list(nodes[1234])
# print(neighs)
#
# for n in list(neighs):
# # 	print(n)
# 	print([m for m in graph.neighbors(n)])

