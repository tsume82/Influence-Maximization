"""
Computation of node2vec embeddings for a given graph
"""

import time
from node2vec import Node2Vec
import argparse
import random
import numpy as np
import utils
import os

from utils import dict2csv
import torch


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Calculation of node2vec embeddings')

	parser.add_argument('--g_nodes', type=int, default=100, help='number of nodes in the graph')
	parser.add_argument('--g_new_edges', type=int, default=2, help='number of new edges in barabasi-albert graphs')
	parser.add_argument('--g_type', default='wiki', choices=['barabasi_albert', 'gaussian_random_partition',
																		'wiki', 'amazon', 'epinions',
																		'twitter', 'facebook', 'CA-GrQc'],
						help='graph type')
	parser.add_argument('--random_seed', type=int, default=44)

	# ---------------------------------------------------------------------------------------------------------------

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

	parser.add_argument('--out_name', default="", help='string that will be inserted in the out file names')

	parser.add_argument('--config_file', type=str, help="Input json file containing configurations parameters")

	args = parser.parse_args()

	if args.config_file is not None:
		import json
		with open(args.config_file, "r") as f:
			in_params = json.load(f)

		ea_args = in_params["script_args"]

		args.__dict__.update(ea_args)

	G = utils.load_graph(g_type=args.g_type, g_nodes=args.g_nodes)

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


	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)
	# Save embeddings for later use
	model.wv.save_word2vec_format(args.out_dir + "/" + args.out_file + args.out_name + ".emb")

	# Save model for later use
	model.save(args.out_dir + "/" + args.out_file + args.out_name + ".model")

	out_dict = args.__dict__
	out_dict["exec_time"] = exec_time

	dict2csv(args=out_dict, csv_name=args.out_dir + "/" + args.log_file + args.out_name)

