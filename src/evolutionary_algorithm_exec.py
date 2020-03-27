"""
evolutionary algorithm execution file: contains methods related to input parameters processing and smart initialization
"""

from functools import partial
import random
import argparse
import time
import networkx as nx
import json

from gensim.models import KeyedVectors

from src.ea.evolutionary_algorithm import ea_influence_maximization
import src.ea.mutators as mutators

from src.spread.monte_carlo import MonteCarlo_simulation as monte_carlo
from src.spread.monte_carlo_mark import MonteCarlo_simulation as monte_carlo_mark
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
from src.spread.monte_carlo_max_hop_mark import MonteCarlo_simulation as monte_carlo_max_hop_mark
from src.spread.two_hop import two_hop_spread as two_hop

# from src.spread_pyx.monte_carlo import MonteCarlo_simulation as monte_carlo
# from src.spread_pyx.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
# from src.spread_pyx.two_hop import two_hop_spread as two_hop

from src.smart_initialization import max_centrality_individual, Community_initialization, degree_random
from src.utils import load_graph, dict2csv, inverse_ncr
from src.nodes_filtering.select_best_spread_nodes import filter_best_nodes as filter_best_spread_nodes
from src.nodes_filtering.select_min_degree_nodes import filter_best_nodes as filter_min_degree_nodes

import src.utils as utils

def create_initial_population(G, args, prng=None, nodes=None):
	"""
	smart initialization techniques
	"""
	# smart initialization
	initial_population = None
	if "community" == args["smart_initialization"]:
		# set for now number of clusters equal to the dimension of seed set
		comm_init = Community_initialization(G, random_seed=args["random_seed"],
											 method=args["community_detection_algorithm"],
											 n_clusters=args["k"] * args["n_clusters"])
		initial_population = \
			comm_init.get_comm_members_random(int(args["population_size"] * args["smart_initialization_percentage"]),
											  k=args["k"], degree=False)
	elif "community_degree" == args["smart_initialization"]:

		# set for now number of clusters equal to the dimension of seed set
		comm_init = Community_initialization(G, random_seed=args["random_seed"],
											 method=args["community_detection_algorithm"],
											 n_clusters=args["k"] * args["n_clusters"])
		initial_population = \
			comm_init.get_comm_members_random(int(args["population_size"] * args["smart_initialization_percentage"]),
											  k=args["k"], degree=True)
	elif "community_degree_spectral" == args["smart_initialization"]:
		comm_init = Community_initialization(G, random_seed=args["random_seed"], method="spectral_clustering",
											 n_clusters=args["k"] * args["n_clusters"])
		initial_population = \
			comm_init.get_comm_members_random(int(args["population_size"] * args["smart_initialization_percentage"]),
											  k=args["k"], degree=True)
	elif "degree_random" == args["smart_initialization"]:
		initial_population = degree_random(args["k"], G,
										   int(args["population_size"] * args["smart_initialization_percentage"]),
										   prng, nodes=nodes)
	elif "degree_random_ranked" == args["smart_initialization"]:
		initial_population = degree_random(args["k"], G,
										   int(args["population_size"] * args["smart_initialization_percentage"]),
										   prng, ranked_probability=True)

	elif args["smart_initialization"] != "none":
		smart_individual = max_centrality_individual(args["k"], G, centrality_metric=args["smart_initialization"])
		initial_population = [smart_individual]

	return initial_population


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def read_arguments():
	"""
	algorithm arguments, it is sufficient to specify all the parameters in the
	.json config file, which should be given as a parameter to the script, it should contain all the other
	script parameters
	"""

	parser = argparse.ArgumentParser(description='Evolutionary algorithm computation')

	parser.add_argument('--k', type=int, default=10, help='seed set size')
	parser.add_argument('--p', type=float, default=0.01, help='probability of influence spread in IC model')
	parser.add_argument('--spread_function', default="monte_carlo_max_hop",
						choices=["monte_carlo", "monte_carlo_max_hop", "two_hop"])
	parser.add_argument('--no_simulations', type=int, default=100, help='number of simulations for spread calculation'
																		' when montecarlo mehtod is used')
	parser.add_argument('--max_hop', type=int, default=3, help='number of max hops for monte carlo max hop function')
	parser.add_argument('--model', default="WC", choices=['IC', 'WC'], help='influence propagation model')
	parser.add_argument('--population_size', type=int, default=4, help='population size of the ea')
	parser.add_argument('--offspring_size', type=int, default=4, help='offspring size of the ea')
	parser.add_argument('--random_seed', type=int, default=0, help='seed to initialize the pseudo-random number '
																	'generation')
	parser.add_argument('--max_generations', type=int, default=50, help='generational budget')

	parser.add_argument('--n_parallel', type=int, default=4,
						help='number of processes to be used for concurrent '
							 'computation')
	parser.add_argument('--g_nodes', type=int, default=100, help='number of nodes in the graph')
	parser.add_argument('--g_new_edges', type=int, default=3, help='number of new edges in barabasi-albert graphs')
	parser.add_argument('--g_type', default='wiki',
						choices=['barabasi_albert', 'gaussian_random_partition',
								'wiki', 'amazon', 'epinions',
								'twitter', 'facebook', 'CA-GrQc', "tiny_wiki",
								"tiny_amazon", "tiny_CA-GrQc", "tiny_wiki_community",
								"tiny_amazon_community", "tiny_CA-GrQc_community"],
						help='graph type')
	parser.add_argument('--g_seed', type=int, default=0, help='random seed of the graph')
	parser.add_argument('--g_file', default=None, help='location of graph file')
	parser.add_argument('--out_dir', default=None,
						help='location of the output directory in case if outfile is preferred'
							 'to have default name')
	parser.add_argument('--smart_initialization', default="degree_random", choices=["none", "degree", "eigenvector", "katz",
																		   "closeness", "betweenness",
																		   "community", "community_degree",
																		   "community_degree_spectral", "degree_random",
																		   "degree_random_ranked"],
						help='if set, an individual containing best nodes according'
							 'to the selected centrality metric will be inesrted'
							 'into the initial population')
	parser.add_argument('--community_detection_algorithm', default="louvain",
						choices=["louvain", "spectral_clustering"],
						help='algorithm to be used for community detection')
	parser.add_argument('--n_clusters', type=int, default=10,
						help="useful only for smart initialization with spectral clustring, "
							 "the scale number of clusters to be used, the actual number of clusters"
							 " will become equal to k*n_clusters")
	parser.add_argument('--smart_initialization_percentage', type=float, default=1,
						help='percentage of "smart" initial population, to be specified when multiple individuals '
							 'technique is used')

	parser.add_argument('--crossover_rate', type=float, default=1.0, help='evolutionary algorithm crossover rate')
	parser.add_argument('--mutation_rate', type=float, default=0.1, help='evolutionary algorithm mutation rate')
	parser.add_argument('--tournament_size', type=int, default=5, help='evolutionary algorithm tournament size')
	parser.add_argument('--num_elites', type=int, default=1, help='evolutionary algorithm num_elites')
	parser.add_argument('--node2vec_file', type=str, default=None, help='evolutionary algorithm node2vec_file')
	parser.add_argument('--min_degree', type=int, default=1,
						help='minimum degree for a node to be inserted into nodes pool in ea')

	parser.add_argument('--mutation_operator', type=str, default="adaptive_mutations", choices=["ea_global_random_mutation",
																					  "ea_local_neighbors_random_mutation",
																					  "ea_local_neighbors_second_degree_mutation",
																					  "ea_global_low_spread",
																					  "ea_global_low_deg_mutation",
																					  "ea_local_approx_spread_mutation",
																					  "ea_local_embeddings_mutation",
																					  "ea_global_subpopulation_mutation",
																					  "ea_adaptive_mutators_alteration",
																					  "ea_local_neighbors_spread_mutation",
																					  "ea_local_additional_spread_mutation",
																					  "ea_local_neighbors_second_degree_mutation_emb",
																					  "ea_global_low_additional_spread",
																					  "ea_differential_evolution_mutation",
																					  "ea_global_activation_mutation",
																					  "ea_local_activation_mutation",
																					  "adaptive_mutations",
																					  ])

	parser.add_argument('--mutators_to_alterate', type=str, nargs='+', default=["ea_local_activation_mutation",
																				#'ea_local_neighbors_second_degree_mutation',
																				#"ea_local_neighbors_second_degree_mutation_emb",
																				#"ea_local_embeddings_mutation",
																				#"ea_local_neighbors_random_mutation",
																				#"ea_local_neighbors_spread_mutation",
																				# "ea_local_additional_spread_mutation",
																				#"ea_local_approx_spread_mutation",
																				# "ea_global_activation_mutation",
																				"ea_global_low_deg_mutation",
																				"ea_global_random_mutation",
																				# "ea_global_low_spread",
																				# "ea_global_low_additional_spread"
																				],
						help='list of mutation methods to alterate')

	parser.add_argument("--moving_avg_len", type=int, default=10,
						help="moving average length for multi-argmed bandit problem")
	parser.add_argument("--filter_best_spread_nodes", type=str2bool, nargs="?", const=True, default=True)
	parser.add_argument("--search_space_size_min", type=int, default=1e9, help="lower bound on the number of combinations")
	parser.add_argument("--search_space_size_max", type=int, default=1e11, help="upper bound on the number of combinations")
	parser.add_argument("--dynamic_population", type=str2bool, nargs="?", const=True, default=True)
	parser.add_argument("--max_generations_percentage_without_improvement", type=float, default=0.5, help="percentage of"
						" the generational budget to use for smart stop condition, ea stops when for this percentage of generations "
						"there is no improvement")
	parser.add_argument('--config_file', type=str, help="Input json file containing configurations parameters")

	args = parser.parse_args()
	args = vars(args)
	if args["config_file"] is not None:
		with open(args["config_file"], "r") as f:
			in_params = json.load(f)

		ea_args = in_params["script_args"]

		ea_args["config_file"] = args["config_file"]
		ea_args["out_dir"] = args["out_dir"]
		# check whether all the parameters are specified in the config file
		if set(args.keys()) != set(ea_args.keys()):
			if len(set(args.keys()).difference(set(ea_args.keys())))>0:
				print("Missing arguments: {}".format(set(args.keys()).difference(set(ea_args.keys()))))
			else:
				print("Unknown arguments: {}".format(set(ea_args.keys()).difference(set(args.keys()))))
			raise KeyError("Arguments error")
		args.update(ea_args)

	# make args read only
	args = utils.make_dict_read_only(args)
	return args


def create_out_dir(args):
	"""
	creation of the out directory and out files names
	"""
	if args["out_dir"] is None:
		out_dir = "."
	else:
		out_dir = args["out_dir"]
		import os

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

	out_name = ".csv"

	population_file = out_dir + "/" + "population" + out_name

	log_file = out_dir + "/" + "log" + out_name

	generations_file = out_dir + "/" + "generations" + out_name

	return population_file, generations_file, log_file


def initialize_fitness_function(G, args, prng):
	"""
	fitness function initialization
	"""
	if args["spread_function"] is None or args["spread_function"] == "monte_carlo":
		spread_function = partial(monte_carlo, no_simulations=args["no_simulations"], p=args["p"], model=args["model"],
								  G=G, random_generator = prng)
	elif args["spread_function"] == "monte_carlo_max_hop":
		spread_function = partial(monte_carlo_max_hop, no_simulations=args["no_simulations"], p=args["p"],
								  model=args["model"], G=G, max_hop=args["max_hop"], random_generator = prng)
	elif args["spread_function"] == "two_hop":
		spread_function = partial(two_hop, G=G, p=args["p"], model=args["model"])

	return spread_function


def initialize_node2vec_model(node2vec_file):
	"""
	initializes node2vec model
	:param node2vec_file:
	:return:
	"""
	if node2vec_file is not None:
		model = KeyedVectors.load_word2vec_format(node2vec_file, binary=False)
	else:
		model = None

	return model


def initialize_stats(generations_file):
	gf = open(generations_file, "w+")
	gf.write(
		"generation number, pop_size, worst, best, median, avg, std, diversity, improvement, number_of_selections\n")
	return gf


def initialize_inidividuls_file(individuals_file):
	ind_f = open(individuals_file, "w")
	ind_f.write("generation number, individual number, fitness, candidate\n")
	return ind_f


def filter_nodes(G, args):
	"""
	selects the most promising nodes from the graph	according to specified input arguments techniques
	:param G:
	:param args:
	:return:
	"""
	# nodes filtering
	nodes = None
	if args["filter_best_spread_nodes"]:

		best_nodes = inverse_ncr(args["search_space_size_min"], args["k"])
		error = (inverse_ncr(args["search_space_size_max"], args["k"]) - best_nodes) / best_nodes
		filter_function = partial(monte_carlo_max_hop, G=G, random_generator=prng, p=args["p"], model=args["model"], max_hop=3, no_simulations=1)
		nodes = filter_best_spread_nodes(G, best_nodes, error, filter_function)

	nodes = filter_min_degree_nodes(G, args["min_degree"], nodes)

	return nodes


if __name__ == "__main__":
	args = read_arguments()

	# load graph
	G = load_graph(args["g_file"], args["g_type"], args["g_nodes"], args["g_new_edges"], args["g_seed"])

	prng = random.Random(args["random_seed"])

	# load mutation function
	mutation_operator = None
	mutators_to_alterate = []
	if args["mutation_operator"] == "adaptive_mutations":
		mutation_operator = mutators.ea_adaptive_mutators_alteration
		for m in args["mutators_to_alterate"]:
			mutators_to_alterate.append(getattr(mutators, m))
	else:
		mutation_operator = getattr(mutators, args["mutation_operator"])

	if mutation_operator == mutators.ea_local_activation_mutation \
			or mutation_operator == mutators.ea_global_activation_mutation \
			or mutators.ea_local_activation_mutation.__name__ in args["mutators_to_alterate"] \
			or mutators.ea_global_activation_mutation.__name__ in args["mutators_to_alterate"]:
		monte_carlo_max_hop = monte_carlo_max_hop_mark
		monte_carlo = monte_carlo_mark

		init_dict = dict()
		for n in G.nodes():
			init_dict[n] = {}
		nx.set_node_attributes(G, init_dict, name="activated_by")

	fitness_function = initialize_fitness_function(G, args, prng)

	population_file, generations_file, log_file = create_out_dir(args)

	nodes = filter_nodes(G, args)
	initial_population = create_initial_population(G, args, prng, nodes)
	node2vec_model = initialize_node2vec_model(args["node2vec_file"])

	generations_file = initialize_stats(generations_file)
	individuals_file = initialize_inidividuls_file(population_file)

	start = time.time()
	best_seed_set, best_spread = ea_influence_maximization(k=args["k"],
														   G=G,
														   pop_size=args["population_size"],
														   offspring_size=args["offspring_size"],
														   max_generations=args["max_generations"],
														   n_processes=args["n_parallel"],
														   prng=prng,
														   initial_population=initial_population,
														   individuals_file=individuals_file,
														   fitness_function=fitness_function,
														   statistics_file=generations_file,
														   crossover_rate=args["crossover_rate"],
														   mutation_rate=args["mutation_rate"],
														   tournament_size=args["tournament_size"],
														   num_elites=args["num_elites"],
														   node2vec_model=node2vec_model,
														   mutators_to_alterate=mutators_to_alterate,
														   mutation_operator=mutation_operator,
														   prop_model=args["model"],
														   p=args["p"],
														   moving_avg_len=args["moving_avg_len"],
														   dynamic_population = args["dynamic_population"],
														   nodes = nodes,
														   max_generations_percentage_without_improvement =
														   args["max_generations_percentage_without_improvement"])

	individuals_file.close()
	generations_file.close()
	exec_time = time.time() - start
	print("Execution time: ", exec_time)

	print("Seed set: ", best_seed_set)
	print("Spread: ", best_spread)

	# best_mc_spread, _ = monte_carlo(G, best_seed_set, args.p, args.no_simulations, args.model, prng)
	best_mc_spread, std1 = monte_carlo(G, best_seed_set, args["p"], 100, args["model"], prng)
	print("Best monte carlo spread: ", best_mc_spread, std1)

	# write experiment log

	out_dict = args.get_copy()
	# if the dataset is one of the tiny datasets, save the ranking archieved
	if "tiny" in out_dict["g_type"]:
		score, total = utils.get_rank_score(best_seed_set, args["g_type"], args["model"], args["k"],
											"monte_carlo",
											args["g_nodes"])
		out_dict["rank_score"] = score
		out_dict["total_combinations"] = total
		out_dict["relative_score"] = score/total
		print("Ranking score: {}/{}".format(score, total))
	out_dict["exec_time"] = exec_time
	out_dict["best_fitness"] = best_spread
	out_dict["best_mc_spread"] = best_mc_spread

	dict2csv(args=out_dict, csv_name=log_file, delimiter=';')
