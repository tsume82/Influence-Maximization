"""
evolutionary algorithm execution file: contains methods related to input parameters processing and smart initialization
"""

from functools import partial
import random
import argparse
import time

from evolutionary_algorithm import ea_influence_maximization
from evolutionary_algorithm import ea_local_embeddings_mutation, ea_local_neighbors_random_mutation, ea_local_neighbors_second_degree_mutation, \
	ea_local_neighbors_second_degree_mutation_emb
from evolutionary_algorithm import ea_gloabal_low_deg_mutation, ea_global_random_mutation

from spread.monte_carlo import MonteCarlo_simulation as monte_carlo
from spread.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
from spread.two_hop import two_hop_spread as two_hop

from smart_initialization import max_centrality_individual, Community_initialization, degree_random

from utils import load_graph, dict2csv


def create_initial_population(G, args, prng=None):
	"""
	smart initialization techniques
	"""
	# smart initialization
	initial_population = None
	if "community" == args.smart_initialization:
		# set for now number of clusters equal to the dimension of seed set
		comm_init = Community_initialization(G, random_seed=args.random_seed, method=args.community_detection_algorithm,
											 n_clusters=args.k * args.n_clusters)
		initial_population = \
			comm_init.get_comm_members_random(int(args.population_size * args.smart_initialization_percentage),
											  k=args.k, degree=False)
	elif "community_degree" == args.smart_initialization:

		# set for now number of clusters equal to the dimension of seed set
		comm_init = Community_initialization(G, random_seed=args.random_seed, method=args.community_detection_algorithm,
											 n_clusters=args.k * args.n_clusters)
		initial_population = \
			comm_init.get_comm_members_random(int(args.population_size * args.smart_initialization_percentage),
											  k=args.k, degree=True)
	elif "community_degree_spectral" == args.smart_initialization:
		comm_init = Community_initialization(G, random_seed=args.random_seed, method="spectral_clustering",
											 n_clusters=args.k * args.n_clusters)
		initial_population = \
			comm_init.get_comm_members_random(int(args.population_size * args.smart_initialization_percentage),
											  k=args.k, degree=True)
	elif "degree_random" == args.smart_initialization:
		initial_population = degree_random(args.k, G, int(args.population_size * args.smart_initialization_percentage),
										   prng)
	elif "degree_random_ranked" == args.smart_initialization:
		initial_population = degree_random(args.k, G, int(args.population_size * args.smart_initialization_percentage),
										   prng, ranked_probability=True)

	elif args.smart_initialization != "none":
		smart_individual = max_centrality_individual(args.k, G, centrality_metric=args.smart_initialization)
		initial_population = [smart_individual]

	return initial_population


def read_arguments():
	"""
	algorithm arguments
	"""
	parser = argparse.ArgumentParser(description='Evolutionary algorithm computation')

	parser.add_argument('--k', type=int, default=10, help='seed set size')
	parser.add_argument('--p', type=float, default=0.01, help='probability of influence spread in IC model')
	parser.add_argument('--spread_function', default="monte_carlo_max_hop",
						choices=["monte_carlo", "monte_carlo_max_hop", "two_hop"])
	parser.add_argument('--no_simulations', type=int, default=100, help='number of simulations for spread calculation'
																		' when montecarlo mehtod is used')
	parser.add_argument('--max_hop', type=int, default=2, help='number of max hops for monte carlo max hop function')
	parser.add_argument('--model', default="IC", choices=['IC', 'WC'], help='type of influence propagation model')
	parser.add_argument('--population_size', type=int, default=16, help='population size of the ea')
	parser.add_argument('--offspring_size', type=int, default=16, help='offspring size of the ea')
	parser.add_argument('--random_seed', type=int, default=43, help='seed to initialize the pseudo-random number '
																	'generation')
	parser.add_argument('--max_generations', type=int, default=30, help='maximum generations')

	parser.add_argument('--n_parallel', type=int, default=1,
						help='number of threads or processes to be used for concurrent '
							 'computation')
	parser.add_argument('--g_nodes', type=int, default=100, help='number of nodes in the graph')
	parser.add_argument('--g_new_edges', type=int, default=3, help='number of new edges in barabasi-albert graphs')
	parser.add_argument('--g_type', default='amazon', choices=['barabasi_albert', 'gaussian_random_partition',
															 'wiki', 'amazon', 'epinions',
															 'twitter', 'facebook', 'CA-GrQc'],
						help='graph type')
	parser.add_argument('--g_seed', type=int, default=0, help='random seed of the graph')
	parser.add_argument('--g_file', default=None, help='location of graph file')
	parser.add_argument('--out_file', default=None, help='location of the output file containing the final population')
	parser.add_argument('--log_file', default=None, help='location of the log file containing info about the run')
	parser.add_argument('--generations_file', default=None, help='location of the log file containing stats from each '
																 'generation population')
	parser.add_argument('--out_name', default=None, help='string that will be inserted in the out file names')
	parser.add_argument('--out_dir', default=None,
						help='location of the output directory in case if outfile is preferred'
							 'to have default name')
	parser.add_argument('--smart_initialization', default="degree", choices=["none", "degree", "eigenvector", "katz",
																					"closeness", "betweenness", "second_order",
																					"community", "community_degree",
																					"community_degree_spectral", "degree_random",
																					"degree_random_ranked"],
						help='if set, an individual containing best nodes according'
							 'to the selected centrality metric will be inesrted'
							 'into the initial population')
	parser.add_argument('--community_detection_algorithm', default="louvain",
						choices=["louvain", "spectral_clustering"],
						help='algorithm to be used for community detection')
	parser.add_argument('--n_clusters', type=int, default=5,
						help="useful only for smart initialization with spectral clustring, "
							 "the scale number of clusters to be used, the actual number of clusters"
							 " will become equal to k*n_clusters")
	parser.add_argument('--smart_initialization_percentage', type=float, default=1,
						help='percentage of "smart" initial population')

	parser.add_argument('--crossover_rate', type=float, default=0.1, help='evolutionary algorithm crossover rate')
	parser.add_argument('--mutation_rate', type=float, default=1.0, help='evolutionary algorithm mutation rate')
	parser.add_argument('--tournament_size', type=int, default=2, help='evolutionary algorithm tournament size')
	parser.add_argument('--num_elites', type=int, default=2, help='evolutionary algorithm num_elites')
	parser.add_argument('--word2vec_file', type=str, default="wiki_embeddings_walk_length_80_.emb", help='evolutionary algorithm word2vec_file')
	parser.add_argument('--max_individual_copies', type=int, default=2, help='max individual duplicates permitted in a population')
	parser.add_argument('--min_degree', type=int, default=0, help='minimum degree for a node to be inserted into nodes pool in ea')
	parser.add_argument('--local_search_rate', type=float, default=0.8, help='evolutionary algorithm local search probability, the global search is set'
																			 'automatically to 1-local_search_rate')

	func_type = type(ea_influence_maximization)
	parser.add_argument('--local_mutation_operator', type=func_type, default=ea_local_neighbors_second_degree_mutation,
						choices=[ea_local_neighbors_second_degree_mutation, ea_local_neighbors_second_degree_mutation_emb, ea_local_embeddings_mutation,
								 ea_local_neighbors_random_mutation], help='local search mutation operator')
	parser.add_argument('--global_mutation_operator', type=func_type, default=ea_gloabal_low_deg_mutation,
						choices=[ea_gloabal_low_deg_mutation, ea_global_random_mutation], help='global search mutation operator')

	args = parser.parse_args()

	return args


def create_out_dir(args):
	"""
	creation of the out directory and out files names
	"""
	if args.out_dir is None:
		out_dir = "."
	else:
		out_dir = args.out_dir
		import os

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

	if args.out_name is None:
		out_name = ".csv"
	else:
		out_name = args.out_name

	if args.out_file is None:
		population_file = out_dir + "/" + "population_" + out_name
	else:
		population_file = args.out_file

	if args.log_file is None:
		log_file = out_dir + "/" + "log_" + out_name
	else:
		log_file = args.log_file

	if args.generations_file is None:
		generations_file = out_dir + "/" + "generations_" + out_name
	else:
		generations_file = args.generations_file

	return population_file, generations_file, log_file


def initialize_fitness_function(G, args):
	"""
	fitness function smart initialization
	"""
	if args.spread_function is None or args.spread_function == "monte_carlo":
		spread_function = partial(monte_carlo, no_simulations=args.no_simulations, p=args.p, model=args.model,
								  G=G)
	elif args.spread_function == "monte_carlo_max_hop":
		spread_function = partial(monte_carlo_max_hop, no_simulations=args.no_simulations, p=args.p,
								  model=args.model, G=G, max_hop=args.max_hop)
	elif args.spread_function == "two_hop":
		spread_function = partial(two_hop, G=G, p=args.p, model=args.model)

	return spread_function


if __name__ == "__main__":
	args = read_arguments()

	G = load_graph(args.g_file, args.g_type, args.g_nodes, args.g_new_edges, args.g_seed)

	prng = random.Random(args.random_seed)

	fitness_function = initialize_fitness_function(G, args)

	population_file, generations_file, log_file = create_out_dir(args)

	initial_population = create_initial_population(G, args, prng)
	start = time.time()

	best_seed_set, best_spread = ea_influence_maximization(k=args.k,
														   G=G,
														   pop_size=args.population_size,
														   offspring_size=args.offspring_size,
														   max_generations=args.max_generations,
														   n_processes=args.n_parallel,
														   prng=prng,
														   initial_population=initial_population,
														   population_file=population_file,
														   fitness_function=fitness_function,
														   generations_file=generations_file,
														   crossover_rate=args.crossover_rate,
														   mutation_rate=args.mutation_rate,
														   tournament_size=args.tournament_size,
														   num_elites=args.num_elites,
														   word2vec_file=args.word2vec_file,
														   min_degree=args.min_degree,
														   max_individual_copies=args.max_individual_copies,
														   local_mutation_rate=args.local_search_rate,
														   local_mutation_operator=args.local_mutation_operator,
														   global_mutation_operator=args.global_mutation_operator)
	exec_time = time.time() - start
	print("Execution time: ", exec_time)

	print("Seed set: ", best_seed_set)
	print("Spread: ", best_spread)

	best_mc_spread, _ = monte_carlo(G, best_seed_set, args.p, args.no_simulations, args.model, prng)
	print("Best monte carlo spread: ", best_mc_spread)

	# write experiment log

	out_dict = args.__dict__
	out_dict["exec_time"] = exec_time
	out_dict["best_fitness"] = best_spread
	out_dict["best_mc_spread"] = best_mc_spread

	dict2csv(args=out_dict, csv_name=log_file)
