"""
evolutionary algorithm execution file: contains methods related to input parameters processing and smart initialization
"""

from functools import partial
import random
import argparse
import time
import networkx as nx

from gensim.models import KeyedVectors

# from evolutionary_algorithm import ea_influence_maximization
from ea.evolutionary_algorithm import ea_influence_maximization
import ea.mutators as mutators

from spread.monte_carlo import MonteCarlo_simulation as monte_carlo
from spread.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
from spread.two_hop import two_hop_spread as two_hop


# from spread_pyx.monte_carlo import MonteCarlo_simulation as monte_carlo
# from spread_pyx.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
# from spread_pyx.two_hop import two_hop_spread as two_hop

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
	algorithm arguments
	"""
	parser = argparse.ArgumentParser(description='Evolutionary algorithm computation')

	parser.add_argument('--k', type=int, default=10, help='seed set size')
	parser.add_argument('--p', type=float, default=0.01, help='probability of influence spread in IC model')
	parser.add_argument('--spread_function', default="monte_carlo_max_hop",
						choices=["monte_carlo", "monte_carlo_max_hop", "two_hop"])
	parser.add_argument('--no_simulations', type=int, default=100, help='number of simulations for spread calculation'
																		' when montecarlo mehtod is used')
	parser.add_argument('--max_hop', type=int, default=3, help='number of max hops for monte carlo max hop function')
	parser.add_argument('--model', default="WC", choices=['IC', 'WC'], help='type of influence propagation model')
	parser.add_argument('--population_size', type=int, default=100, help='population size of the ea')
	parser.add_argument('--offspring_size', type=int, default=100, help='offspring size of the ea')
	parser.add_argument('--random_seed', type=int, default=43, help='seed to initialize the pseudo-random number '
																	'generation')
	parser.add_argument('--max_generations', type=int, default=100, help='maximum generations')

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
	parser.add_argument('--smart_initialization', default="none", choices=["none", "degree", "eigenvector", "katz",
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
	parser.add_argument('--n_clusters', type=int, default=10,
						help="useful only for smart initialization with spectral clustring, "
							 "the scale number of clusters to be used, the actual number of clusters"
							 " will become equal to k*n_clusters")
	parser.add_argument('--smart_initialization_percentage', type=float, default=0.7,
						help='percentage of "smart" initial population')

	parser.add_argument('--crossover_rate', type=float, default=1.0, help='evolutionary algorithm crossover rate')
	parser.add_argument('--mutation_rate', type=float, default=0.1, help='evolutionary algorithm mutation rate')
	parser.add_argument('--tournament_size', type=int, default=5, help='evolutionary algorithm tournament size')
	parser.add_argument('--num_elites', type=int, default=2, help='evolutionary algorithm num_elites')
	parser.add_argument('--node2vec_file', type=str, default=None, help='evolutionary algorithm node2vec_file')
	parser.add_argument('--max_individual_copies', type=int, default=1, help='max individual duplicates permitted in a population')
	parser.add_argument('--min_degree', type=int, default=0, help='minimum degree for a node to be inserted into nodes pool in ea')
	parser.add_argument('--local_search_rate', type=float, default=1, help='evolutionary algorithm local search probability, the global search is set'
																			 'automatically to 1-local_search_rate')

	parser.add_argument('--local_mutation_operator', type=str, default='ea_local_neighbors_random_mutation',
											choices=['ea_local_neighbors_second_degree_mutation', "ea_local_neighbors_second_degree_mutation_emb", "ea_local_embeddings_mutation",
								 "ea_local_neighbors_random_mutation", "ea_local_neighbors_spread_mutation",
								 "ea_local_additional_spread_mutation", "ea_local_approx_spread_mutation"], help='local search mutation operator')
	parser.add_argument('--global_mutation_operator', type=str, default="ea_global_random_mutation",
											choices=["ea_global_low_deg_mutation", "ea_global_random_mutation", "ea_differential_evolution_mutation",
								 "ea_global_low_spread", "ea_global_low_additional_spread", "ea_global_subpopulation_mutation"], help='global search mutation operator')

	parser.add_argument("--adaptive_local_rate", type=str2bool, nargs='?',
						const=True, default=False,
						help="ee.")

	args = parser.parse_args()

	# load mutation functions
	args.local_mutation_operator = getattr(mutators, args.local_mutation_operator)
	args.global_mutation_operator = getattr(mutators, args.global_mutation_operator)

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
	gf.write("generation number, pop_size, worst, best, median, avg, std, diversity, improvement\n")
	return gf


def initialize_inidividuls_file(individuals_file):
	ind_f = open(individuals_file, "w")
	ind_f.write("generation number, individual number, fitness, candidate\n")
	return ind_f


if __name__ == "__main__":
	args = read_arguments()

	G = load_graph(args.g_file, args.g_type, args.g_nodes, args.g_new_edges, args.g_seed)

	prng = random.Random(args.random_seed)

	fitness_function = initialize_fitness_function(G, args)

	population_file, generations_file, log_file = create_out_dir(args)
	initial_population = create_initial_population(G, args, prng)

	node2vec_model = initialize_node2vec_model(args.node2vec_file)

	generations_file = initialize_stats(generations_file)
	individuals_file = initialize_inidividuls_file(population_file)

	start = time.time()
	best_seed_set, best_spread = ea_influence_maximization(k=args.k,
														   G=G,
														   pop_size=args.population_size,
														   offspring_size=args.offspring_size,
														   max_generations=args.max_generations,
														   n_processes=args.n_parallel,
														   prng=prng,
														   initial_population=initial_population,
														   individuals_file=individuals_file,
														   fitness_function=fitness_function,
														   statistics_file=generations_file,
														   crossover_rate=args.crossover_rate,
														   mutation_rate=args.mutation_rate,
														   tournament_size=args.tournament_size,
														   num_elites=args.num_elites,
														   node2vec_model=node2vec_model,
														   min_degree=args.min_degree,
														   max_individual_copies=args.max_individual_copies,
														   local_mutation_rate=args.local_search_rate,
														   local_mutation_operator=args.local_mutation_operator,
														   global_mutation_operator=args.global_mutation_operator,
														   adaptive_local_rate=args.adaptive_local_rate)

	individuals_file.close()
	generations_file.close()
	exec_time = time.time() - start
	print("Execution time: ", exec_time)

	print("Seed set: ", best_seed_set)
	print("Spread: ", best_spread)

	# best_mc_spread, _ = monte_carlo(G, best_seed_set, args.p, args.no_simulations, args.model, prng)
	best_mc_spread, _ = monte_carlo(G, best_seed_set, args.p, args.no_simulations, args.model, prng)
	print("Best monte carlo spread: ", best_mc_spread)

	# write experiment log

	out_dict = args.__dict__
	out_dict["exec_time"] = exec_time
	out_dict["best_fitness"] = best_spread
	out_dict["best_mc_spread"] = best_mc_spread

	dict2csv(args=out_dict, csv_name=log_file)
