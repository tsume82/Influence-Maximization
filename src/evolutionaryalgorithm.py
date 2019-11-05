"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithms for influence maximization. 
Ideally, it will eventually contain both the single-objective (maximize influence with a fixed 
amount of seed nodes) and multi-objective (maximize influence, minimize number of seed nodes) versions. 
This relies upon the inspyred Python library for evolutionary algorithms."""

# general libraries
import inspyred
import random
from functools import partial
import copy
import numpy as np

from time import time, strftime

from utils import add_weights_WC, add_weights_IC, dict2csv

# local libraries

# spread function, seed set argument of all function should be named "A"
from spread.monte_carlo import MonteCarlo_simulation as monte_carlo
from spread.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
from spread.two_hop import two_hop_spread as two_hop

from smart_initialization import max_centrality_individual

from load import read_graph


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
	reutrns diversity of a given population
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


def ea_observer(population, num_generations, num_evaluations, args):
	time_previous_generation = args['time_previous_generation']
	currentTime = time()
	timeElapsed = currentTime - time_previous_generation
	args['time_previous_generation'] = currentTime

	best = max(population)
	# logging.info('[{0:.2f} s] Generation {1:6} -- {2}'.format(timeElapsed, num_generations, best.fitness))

	# TODO write current state of the ALGORITHM to a file (e.g. random number generator, time elapsed, stuff like that)
	# write current state of the population to a file
	population_file = args["population_file"]
	generations_file = args["generations_file"]

	# find the longest individual
	max_length = len(max(population, key=lambda x: len(x.candidate)).candidate)

	# compute some generation statistics
	generation_stats = {}
	prev_best = args["prev_population_best"]
	current_best = max(population).fitness
	if prev_best > 0:
		generation_stats["improvement"] = (current_best - prev_best)/prev_best
	else:
		generation_stats["improvement"] = 0
	args["prev_population_best"] = current_best

	with open(generations_file, "a") as fg:
		fg.write("{},".format(num_generations))
		fg.write("{},".format(diversity(population)))
		fg.write("{},".format(generation_stats["improvement"]))
		fg.write("{}\n".format(current_best))

	with open(population_file, "w") as fp:
		# header, of length equal to the maximum individual length in the population
		fp.write("n_nodes,influence")
		for i in range(0, max_length): fp.write(",n%d" % i)
		fp.write("\n")

		# and now, we write stuff, individual by individual
		for individual in population:

			# check if fitness is an iterable collection (e.g. a list) or just a single value
			if hasattr(individual.fitness, "__iter__"):
				fp.write("%d,%.4f" % (1.0 / individual.fitness[1], individual.fitness[0]))
			else:
				fp.write("%d,%.4f" % (len(set(individual.candidate)), individual.fitness))

			for node in individual.candidate:
				fp.write(",%d" % node)

			for i in range(len(individual.candidate), max_length - len(individual.candidate)):
				fp.write(",")

			fp.write("\n")

	# diversity and % of improvement

	return


# @inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def ea_alteration_mutation(random, candidate, args):
	# print("nsga2alterationMutation received this candidate:", candidate)
	nodes = args["nodes"]

	mutatedIndividual = list(set(candidate))

	# choose random place
	gene = random.randint(0, len(mutatedIndividual) - 1)
	mutatedIndividual[gene] = nodes[random.randint(0, len(nodes) - 1)]

	return mutatedIndividual


"""Single-objective evolutionary influence maximization. Parameters:
    k: seed set size
    G: networkx graph
    p: probability of influence spread 
    no_simulations: number of simulations
    model: type of influence propagation model
    population_size: population of the EA (default: value)
    offspring_size: offspring of the EA (default: value)
    max_generations: maximum generations (default: value)
    n_parallel: number of threads or processes to be used for concurrent evaluations (default: 1)
    random_seed: seed to initialize the pseudo-random number generation (default: time)
    initial_population: individuals (seed sets) to be added to the initial population (the rest will be randomly generated)
    
    multithread: if true multithreading is used to parallelize execution, otherwise multiprocessing is used
    """


def ea_influence_maximization(k, G, p, no_simulations, model, population_size=100, offspring_size=100,
							  max_generations=100, n_parallel=1, random_generator=None, initial_population=None,
							  population_file=None, spread_function=None, max_hop=None,
							  generations_file=None):
	# initialize a generic evolutionary algorithm
	# logging.debug("Initializing Evolutionary Algorithm...")
	if random_generator is None:
		random_generator = random.Random()

	# check if some of the optional parameters are set; otherwise, use default values
	nodes = list(G.nodes)

	if spread_function is None or spread_function == "monte_carlo":
		spread_function = partial(monte_carlo, no_simulations=no_simulations, p=p, model=model,
								  G=G)
	elif spread_function == "monte_carlo_max_hop":
		spread_function = partial(monte_carlo_max_hop, no_simulations=no_simulations, p=p,
								  model=model, G=G, max_hop=max_hop)
	elif spread_function == "two_hop":
		spread_function = partial(two_hop, G=G, p=p, model=model)

	# initialize generations file
	with open(generations_file, "w") as gf:
		gf.write("num_genrations,diversity,improvement,best_fitness\n")

	# instantiate a basic EvolutionaryComputation object, that is "empty" (no default methods defined for any component)
	# so we will need to define every method
	ea = inspyred.ec.EvolutionaryComputation(prng)
	ea.observer = ea_observer
	ea.variator = [ea_super_operator]
	ea.terminator = inspyred.ec.terminators.generation_termination
	ea.selector = inspyred.ec.selectors.tournament_selection  # default size is 2
	ea.replacer = inspyred.ec.replacers.plus_replacement

	generations_stats = []
	# start the evolutionary process
	final_population = ea.evolve(
		generator=ea_generator,
		evaluator=ea_evaluator,
		maximize=True,
		seeds=initial_population,
		pop_size=population_size,
		num_selected=offspring_size,
		max_generations=max_generations,

		# all arguments below will go inside the dictionary 'args'
		k=k,
		G=G,
		p=p,
		model=model,
		no_simulations=no_simulations,
		nodes=nodes,
		n_parallel=n_parallel,
		population_file=population_file,
		generations_file = generations_file,
		time_previous_generation=time(),  # this will be updated in the observer
		spread_function=spread_function,
		random_generator=random_generator,
		prev_population_best = -1,
	)

	best_individual = max(final_population)
	best_seed_set = best_individual.candidate
	best_spread = best_individual.fitness

	return best_seed_set, best_spread


@inspyred.ec.generators.diversify  # decorator that makes it impossible to generate copies
def ea_generator(random, args):
	# k is the size of the seed sets
	k = args["k"]
	nodes = args["nodes"]

	# extract random number in 1,max_seed_nodes
	individual = [0] * k
	# logging.debug("Creating individual of size %d, with genes ranging from %d to %d" % (k, nodes[0], nodes[-1]))
	for i in range(0, k): individual[i] = nodes[random.randint(0, len(nodes) - 1)]
	# logging.debug(individual)

	return individual


@inspyred.ec.variators.crossover
# decorator that defines the operator as a crossover, even if it isn't in this case :-)
def ea_super_operator(random, candidate1, candidate2, args):
	k = args["k"]
	children = []

	# uniform choice of operator
	randomChoice = random.randint(0, 1)

	# one-point crossover or mutation that swaps exactly one node with another
	if randomChoice == 0:
		children = inspyred.ec.variators.n_point_crossover(random, [list(candidate1), list(candidate2)], args)
	elif randomChoice == 1:
		children.append(ea_alteration_mutation(random, list(candidate1), args))

	# this should probably be commented or sent to logging
	# for c in children: logging.debug(
	# 	"randomChoice=%d : from parent of size %d, created child of size %d" % (randomChoice, len(candidate1), len(c)))

	# purge the children from "None" and arrays of the wrong size
	children = [c for c in children if c is not None and len(set(c)) == k]

	return children


def ea_evaluator(candidates, args):
	n_parallel = args["n_parallel"]
	spread_function = args["spread_function"]
	random_generator = args["random_generator"]

	# -------------- multiprocessing ----------------
	from multiprocessing import Pool
	process_pool = Pool(n_parallel)

	tasks = []
	for A in candidates:
		tasks.append([A, random_generator.random()])
	from functools import partial
	# multiprocessing pool imap function accepts only one argument at a time, create partial function with
	# constant parameters
	f = partial(ea_evaluator_processed, spread_function=spread_function)
	fitness = list(process_pool.imap(f, tasks))

	return fitness


def ea_evaluator_processed(args, spread_function):
	A, random_seed = args
	A = set(A)
	# run spread simulation
	if spread_function.func != two_hop:
		influence_mean, influence_std = spread_function(A=A, random_generator=random.Random(random_seed))
	else:
		influence_mean = spread_function(A=A)
	return influence_mean


# this main here is just to test the current implementation
if __name__ == "__main__":
	# initialize logging
	# import logging
	#
	# logger = logging.getLogger('')
	# logger.setLevel(logging.DEBUG)  # TODO switch between INFO and DEBUG for less or more in-depth logging
	# formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
	#
	# ch = logging.StreamHandler()
	# ch.setLevel(logging.DEBUG)
	# ch.setFormatter(formatter)
	# logger.addHandler(ch)

	# reading arguments

	import argparse

	parser = argparse.ArgumentParser(description='Evolutionary algorithm computation')

	parser.add_argument('--k', type=int, default=10, help='seed set size')
	parser.add_argument('--p', type=float, default=0.1, help='probability of influence spread in IC model')
	parser.add_argument('--spread_function', default="monte_carlo",
						choices=["monte_carlo", "monte_carlo_max_hop", "two_hop"])
	parser.add_argument('--no_simulations', type=int, default=100, help='number of simulations for spread calculation'
																		' when montecarlo mehtod is used')
	parser.add_argument('--max_hop', type=int, default=2, help='number of max hops for monte carlo max hop function')
	parser.add_argument('--model', default="IC", choices=['IC', 'WC'], help='type of influence propagation model')
	parser.add_argument('--population_size', type=int, default=16, help='population size of the ea')
	parser.add_argument('--offspring_size', type=int, default=16, help='offspring size of the ea')
	parser.add_argument('--random_seed', type=int, default=44, help='seed to initialize the pseudo-random number '
																	'generation')
	parser.add_argument('--max_generations', type=int, default=10, help='maximum generations')

	parser.add_argument('--n_parallel', type=int, default=4,
						help='number of threads or processes to be used for concurrent '
							 'computation')
	parser.add_argument('--g_nodes', type=int, default=10000, help='number of nodes in the graph')
	parser.add_argument('--g_new_edges', type=int, default=3, help='number of new edges in barabasi-albert graphs')
	parser.add_argument('--g_type', default='barabasi_albert', choices=['barabasi_albert', 'gaussian_random_partition',
																		'wiki', 'amazon',
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
	parser.add_argument('--smart_initialization', default="", choices=["", "degree", "eigenvector", "katz", "closeness",
																	   "betweenness", "second_order"],
						help='if set, an individual containing best nodes according'
							 'to the selected centrality metric will be inesrted'
							 'into the initial population')

	args = parser.parse_args()

	if args.g_file is not None:
		import load

		G = load.read_graph(args.g_file)
	else:
		import networkx as nx

		if args.g_type == "barabasi_albert":
			G = nx.generators.barabasi_albert_graph(args.g_nodes, args.g_new_edges, seed=args.g_seed)
		elif args.g_type == "gaussian_random_partition":
			G = nx.gaussian_random_partition_graph(n=args.g_nodes, s=10, v=10, p_in=0.25, p_out=0.1, seed=args.g_seed)
		elif args.g_type == "wiki":
			G = read_graph("../graphs/wiki-Vote.txt", directed=True)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "amazon":
			G = read_graph("../graphs/amazon0302.txt", directed=True)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "twitter":
			G = read_graph("../graphs/twitter_combined.txt", directed=True)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "facebook":
			G = read_graph("../graphs/facebook_combined.txt", directed=False)
			args.g_nodes = len(G.nodes())
		elif args.g_type == "CA-GrQc":
			G = read_graph("../graphs/CA-GrQc.txt", directed=True)
			args.g_nodes = len(G.nodes())
	# random generator
	prng = random.Random()
	if args.random_seed is None:
		random_seed = time()
	else:
		random_seed = args.random_seed
	# logging.debug("Random number generator seeded with %s" % str(args.random_seed))
	prng.seed(random_seed)

	# out file names / directory creation

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

	# smart initialization
	initial_population = None
	if args.smart_initialization != "":
		smart_individual = max_centrality_individual(args.k, G, centrality_metric=args.smart_initialization)
		initial_population = [smart_individual]

	print("Graph info")
	print(nx.classes.function.info(G))

	start = time()
	best_seed_set, best_spread = ea_influence_maximization(k=args.k, G=G, p=args.p, no_simulations=args.no_simulations,
														   model=args.model, population_size=args.population_size,
														   offspring_size=args.offspring_size,
														   max_generations=args.max_generations,
														   n_parallel=args.n_parallel, random_generator=prng,
														   initial_population=initial_population,
														   population_file=population_file,
														   spread_function=args.spread_function, max_hop=args.max_hop,
														   generations_file=generations_file)
	exec_time = time() - start

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
