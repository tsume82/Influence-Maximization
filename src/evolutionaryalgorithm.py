"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithms for influence maximization. Ideally, it will eventually contain both the single-objective (maximize influence with a fixed amount of seed nodes) and multi-objective (maximize influence, minimize number of seed nodes) versions. This relies upon the inspyred Python library for evolutionary algorithms."""

# general libraries
import inspyred
import logging
import random

from time import time, strftime

# local libraries
import spread


def ea_observer(population, num_generations, num_evaluations, args):
	time_previous_generation = args['time_previous_generation']
	currentTime = time()
	timeElapsed = currentTime - time_previous_generation
	args['time_previous_generation'] = currentTime

	best = max(population)
	logging.info('[{0:.2f} s] Generation {1:6} -- {2}'.format(timeElapsed, num_generations, best.fitness))

	# TODO write current state of the ALGORITHM to a file (e.g. random number generator, time elapsed, stuff like that)
	# write current state of the population to a file
	population_file = args["population_file"]

	# find the longest individual
	max_length = len(max(population, key=lambda x: len(x.candidate)).candidate)

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
							  max_generations=100, n_parallel=1, random_seed=None, initial_population=None,
							  population_file=None, multithread=False):
	# initialize a generic evolutionary algorithm
	logging.debug("Initializing Evolutionary Algorithm...")
	prng = random.Random()
	if random_seed == None: random_seed = time()
	logging.debug("Random number generator seeded with %s" % str(random_seed))
	prng.seed(random_seed)

	# check if some of the optional parameters are set; otherwise, use default values
	nodes = list(G.nodes)
	if population_file == None:
		ct = time()
		population_file = strftime("%Y-%m-%d-%H-%M-%S-population.csv")

	# instantiate a basic EvolutionaryComputation object, that is "empty" (no default methods defined for any component)
	# so we will need to define every method
	ea = inspyred.ec.EvolutionaryComputation(prng)
	ea.observer = ea_observer
	ea.variator = [ea_super_operator]
	ea.terminator = inspyred.ec.terminators.generation_termination
	ea.selector = inspyred.ec.selectors.tournament_selection  # default size is 2
	ea.replacer = inspyred.ec.replacers.plus_replacement

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
		time_previous_generation=time(),  # this will be updated in the observer
		random_seed=random_seed,
		multithread=multithread
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
	logging.debug("Creating individual of size %d, with genes ranging from %d to %d" % (k, nodes[0], nodes[-1]))
	for i in range(0, k): individual[i] = nodes[random.randint(0, len(nodes) - 1)]
	logging.debug(individual)

	return individual


@inspyred.ec.variators.crossover  # decorator that defines the operator as a crossover, even if it isn't in this case :-)
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
	for c in children: logging.debug(
		"randomChoice=%d : from parent of size %d, created child of size %d" % (randomChoice, len(candidate1), len(c)))

	# purge the children from "None" and arrays of the wrong size
	children = [c for c in children if c is not None and len(set(c)) == k]

	return children


def ea_evaluator(candidates, args):
	n_parallel = args["n_parallel"]
	G = args["G"]
	p = args["p"]
	model = args["model"]
	no_simulations = args["no_simulations"]
	random_seed = args["random_seed"]

	# we start with a list where every element is None
	fitness = [None] * len(candidates)

	# depending on how many threads we have at our disposal,
	# we use a different methodology
	# if we just have one thread, let's just evaluate individuals old style
	if n_parallel == 1:
		for index, A in enumerate(candidates):
			# TODO sort phenotype, use cache...? or manage sorting directly during individual creation? see lines 108-142 in src_OLD/multiObjective-inspyred/sn-inflmax-inspyred.py
			# TODO maybe if we make sure that candidates are already sets before getting here, we could save some computational time
			A_set = set(A)

			# TODO consider std inside the fitness in some way?
			influence_mean, influence_std = spread.MonteCarlo_simulation(G, A_set, p, no_simulations, model,
																		 random_seed)
			fitness[index] = influence_mean

	else:
		if args["multithread"]:
			# create a threadpool, using the local module
			import threadpool
			thread_pool = threadpool.ThreadPool(n_parallel)

			# create thread lock, to be used for concurrency
			import threading
			thread_lock = threading.Lock()

			# create list of tasks for the thread pool, using the threaded evaluation function
			tasks = [(G, p, A, no_simulations, model, fitness, index, thread_lock, random_seed) for index, A in
					 enumerate(candidates)]

			thread_pool.map(ea_evaluator_threaded, tasks)

			# start thread pool and wait for conclusion
			thread_pool.wait_completion()
		# thread_pool.
		else:
			# -------------- multiprocessing ----------------
			from multiprocessing import Pool
			process_pool = Pool(n_parallel)

			tasks = [A for A in candidates]
			from functools import partial
			# multiprocessing pool imap function accepts only one argument at a time, create partial function with
			# constant parameters
			f = partial(ea_evaluator_processed, G=G, p=p, no_simulations=no_simulations, model=model,
						random_seed=random_seed)
			fitness = list(process_pool.imap(f, tasks))

	return fitness


def ea_evaluator_threaded(G, p, A, no_simulations, model, fitness, index, thread_lock, random_seed, thread_id):
	# TODO not sure that this is needed
	A_set = set(A)

	# run spread simulation
	influence_mean, influence_std = spread.MonteCarlo_simulation(G, A_set, p, no_simulations, model, random_seed)

	# lock shared resource, write in it, release
	thread_lock.acquire()
	fitness[index] = influence_mean
	thread_lock.release()

	return


def ea_evaluator_processed(A, G, p, no_simulations, model, random_seed):
	A = set(A)
	# run spread simulation
	influence_mean, influence_std = spread.MonteCarlo_simulation(G, A, p, no_simulations, model, random_seed)
	return influence_mean


# this main here is just to test the current implementation
if __name__ == "__main__":
	# initialize logging
	import logging

	logger = logging.getLogger('')
	logger.setLevel(logging.DEBUG)  # TODO switch between INFO and DEBUG for less or more in-depth logging
	formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')

	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	# reading arguments

	import argparse

	parser = argparse.ArgumentParser(description='Evolutionary algorithm computation')

	parser.add_argument('--k', type=int, default=5, help='seed set size')
	parser.add_argument('--p', type=float, default=0.1, help='probability of influence spread in IC model')
	parser.add_argument('--no_simulations', type=int, default=100, help='number of simulations for spread calculation'
																		' when montecarlo mehtod is used')
	parser.add_argument('--model', default="IC", choices=['IC', 'WC'], help='type of influence propagation model')
	parser.add_argument('--population_size', type=int, default=16, help='population size of the ea')
	parser.add_argument('--offspring_size', type=int, default=16, help='offspring size of the ea')
	parser.add_argument('--random_seed', type=int, default=44, help='seed to initialize the pseudo-random number '
																	'generation')
	parser.add_argument('--max_generations', type=int, default=10, help='maximum generations')
	parser.add_argument('--n_parallel', type=int, default=1,
						help='number of threads or processes to be used for concurrent '
							 'computation')
	parser.add_argument('--multithread', type=bool, default=False, help='if true multithreading is used to parallelize'
																		' execution, otherwise multiprocessing is used')
	parser.add_argument('--g_nodes', type=int, default=100, help='number of nodes in the graph')
	parser.add_argument('--g_new_edges', type=int, default=3, help='number of new edges in barabasi-albert graphs')
	parser.add_argument('--g_type', default='barabasi_albert', choices=['barabasi_albert'], help='graph type')
	parser.add_argument('--g_seed', type=int, default=0, help='random seed of the graph')
	parser.add_argument('--g_file', default=None, help='location of graph file')
	parser.add_argument('--out_file', default=None, help='location of the output file containing the final population')

	args = parser.parse_args()

	if args.g_file is not None:
		import load

		G = load.read_graph(args.g_file)
	# G = load.read_graph("graphs/facebook_combined.txt")
	else:
		import networkx as nx

		if args.g_type == "barabasi_albert":
			G = nx.generators.barabasi_albert_graph(args.g_nodes, args.g_new_edges, seed=args.g_seed)

	start = time()
	best_seed_set, best_spread = ea_influence_maximization(k=args.k, G=G, p=args.p, no_simulations=args.no_simulations,
														   model=args.model, population_size=args.population_size,
														   offspring_size=args.offspring_size,
														   max_generations=args.max_generations,
														   n_parallel=args.n_parallel, random_seed=args.random_seed,
														   population_file=args.out_file, multithread=args.multithread)
	exec_time = time() - start
	print("Execution time: ", exec_time)

	print("Seed set: ", best_seed_set)
	print("Spread: ", best_spread)
