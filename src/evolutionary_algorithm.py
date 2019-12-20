"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithm for influence maximization, a single-objective version. 
This relies upon the inspyred Python library for evolutionary algorithms."""

import inspyred
import random
import numpy as np
from gensim.models import KeyedVectors

from spread.two_hop import two_hop_spread as two_hop


from utils import common_elements, diversity, individuals_diversity


# population random initialization
@inspyred.ec.generators.diversify
def ea_generator(prng, args):
	"""
	randomly generates an individual without node repetitions
	"""
	k = args["k"]
	nodes = args["nodes"].copy()

	individual = [0] * k
	for i in range(0, k):
		idx = prng.randint(0, len(nodes) - 1)
		individual[i] = nodes.pop(idx)

	return individual


# -------------------------------------- mutation operators ---------------------------------
# @inspyred.ec.variators.mutator
def ea_global_random_mutation(prng, candidate, args):
	"""
	randomly mutates one gene of the individual
	"""
	nodes = args["nodes"].copy()
	# avoid nodes repetitions
	for c in candidate:
		nodes.remove(c)
	mutatedIndividual = list(set(candidate))

	# choose random place
	gene = prng.randint(0, len(mutatedIndividual) - 1)
	mutatedIndividual[gene] = nodes[prng.randint(0, len(nodes) - 1)]

	return mutatedIndividual


def ea_local_neighbors_random_mutation(prng, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors
	"""
	mutatedIndividual = list(set(candidate))

	# choose random place
	gene = prng.randint(0, len(mutatedIndividual) - 1)
	# choose among neighbours of the selected node
	nodes = list(args["G"].neighbors(mutatedIndividual[gene]))
	# avoid nodes repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)
	if len(nodes) > 0:
		mutatedIndividual[gene] = nodes[prng.randint(0, len(nodes) - 1)]

	return mutatedIndividual


def ea_local_embeddings_mutation(prng, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's nearest nodes according to their embeddings
	"""
	mutatedIndividual = list(set(candidate))

	# choose random place
	gene = prng.randint(0, len(mutatedIndividual) - 1)

	# choose among the most similar nodes according to the embedding
	nodes = args["model"].wv.most_similar(str(mutatedIndividual[gene]))
	nodes = [int(n[0]) for n in nodes]
	# avoid nodes repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)
	if len(nodes) > 0:
		mutatedIndividual[gene] = nodes[prng.randint(0, len(nodes) - 1)]

	return mutatedIndividual


def ea_local_neighbors_second_degree_mutation(prng, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, but according to second degree probability
	"""
	#TODO: approximate better second degree? use monte carlo second degree max inf? use probabilities weights?
	mutatedIndividual = list(set(candidate))

	# choose random place
	gene = prng.randint(0, len(mutatedIndividual) - 1)
	# choose among neighbours of the selected node
	nodes = list(args["G"].neighbors(mutatedIndividual[gene]))

	# avoid nodes repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)

	if len(nodes) > 0:
		# calculate second degree of each of the neighbors
		second_degrees = []
		for node in nodes:
			sec_degree = 0
			sec_degree += len(nodes)
			node_neighs = list(args["G"].neighbors(node))
			for node_neigh in node_neighs:
				# !very roughly approximated, may include repetitions
				sec_degree += len(list(args["G"].neighbors(node_neigh)))
			second_degrees.append(sec_degree)
		probs = np.array(second_degrees) / max(second_degrees)
		idx = prng.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]

	return mutatedIndividual


def ea_local_neighbors_second_degree_mutation_emb(prng, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, but according to second degree probability
	"""
	# TODO calcolare le probabilità corrispondenti alla two hop spread del nodo, oppure alla somma di probabilità
	# di propagazione al posto delle degrees
	mutatedIndividual = list(set(candidate))

	# choose random place
	gene = prng.randint(0, len(mutatedIndividual) - 1)
	# choose among neighbours of the selected node
	# nodes = list(args["G"].neighbors(mutatedIndividual[gene]))

	# choose among the most similar nodes according to the embedding
	nodes = args["model"].wv.most_similar(str(mutatedIndividual[gene]))
	nodes = [int(n[0]) for n in nodes]
	# avoid nodes repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)
	if len(nodes) > 0:
		# calculate second degree of each of the neighbors
		second_degrees = []
		for node in nodes:
			sec_degree = 0
			sec_degree += len(nodes)
			node_neighs = list(args["G"].neighbors(node))
			for node_neigh in node_neighs:
				# !very roughly approximated, may include repetitions
				sec_degree += len(list(args["G"].neighbors(node_neigh)))
			second_degrees.append(sec_degree)
		probs = np.array(second_degrees) / max(second_degrees)
		idx = prng.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]

	return mutatedIndividual


def ea_gloabal_low_deg_mutation(prng, candidate, args):
	"""
	the probability to select the gene to mutate depends on its degree
	"""
	nodes = args["nodes"].copy()
	# avoid nodes repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)
	mutatedIndividual = list(set(candidate))

	# choose random place
	probs = []
	for node in mutatedIndividual:
		probs.append(len(list(args["G"].neighbors(node))))

	probs = np.array(probs) / max(probs)
	probs = 1 - probs

	gene = prng.choices(range(0, len(mutatedIndividual)), probs)[0]
	mutatedIndividual[gene] = nodes[prng.randint(0, len(nodes) - 1)]

	return mutatedIndividual


# -------------------------------------- crossover operators ---------------------------------

# 1-point crossover
# @inspyred.ec.variators.crossover
def ea_one_point_crossover(prng, candidate1, candidate2, args):
	"""
	applies 1-point crossover by avoiding repetitions
	"""
	# see common elements
	common = list(set(candidate1).intersection(candidate2))

	# if two candidates are same
	if len(common) == len(candidate1):
		return [candidate1]

	# move common elements to the "beginning" of the list
	for c in common:
		candidate1.insert(0, candidate1.pop(candidate1.index(c)))
		candidate2.insert(0, candidate2.pop(candidate2.index(c)))

	# choose swap position, by avoiding common elements
	swap_idx = prng.randint(len(common), len(candidate1) - 1)
	swap = candidate1[swap_idx:]
	candidate1[swap_idx:] = candidate2[swap_idx:]
	candidate2[swap_idx:] = swap

	return [candidate1, candidate2]


# -----------------------------------crossover & mutations combinations----------------------

@inspyred.ec.variators.crossover
def ea_super_operator(prng, candidate1, candidate2, args):
	"""
	randomly applies crossover or mutation operator
	"""
	k = args["k"]
	local = args["local_mutation_rate"]
	children = []

	# uniform choice of operator
	randomChoice = prng.random()

	# one-point crossover or mutation that swaps exactly one node with another
	if randomChoice < args["crossover_rate"]:
		# children = inspyred.ec.variators.n_point_crossover(prng, [list(candidate1), list(candidate2)], args)
		children = ea_one_point_crossover(prng, list(candidate1), list(candidate2), args)

	randomChoice = prng.random()
	if randomChoice < args["mutation_rate"]:
		mut = prng.random()
		if mut < local:
			mutation = args["local_mutation_operator"]
		else:
			mutation = args["global_mutation_operator"]
		c1_mutated = mutation(prng, list(candidate1), args)
		c2_mutated = mutation(prng, list(candidate2), args)
		if common_elements(c1_mutated, candidate1) < len(candidate1):
			children.append(c1_mutated)
		else:
			print("wewe")

		if common_elements(c2_mutated, candidate2) < len(candidate2) and common_elements(c2_mutated, c1_mutated) < len(
				candidate2):
			children.append(c2_mutated)
		else:
			print("wewe")
	# purge the children from "None" and arrays of the wrong size
	l = len(children)
	children = [c for c in children if c is not None and len(set(c)) == k]
	if l != len(children):
		print("this message should not be printed")
	return children


# -------------------------------------- replacer operators ---------------------------------

# @inspyred.ec.replacers.plus_replacement
def ea_replacer(random, population, parents, offspring, args):
	"""
	selection of the new population: parents + offspring are both used
	"""
	n = args["max_individual_copies"]
	n_elites = args["num_elites"]

	# add elites from the old population
	pool = list(population)
	pool.sort(reverse=True)
	elites = pool[:n_elites]

	pool = list(offspring)
	pool.extend(parents)
	pool.extend(elites)

	for individual in pool:
		n_i = 1
		ind1 = set(individual.candidate)
		same_individuals = []
		pool2 = pool.copy()
		pool2.remove(individual)
		for individual2 in pool2:
			ind2 = set(individual2.candidate)
			if ind1 == ind2:
				n_i += 1
				same_individuals.append(individual2)
		if n_i > n:
			# remove all the "extra" occurrences from parents,
			# attention here: sometimes individuals with same candidates have different monte carlo fitness evaluations,
			# removing some of them may introduce loss of the individual with the highiest fitness
			for _ in range(n_i - n):
				# remove one individual having as candidate 'individual' candidate
				pool.remove(same_individuals.pop())

	pool.sort(reverse=True)
	survivors = pool[:len(population)]
	return survivors


# ----------------------------------------- observers ---------------------------------------

def ea_observer1(population, num_generations, num_evaluations, args):
	"""
	debug info, printing to stout some generational info
	"""
	# to access to evolutionary computation stuff
	# print(args["_ec"])
	div = diversity(population)
	print("generation {}: diversity {}".format(num_generations, div))
	ind_div = individuals_diversity(population)
	print("generation {}: individuals diversity {}".format(num_generations, ind_div))
	if ind_div < 0.5:
		for ind in population:
			print(ind)
		exit(0)
	return


def ea_observer2(population, num_generations, num_evaluations, args):
	"""
	printing generational log to out files
	"""

	# write current state of the population to a file
	population_file = args["population_file"]
	generations_file = args["generations_file"]

	# compute some generation statistics
	generation_stats = {}
	prev_best = args["prev_population_best"]
	current_best = max(population).fitness
	if prev_best > 0:
		generation_stats["improvement"] = (current_best - prev_best) / prev_best
	else:
		generation_stats["improvement"] = 0
	args["prev_population_best"] = current_best

	with open(generations_file, "a") as fg:
		fg.write("{},".format(num_generations))
		fg.write("{},".format(diversity(population)))
		fg.write("{},".format(generation_stats["improvement"]))
		fg.write("{}\n".format(current_best))

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


# ----------------------------------------- fitness evaluation ------------------------------

def ea_evaluator(candidates, args):
	"""
	evaluation of candidates' fitness
	"""
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


# ----------------------------------------- genetic algorithm -------------------------------

def ea_influence_maximization(k, G, fitness_function, pop_size, offspring_size, max_generations, prng,
							  crossover_rate=0, mutation_rate=1, n_processes=16, initial_population=[],
							  population_file=None, generations_file=None, tournament_size=2, num_elites=2,
							  word2vec_file=None, min_degree=2,
							  max_individual_copies=2, local_mutation_rate=0.5,
							  local_mutation_operator=ea_local_neighbors_second_degree_mutation,
							  global_mutation_operator=ea_gloabal_low_deg_mutation):
	# initialize generations file
	with open(generations_file, "w") as gf:
		gf.write("num_genrations,diversity,improvement,best_fitness\n")

	# nodes available for individuals construction
	nodes = list(G.nodes)
	# remove nodes with low degrees
	for node in nodes:
		if len(list(G.neighbors(node))) < min_degree:
			nodes.remove(node)

	ea = inspyred.ec.EvolutionaryComputation(prng)
	# ea = inspyred.ec.GA(prng)

	# each observer should be called at each generation
	ea.observer = [ea_observer1, ea_observer2]

	ea.terminator = inspyred.ec.terminators.generation_termination

	# crossover & mutation
	# called with probability 1, in our case for each different couple of parents, without repetitions:
	# es for populazion of size 16 the method is called 8 times for 8 random couples
	ea.variator = [ea_super_operator]

	ea.selector = inspyred.ec.selectors.tournament_selection
	# ea.selector = inspyred.ec.selectors.rank_selection

	# try another replacement operators you can find here : https://pythonhosted.org/inspyred/reference.html
	# ea.replacer = inspyred.ec.replacers.plus_replacement
	ea.replacer = ea_replacer
	if word2vec_file is not None:
		model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
	else:
		model = None

	final_population = ea.evolve(
		generator=ea_generator,
		evaluator=ea_evaluator,
		maximize=True,
		seeds=initial_population,
		pop_size=pop_size,
		num_selected=offspring_size,
		max_generations=max_generations,
		tournament_size=tournament_size,
		num_elites=num_elites,
		crossover_rate=crossover_rate,
		mutation_rate=mutation_rate,
		k=k,
		nodes=nodes,
		n_parallel=n_processes,
		spread_function=fitness_function,
		random_generator=prng,
		population_file=population_file,
		generations_file=generations_file,
		prev_population_best=-1,
		G=G,
		model=model,
		max_individual_copies=max_individual_copies,
		local_mutation_rate=local_mutation_rate,
		local_mutation_operator=local_mutation_operator,
		global_mutation_operator=global_mutation_operator
	)

	best_individual = max(final_population)
	best_seed_set = best_individual.candidate
	best_spread = best_individual.fitness

	return best_seed_set, best_spread

# from spread.monte_carlo import MonteCarlo_simulation as monte_carlo
# from utils import load_graph
# G = load_graph(g_type="wiki")
# means = []
# stds = []
# for _ in range(1000):
# 	m, s = monte_carlo(G, [1151, 766, 5524, 1166, 1549, 11, 457, 2565, 2688, 3028], 0.1, 100, "WC")
# 	means.append(m)
# 	stds.append(s)
#
# print(max(means))
# print(stds)