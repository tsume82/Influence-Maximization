import inspyred
import networkx as nx

from heapq import nlargest

from ea.mutators import ea_global_local_alteration
from ea.crossovers import ea_one_point_crossover
from ea.observers import ea_observer1, ea_observer2
from ea.evaluators import one_process_evaluator, multiprocess_evaluator
from ea.generators import generator, subpopulation_generator
import ea.mutators as mutators
# from ea.generators import subpopulation_generator as generator
from ea.replacers import ea_replacer
from ea.terminators import generation_termination
from multi_armed_bandit import Multi_armed_bandit
from spread.monte_carlo_max_hop import MonteCarlo_simulation as mc
from select_best_spread_nodes import filter_best_nodes
from functools import partial
from utils import inverse_ncr

from smart_initialization import degree_random


#TODO: remove from here?
def filter_nodes(G, min_degree, nodes=None):
	"""
	selects nodes with degree at least high as min_degree
	:param G:
	:param min_degree:
	:return:
	"""
	if nodes is None:
		nodes = list(G.nodes())
	if min_degree > 0:
		for node in nodes:
			if G.out_degree(node) < min_degree:
				nodes.remove(node)

	return nodes


def filter_max_spread(G, prng, best_percentage=0.001):
	nodes = list(G.nodes())
	spreads = {}
	for n in nodes:
		i = mc(G=G, A=[n], p=0.01, model="WC", max_hop=3, random_generator=prng, no_simulations=5)
		spreads[n] = i[0]
	# select best percentage
	nodes = nlargest(int(len(G)*best_percentage), spreads, key=spreads.get)
	return nodes


@inspyred.ec.variators.crossover
def ea_variator(prng, candidate1, candidate2, args):
	randomChoice = prng.random()

	# one-point crossover or mutation that swaps exactly one node with another
	children = []
	if randomChoice < args["crossover_rate"]:
		res = ea_one_point_crossover(prng, candidate1, candidate2, args)
		for mut in res:
			children.append(mut)

	randomChoice = prng.random()
	if randomChoice < args["mutation_rate"]:
		mutatedIndividual1 = args["mutation_operator"](prng, [candidate1], args)
		mutatedIndividual2 = args["mutation_operator"](prng, [candidate2], args)
		children.append(mutatedIndividual1[0])
		children.append(mutatedIndividual2[0])
	return children


def ea_influence_maximization(k, G, fitness_function, pop_size, offspring_size, max_generations, prng,
							  crossover_rate=0, mutation_rate=1, n_processes=1, initial_population=[],
							  individuals_file=None, statistics_file=None, tournament_size=2, num_elites=2,
							  node2vec_model=None, min_degree=2,
							  max_individual_copies=2, local_mutation_rate=0.5,
							  local_mutation_operator=None,
							  global_mutation_operator=None,
							  adaptive_local_rate=True, mutators_to_alterate=[],
							  mutation_operator=ea_global_local_alteration, prop_model="WC", p=0.01,
							  exploration_weight=1, moving_avg_len=100, best_nodes_percentage=0.01,
							  filter_best_spread_nodes=False, dynamic_population=False,
							  adaptive_mutations=False, smart_initialization = None, smart_initialization_percentage=0.7):

	ea = inspyred.ec.EvolutionaryComputation(prng)

	# observers: provide various logging features
	ea.observer = [inspyred.ec.observers.stats_observer,
					   inspyred.ec.observers.file_observer,
				   ea_observer1, ea_observer2]

	#  selection operator
	ea.selector = inspyred.ec.selectors.tournament_selection

	ea.variator = [ea_one_point_crossover, mutation_operator]

	# replacement operator
	ea.replacer = inspyred.ec.replacers.generational_replacement

	# termination condition
	ea.terminator = [inspyred.ec.terminators.no_improvement_termination, generation_termination]

	# population evaluator
	if n_processes == 1:
		evaluator = one_process_evaluator
	else:
		evaluator = multiprocess_evaluator

	# --------------------------------------------------------------------------- #


	# nodes = filter_max_spread(G, prng, best_nodes_percentage)
	nodes = None
	if filter_best_spread_nodes:
		search_space_size_min = 1e9
		search_space_size_max = 1e11

		best_nodes = inverse_ncr(search_space_size_min, k)
		error = (inverse_ncr(search_space_size_max, k) - best_nodes) / best_nodes
		filter_function = partial(mc, G=G, random_generator=prng, p=p, model=prop_model, max_hop=3, no_simulations=1)
		nodes = filter_best_nodes(G, best_nodes, error, filter_function)

	nodes = filter_nodes(G, min_degree, nodes)

	if smart_initialization == "degree_random":
		initial_population = degree_random(k, G,
										    pop_size,
										   	prng, nodes=nodes)

	bounder = inspyred.ec.DiscreteBounder(nodes)

	# if global_mutation_operator == mutators.ea_global_subpopulation_mutation or mutators.ea_global_subpopulation_mutation in mutators_to_alterate:
	# 	gen = subpopulation_generator
	# 	seeds = prng.sample(nodes, k)
	# 	voronoi_cells = nx.algorithms.voronoi_cells(G, seeds)
	# else:
	gen = generator
	voronoi_cells = None

	# run the EA
	mab = None
	if adaptive_mutations:
		mab = Multi_armed_bandit(mutators_to_alterate, exploration_weight, moving_avg_len)

	final_pop = ea.evolve(generator=gen,
						  evaluator=evaluator,
						  bounder= bounder,
						  maximize=True,
						  pop_size=pop_size,
						  start_size=10,
						  generations_budget=max_generations,
						  max_generations=max_generations*0.1,
						  # max_evaluations=100,
						  tournament_size=tournament_size,
						  mutation_rate=mutation_rate,
						  crossover_rate=crossover_rate,
						  num_selected=offspring_size,
						  num_elites=num_elites,
						  k=k,
						  G=G,
						  fitness_function=fitness_function,
						  prng=prng,
						  seeds=initial_population,
						  nodes=nodes,
						  n_parallel=n_processes,
						  individuals_file=individuals_file,
						  statistics_file=statistics_file,
						  prev_population_best=-1,
						  local_mutation_operator=local_mutation_operator,
						  global_mutation_operator=global_mutation_operator,
						  local_mutation_rate=local_mutation_rate,
						  adaptive_local_rate=adaptive_local_rate,
						  model=node2vec_model,
						  prop_model = prop_model,
						  max_individual_copies=max_individual_copies,
						  voronoi_cells=voronoi_cells,
						  mutators_to_alterate=mutators_to_alterate,
						  mab = mab,
						  p=p,
						  mutation_operator=mutation_operator,
						  offspring_fitness = {},
						  individuals_pool = [],
						  dynamic_population = dynamic_population)

	best_individual = max(final_pop)
	best_seed_set = best_individual.candidate
	best_spread = best_individual.fitness

	return best_seed_set, best_spread
