import inspyred
import networkx as nx

from ea.mutators import ea_global_local_alteration
from ea.crossovers import ea_one_point_crossover
from ea.observers import ea_observer1, ea_observer2
from ea.evaluators import one_process_evaluator, multiprocess_evaluator
from ea.generators import generator, subpopulation_generator
import ea.mutators as mutators
# from ea.generators import subpopulation_generator as generator
from ea.replacers import ea_replacer
from multi_armed_bandit import Multi_armed_bandit


#TODO: remove from here?
def filter_nodes(G, min_degree):
	"""
	selects nodes with degree at least high as min_degree
	:param G:
	:param min_degree:
	:return:
	"""
	nodes = list(G.nodes())
	if min_degree > 0:
		for node in nodes:
			if G.out_degree(node) < min_degree:
				nodes.remove(node)

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
	# print(children)
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
							  exploration_weight=1, moving_avg_len=100):

	ea = inspyred.ec.EvolutionaryComputation(prng)

	# observers: provide various logging features
	ea.observer = [inspyred.ec.observers.stats_observer,
					   inspyred.ec.observers.file_observer,
				   ea_observer1, ea_observer2]

	#  selection operator
	ea.selector = inspyred.ec.selectors.tournament_selection

	# # variation operators (mutation/crossover)
	# ea.variator = [inspyred.ec.variators.n_point_crossover,
	# 			   inspyred.ec.variators.random_reset_mutation]

	# ea.variator = [ea_one_point_crossover,
	# 			   mutation_operator]

	ea.variator = [ea_one_point_crossover, mutation_operator]
	# ea.variator = [mutation_operator]
	# ea.variator = ea_variator


	# replacement operator
	ea.replacer = inspyred.ec.replacers.generational_replacement
	# ea.replacer = inspyred.ec.replacers.comma_replacement
	# ea.replacer = ea_replacer

	# termination condition
	ea.terminator = inspyred.ec.terminators.generation_termination

	# population evaluator
	if n_processes == 1:
		evaluator = one_process_evaluator
	else:
		evaluator = multiprocess_evaluator

	# --------------------------------------------------------------------------- #

	nodes = filter_nodes(G, min_degree)

	bounder = inspyred.ec.DiscreteBounder(nodes)

	if global_mutation_operator == mutators.ea_global_subpopulation_mutation:
		gen = subpopulation_generator
		seeds = prng.sample(nodes, k)
		voronoi_cells = nx.algorithms.voronoi_cells(G, seeds)
	else:
		gen = generator
		voronoi_cells = None

	# run the EA
	mab = Multi_armed_bandit(mutators_to_alterate, exploration_weight, moving_avg_len)

	final_pop = ea.evolve(generator=gen,
						  evaluator=evaluator,
						  bounder= bounder,
						  maximize=True,
						  pop_size=pop_size,
						  max_generations=max_generations,
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
						  offspring_fitness = {})

	best_individual = max(final_pop)
	best_seed_set = best_individual.candidate
	best_spread = best_individual.fitness

	return best_seed_set, best_spread
