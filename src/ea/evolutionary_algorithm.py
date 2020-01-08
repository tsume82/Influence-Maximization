import inspyred


from ea.mutators import ea_global_local_alteration
from ea.crossovers import ea_one_point_crossover
from ea.observers import ea_observer1, ea_observer2
from ea.evaluators import one_process_evaluator, multiprocess_evaluator
from ea.generators import generator
from ea.replacers import ea_replacer


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


def ea_influence_maximization(k, G, fitness_function, pop_size, offspring_size, max_generations, prng,
							  crossover_rate=0, mutation_rate=1, n_processes=1, initial_population=[],
							  individuals_file=None, statistics_file=None, tournament_size=2, num_elites=2,
							  node2vec_model=None, min_degree=2,
							  max_individual_copies=2, local_mutation_rate=0.5,
							  local_mutation_operator=None,
							  global_mutation_operator=None,
							  adaptive_local_rate=True):

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

	ea.variator = [ea_one_point_crossover,
				   ea_global_local_alteration]

	# replacement operator
	ea.replacer = inspyred.ec.replacers.generational_replacement
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

	# run the EA
	final_pop = ea.evolve(generator=generator,
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
						  max_individual_copies=max_individual_copies)

	best_individual = max(final_pop)
	best_seed_set = best_individual.candidate
	best_spread = best_individual.fitness

	return best_seed_set, best_spread
