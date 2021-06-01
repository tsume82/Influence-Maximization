import inspyred
import networkx as nx

from src.ea.crossovers import ea_one_point_crossover
from src.ea.observers import ea_observer0, ea_observer1, ea_observer2
from src.ea.evaluators import one_process_evaluator, multiprocess_evaluator
from src.ea.generators import generator, subpopulation_generator
import src.ea.mutators as mutators
# from src.ea.generators import subpopulation_generator as generator
from src.ea.terminators import generation_termination
from src.multi_armed_bandit import Multi_armed_bandit


# @inspyred.ec.variators.crossover
# def ea_variator(prng, candidate1, candidate2, args):
# 	randomChoice = prng.random()
#
# 	# one-point crossover or mutation that swaps exactly one node with another
# 	children = []
# 	if randomChoice < args["crossover_rate"]:
# 		res = ea_one_point_crossover(prng, candidate1, candidate2, args)
# 		for mut in res:
# 			children.append(mut)
#
# 	randomChoice = prng.random()
# 	if randomChoice < args["mutation_rate"]:
# 		mutatedIndividual1 = args["mutation_operator"](prng, [candidate1], args)
# 		mutatedIndividual2 = args["mutation_operator"](prng, [candidate2], args)
# 		children.append(mutatedIndividual1[0])
# 		children.append(mutatedIndividual2[0])
# 	return children


def ea_influence_maximization(k,
							  G,
							  fitness_function,
							  pop_size,
							  offspring_size,
							  max_generations,
							  prng,
							  crossover_rate=1.0,
							  mutation_rate=0.1,
							  n_processes=1,
							  initial_population=[],
							  individuals_file=None,
							  statistics_file=None,
							  tournament_size=2,
							  num_elites=2,
							  node2vec_model=None,
							  mutators_to_alterate=[],
							  mutation_operator=None,
							  prop_model="WC",
							  p=0.01,
							  moving_avg_len=100,
							  dynamic_population=False,
							  nodes=None,
							  max_generations_percentage_without_improvement = 0.1):

	ea = inspyred.ec.EvolutionaryComputation(prng)

	# observers: provide various logging features
	ea.observer = [inspyred.ec.observers.stats_observer,
					   inspyred.ec.observers.file_observer,
				   ea_observer0, ea_observer1, ea_observer2]

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

	bounder = inspyred.ec.DiscreteBounder(nodes)

	if mutation_operator == mutators.ea_global_subpopulation_mutation \
			or mutators.ea_global_subpopulation_mutation in mutators_to_alterate:
		gen = subpopulation_generator
		seeds = prng.sample(nodes, k)
		voronoi_cells = nx.algorithms.voronoi_cells(G, seeds)
	else:
		gen = generator
		voronoi_cells = None

	# run the EA
	mab = None
	if mutation_operator == mutators.ea_adaptive_mutators_alteration:
		init_exploration_weight = 1
		mab = Multi_armed_bandit(mutators_to_alterate, init_exploration_weight, moving_avg_len)


	final_pop = ea.evolve(generator=gen,
						  evaluator=evaluator,
						  bounder= bounder,
						  maximize=True,
						  pop_size=pop_size,
						  min_pop_size=pop_size,
						  generations_budget=max_generations,
						  max_generations=int(max_generations*max_generations_percentage_without_improvement),
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
						  model=node2vec_model,
						  prop_model = prop_model,
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
