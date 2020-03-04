from utils import diversity, individuals_diversity
import os
import inspyred.ec
from ea.generators import generator_new_nodes


def ea_observer1(population, num_generations, num_evaluations, args):
	"""
	debug info, printing to stdout some generational info
	"""
	# to access to evolutionary computation stuff
	div = diversity(population)
	print("generation {}: diversity {}".format(num_generations, div))
	# if adaptive, set local mutation rate to the nodes diversity
	if args["adaptive_local_rate"]:
		args["local_mutation_rate"] = div
	ind_div = individuals_diversity(population)
	print("generation {}: individuals diversity {}".format(num_generations, ind_div))

	# check for repetitions
	for ind in population:
		if len(set(ind.candidate)) != len(ind.candidate):
			raise NameError("Nodes repetition inside an individual")
	# reset offspring fitnesses
	args["offspring_fitness"] = {}
	# exploration weight exponential decay
	#TODO: use a logarithm?
	if args["mab"] is not None:
		args["mab"].exploration_weight = 1/(num_generations+1)**(3)
		print("Mab selections: {}".format(args["mab"].n_selections))
	print("Population size: {}".format(len(population)))

	if args["dynamic_population"]:
		if len(population) < 100:
			if "improvement" in args.keys():
				if sum(args["improvement"]) == 0:
					new_individuals = min(int(1/div), 10)
					for _ in range(new_individuals):
						# candidate = args["_ec"].generator(args["prng"], args)
						candidate = generator_new_nodes(args["prng"], args)
						args["_ec"].population.append(inspyred.ec.Individual(candidate=candidate))
						args["_ec"].population[-1].fitness = args["fitness_function"](A=candidate)[0]
					args["num_selected"] = len(args["_ec"].population)
			else:
				args["improvement"] = [0]*3
	return


def ea_observer2(population, num_generations, num_evaluations, args):
	"""
	printing generational log to out files
	"""

	# write current state of the population to a file
	sf = args["statistics_file"]

	# compute some generation statistics
	generation_stats = {}
	prev_best = args["prev_population_best"]
	current_best = max(population).fitness
	if prev_best > 0:
		generation_stats["improvement"] = (current_best - prev_best) / prev_best
	else:
		generation_stats["improvement"] = 0
	args["prev_population_best"] = current_best

	sf.seek(sf.tell()-1, os.SEEK_SET)

	sf.write(",{},".format(diversity(population)))
	sf.write("{},".format(generation_stats["improvement"]))
	if args["mab"] is not None:
		sf.write("{}\n".format(args["mab"].n_selections))
	if args["dynamic_population"]:
		args["improvement"].pop(0)
		args["improvement"].append(generation_stats["improvement"])

	return
