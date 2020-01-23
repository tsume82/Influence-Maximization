from utils import diversity, individuals_diversity
import os


def ea_observer1(population, num_generations, num_evaluations, args):
	"""
	debug info, printing to stout some generational info
	"""
	# to access to evolutionary computation stuff
	div = diversity(population)
	print("generation {}: diversity {}".format(num_generations, div))
	# if adaptive, set local mutation rate to the nodes diversity
	if args["adaptive_local_rate"]:
		args["local_mutation_rate"] = div
	ind_div = individuals_diversity(population)
	print("generation {}: individuals diversity {}".format(num_generations, ind_div))
	print("Mutations reward {}".format(args["mab"].sums_of_reward))

	# check for repetitions
	for ind in population:
		if len(set(ind.candidate)) != len(ind.candidate):
			raise NameError("Nodes repetition inside an individual")
	# reset offspring fitnesses
	args["offspring_fitness"] = {}
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
	sf.write("{}\n".format(args["mab"].n_selections))

	return
