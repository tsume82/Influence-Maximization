from utils import diversity, individuals_diversity


def ea_observer1(population, num_generations, num_evaluations, args):
	"""
	debug info, printing to stout some generational info
	"""
	# to access to evolutionary computation stuff
	div = diversity(population)
	print("generation {}: diversity {}".format(num_generations, div))
	# if adaptive, set local mutation rate to the nodes diversity
	# if args["adaptive_local_rate"]:
	# 	args["local_mutation_rate"] = div
	ind_div = individuals_diversity(population)
	print("generation {}: individuals diversity {}".format(num_generations, ind_div))

	# save the current population in args
	# args["population"] = population

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
