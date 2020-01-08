import inspyred

# @inspyred.ec.replacers.plus_replacement
def ea_replacer(random, population, parents, offspring, args):
	"""
	selection of the new population: parents + offspring are both used
	"""
	n = args["max_individual_copies"]
	n_elites = args["num_elites"]

	# print(len(offspring))
	# exit(0)

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