import inspyred
import numpy as np

# def ea_global_random_mutation(prng, candidate, args):

def ea_global_random_mutation(prng, candidate, args):

	"""
	randomly mutates one gene of the individual
	"""
	nodes = args["_ec"].bounder.values

	mutatedIndividual = list(set(candidate))

	# choose random place
	gene = prng.randint(0, len(mutatedIndividual) - 1)
	mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

	# avoid repetitions
	while mutated_node in mutatedIndividual:
		mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

	mutatedIndividual[gene] = mutated_node

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
		mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

		mutatedIndividual[gene] = mutated_node
	else:
		mutatedIndividual = ea_global_random_mutation(prng, candidate, args)

	return mutatedIndividual


def ea_local_neighbors_second_degree_mutation(prng, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, but according to second degree probability
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
		# calculate second degree of each of the neighbors
		second_degrees = []
		for node in nodes:
			sec_degree = 0
			sec_degree += len(nodes)
			node_neighs = list(args["G"].neighbors(node))
			for node_neigh in node_neighs:
				# !very roughly approximated, may include repetitions
				neighbors_of_neighbors = list(args["G"].neighbors(node_neigh))
				sec_degree += len(neighbors_of_neighbors)
			second_degrees.append(sec_degree)
		probs = np.array(second_degrees) / max(second_degrees)
		idx = prng.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		mutatedIndividual = ea_global_random_mutation(prng, candidate, args)
	return mutatedIndividual


def ea_global_low_spread(prng, candidate, args):
	"""
	the probability to select the gene to mutate depends on its spread
	:param prng:
	:param candidate:
	:param args:
	:return:
	"""
	nodes = args["_ec"].bounder.values
	mutatedIndividual = list(set(candidate))

	# choose random place
	probs = []
	for node in mutatedIndividual:
		spread = args["fitness_function"](A=[node], random_generator=prng)[0]
		probs.append(spread)

	probs = np.array(probs) / max(probs)
	probs = 1 - probs

	gene = prng.choices(range(0, len(mutatedIndividual)), probs)[0]
	mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

	# avoid repetitions
	while mutated_node in mutatedIndividual:
		mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

	mutatedIndividual[gene] = mutated_node

	return mutatedIndividual


def ea_global_low_deg_mutation(prng, candidate, args):
	"""
	the probability to select the gene to mutate depends on its degree
	"""

	nodes = args["_ec"].bounder.values

	mutatedIndividual = list(set(candidate))

	# choose random place
	probs = []
	for node in mutatedIndividual:
		probs.append(args["G"].out_degree(node))

	probs = np.array(probs) / max(probs)
	probs = 1 - probs

	gene = prng.choices(range(0, len(mutatedIndividual)), probs)[0]
	mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

	# avoid repetitions
	while mutated_node in mutatedIndividual:
		mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

	mutatedIndividual[gene] = mutated_node

	return mutatedIndividual


@inspyred.ec.variators.mutator
def ea_global_local_alteration(prng, candidate, args):
	"""
	this method calls with certain probability global and local mutations, those must be specified in args as
	parameters
	:param prng:
	:param candidate:
	:param args:
	:return:
	"""
	mut = prng.random()
	if mut < args["local_mutation_rate"]:
		mutation = args["local_mutation_operator"]
	else:
		mutation = args["global_mutation_operator"]

	mutatedIndividual = mutation(prng, candidate, args)

	return mutatedIndividual


def ea_local_approx_spread_mutation(prng, candidate, args):
	"""
	selects among neighbours neighbor with maximum approximated degree
	:param prng:
	:param candidate:
	:param args:
	:return:
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
		# calculate second degree of each of the neighbors
		approx_spreads = []
		for node in nodes:
			approx_spread = 0
			node_neighs = list(args["G"].neighbors(node))
			for node_neigh in node_neighs:
				# !very roughly approximated, may include repetitions
				neighbors_of_neighbors = list(args["G"].neighbors(node_neigh))
				for node_neigh_neigh in neighbors_of_neighbors:
					approx_spread += 1/(args["G"].in_degree(node_neigh_neigh))
			approx_spreads.append(approx_spread)
		probs = np.array(approx_spreads) / max(approx_spreads)
		idx = prng.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		mutatedIndividual = ea_global_random_mutation(prng, candidate, args)
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
	else:
		mutatedIndividual = ea_global_random_mutation(prng, candidate, args)

	return mutatedIndividual


def ea_global_subpopulation_mutation(prng, candidate, args):

	"""
	randomly mutates one gene of the individual
	"""
	# nodes = args["_ec"].bounder.values

	mutatedIndividual = list(set(candidate))

	# choose random place
	gene = prng.randint(0, len(mutatedIndividual) - 1)
	nodes = list(args["voronoi_cells"][list(args["voronoi_cells"].keys())[gene]])
	mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

	# avoid repetitions
	while mutated_node in mutatedIndividual:
		mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

	mutatedIndividual[gene] = mutated_node

	return mutatedIndividual

# ----------------------------


def ea_local_neighbors_spread_mutation(prng, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, which is chosen according to their
	two hop spread probability
	:param prng:
	:param candidate:
	:param args:
	:return:
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
		spreads = []
		for node in nodes:
			spread = args["fitness_function"]( A=[node], random_generator=args["prng"])
			spreads.append(spread)
		# print(spreads)
		# exit(0)
		probs = np.array(spreads) / max(spreads)
		idx = prng.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		mutatedIndividual = ea_global_random_mutation(prng, candidate, args)
	return mutatedIndividual


def ea_local_additional_spread_mutation(prng, candidate, args):
	"""

	:param prng:
	:param candidate:
	:param args:
	:return:
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
		spreads = []

		mutatedIndividual_without = mutatedIndividual.copy()
		mutatedIndividual_without.remove(mutatedIndividual[gene])
		spread_without = args["fitness_function"](A=mutatedIndividual_without, random_generator=prng)[0]
		for node in nodes:
			mutatedIndividual_with = mutatedIndividual_without.copy()
			mutatedIndividual_with.append(node)
			spread_with = args["fitness_function"](A=mutatedIndividual_with, random_generator=prng)[0]
			additional_spread = spread_with - spread_without
			spreads.append(additional_spread)
		# print(spreads)
		# exit(0)
		probs = np.array(spreads) / max(spreads)
		idx = prng.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		mutatedIndividual = ea_global_random_mutation(prng, candidate, args)
	return mutatedIndividual


def ea_local_neighbors_second_degree_mutation_emb(prng, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, but according to second degree probability
	"""
	#TODO calcolare le probabilità corrispondenti alla two hop spread del nodo, oppure alla somma di probabilità
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
	else:
		mutatedIndividual = ea_global_random_mutation(prng, candidate, args)

	return mutatedIndividual


def ea_global_low_additional_spread(prng, candidate, args):
	"""
	the probability to select the gene to mutate depends on its "marginal" spread: the improvement that the node adds
	when added to the other genes in the individual
	:param prng:
	:param candidate:
	:param args:
	:return:
	"""
	nodes = args["nodes"].copy()
	# avoid nodes repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)
	mutatedIndividual = list(set(candidate))

	spread_individual = args["fitness_function"](A=mutatedIndividual, random_generator=prng)[0]
	# choose random place
	probs = []
	for node in mutatedIndividual:
		mutatedIndividual_without = mutatedIndividual.copy()
		mutatedIndividual_without.remove(node)

		spread_without = args["fitness_function"](A=mutatedIndividual_without, random_generator=prng)[0]
		additional_spread = spread_individual - spread_without
		probs.append(additional_spread)

	probs = np.array(probs) / max(probs)
	probs = 1 - probs

	gene = prng.choices(range(0, len(mutatedIndividual)), probs)[0]
	mutatedIndividual[gene] = nodes[prng.randint(0, len(nodes) - 1)]

	return mutatedIndividual


def ea_differential_evolution_mutation(prng, candidate, args):
	"""
	differential evolution mutation: x = x + (a - b)
	"""
	# pick two random individuals a and b

	population = args["population"].copy()
	population = [p.candidate for p in population]
	if candidate in population: population.remove(candidate)

	A = population[prng.randint(0, len(population)-1)]
	population.remove(A)
	B = population[prng.randint(0, len(population)-1)]

	mutatedIndividual = []
	for n, a, b in zip(candidate, A, B):
		n1 = args["model"].most_similar(positive=[str(a), str(n)], negative=[str(b)], topn=1)[0][0]
		n1 = int(n1)
		mutatedIndividual.append(n1)

	return mutatedIndividual
