import inspyred
import numpy as np


def get_nodes_without_repetitions(candidate, args):
	"""
	removes candidate nodes from the pool of nodes
	:param candidate:
	:param args:
	:return:
	"""
	nodes = args["_ec"].bounder.values.copy()

	for c in candidate:
		if c in nodes: nodes.remove(c)
	return nodes


def get_nodes_neighbours_without_repetitions(node, candidate, args):
	"""
	returns nodes neighbours without nodes in candidate
	:param node:
	:param candidate:
	:param args:
	:return:
	"""
	nodes = list(args["G"].neighbors(node))
	# avoid nodes repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)
	return nodes


def get_node2vec_neighbors_without_repetitions(node, candidate, args):
	"""
	returns the most similar nodes accorting to the node2vec embeddings
	:param node:
	:param candidate:
	:param args:
	:return:
	"""
	nodes = args["model"].wv.most_similar(str(node))
	nodes = [int(n[0]) for n in nodes]
	# avoid nodes repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)
	return nodes


def approximated_spread(node, args):
	"""
	influence spread approximation : the spread is approximated by calculating the probability of activation of the neighbors
	and neighbors of neighbors of the node, where neighbors of neighbors are roughly approximated
	:param node:
	:param args:
	:return:
	"""

	approx_spread = 0
	node_neighs = list(args["G"].neighbors(node))
	# for each node neighbor calculate it's contribution
	for node_neigh in node_neighs:
		# very roughly approximated! may include repetitions
		neighbors_of_neighbors = list(args["G"].neighbors(node_neigh))
		if args["prop_model"] == "WC":
			approx_spread += 1/args["G"].in_degree(node_neigh)
		else:
			approx_spread += args["p"]
		for node_neigh_neigh in neighbors_of_neighbors:
			if args["prop_model"] == "WC":
				approx_spread += (1 / (args["G"].in_degree(node_neigh_neigh)))*(1/args["G"].in_degree(node_neigh))
			else:
				approx_spread += args["p"]**2
	return approx_spread


def eval_fitness(seed_set, random, args):
	"""
	evaluates fitness of the seed set
	:param seed_set:
	:param random:
	:return:
	"""
	spread = args["fitness_function"](A=seed_set, random_generator=random)
	# if we are using monteCarlo simulations which returns mean and std
	if len(spread) > 0:
		spread = spread[0]
	return spread


# @inspyred.ec.variators.mutator
def ea_global_random_mutation(prng, candidate, args):
	"""
	randomly mutates one gene of the individual with one random node of the graph
	"""

	nodes = get_nodes_without_repetitions(candidate, args)

	mutatedIndividual = candidate.copy()
	# choose random gene
	gene = prng.randint(0, len(mutatedIndividual) - 1)
	mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

	mutatedIndividual[gene] = mutated_node
	return mutatedIndividual


def ea_local_neighbors_random_mutation(random, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors
	"""
	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)

	# choose among neighbours of the selected node
	nodes = get_nodes_neighbours_without_repetitions(mutatedIndividual[gene], candidate, args)

	if len(nodes) > 0:
		mutated_node = nodes[random.randint(0, len(nodes) - 1)]
		mutatedIndividual[gene] = mutated_node
	else:
		# if we don't have neighbors to choose from, global mutation
		mutatedIndividual = ea_global_random_mutation(random, candidate, args)

	return mutatedIndividual


def ea_local_neighbors_second_degree_mutation(random, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, but according to second degree probability
	"""
	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)
	# choose among neighbours of the selected node
	nodes = get_nodes_neighbours_without_repetitions(mutatedIndividual[gene], candidate, args)

	if len(nodes) > 0:
		# calculate second degree of each of the neighbors
		second_degrees = []
		for node in nodes:
			sec_degree = 0
			sec_degree += len(nodes)
			node_neighs = list(args["G"].neighbors(node))
			for node_neigh in node_neighs:
				# !very roughly approximated to reduce computation time, may include repetitions
				neighbors_of_neighbors = list(args["G"].neighbors(node_neigh))
				sec_degree += len(neighbors_of_neighbors)
			second_degrees.append(sec_degree)
		probs = np.array(second_degrees) / max(second_degrees)
		idx = random.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		# if we don't have neighbors to choose from, global mutation
		mutatedIndividual = ea_global_random_mutation(random, candidate, args)
	return mutatedIndividual


def ea_global_low_spread(random, candidate, args):
	"""
	the probability to select the gene to mutate depends on its spread
	:param random:
	:param candidate:
	:param args:
	:return:
	"""
	nodes = get_nodes_without_repetitions(candidate, args)
	mutatedIndividual = candidate

	# choose random gene
	probs = []
	for node in mutatedIndividual:
		spread = eval_fitness([node], random, args)
		probs.append(spread)
	probs = np.array(probs) / max(probs)
	probs = 1 - probs

	gene = random.choices(range(0, len(mutatedIndividual)), probs)[0]
	mutated_node = nodes[random.randint(0, len(nodes) - 1)]

	mutatedIndividual[gene] = mutated_node

	return mutatedIndividual


def ea_global_low_deg_mutation(random, candidate, args):
	"""
	the probability to select the gene to mutate depends on its degree
	"""

	nodes = get_nodes_without_repetitions(candidate, args)
	mutatedIndividual = candidate

	# choose random gene
	probs = []
	for node in mutatedIndividual:
		probs.append(args["G"].out_degree(node))

	probs = np.array(probs) / max(probs)
	probs = 1 - probs

	gene = random.choices(range(0, len(mutatedIndividual)), probs)[0]
	mutated_node = nodes[random.randint(0, len(nodes) - 1)]

	mutatedIndividual[gene] = mutated_node

	return mutatedIndividual


def ea_local_approx_spread_mutation(random, candidate, args):
	"""
	selects a neighbor accorting to the maximum approximated spread probability
	:param random:
	:param candidate:
	:param args:
	:return:
	"""
	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)
	# choose among neighbours of the selected node
	nodes = get_nodes_neighbours_without_repetitions(mutatedIndividual[gene], candidate, args)

	if len(nodes) > 0:
		# calculate second degree of each of the neighbors
		approx_spreads = []
		for node in nodes:
			approx_spread = approximated_spread(node, args)
			approx_spreads.append(approx_spread)
		probs = np.array(approx_spreads) / max(approx_spreads)
		idx = random.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		# if we don't have neighbors to choose from, global mutation
		mutatedIndividual = ea_global_random_mutation(random, candidate, args)
	return mutatedIndividual


def ea_local_embeddings_mutation(random, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's nearest nodes according to their embeddings
	"""
	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)

	nodes = get_node2vec_neighbors_without_repetitions(mutatedIndividual[gene], candidate, args)
	if len(nodes) > 0:
		mutatedIndividual[gene] = nodes[random.randint(0, len(nodes) - 1)]
	else:
		# if we don't have neighbors to choose from, global mutation
		mutatedIndividual = ea_global_random_mutation(random, candidate, args)

	return mutatedIndividual


def ea_global_subpopulation_mutation(random, candidate, args):

	"""
	randomly mutates one gene of the individual with one of the nodes from the subpopulation assigned to that gene
	"""

	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)
	nodes = list(args["voronoi_cells"][list(args["voronoi_cells"].keys())[gene]])
	nodes = nodes.copy()
	# avoid repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)
	mutated_node = nodes[random.randint(0, len(nodes) - 1)]

	mutatedIndividual[gene] = mutated_node

	return mutatedIndividual

#TODO: vederee se potrebbe essere l'if in fondo
@inspyred.ec.variators.mutator
def ea_adaptive_mutators_alteration(random, candidate, args):
	"""
	this method calls with certain probability global and local mutations, those must be specified in args as
	parameters
	:param random:
	:param candidate:
	:param args:
	:return:
	"""
	if tuple(set(candidate)) not in args["offspring_fitness"].keys():
		old_fitness=args["fitness_function"](A=candidate, random_generator=random)[0]
		args["offspring_fitness"][tuple(set(candidate))]=old_fitness
	else:
		old_fitness = args["offspring_fitness"][tuple(set(candidate))]

	mutation = args["mab"].select_action()
	mutatedIndividual = mutation(random, candidate, args)
	if tuple(set(mutatedIndividual)) not in args["offspring_fitness"].keys():
		new_fitness = args["fitness_function"](A=mutatedIndividual, random_generator=random)[0]
		# save new fitness to the results
		args["offspring_fitness"][tuple(set(mutatedIndividual))]=new_fitness
	else:
		new_fitness = args["offspring_fitness"][tuple(set(mutatedIndividual))]
	improvement = (new_fitness - old_fitness) / old_fitness
	reward = improvement if improvement > 0 else 0
	args["mab"].update_reward(reward)
	# if improvement > 0:
	return mutatedIndividual
	# return candidate


def ea_local_neighbors_spread_mutation(random, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, which is chosen according to their
	spread probability
	:param random:
	:param candidate:
	:param args:
	:return:
	"""
	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)
	nodes = get_nodes_neighbours_without_repetitions(mutatedIndividual[gene], candidate, args)

	if len(nodes) > 0:
		spreads = []
		for node in nodes:
			spread = eval_fitness([node], random, args)
			spreads.append(spread)
		probs = np.array(spreads) / max(spreads)
		idx = random.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		# if we don't have neighbors to choose from, global mutation
		mutatedIndividual = ea_global_random_mutation(random, candidate, args)
	return mutatedIndividual


def ea_local_additional_spread_mutation(random, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, which is chosen according to their additional
	spread probability
	:param random:
	:param candidate:
	:param args:
	:return:
	"""
	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)

	nodes = get_nodes_neighbours_without_repetitions(mutatedIndividual[gene], candidate, args)
	if len(nodes) > 0:
		spreads = []
		# get the seed set without the selected node
		mutatedIndividual_without = mutatedIndividual.copy()
		mutatedIndividual_without.remove(mutatedIndividual[gene])
		# calculate its fitness function
		spread_without = eval_fitness(mutatedIndividual_without, random, args)
		# for each neighbor, calculate the spread increase it would generate
		for node in nodes:
			mutatedIndividual_with = mutatedIndividual_without.copy()
			mutatedIndividual_with.append(node)
			spread_with = eval_fitness(mutatedIndividual_with, random, args)
			additional_spread = spread_with - spread_without
			spreads.append(additional_spread)
		probs = np.array(spreads) / max(spreads)
		idx = random.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		# if we don't have neighbors to choose from, global mutation
		mutatedIndividual = ea_global_random_mutation(random, candidate, args)
	return mutatedIndividual


def ea_local_neighbors_second_degree_mutation_emb(random, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, but according to the second degree probability
	"""
	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)

	nodes = get_node2vec_neighbors_without_repetitions(mutatedIndividual[gene], candidate, args)
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
		idx = random.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		# if we don't have neighbors to choose from, global mutation
		mutatedIndividual = ea_global_random_mutation(random, candidate, args)

	return mutatedIndividual


def ea_global_low_additional_spread(random, candidate, args):
	"""
	the probability to select the gene to mutate depends on its "marginal" spread: the improvement that the node adds
	when added to the other genes in the individual
	:param random:
	:param candidate:
	:param args:
	:return:
	"""
	mutatedIndividual = candidate
	nodes = get_nodes_without_repetitions(candidate, args)
	spread_individual = eval_fitness(mutatedIndividual, random, args)

	probs = []
	for node in mutatedIndividual:
		mutatedIndividual_without = mutatedIndividual.copy()
		mutatedIndividual_without.remove(node)

		spread_without = eval_fitness(mutatedIndividual_without, random, args)
		additional_spread = spread_individual - spread_without
		probs.append(additional_spread)

	probs = np.array(probs) / max(probs)
	probs = 1 - probs

	gene = random.choices(range(0, len(mutatedIndividual)), probs)[0]
	mutatedIndividual[gene] = nodes[random.randint(0, len(nodes) - 1)]

	return mutatedIndividual


def ea_differential_evolution_mutation(random, candidate, args):
	"""
	differential evolution mutation: x = x + (a - b)
	"""
	# pick two random individuals a and b
	population = args["_ec"].population
	population = [p.candidate for p in population]
	if candidate in population: population.remove(candidate)

	A = population[random.randint(0, len(population)-1)]
	population.remove(A)
	B = population[random.randint(0, len(population)-1)]

	mutatedIndividual = []
	for n, a, b in zip(candidate, A, B):
		n1 = args["model"].most_similar(positive=[str(a), str(n)], negative=[str(b)], topn=1)[0][0]
		n1 = int(n1)
		mutatedIndividual.append(n1)

	return mutatedIndividual


def ea_global_activation_mutation(random, candidate, args):
	"""
	mutates one gene of the individual with one random node, which was both never activated by one of the candidate nodes
	and it never activated none of the candidate nodes
	"""

	mutatedIndividual = candidate
	nodes = get_nodes_without_repetitions(candidate, args)

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)
	mutated_node = nodes[random.randint(0, len(nodes) - 1)]

	# avoid repetitions
	ok = False
	while not ok:
		mutated_node = nodes[random.randint(0, len(nodes) - 1)]
		ok = True
		G_nodes = args["G"].nodes
		for node in candidate:
			if node in G_nodes[mutated_node]["activated_by"].keys():
				ok = False
			if mutated_node in G_nodes[node]["activated_by"].keys():
				ok = False

	mutatedIndividual[gene] = mutated_node

	return mutatedIndividual


def ea_local_activation_mutation(random, candidate, args):
	"""
	mutates the gene with the node by which it was activated the biggest amount of times
	"""
	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)

	old_node = candidate[gene]
	G_nodes = args["G"].nodes
	if len(G_nodes[old_node]["activated_by"]) > 0:
		nodes = list(G_nodes[old_node]["activated_by"].keys())
		for c in candidate:
			if c in nodes: nodes.remove(c)
		if len(nodes) > 0:
			# update the probabilities
			probabilities = []
			for node in nodes:
				probabilities.append(G_nodes[old_node]["activated_by"][node])
			# probabilities = list(G_nodes[old_node]["activated_by"].values())
			probabilities = np.array(probabilities)
			probabilities[np.argmax(probabilities)] *= 10
			mutated_node = random.choices(nodes, probabilities)[0]

			mutatedIndividual[gene] = mutated_node
			return mutatedIndividual

	# if we don't have activation nodes to choose from, global mutation
	mutatedIndividual = ea_global_random_mutation(random, candidate, args)

	return mutatedIndividual

