

def generator(random, args):
	"""
	simple random generator: generates individual by sampling random nodes
	:param random:
	:param args:
	:return:
	"""
	#TODO: sostituire pool di nodi da bounder, dove i nodi nel caso vengono filtrati
	# return random.sample(args["G"].nodes(), args["k"])
	return random.sample(args["nodes"], args["k"])


def subpopulation_generator(random, args):
	"""
	for each dimension selects node from one cell
	:param random:
	:param args:
	:return:
	"""
	individual = []
	voronoi_cells = args["voronoi_cells"]
	for i in range(args["k"]):
		nodes = voronoi_cells[list(voronoi_cells.keys())[i]]
		individual.append(random.sample(nodes, 1)[0])

	return individual