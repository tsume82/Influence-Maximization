

def generator(random, args):
	"""
	simple random generator: generates individual by sampling random nodes
	:param random:
	:param args:
	:return:
	"""
	#TODO: sostituire pool di nodi da bounder, dove i nodi nel caso vengono filtrati
	return random.sample(args["G"].nodes(), args["k"])
