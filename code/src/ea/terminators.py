""" Method which determines when the optimisation run should terminate.

	generation_termination ends the run when a configurable number of generations were reached.
"""

def generation_termination(population, num_generations, num_evaluations, args):
	"""
	generation termination function
	key args argument: generations_budget
	:param population:
	:param num_generations:
	:param num_evaluations:
	:param args:
	:return:
	"""
	return num_generations == args["generations_budget"]

