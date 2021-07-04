""" Methods for evaluating the fitness of a candidate solution. """

from multiprocessing import Pool
import random

from src.spread.two_hop import two_hop_spread as two_hop


def multiprocess_evaluator(candidates, args):
	"""
	evaluation of candidates' fitness using multiprocessing pool
	"""
	n_parallel = args["n_parallel"]
	fitness_function = args["fitness_function"]
	random_generator = args["prng"]

	# -------------- multiprocessing ----------------
	process_pool = Pool(n_parallel)

	tasks = []
	for A in candidates:
		tasks.append([A, random_generator.random()])
	from functools import partial
	# multiprocessing pool imap function accepts only one argument at a time, create partial function with
	# constant parameters
	f = partial(ea_evaluator_processed, fitness_function=fitness_function)
	fitness = list(process_pool.imap(f, tasks))
	# print(fitness)
	return fitness


def ea_evaluator_processed(args, fitness_function):
	A, random_seed = args
	A = set(A)
	# run spread simulation
	if fitness_function.func != two_hop:
		influence_mean, influence_std = fitness_function(A=A, random_generator=random.Random(random_seed))
		# influence_mean, influence_std = fitness_function(A=A)

	else:
		influence_mean = fitness_function(A=A)
	return influence_mean


def one_process_evaluator(candidates, args):
	"""
	simple one processed candidates' evaluator
	:param candidates:
	:param args:
	:return:
	"""
	fitness = []

	for candidate in candidates:
		# fit = evaluate(args["G"], candidate, args["p"], args["no_simulations"], args["model"])
		# if tuple(set(candidate)) in args["offspring_fitness"].keys():
		# 	fit = args["offspring_fitness"][tuple(set(candidate))]
		# else:
		if args["fitness_function"].func.__name__ != "two_hop_spread":
			fit = args["fitness_function"](A=candidate)[0]
		else:
			fit = args["fitness_function"](A=candidate)
		fitness.append(fit)
	return fitness
