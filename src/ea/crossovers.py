import inspyred
from ea.mutators import ea_global_random_mutation


@inspyred.ec.variators.crossover
def ea_one_point_crossover(prng, candidate1, candidate2, args):
	"""
	applies 1-point crossover by avoiding repetitions
	"""
	# see common elements
	common = list(set(candidate1).intersection(set(candidate2)))
	max_trials = 5
	while (len(candidate1) - len(common)) < 2 and max_trials > 0:
		# while candidates have less then 2 nodes in common crossover would not produce any new candidates,
		# so mutation is forced
		if (len(candidate1) - len(common) == 1):
			candidate1 = ea_global_random_mutation(prng, candidate1, args)
			candidate2 = ea_global_random_mutation(prng, candidate2, args)
		elif (len(candidate1)==len(common)):
			for _ in range(2):
				candidate1 = ea_global_random_mutation(prng, candidate1, args)
				candidate2 = ea_global_random_mutation(prng, candidate2, args)
		max_trials -= 1
		common = list(set(candidate1).intersection(set(candidate2)))

	if max_trials==0:
		return [candidate2, candidate1]

	candidate1_to_swap = candidate1.copy()
	candidate2_to_swap = candidate2.copy()
	c1_common = {}
	c2_common = {}

	# get the nodes of each candidate that can be swapped
	for c in common:
		candidate1_to_swap.pop(candidate1_to_swap.index(c))
		candidate2_to_swap.pop(candidate2_to_swap.index(c))
		idx1 = candidate1.index(c)
		idx2 = candidate2.index(c)
		c1_common[idx1] = c
		c2_common[idx2] = c

	# choose swap position

	swap_idx = prng.randint(1, len(candidate1_to_swap) - 1)
	swap = candidate1_to_swap[swap_idx:]
	candidate1_to_swap[swap_idx:] = candidate2_to_swap[swap_idx:]
	candidate2_to_swap[swap_idx:] = swap

	for (idx, c) in c1_common.items():
		candidate1_to_swap.insert(idx, c)
	for (idx, c) in c2_common.items():
		candidate2_to_swap.insert(idx, c)

	# if set(candidate1_to_swap) == set(candidate2) or set(candidate1_to_swap)==set(candidate1):
	# 	print(candidate1_to_swap, candidate2_to_swap)
	# 	exit(0)

	return [candidate1_to_swap, candidate2_to_swap]
