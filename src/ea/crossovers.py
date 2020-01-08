import inspyred


@inspyred.ec.variators.crossover
def ea_one_point_crossover(prng, candidate1, candidate2, args):
	"""
	applies 1-point crossover by avoiding repetitions
	"""
	# see common elements
	common = list(set(candidate1).intersection(candidate2))

	# if two candidates are same
	if len(common) == len(candidate1):
		return [candidate1]

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
	swap_idx = prng.randint(0, len(candidate1_to_swap) - 1)
	swap = candidate1_to_swap[swap_idx:]
	candidate1_to_swap[swap_idx:] = candidate2_to_swap[swap_idx:]
	candidate2_to_swap[swap_idx:] = swap

	for (idx, c) in c1_common.items():
		candidate1_to_swap.insert(idx, c)
	for (idx, c) in c2_common.items():
		candidate2_to_swap.insert(idx, c)

	return [candidate1_to_swap, candidate2_to_swap]
