# 1-hop and 2-hop spread
# implementation of the formula defined in "A Fast Approximation for Influence Maximization in Large Social Networks",
# authors Chin-Wan Chung, Jong-Ryul Lee, 2014
# link: https://dl.acm.org/citation.cfm?id=2567948.2580063

import networkx as nx


def degree_function(G):
	"""
	degree function for WC model probability calculation
	:param G:
	:return:
	"""
	if nx.is_directed(G):
		function = G.in_degree
	else:
		function = G.degree
	return function


def neighbours(G, s):
	"""
	neighbours of node s
	:param G: graph G
	:param s: label of the node
	:return: list with neighbours of s
	"""
	return list(dict(G.adjacency())[s])


def n_neighbours(G, s):
	"""
	number of neighbours of node s
	:param G:
	:param s:
	:return:
	"""
	return len(neighbours(G, s))


def one_hop_spread_single_node(G, s, p, model):
	"""

	:param G: networkx graph
	:param s: node label
	:param p: propagation probability
	:param model: propagation model
	:return:
	"""
	my_degree_function = degree_function(G)
	if model == "IC":
		return 1 + n_neighbours(G, s)*p
	elif model == "WC":
		result = 1
		Cs = neighbours(G, s)
		for c in Cs:
			result += 1.0/my_degree_function(c)
		return result


def _chi(G, S, p, model):
	"""
	returns chi value of seed set S
	:param G: netowrkx graph
	:param S: seed set
	:param p: propagation probability
	:param model: propagation model
	:return:
	"""
	result = 0
	my_degree_function = degree_function(G)
	for s in S:
		Cs = set(neighbours(G, s))
		Cs = Cs.difference(S)
		for c in Cs:
			Cc = set(neighbours(G, c))
			Cc = Cc.intersection(S).difference({s})
			for d in Cc:
				if model == "IC":
					ps_c = p
					pc_d = p
				elif model == "WC":
					ps_c = 1.0/my_degree_function(c)
					pc_d = 1.0/my_degree_function(d)
				result += ps_c*pc_d
	return result


def two_hop_spread(G, A, p, model):
	"""
	two hop spread of initial seed set S under IC or WC propagation models
	:param G: networkx graph
	:param A: initial seed set, list
	:param p: propagation probability
	:param model: propagation model
	:return:
	"""
	# just to keep the paper notation
	S = A
	k = len(S)
	result = 0
	result -= _chi(G, S, p, model)
	my_degree_function = degree_function(G)
	if k == 1:
		result += 1
		s = list(S)[0]
		Cs = set(neighbours(G, s))
		for c in Cs:
			sigma_c = one_hop_spread_single_node(G, c, p, model)
			if model == "IC":
				ps_c = p
				pc_s = p
			elif model == "WC":
				ps_c = 1.0/my_degree_function(c)
				pc_s = 1.0/my_degree_function(s)
			result += ps_c*(sigma_c-pc_s)
	else:
		for s in S:
			result += two_hop_spread(G, [s], p, model)[0]
			Cs = set(neighbours(G, s))
			Cs = Cs.intersection(S)
			for c in Cs:
				sigma_c = one_hop_spread_single_node(G, c, p, model)
				if model == "IC":
					ps_c = p
					pc_s = p
				elif model == "WC":
					ps_c = 1.0 / my_degree_function(c)
					pc_s = 1.0 / my_degree_function(s)
				result -= ps_c*(sigma_c-pc_s)

	return result, 0


if __name__ == "__main__":
	import networkx as nx
	G = nx.generators.random_graphs.barabasi_albert_graph(100, 5, seed=0)
	A = list(range(0, 20, 2))
	two_h1 = two_hop_spread(G, A, 0.4, "IC")
	two_h2 = two_hop_spread(G, A, 0.4, "WC")

	print("IC two-hop spread: ", two_h1)
	print("WC two-hop spread: ", two_h2)
