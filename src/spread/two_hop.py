# 1-hop and 2-hop spread
# implementation of the formula defined in "A Fast Approximation for Influence Maximization in Large Social Networks",
# authors Chin-Wan Chung, Jong-Ryul Lee, 2014
# link: https://dl.acm.org/citation.cfm?id=2567948.2580063

import networkx as nx


def neighbours(G, s):
	"""
	neighbours of node s
	:param G: graph G
	:param s: label of the node
	:return: list with neighbours of s
	"""
	return list(dict(G.adjacency())[s])


def one_hop_spread_single_node(G, s):
	"""
	:param G: netowrkx weighted directed graphs, where weights are spread probabilities
	:param s: node label
	:return:
	"""
	result = 1
	Cs = neighbours(G, s)
	for c in Cs:
		ps_c = G[s][c]["weight"]
		result += ps_c
	return result


def _chi(G, S):
	"""
	returns chi value of seed set S
	:param G: netowrkx weighted directed graphs, where weights are spread probabilities
	:param S: seed set
	:return:
	"""
	result = 0
	for s in S:
		Cs = set(neighbours(G, s))
		Cs = Cs.difference(S)
		for c in Cs:
			Cc = set(neighbours(G, c))
			Cc = Cc.intersection(S).difference({s})
			for d in Cc:
				ps_c = G[s][c]["weight"]
				pc_d = G[c][d]["weight"]
				result += ps_c*pc_d
	return result


def two_hop_spread(G, A):
	"""
	two hop spread of initial seed set S under IC or WC propagation models
	:param G: networkx graph
	:param A: initial seed set, list
	:return:
	"""
	# just to keep the paper notation
	S = A
	k = len(S)
	result = 0
	result -= _chi(G, S)
	if k == 1:
		result += 1
		s = list(S)[0]
		Cs = set(neighbours(G, s))
		for c in Cs:
			sigma_c = one_hop_spread_single_node(G, c)
			ps_c = G[s][c]["weight"]
			pc_s = G[c][s]["weight"]
			result += ps_c*(sigma_c-pc_s)
	else:
		for s in S:
			result += two_hop_spread(G, [s])
			Cs = set(neighbours(G, s))
			Cs = Cs.intersection(S)
			for c in Cs:
				sigma_c = one_hop_spread_single_node(G, c)
				ps_c = G[s][c]["weight"]
				pc_s = G[c][s]["weight"]
				result -= ps_c*(sigma_c-pc_s)

	return result


if __name__ == "__main__":
	import networkx as nx
	G = nx.generators.random_graphs.barabasi_albert_graph(100, 5, seed=0)
	A = list(range(0, 20, 2))
	from utils import add_weights_IC, add_weights_WC
	two_h1 = two_hop_spread(add_weights_IC(G, 0.4), A)
	two_h2 = two_hop_spread(add_weights_WC(G), A)
	print("IC two-hop spread: ", two_h1)
	print("WC two-hop spread: ", two_h2)


