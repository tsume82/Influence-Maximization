import networkx as nx


def max_centrality_individual(k, G, centrality_metric="degree"):
	"""
	returns k nodes with the highest centrality metric
	:param k:
	:param G: networkx graph
	:param centrality_metric: centrality metric string, to be chosen from "degree", "eigenvector", "katz", "closeness",
							  "betweenness", "second_order"
	:return:
	"""
	if centrality_metric == "degree":
		nodes_centrality = nx.degree_centrality(G)
	elif centrality_metric == "eigenvector":
		nodes_centrality = nx.eigenvector_centrality_numpy(G)
	elif centrality_metric == "katz":
		nodes_centrality = nx.katz_centrality_numpy(G)
	elif centrality_metric == "closeness":
		nodes_centrality = nx.closeness_centrality(G)
	elif centrality_metric == "betweenness":
		nodes_centrality = nx.betweenness_centrality(G)
	elif centrality_metric == "second_order":
		nodes_centrality = nx.second_order_centrality(G)

	sorted_nodes_centrality = dict(sorted(nodes_centrality.items(), key=lambda nodes_centrality: nodes_centrality[1],
										  reverse=True))
	return list(sorted_nodes_centrality)[:k]


if __name__ == "__main__":
	G = nx.generators.barabasi_albert_graph(10, 2, 0)
	S = max_centrality_individual(4, G, "second_order")
	print(S)