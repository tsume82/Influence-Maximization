import logging
import networkx as nx

""" Graph loading """


def read_graph(filename, directed=False, nodetype=int):
	graph_class = nx.DiGraph() if directed else nx.Graph()
	G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)
	return G


if __name__ == '__main__':
	logger = logging.getLogger('')
	logger.setLevel(logging.DEBUG)

	filename = "../graphs/wiki-Vote.txt"
	directed = True
	G = read_graph(filename, directed=directed)

	msg = ' '.join(["Read from file", filename, "the", "directed" if directed else "undirected", "graph\n",
					nx.classes.function.info(G)])

	logging.info(msg)
