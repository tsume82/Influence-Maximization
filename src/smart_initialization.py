import networkx as nx
import community
import numpy as np


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


class Community_initialization:
	def __init__(self, G, random_seed=None):
		self.G_original = G.copy().to_undirected()
		self.G = self.G_original.copy()
		self.prng = np.random.RandomState(random_seed)
		self.comm_part = community.best_partition(self.G, random_state=self.prng)
		comm_idxs = set(self.comm_part.values())
		# dictionary containing community indexes as keys and graph labels as values
		self.communities_original = dict()
		for comm_idx in comm_idxs:
			self.communities_original[comm_idx] = np.array([k for k, v in self.comm_part.items() if v == comm_idx])

		self.degree = None

	# def _spectral_clustering(self, G, n_clusters):

	def _get_random_degree_node(self, nodes, n):
		"""
		returns n randomly selected nodes from the given nodes, a probability of a node to be chosen is proportional
		to its degree
		:param nodes:
		:return:
		"""
		# if self.degree is None:
		self.degree = nx.degree_centrality(self.G)

		nodes_degrees = np.array([self.degree[node] for node in nodes])

		probabilities = nodes_degrees / sum(nodes_degrees)
		best_degree_nodes = self.prng.choice(nodes, n, p=probabilities)
		return best_degree_nodes

	def get_comm_members_random(self, n, k, degree=False):
		"""
		returns n members randomly chosen from communities, the probability to be taken
		from a community is proportional to its size
		:param n:
		:param degree: if true, for each randomly selected community the probability to select a node is
		proportional to its degree
		:return:
		"""
		result = []
		for i in range(n):
			indiv = []
			# reset values
			self.G = self.G_original.copy()
			self.communities = self.communities_original.copy()
			for _ in range(k):
				# recalculate communities and probabilities
				max = len(self.communities.keys())
				probabilites = np.zeros(len(self.communities.keys()))
				tot_len = 0
				for i in self.communities:
					len_comm = len(self.communities[i])
					tot_len += len_comm
					probabilites[i] = len_comm
				probabilites /= tot_len

				# choose a community
				idx = self.prng.choice(max, 1, p=probabilites)[0]
				# choose a node inside the community
				if degree:
					comm_random_indiv = self._get_random_degree_node(self.communities[idx], 1)[0]
				else:
					comm_random_indiv = self.prng.choice(self.communities[idx])
				indiv.append(comm_random_indiv)

				# remove the node from communities and graph
				self.G.remove_node(comm_random_indiv)
				if len(self.communities[idx]) == 1:
					del self.communities[idx]
				else:
					self.communities[idx] = np.delete(self.communities[idx], np.where(self.communities[idx] == comm_random_indiv))
			result.append(indiv)
		return result


if __name__ == "__main__":
	G = nx.generators.barabasi_albert_graph(100, 2, 0)
	S = max_centrality_individual(4, G, "second_order")
	# print(S)
	C_i = Community_initialization(G, 2)
	inds = C_i.get_comm_members_random(4, 5, degree=True)
	print(inds)

	# print(C_i.sample_min_repetitions(4, 10))
	# print(C_i.get_random_members(2, 3))
	# print(C_i.get_comm_members_degree(probabilistic=True))
