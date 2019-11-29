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




# def random_community_member(comm_part, i):
# 	"""
# 	returns random community member given the community partition and community number i
# 	:param comm_part:
# 	:param i:
# 	:return:
# 	"""
# 	# check if community i exists
# 	print (comm_part.values())


class Community_initialization:
	def __init__(self, G, random_seed=None):
		self.G = G.to_undirected()
		self.prng = np.random.RandomState(random_seed)
		self.comm_part = community.best_partition(self.G, random_state=self.prng)
		comm_idxs = set(self.comm_part.values())
		# dictionary containing community indexes as keys and graph labels as values
		self.communities = dict()
		for comm_idx in comm_idxs:
			self.communities[comm_idx] = np.array([k for k, v in self.comm_part.items() if v == comm_idx])

		self.degree=None

	def sample_min_repetitions(self, max, n):
		"""
		samples n numbers from range by repeating a minimum number of times
		:param max:
		:param probabilities:
		:param n:
		:return:
		"""

		chunks = int(n/max)
		result = []
		a = np.arange(max)
		for _ in range(chunks):
			self.prng.shuffle(a)
			result.append(list(a.copy()))
		# resulting chunk
		diff = n - chunks*max
		self.prng.shuffle(a)
		result.append(list(a.copy()[:diff]))
		result = [item for sublist in result for item in sublist]
		return result

	def get_random_members(self, comm_idx, n):
		"""
		get n random members from the community having comm_idx as index
		:param comm_idx:
		:param n:
		:return:
		"""
		return self.prng.choice(self.communities[comm_idx], n)

	def _get_random_degree_node(self, nodes, n):
		"""
		returns n randomly selected nodes from the given nodes, a probability of a node to be chosen is proportional
		to its degree
		:param nodes:
		:return:
		"""
		if self.degree is None:
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
		probabilites = np.zeros(len(self.communities.keys()))
		tot_len = 0
		for i in self.communities:
			len_comm = len(self.communities[i])
			tot_len += len_comm
			probabilites[i] = len_comm

		probabilites /= tot_len
		result = []
		max = len(self.communities.keys())
		for i in range(n):
			indiv = []
			while len(indiv) < k:
				comm_idx = self.prng.choice(max, k, p=probabilites)
				# comm_idx = self.sample_min_repetitions(max, k)
				for idx in comm_idx:
					if degree:
						comm_random_indiv = self._get_random_degree_node(self.communities[idx], 1)[0]
					else:
						comm_random_indiv = self.prng.choice(self.communities[idx])
					if comm_random_indiv not in indiv and len(indiv) < k:
						indiv.append(comm_random_indiv)
			result.append(indiv)
		return result

	def get_comm_members_degree(self, probabilistic=False):
		"""
		returns one member for each community with the highest degree centrality
		:probabilistic: if true returns for each community an individual with probability to be chosen
		correspondent to its degree
		:return:
		"""
		if self.degree is None:
			self.degree = nx.degree_centrality(self.G)
		best_degree_indivs = []
		for k in self.communities:
			comm_nodes = self.communities[k]
			comm_nodes_degrees = np.array([self.degree[comm_node] for comm_node in comm_nodes])
			if probabilistic:
				best_comm_node = self._get_random_degree_node(comm_nodes, 1)[0]
			else:
				best_comm_node = comm_nodes[comm_nodes_degrees.argmax()]
			best_degree_indivs.append(best_comm_node)

		return best_degree_indivs


if __name__ == "__main__":
	G = nx.generators.barabasi_albert_graph(100, 2, 0)
	S = max_centrality_individual(4, G, "second_order")
	# print(S)
	C_i = Community_initialization(G, 2)
	inds = C_i.get_comm_members_random(4, 5, degree=False)
	print(inds)

	# print(C_i.sample_min_repetitions(4, 10))
	# print(C_i.get_random_members(2, 3))
	# print(C_i.get_comm_members_degree(probabilistic=True))

