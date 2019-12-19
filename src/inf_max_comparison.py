import networkx as nx
from functools import partial
import random
import time

from heuristics import general_greedy, CELF
from evolutionary_algorithm import ea_influence_maximization
from spread.monte_carlo import MonteCarlo_simulation as monte_carlo
from spread.monte_carlo_max_hop import MonteCarlo_simulation as monte_carlo_max_hop
from utils import load_graph

if __name__=="__main__":
	# G = nx.generators.barabasi_albert_graph(3000, 3, seed=0)
	G = load_graph(g_type="wiki")
	print("wewe")
	k=10
	monte_c = partial(monte_carlo, no_simulations=100, p=0.01, model="IC", G=G)
	monte_c_max_hop = partial(monte_carlo_max_hop, max_hop=2, no_simulations=100, p=0.01, model="IC", G=G)
	prng = random.Random(0)
	start = time.time()
	ea_res = ea_influence_maximization(k, G, monte_c_max_hop, 128, 128, 20, prng)
	stop = time.time()
	print(ea_res)
	print("Evolutionary algorithm: {}".format(monte_c(A=ea_res[0], random_generator=prng)))
	print("Runnging time: {}".format(stop-start))

	# start = time.time()
	# greedy_res = general_greedy(k, G, 0.1, 100, "IC", prng)
	# stop = time.time()
	# print("Greedy res: {}".format(monte_c(A=greedy_res, random_generator=prng)))
	# print("Runnging time: {}".format(stop-start))

	# start = time.time()
	# celf_res = CELF(k, G, 0.1, 100, "IC", prng)
	# stop = time.time()
	# print("CELF res: {}".format(monte_c(A=celf_res, random_generator=prng)))
	# print("Runnging time: {}".format(stop-start))
