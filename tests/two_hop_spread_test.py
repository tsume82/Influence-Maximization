import unittest
import networkx as nx

from src.spread.two_hop import two_hop_spread, one_hop_spread_single_node
from src.utils import add_weights_WC, add_weights_IC


class TestTwoHopSpread(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super(TestTwoHopSpread, self).__init__(*args, **kwargs)
		G = nx.generators.random_graphs.barabasi_albert_graph(5, 3, seed=0)
		self.G_IC = add_weights_IC(G, 0.1)
		self.G_WC = add_weights_WC(G)

		G2 = nx.generators.random_graphs.barabasi_albert_graph(7, 2, seed=1)
		self.G2_IC = add_weights_IC(G2, 0.1)
		self.G2_WC = add_weights_WC(G2)

	def test_one_hop(self):
		one_hop_spr = one_hop_spread_single_node(self.G_IC, 4, 0.1, "IC")
		# eliminate error due to machine approximation
		one_hop_spr = round(one_hop_spr, 5)
		self.assertEqual(one_hop_spr, 1.3)

		one_hop_spr = one_hop_spread_single_node(self.G_WC, 4, 0.1, "WC")
		one_hop_spr = round(one_hop_spr, 5)
		self.assertEqual(one_hop_spr, 2.25)

	def test_two_hop(self):
		S = [4, 1]
		two_hop_spr = two_hop_spread(self.G_IC, S, 0.1, "IC")
		two_hop_spr = round(two_hop_spr, 5)
		self.assertEqual(two_hop_spr, 2.46)

		two_hop_spr = two_hop_spread(self.G_WC, S, 0.1, "WC")
		two_hop_spr = round(two_hop_spr, 5)
		self.assertEqual(two_hop_spr, 4.25)

		S2 = [0, 2, 5]
		two_hop_spr = two_hop_spread(self.G2_IC, S2, 0.1, "IC")
		two_hop_spr = round(two_hop_spr, 5)
		self.assertEqual(two_hop_spr, 3.69)

		two_hop_spr = two_hop_spread(self.G2_WC, S2, 0.1, "WC")
		two_hop_spr = round(two_hop_spr, 5)
		self.assertEqual(two_hop_spr, 5.95833)
