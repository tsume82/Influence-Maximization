import unittest

from src.utils import args2cmd


class TestMultiprocessing(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super(TestMultiprocessing, self).__init__(*args, **kwargs)
		self.args = dict()
		self.args["k"] = 5
		self.args["p"] = 0.1
		self.args["no_simulations"] = 10
		self.args["model"] = "IC"
		self.args["population_size"] = 16
		self.args["offspring_size"] = 16
		self.args["random_seed"] = 0
		self.args["max_generations"] = 10
		self.args["n_parallel"] = 1
		self.args["multithread"] = False
		self.args["g_nodes"] = 100
		self.args["g_new_edges"] = 3
		self.args["g_type"] = "barabasi_albert"
		self.args["g_seed"] = 0
		self.args["out_file"] = "res.csv"

		self.script = "./src/evolutionaryalgorithm.py"

	def test_results(self):
		import subprocess
		import time
		import filecmp

		models = ["IC"]

		for model in models:
			self.args["model"] = model
			# single process single threaded

			# multiprocess
			self.args["n_parallel"] = 4
			self.args["out_file"] = "res1.csv"
			cmd = args2cmd(self.args, self.script)
			print(cmd)
			subprocess.call(cmd.split())

			# single process
			self.args["n_parallel"] = 1
			self.args["out_file"] = "res2.csv"
			cmd = args2cmd(self.args, self.script)
			subprocess.call(cmd.split())

			# 5 seconds should be enough
			time.sleep(5)

			files_equal = filecmp.cmp('res1.csv', 'res2.csv')

			subprocess.call("rm res1.csv res2.csv".split())

			self.assertTrue(files_equal)
