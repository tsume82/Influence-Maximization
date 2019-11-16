import os
import json

from src.utils import make_dict_read_only


def default_values_dict():
	args = dict()
	args["k"] = 5
	args["p"] = 0.1
	args["spread_function"] = "monte_carlo_max_hop"
	args["no_simulations"] = 100
	args["max_hop"] = 2
	# args["model"] =
	args["population_size"] = 16
	args["offspring_size"] = 16
	args["random_seed"] = 42
	args["max_generations"] = 10
	args["n_parallel"] = 4
	args["g_nodes"] = 1000
	args["g_new_edges"] = 3
	args["g_type"] = 'barabasi_albert'
	args["g_seed"] = 0
	# args["g_file"] =
	# args["out_file"] =

	# args["out_dir"] =

	return make_dict_read_only(args)


n_repetitions = 1
random_seeds = range(5)

script = "evolutionaryalgorithm.py"

K = [1, 2, 5, 7, 10]
P = [0.1, 0.3, 0.5, 0.7, 0.9]
Smart_init = ["degree", "eigenvector", "katz", "closeness",
							  "betweenness", "second_order", "none"]


vars2compare = ['p', 'k']
values = [P, K]
models = ["IC", "WC"]
graph_types = ["barabasi_albert", "wiki", "epinions", "amazon"]
for smart_init in Smart_init:
	for i, var in enumerate(vars2compare):
		for value in values[i]:
			for model in models:
				for seed in random_seeds:
					for graph_t in graph_types:
						# try with and without smart initialization
						for s_i in [smart_init]:
							args = default_values_dict().get_copy()
							args["smart_initialization"] = s_i
							args[var] = value
							args["model"] = model
							args["out_name"] = "{}_{}_".format(var, value)
							args["random_seed"] = seed
							args["g_type"] = graph_t
							# write out_file
							in_dir = "./in/" + graph_t + "/" + var + "/" + model + "/" + s_i + "/" + "{}".format(value)
							if not os.path.exists(in_dir):
								os.makedirs(in_dir)

							data = dict()
							data["script"] = script
							data["n_repetitions"] = n_repetitions
							data["script_args"] = args

							with open(in_dir + '/' + 'seed_{}_'.format(seed) + 'exp_in.json', 'w') as outfile:
								json.dump(data, outfile, indent=4)