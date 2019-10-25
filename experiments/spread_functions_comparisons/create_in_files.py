import os
import json

from src.utils import make_dict_read_only


def default_values_dict():
	args = dict()
	args["k"] = 5
	args["p"] = 0.3
	# args["spread_function"] =
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
Pop_size = [8, 16, 32, 64, 128]
Max_gen = [5, 10, 15, 20, 30]
MC_simulations = [10, 50, 100, 500, 1000]


vars2compare = ['p', 'k', 'population_size', 'max_generations', 'no_simulations']
values = [P, K, Pop_size, Max_gen, MC_simulations]
spread_functions = ["monte_carlo", "monte_carlo_max_hop", "two_hop"]
models = ["IC", "WC"]
graph_types = ["barabasi_albert", "gaussian_random_partition"]
for i, var in enumerate(vars2compare):
	for value in values[i]:
		for spr_func in spread_functions:
			for model in models:
				for seed in random_seeds:
					for graph_t in graph_types:
						args = default_values_dict().get_copy()
						args[var] = value
						if var == 'population_size':
							args['offspring_size'] = value
						args["spread_function"] = spr_func
						args["model"] = model
						args["out_name"] = "{}_{}_".format(var, value)
						args["random_seed"] = seed
						args["g_type"] = graph_t

						# write out_file
						in_dir = "./in/" + graph_t + "/" + var + "/" + model + "/" + spr_func + "/" + "{}".format(value)
						if not os.path.exists(in_dir):
							os.makedirs(in_dir)

						data = dict()
						data["script"] = script
						data["n_repetitions"] = n_repetitions
						data["script_args"] = args

						with open(in_dir + '/' + 'seed_{}_'.format(seed) + 'exp_in.json', 'w') as outfile:
							json.dump(data, outfile, indent=4)