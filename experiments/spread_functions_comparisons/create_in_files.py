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
	# args["multithread"] = False
	args["g_nodes"] = 1000
	args["g_new_edges"] = 3
	args["g_type"] = 'barabasi_albert'
	args["g_seed"] = 0
	# args["g_file"] =
	# args["out_file"] =

	# args["out_dir"] =
	args["print_mc_best"] = True

	return make_dict_read_only(args)


n_repetitions = 5
script = "evolutionaryalgorithm.py"


# K = [1, 2, 5, 7, 10]
# P = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

K = [1, 2]
P = [0.1, 0.2]


vars2compare = ['p', 'k']
values = [P, K]
spread_functions = ["monte_carlo", "monte_carlo_max_hop", "two_hop"]
models = ["IC", "WC"]
for i, var in enumerate(vars2compare):
	for value in values[i]:
		for spr_func in spread_functions:
			for model in models:
				args = default_values_dict().get_copy()
				args[var] = value
				args["spread_function"] = spr_func
				args["model"] = model
				args["out_name"] = "{}_{}_".format(var, value)

				# write out_file
				in_dir = "./in/" + var + "/" + model + "/" + "{}".format(value) + "/" + spr_func
				if not os.path.exists(in_dir):
					os.makedirs(in_dir)

				data = dict()
				data["script"] = script
				data["n_repetitions"] = n_repetitions
				data["script_args"] = args

				with open(in_dir + '/' + 'exp_in.json', 'w') as outfile:
					json.dump(data, outfile, indent=4)