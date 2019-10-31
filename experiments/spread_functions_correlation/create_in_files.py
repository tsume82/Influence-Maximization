import os
import json

from src.utils import make_dict_read_only


def default_values_dict():
	args = dict()
	args["k"] = 5
	args["p"] = 0.2
	args["n"] = 100
	args["no_simulations"] = 100
	args["max_hop"] = 2
	args["random_seed"] = 42
	args["g_nodes"] = 1000
	args["g_new_edges"] = 2
	args["g_type"] = 'barabasi_albert'
	args["g_seed"] = 0

	return make_dict_read_only(args)


n_repetitions = 1
random_seeds = range(10)

script="spread_computation.py"

K = range(1, 22, 2)
G_nodes = [50, 100, 200, 500, 1000, 2000]

vars = ["k", "g_nodes"]
values = [K, G_nodes]

for i, var in enumerate(vars):
	for value in values[i]:
		for model in ["IC", "WC"]:
			for seed in random_seeds:
				args = default_values_dict().get_copy()
				args[var] = value
				args["model"] = model
				args["random_seed"] = seed
				in_dir = "./in/" + args["g_type"] + "/" + var + "/" + model + "/" + "{}".format(value)
				if not os.path.exists(in_dir):
					os.makedirs(in_dir)

				data = dict()
				data["script"] = script
				data["n_repetitions"] = n_repetitions
				data["script_args"] = args

				with open(in_dir + '/' + 'seed_{}_'.format(seed) + 'exp_in.json', 'w') as outfile:
					json.dump(data, outfile, indent=4)

# WC specific
G_new_edges = range(1, 20, 2)

for g_new_edges in G_new_edges:
	for seed in random_seeds:
		args = default_values_dict().get_copy()
		args["g_new_edges"] = g_new_edges
		args["random_seed"] = seed
		args["model"] = "WC"
		in_dir = "./in/" + args["g_type"] + "/" + "g_new_edges" + "/" + "WC" + "/" + "{}".format(g_new_edges)
		if not os.path.exists(in_dir):
			os.makedirs(in_dir)

		data = dict()
		data["script"] = script
		data["n_repetitions"] = n_repetitions
		data["script_args"] = args

		with open(in_dir + '/' + 'seed_{}_'.format(seed) + 'exp_in.json', 'w') as outfile:
			json.dump(data, outfile, indent=4)


# IC specific
P = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for p in P:
	for seed in random_seeds:
		args = default_values_dict().get_copy()
		args["p"] = p
		args["random_seed"] = seed
		args["model"] = "IC"

		in_dir = "./in/" + args["g_type"] + "/" + "p" + "/" + "IC" + "/" + "{}".format(p)
		if not os.path.exists(in_dir):
			os.makedirs(in_dir)

		data = dict()
		data["script"] = script
		data["n_repetitions"] = n_repetitions
		data["script_args"] = args

		with open(in_dir + '/' + 'seed_{}_'.format(seed) + 'exp_in.json', 'w') as outfile:
			json.dump(data, outfile, indent=4)

