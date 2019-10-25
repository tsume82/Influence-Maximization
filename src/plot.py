"""
plotting of the experiments output
"""

import matplotlib.pyplot as plt
import os
import pandas as pd


def get_path_level(dir, subdir):
	level = subdir.replace(dir, "").count("/")
	return level


def traverse_level(dir, level):
	out = []
	for sub_dir_path, sub_dir_rel_path, files in os.walk(dir):
		lev = get_path_level(dir, sub_dir_path)
		if lev == level:
			out.append([sub_dir_path, sub_dir_rel_path, files])
	return out


def plot_dir(out_dir, levels_function, name="", subplot_row=None, rows= None, cols=None, csv_prefix="", sub_Y=[],
			 F=None, x2plot=None):
	"""

	:param out_dir:
	:param levels_function: for each level specify function among {"separate", "subplot"}
	:return:
	"""
	if len(levels_function) > 0:

		if "separate" in levels_function[0]:
			sub_dirs = traverse_level(out_dir, 1)
			for sub_dir, _, _ in sub_dirs:
				if len(levels_function) == 1:
					next_levels_function = []
				else:
					next_levels_function = levels_function[1:]
				if "x" in levels_function[0]:
					x2plot = sub_dir.replace(out_dir + "/", "")
				plot_dir(sub_dir, next_levels_function, name + "_" + sub_dir.replace(out_dir + "/", ""),
						 csv_prefix=csv_prefix, sub_Y=sub_Y, F=F, x2plot=x2plot)
		if "subplot" in levels_function[0]:
			sub_dirs = traverse_level(out_dir, 1)
			plt.suptitle(name, color='red', x=0.1, y=0.995)
			rows = len(sub_dirs)

			for i, (sub_dir, dirs, _) in enumerate(sub_dirs):
				if len(levels_function) == 1:
					next_levels_function = []
					cols = 1 if len(sub_Y) == 0 else len(sub_Y)
				else:
					next_levels_function = levels_function[1:]
					if "subplot" in levels_function[1]:
						cols = len(dirs)
				plot_dir(sub_dir, next_levels_function, sub_dir.replace(out_dir + "/", ""), subplot_row=i+1, rows=rows,
						 cols=cols,
						 csv_prefix=csv_prefix, sub_Y=sub_Y, F=F, x2plot=x2plot)
			plt.show()

	else:
		# collect resutls
		# concatenate all csv files in all subdirectories
		all_files = []
		for sub_dir_path, sub_dir_rel_path, files in os.walk(out_dir):
			for f in files:
				if (csv_prefix in f and ".csv" in f):
					all_files.append(sub_dir_path + "/" + f)
		# print(all_files)
		li = []

		for filename in all_files:
			df = pd.read_csv(filename, index_col=None, header=0)
			li.append(df)

		frame = pd.concat(li, axis=0, ignore_index=True)

		if subplot_row is not None:
			for j, y in enumerate(sub_Y):
				plt.subplot(rows, cols, (subplot_row-1) * cols + j+1)
				plt.title(name)
				plt.ylabel(y)
				plt.xlabel(x2plot)
				x = frame[x2plot].unique()
				f = F[0]
				V = F[1]
				for v in V:
					data2plot = frame[frame[f] == v]
					mean = data2plot.groupby(x2plot)[y].mean()
					std = data2plot.groupby(x2plot)[y].std()
					plt.plot(x, mean, label=v)
					plt.fill_between(x, mean - std, mean + std, alpha=0.2)
					plt.legend(loc='best')
		return


if __name__ == "__main__":
	plot_dir("../experiments/spread_functions_comparisons/out", ["separate_x", "subplot"], name="comparison",
			 csv_prefix="log", sub_Y=["exec_time", "best_mc_spread"], F=("spread_function", ["monte_carlo", "monte_carlo_max_hop", "two_hop"]) )