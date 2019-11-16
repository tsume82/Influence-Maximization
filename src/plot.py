"""
plotting of the experiments output
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
from utils import traverse_level


def invert_levels(dir, level1, level2):
	"""
	invert two directory levels and changes all files paths, level1 and level2 must be two consecutive levels
	:param dir:
	:param level1: int indicating relative level1 with respect to dir
	:param level2: int indicating relative level2 with respect to dir
	:return:
	"""
	if (level2 - level1) != 1:
		raise NotImplementedError

	lev1 = traverse_level(dir, level1 - 1)
	for dir1, sub_dir_rel1, _ in lev1:
		for sub_dir_r1 in sub_dir_rel1:
			lev2 = traverse_level(dir1 + "/" + sub_dir_r1, 0)
			for dir2, sub_dir_rel2, _ in lev2:
				for sub_dir_r2 in sub_dir_rel2:
					# create reverse structure
					# lev1/lev2 --> lev2/lev1
					inverted_dir = dir1 + "/" + sub_dir_r2 + "/" + sub_dir_r1
					if not os.path.exists(inverted_dir):
						os.makedirs(inverted_dir)
					non_inverted_dir = dir1 + "/" + sub_dir_r1 + "/" + sub_dir_r2
					# move all subdirectories in the inverted directory
					lev3 = traverse_level(non_inverted_dir, 1)
					for dir3, _, _ in lev3:
						shutil.move(dir3, inverted_dir)
					# remove old structure dirs
				shutil.rmtree(dir2)


def collect_results(out_dir, csv_prefix="", generations_stats=False):
	"""
	merges all csv files in all subdirectories of out_dir recursively in unique pandas
	dataframe
	:param out_dir:
	:param file_prefix: only files with this prefix will be merged
	:return: pandas dataframe
	"""
	all_files = []
	for sub_dir_path, sub_dir_rel_path, files in os.walk(out_dir):
		for f in files:
			if (csv_prefix in f): #and ".csv" in f):
				all_files.append(sub_dir_path + "/" + f)
	li = []

	for filename in all_files:

		df = pd.read_csv(filename, index_col=None, header=0)
		if generations_stats:
			path = filename.replace(filename.split("/")[-1], "")
			path_files = os.listdir(path)
			gen_file = None

			for f in path_files:
				if f.startswith("generations"):
					gen_file = f
			df_gen = pd.read_csv(path + gen_file, index_col=None, header=0)
			diversity = df_gen["diversity"].mean()
			improvement = df_gen["improvement"].mean()
			df["diversity"] = diversity
			df["improvement"] = improvement

		li.append(df)

	frame = pd.concat(li, axis=0, ignore_index=True)
	return frame


def plt_subplot(df, subplot_row, subplot_col, rows, cols, var2plot, y, F=None, title=""):
	"""
	plots a subplot given the dataframe, variable to plot
	:param df:
	:param subplot_row:
	:param subplot_col:
	:param rows:
	:param cols:
	:param var2plot:
	:param y:
	:param F:
	:return:
	"""
	# x = sorted(df[var2plot].unique())
	if type(y) is dict:
		y_label = list(y.keys())[0]
		Y = list(y.values())[0]
		print(y_label)
		print(Y)

	else:
		Y = [y]
		y_label = y

	for y in Y:
		if F is not None:
			f = F[0]
			V = F[1]
		else:
			f = None
			V = [None]
		for v in V:
			ax = plt.subplot(rows, cols, subplot_row * cols + subplot_col)
			ax.title.set_text(title)
			if f is not None:
				data2plot = df[df[f] == v]
			else:
				data2plot = df

			x = sorted(data2plot[var2plot].unique())

			mean = data2plot.groupby(var2plot)[y].mean()[x].to_numpy()
			std = data2plot.groupby(var2plot)[y].std()[x].to_numpy()
			if v is not None:
				plt.plot(x, mean, label=v)
			elif len(Y)>1:
				plt.plot(x, mean, label=y)
			else:
				plt.plot(x, mean)
			plt.fill_between(x, mean - std, mean + std, alpha=0.2)
			plt.ylabel(y_label)
			plt.xlabel(var2plot)
			plt.legend(loc='best')


def plot_dir(out_dir, levels_function=[], name="", subplot_row=None, rows=None, cols=None, csv_prefix="", sub_Y=[],
			 F=None, x2plot=None, res_dir=".", generations_stats=False):
	"""

	:param out_dir:
	:param levels_function: for each level specify function among {"separate", "subplot"}
	:return:
	"""
	if len(levels_function) > 0:
		# check level function
		level_function = levels_function[0]
		if "separate" in level_function:
			# plot separately all subdirectories
			sub_dirs = traverse_level(out_dir, 1)
			for sub_dir, _, _ in sub_dirs:
				if "x" in level_function:
					x2plot = sub_dir.replace(out_dir + "/", "")
				plot_dir(sub_dir, levels_function[1:], name + "_" + sub_dir.replace(out_dir + "/", ""),
						 csv_prefix=csv_prefix, sub_Y=sub_Y, F=F, x2plot=x2plot, res_dir=res_dir)

		if "subplot" in level_function:
			# plot as subplots all subdirectories
			sub_dirs = traverse_level(out_dir, 1)
			if name[0] == "_":
				name = name[1:]
			plt.suptitle(name, color='red', x=0.5, y=0.995)
			# each row corresponds to a subdirectory
			rows = len(sub_dirs)

			for i, (sub_dir, dirs, _) in enumerate(sub_dirs):
				# assumption for now: we cannot have two subsequent subplot levels
				if len(levels_function) > 1:
					print("check next level for x variable? under which scenario accepted next level?")
					raise NotImplementedError
				# print(sub_dir)
				# print(i)
				plot_dir(sub_dir, levels_function[1:], name + "_" +  sub_dir.replace(out_dir + "/", ""), subplot_row=i, rows=rows,
						 cols=cols,
						 csv_prefix=csv_prefix, sub_Y=sub_Y, F=F, x2plot=x2plot, res_dir=res_dir)

			if not os.path.exists(res_dir):
				os.makedirs(res_dir)
			plt.tight_layout(h_pad=5)
			plt.savefig(res_dir + "/" + name + ".pdf")
			plt.show()

	else:
		# data to print
		df = collect_results(out_dir, csv_prefix, generations_stats)
		if subplot_row is not None:
			cols = 1 if len(sub_Y) == 0 else len(sub_Y)
			for j, y in enumerate(sub_Y):
				plt_subplot(df, subplot_row, j + 1, rows, cols, x2plot, y, F, title=name)
		# plot_subplot(df, name, rows=rows, subplot_row=subplot_row, sub_Y=sub_Y, F=F, x2plot=x2plot)
		else:
			if name[0] == "_":
				name = name[1:]
			plt.suptitle(name, color='red', x=0.5, y=0.995)
			rows = len(sub_Y)
			cols = 1
			for i, y in enumerate(sub_Y):
				plt_subplot(df, i, subplot_col=1, rows=rows, cols=cols, var2plot=x2plot, y=y, F=F)
			if not os.path.exists(res_dir):
				os.makedirs(res_dir)
			plt.tight_layout(h_pad=5)
			plt.savefig(res_dir + "/" + name + ".pdf")
			plt.show()
		return


if __name__ == "__main__":
	# invert_levels("../experiments/smart_initialization_comparison/out/barabasi_albert", 2, 3)

	# plot_dir("../experiments/test", x2plot="k", csv_prefix="log", sub_Y=["exec_time", "best_mc_spread"],
	# 		 F=("spread_function", ["monte_carlo"]))

	# )
	# plot_dir(out_dir="../experiments/smart_initialization_comparison/out/barabasi_albert",
	# 		 levels_function=["separate_x", "subplot"], csv_prefix="log", sub_Y=["best_mc_spread"],
	# 		 F=("smart_initialization", ["betweenness", "closeness", "degree",
	# 									 "second_order"]),
	# 		 res_dir="../experiments/smart_initialization_comparison/out/plots2")

	plot_dir(out_dir="../experiments/spread_functions_correlation/out",
			 levels_function=["separate","separate_x", "separate"],
			 csv_prefix="log",
			 sub_Y=["mc_th_corr", "mc_mcmh_corr", "mc_std"],
			 res_dir="../experiments/spread_functions_correlation/out/plots")

