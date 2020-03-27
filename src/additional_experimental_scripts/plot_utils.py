"""
plotting of the experiments output
"""

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


def collect_results(out_dir, csv_prefix="", generations_stats=False, delimiter=','):
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

		df = pd.read_csv(filename, index_col=None, header=0, delimiter=delimiter)
		if generations_stats:
			path = filename.replace(filename.split("/")[-1], "")
			path_files = os.listdir(path)
			gen_file = None

			for f in path_files:
				if f.startswith("generations"):
					gen_file = f
			df_gen = pd.read_csv(path + gen_file, index_col=None, header=0, delimiter=delimiter)
			diversity = df_gen["diversity"].mean()
			improvement = df_gen["improvement"].mean()
			df["diversity"] = diversity
			df["improvement"] = improvement

		li.append(df)

	frame = pd.concat(li, axis=0, ignore_index=True)
	return frame
