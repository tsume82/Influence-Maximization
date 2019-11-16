"""
splits jobs of the desired subdirectory level
"""
import os
from src.utils import traverse_level

in_dir = "../experiments/smart_initialization_comparison/in"
split_level = 2

sub_directories = []
level_dirs = traverse_level(in_dir, split_level)
for lev_dir, _, _ in level_dirs:
	sub_directories.append(lev_dir)


directory = in_dir.replace("..", "").replace("/", "_")
if not os.path.exists(directory):
	os.makedirs(directory)

n_cpus = 4
mem = 24
walltime = 24
queue = "common"

for sub_dir in sub_directories:
	shell_text = """#!/bin/bash

#PBS -l select=1:ncpus={}:mem={}gb

# set max execution time
#PBS -l walltime={}:00:0

# set the queue
#PBS -q {}_cpuQ
module load python-3.7.2
pip3 install inspyred
pip3 install networkx
pip3 install numpy
cd Influence-Maximization/src/
python3 experiments.py --exp_dir={} --hpc=True
	""".format(n_cpus, mem, walltime, queue, sub_dir)
	filename = sub_dir.replace(in_dir, "").replace("/", "_") + ".sh"
	with open(directory + "/" + filename, "w") as f:
		f.write(shell_text)
