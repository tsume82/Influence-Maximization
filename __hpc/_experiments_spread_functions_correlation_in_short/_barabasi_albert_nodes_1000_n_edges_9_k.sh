#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb

# set max execution time
#PBS -l walltime=6:00:0

# set the queue
#PBS -q short_cpuQ
module load python-3.7.2
pip3 install inspyred
pip3 install networkx
pip3 install numpy
cd Influence-Maximization/src/
python3 experiments.py --exp_dir=../experiments/spread_functions_correlation/in/barabasi_albert_nodes_1000_n_edges_9/k --hpc=True
	