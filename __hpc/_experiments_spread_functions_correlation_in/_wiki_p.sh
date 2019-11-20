#!/bin/bash

#PBS -l select=1:ncpus=4:mem=24gb

# set max execution time
#PBS -l walltime=24:00:0

# set the queue
#PBS -q common_cpuQ
module load python-3.7.2
pip3 install inspyred
pip3 install networkx
pip3 install numpy
cd Influence-Maximization/src/
python3 experiments.py --exp_dir=../experiments/spread_functions_correlation/in/wiki/p --hpc=True
	