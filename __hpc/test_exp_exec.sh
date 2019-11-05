#!/bin/bash

# set the number of nodes and the RAM and the core used by th nodes
# In the example 4 cores and 10 GB of RAM for each of the 1 nodes

#PBS -l select=1:ncpus=4:mem=10gb

# set max execution time
#PBS -l walltime=2:00:0

# set the queue
#PBS -q short_cpuQ
module load python-3.7.2
pip3 install inspyred
pip3 install networkx
pip3 install numpy
cd Influence-Maximization/src/
python3 experiments.py --exp_dir=../experiments/test --hpc=True
