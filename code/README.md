# Influence-Maximization
Code and data for experiments with evolutionary influence maximization

This repository contains source code and some experimental scripts for experimental setups. 
The inputs, outputs, plots and other experiment-specific files are contained in another subrepository: 
[experiments](https://github.com/katerynak/Influence-Maximization-experiments).

Influence-Maximization repository is organized as follows: 

![repo_organization](figures/repo_organization.png)

* src: source code containing code of the implemented features
* src_OLD: old source code in case it comes at hand
* tests: some tests needed to check whether some implemented functions behave properly
* experiments: subrepository containing inputs and outputs of each experimental setup
* datasets: contains script that downloads datasets

The src module contains the following submodules:

* ea: code related to each of the components of the evolutionary algorithm
* spread: implementations of fitness functions approximations
* spread_pyx: cython code and compilation script
* nodes_filtering: implementation of different nodes filtering strategies
* additional_experimental_scripts: experimental setup scripts of all old experiments

and files:
* evolutionary_algorithm_exec: script containing the execution pipeline: graph loading, 
nodes filtering, smart initialization, execution of the ea algorithm with appropriate parameters,
 output logging..
 * heuristics: implementation of other heuristics we compare with
 * multi_armed_bandit: implementation of the UCB1 algorithm for multi-armed-bandit problem (used for 
 dynamic mutation selection)
 * smart_initialization: implementation of different smart initialization techniques
 * utils: mix of helpful functions
 * experiments: script used for execution of multiple experiments
 * load.py: graph loading code
 * mo_evolutionaryalgorithm: multi-objective old evolutionary algorithm code
 

To run the whole pipeline:

``
cd Influence-Maximization;
python -m src.evolutionary_algorithm_exec --config_file src/example_experimental_setup.json
``

You need to specify all the hyperparameters of the algorithm in a .json config
file and pass it as a parameter of the script. The results are stored in a
directory specified as a script argument (`--out_dir`).

## Limitations

Pay attention: when you change the `mutation_operator` parameter, currently you
need to add a decorator `@inspyred.ec.variators.mutator` to the selected mutation
function in `src/ea/mutators.py`.

Python and libraries versions currently used:

* python 3.6.8
* inspyred                  1.0.1 
* networkx                  2.3 
* numpy                     1.16.3 
* cython                    0.29.13 
* node2vec                  0.3.1 
* gensim                    3.8.1 
* pandas                    0.24.2 
