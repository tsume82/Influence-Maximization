#!/usr/bin/env bash

# get the datasets used in this repo from snap

# wiki-Vote
wget http://snap.stanford.edu/data/wiki-Vote.txt.gz
gunzip wiki-Vote.txt.gz

# amazon
wget http://snap.stanford.edu/data/amazon0302.txt.gz
gunzip amazon0302.txt.gz

# epinions
wget http://snap.stanford.edu/data/soc-Epinions1.txt.gz
gunzip soc-Epinions1.txt.gz

# CA-GrQc
wget http://snap.stanford.edu/data/ca-GrQc.txt.gz
gunzip ca-GrQc.txt.gz