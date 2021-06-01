# hash:sha256:fcd9e6079c07a24ac3f7f633cf4e450fa2cb77ea260f546ad54bb1e6605a0454
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget=1.20.3-1ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    community==1.0.0b1 \
    cython==0.29.23 \
    gensim==4.0.1 \
    inspyred==1.0.1 \
    matplotlib==3.4.2 \
    networkx==2.5.1 \
    node2vec==0.4.3 \
    numpy==1.20.3 \
    pandas==1.2.4 \
    python-levenshtein==0.12.2 \
    sklearn==0.0
