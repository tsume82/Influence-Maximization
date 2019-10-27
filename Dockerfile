FROM python:3.7

RUN mkdir -p /Influence-Maximization
RUN pip3 install inspyred
RUN pip3 install networkx
RUN pip3 install numpy

WORKDIR /Influence-Maximization/src

COPY ./ /Influence-Maximization

ENTRYPOINT python /Influence-Maximization/src/experiments.py --exp_dir=../experiments/smart_initialization_comparison

# docker build . -t inf-max
# docker run --name "inf-max" inf-max
# docker start inf-max
# docker ps -a
# docker stop inf-max
# docker rm inf-max
# docker rm -f $(docker ps -aq)
# docker run --rm -v $(pwd):/Influence-Maximization/:rw inf-max