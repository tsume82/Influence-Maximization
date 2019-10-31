FROM python:3.7

RUN mkdir -p /Influence-Maximization
RUN pip3 install inspyred
RUN pip3 install networkx
RUN pip3 install numpy
RUN pip3 install pandas

WORKDIR /Influence-Maximization/src

COPY ./ /Influence-Maximization

ENTRYPOINT python /Influence-Maximization/src/experiments.py --exp_dir=../experiments/spread_functions_correlation
#ENTRYPOINT rm -r /Influence-Maximization/experiments/smart_initialization_comparison/out

# docker build . -t inf-max
# docker start inf-max
# docker ps -a
# docker stop inf-max
# docker rm inf-max
# docker rm -f $(docker ps -aq)
# docker run -it --rm -v $(pwd):/Influence-Maximization/:rw --user $(id -u):$(id -g) inf-max
