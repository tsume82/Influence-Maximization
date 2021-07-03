import networkx as nx
import random
import numpy
import math

""" Spread models """

""" Simulation of spread for Independent Cascade (IC) and Weighted Cascade (WC). 
        Suits (un)directed graphs. 
        Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
"""


def IC_model(G, a, p, random_generator, max_hop=float("inf"), mark=False):  # a: the set of initial active nodes
        # p: the system-wide probability of influence on an edge, in [0,1]
        # max_hop: maximum number of hops (optional), default is infinity
        # mark: boolean flag, whether we should write annotations in nodes (optional), default is False

        A = set(a)  # A: the set of active nodes, initially a
        B = set(a)  # B: the set of nodes activated in the last completed iteration
        converged = False
        hop = 0
        my_degree_function = 0

        # if 'mark' is set to True, let's prepare for annotations
        if mark == True:
            if nx.is_directed(G):
                my_degree_function = G.in_degree
            else :
                my_degree_function = G.degree

        while (not converged) and (hop < max_hop):
                nextB = set()
                for n in B:
                        for m in set(G.neighbors(n)) - A:  # G.neighbors follows A-B and A->B (successor) edges
                                prob = random_generator.random()  # in the range [0.0, 1.0)
                                # this below is a data structure only used if we are marking activations
                                activations = {}
                                    
                                if prob <= p:
                                        nextB.add(m)

                                        # this part below is only activated if we are marking activations
                                        if mark == True:
                                            if m not in activations.keys():
                                                    activations[m] = [n]
                                            else:
                                                    activations[m].append(n)

                                            # append also nodes which activated their activators
                                            if n in activations.keys():
                                                    for act in activations[n]:
                                                            activations[m].append(act)
                                                            if act in activations.keys():
                                                                    for act2 in activations[act]:
                                                                            activations[m].append(act2)

                                            # update the graph
                                            for a in activations[m]:
                                                    if a not in G.nodes[m]['activated_by'].keys():
                                                            G.nodes[m]['activated_by'][a] = 1
                                                    else:
                                                            G.nodes[m]['activated_by'][a] += 1
                B = set(nextB)
                if not B:
                        converged = True
                A |= B
                hop += 1

        return len(A)


def WC_model(G, a, random_generator, max_hop=float("inf"), mark=False):  # a: the set of initial active nodes
        # each edge from node u to v is assigned probability 1/in-degree(v) of activating v

        A = set(a)  # A: the set of active nodes, initially a
        B = set(a)  # B: the set of nodes activated in the last completed iteration
        converged = False
        hop = 0

        if nx.is_directed(G):
                my_degree_function = G.in_degree
        else:
                my_degree_function = G.degree

        while (not converged) and (hop < max_hop):
                nextB = set()
                for n in B:
                        for m in set(G.neighbors(n)) - A:
                                prob = random_generator.random()  # in the range [0.0, 1.0)
                                p = 1.0 / my_degree_function(m)
                                
                                # this data structure below is only used if we are marking activations
                                activations = {}
                            
                                if prob <= p:
                                        nextB.add(m)
                                        if mark == True:
                                            if m not in activations.keys():
                                                    activations[m] = [n]
                                            else:
                                                    activations[m].append(n)
                                            # append also nodes which activated their activators
                                            if n in activations.keys():
                                                    for act in activations[n]:
                                                            activations[m].append(act)
                                                            if act in activations.keys():
                                                                    for act2 in activations[act]:
                                                                            activations[m].append(act2)
                                            # update the graph
                                            for a in activations[m]:
                                                    if a not in G.nodes[m]['activated_by'].keys():
                                                            G.nodes[m]['activated_by'][a] = 1
                                                    else:
                                                            G.nodes[m]['activated_by'][a] += 1

                B = set(nextB)
                if not B:
                        converged = True
                A |= B
                hop += 1

        return len(A)


""" Evaluates a given seed set A, simulated "no_simulations" times.
        Returns a tuple: (the mean, the stdev).
"""


def MonteCarlo_simulation(G, A, p, no_simulations, model, random_generator=None):
        if random_generator is None:
                random_generator = random.Random()

        results = []

        if model == 'WC':
                for i in range(no_simulations):
                        results.append(WC_model(G, A, random_generator))
        elif model == 'IC':
                for i in range(no_simulations):
                        results.append(IC_model(G, A, p, random_generator))

        return (numpy.mean(results), numpy.std(results))


if __name__ == "__main__":
        G = nx.path_graph(100)
        print(nx.classes.function.info(G))
        print("Simulation using IC model:")
        print(MonteCarlo_simulation(G, [0, 2, 4, 6, 8, 10], 0.1, 100, 'IC', random.Random(0)))
        print("Simulation using WC model:")
        print(MonteCarlo_simulation(G, [0, 2, 4, 6, 8, 10], 0.1, 100, 'WC', random.Random(0)))
