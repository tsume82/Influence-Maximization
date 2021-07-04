import networkx as nx
import random
import numpy

# in order to avoid redundancy, the models are now taken from the generalized version in monte_carlo.py
from src.spread.monte_carlo import WC_model, IC_model

""" Spread models """

""" Simulation of approximated spread for Independent Cascade (IC) and Weighted Cascade (WC). Nodes activators (distant
        up to two hops) are saved as nodes' attributes.  
        Suits (un)directed graphs. 
        Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
"""


#def IC_model(G, a, p, max_hop, random_generator):  # a: the set of initial active nodes
#        # p: the system-wide probability of influence on an edge, in [0,1]
#        A = set(a)  # A: the set of active nodes, initially a
#        B = set(a)  # B: the set of nodes activated in the last completed iteration
#        converged = False
#        while (not converged) and (max_hop > 0):
#                nextB = set()
#                for n in B:
#                        for m in set(G.neighbors(n)) - A:  # G.neighbors follows A-B and A->B (successor) edges
#                                prob = random_generator.random()  # in the range [0.0, 1.0)
#                                activations = {}
#                                if prob <= p:
#                                        nextB.add(m)
#                                        if m not in activations.keys():
#                                                activations[m] = [n]
#                                        else:
#                                                activations[m].append(n)
#                                        # append also nodes which activated their activators
#                                        if n in activations.keys():
#                                                for act in activations[n]:
#                                                       activations[m].append(act)
#                                                        if act in activations.keys():
#                                                                for act2 in activations[act]:
#                                                                        activations[m].append(act2)
#
#                                        # update the graph
#                                        for a in activations[m]:
#                                                if a not in G.nodes[m]['activated_by'].keys():
#                                                        G.nodes[m]['activated_by'][a] = 1
#                                                else:
#                                                        G.nodes[m]['activated_by'][a] += 1
#
#                B = set(nextB)
#                if not B:
#                        converged = True
#                A |= B
#                max_hop -= 1
#
#        return len(A)


#def WC_model(G, a, max_hop, random_generator):  # a: the set of initial active nodes
#        # each edge from node u to v is assigned probability 1/in-degree(v) of activating v
#        A = set(a)  # A: the set of active nodes, initially a
#        B = set(a)  # B: the set of nodes activated in the last completed iteration
#        converged = False
#
#        if nx.is_directed(G):
#                my_degree_function = G.in_degree
#        else:
#                my_degree_function = G.degree
#
#        while (not converged) and (max_hop > 0):
#                nextB = set()
#                for n in B:
#                        for m in set(G.neighbors(n)) - A:
#                                prob = random_generator.random()  # in the range [0.0, 1.0)
#                                p = 1.0 / my_degree_function(m)
#                                activations = {}
#                                if prob <= p:
#                                        nextB.add(m)
#                                        if m not in activations.keys():
#                                                activations[m] = [n]
#                                        else:
#                                                activations[m].append(n)
#                                        # append also nodes which activated their activators
#                                        if n in activations.keys():
#                                                for act in activations[n]:
#                                                        activations[m].append(act)
#                                                        if act in activations.keys():
#                                                                for act2 in activations[act]:
#                                                                        activations[m].append(act2)
#                                        # update the graph
#                                        for a in activations[m]:
#                                                if a not in G.nodes[m]['activated_by'].keys():
#                                                        G.nodes[m]['activated_by'][a] = 1
#                                                else:
#                                                        G.nodes[m]['activated_by'][a] += 1
#                                        # for a in A:
#
#                                                # if a not in G.nodes[m]['activated_by'].keys():
#                                                #       G.nodes[m]['activated_by'][a] = 1
#                                                # else:
#                                                #       G.nodes[m]['activated_by'][a] += 1
#                B = set(nextB)
#                if not B:
#                        converged = True
#                A |= B
#                max_hop -= 1
#
#        return len(A)


def MonteCarlo_simulation(G, A, p, no_simulations, model, max_hop, random_generator=None):
        """
        calculates approximated influence spread of a given seed set A, with
        information propagation limited to a maximum number of hops
        example: with max_hops = 2 only neighbours and neighbours of neighbours can be activated
        :param G: networkx input graph
        :param A: seed set
        :param p: probability of influence spread (IC model)
        :param no_simulations: number of spread function simulations
        :param model: propagation model
        :param max_hops: maximum number of hops
        :return:
        """
        if random_generator is None:
                random_generator = random.Random()

        results = []

        if model == 'WC':
                for i in range(no_simulations):
                        results.append(WC_model(G, A, random_generator, max_hop=max_hop, mark=True))
        elif model == 'IC':
                for i in range(no_simulations):
                        results.append(IC_model(G, A, p, random_generator, max_hop=max_hop, mark=True))

        return (numpy.mean(results), numpy.std(results))


if __name__ == "__main__":
        G = nx.path_graph(100)
        print(nx.classes.function.info(G))
        init_dict = dict()
        for n in G.nodes():
                init_dict[n] = {}
        nx.set_node_attributes(G, init_dict, name="activated_by")
        print("Simulation using IC model:")
        print(MonteCarlo_simulation(G, [0, 2, 4, 6, 8, 10], 0.7, 100, 'IC', 5, random.Random(0)))
        print("Simulation using WC model:")
        print(MonteCarlo_simulation(G, [0, 2, 4, 6, 8, 10], 0.7, 100, 'WC', 5, random.Random(0)))
