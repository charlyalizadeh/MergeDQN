import numpy as np
import networkx as nx

def nintersect(c1, c2):
    return sum([c in c2 for c in c1])

def get_clique_tree(cliques):
    clique_graph = nx.Graph()
    keys = list(cliques.keys())
    edge_id = 0
    for i in range(len(keys) - 1):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            clique_graph.add_node(i, id=k1)
            clique_graph.add_node(j, id=k2)
            weight = nintersect(cliques[k1], cliques[k2])
            if weight != 0:
                clique_graph.add_edge(i, j, weight=weight, id=str(edge_id), src=k1, dst=k2)
                edge_id += 1
    clique_tree = nx.maximum_spanning_tree(clique_graph)
    return clique_tree

def neighbors(adj, v, exclude=[]):
	return [i for i in range(adj.shape[0]) if (adj[v, i] != 0 or adj[i, v] != 0) and not (i in exclude)]

def compute_merge_cost(c1, c2):
    di = len(c1)
    dk = len(c2)
    sik = len(np.intersect1d(c1, c2))
    dik = di + dk - sik
    delta_ik = dik * (2 * dik + 1) - di * (2 * di + 1) - dk * (2 * dk + 1) - sik * (2 * sik + 1)
    return delta_ik

def compute_merge_cost_all(cliques, clique_tree):
    costs = []
    for i in range(clique_tree.shape[0]-1):
        for k in range(i+1, clique_tree.shape[0]):
            if clique_tree[i, k] != 0:
                costs.append([i, k, compute_merge_cost(cliques[i], cliques[k])])
    return costs
