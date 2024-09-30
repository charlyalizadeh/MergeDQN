import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from merge import get_clique_tree, compute_merge_cost, nintersect
import matplotlib.pyplot as plt
from math import exp
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def read_graph(path):
    return np.loadtxt(path, dtype=int, delimiter=",")

"""
    Set an attribute named "id" for each node corresponding to their order
"""
def set_node_id(graph):
    attributes = {i: str(i) for i in range(graph.number_of_nodes())}
    nx.set_node_attributes(graph, attributes, name="id")


class CliqueTreeState:
    def __init__(self, graph_path):
        # Retrieve cliques
        self.graph = read_graph(graph_path)
        self.nxgraph = nx.from_numpy_array(self.graph)
        set_node_id(self.nxgraph)
        self.cliques = list(nx.find_cliques(self.nxgraph))
        self.cliques = {str(i):c for i, c in enumerate(self.cliques)}

        # Setup cliquetree
        self.nxclique_tree = get_clique_tree(self.cliques)

        # Setup edge list (constant)
        self.edges_const = [(e[2]["src"], e[2]["dst"]) for e in self.nxclique_tree.edges(data=True)]
        self.true_id = {str(i):str(i) for i, c in enumerate(self.cliques)}

    def get_clique(self, index):
        return self.cliques[self.true_id[index]]

    def get_clique_array(self):
        return list(self.cliques.values())

    """
    Each action corresponds to an edge in the original cliquetree. However the cliquetree
    evolves after each action taken, so we need to keep track of where the original cliques
    are in the new cliques dictionnary.
    To do so we use `self.true_id` which takes as input a node id in the original cliquetree
    and return the current clique id containing this orignal clique.
    """
    def update(self, action):
        i, k = self.edges_const[action]
        clique = list(np.union1d(self.get_clique(i), self.get_clique(k)))
        self.cliques[self.true_id[i]] = clique
        self.cliques[self.true_id[k]] = clique
        if int(i) < int(k):
            self.true_id[k] = self.true_id[i]
        else:
            self.true_id[i] = self.true_id[k]
        return clique

    def get_number_of_cliques(self):
        return len(set(self.true_id.values()))

class CliqueTreeEnv(gym.Env):
    def __init__(self, env_config):
        self.nneighbor = env_config["nneighbor"]
        self.nfeature = 1

        self.graph_path = env_config["graph_path"]
        self.state = CliqueTreeState(self.graph_path)
        self.nedge = self.state.nxclique_tree.number_of_edges()
        self.stop_treshold = int(0.5 * len(self.state.cliques))
        self.action_done = []
        self.total_reward = 0
        self.all_rewards = []

        # Use to divide every reward by the first reward
        self.first_reward_set = False
        self.first_reward = 1

        self.observation_space = spaces.Box(
            low=-1000.0,
            high=2**63 - 2,
            shape=(self.nedge * (self.nneighbor + 1) * self.nfeature,),
            dtype=np.int64
        )
        self.observation = []
        for e in self.state.edges_const:
            self.observation.extend([
                len(self.state.cliques[e[0]]),
                len(self.state.cliques[e[1]]),
                nintersect(self.state.cliques[e[0]], self.state.cliques[e[1]])
            ])
        self.observation = np.array(self.observation)
        self.action_space = spaces.Discrete(self.nedge)
        self.action_done = []

    def update_observation(self):
        for i, e in enumerate(self.state.edges_const):
            if i in self.action_done:
                self.observation[i * 3] = 0
                self.observation[i * 3 + 1] = 0
                self.observation[i * 3 + 2] = 0
                continue
            c1 = self.state.cliques[self.state.true_id[e[0]]]
            c2 = self.state.cliques[self.state.true_id[e[1]]]
            self.observation[i * 3] = len(c1)
            self.observation[i * 3 + 1] = len(c2)
            self.observation[i * 3 + 2] = nintersect(c1, c2)

    def step(self, action):
        if action in self.action_done:
            return self.observation, 0, False, False, {}
        self.action_done.append(action)
        i, k = self.state.edges_const[action]
        molzahn = compute_merge_cost(self.state.get_clique(i), self.state.get_clique(k))
        reward = -molzahn
        if not self.first_reward_set:
            self.first_reward = reward
            self.first_reward_set = True
        reward /= self.first_reward
        self.all_rewards.append(reward)
        self.total_reward += reward
        self.state.update(action)
        self.update_observation()
        terminated = self.state.get_number_of_cliques() <= self.stop_treshold or len(self.action_done) == self.nedge
        return self.observation, reward, terminated, False, {}

    def reset(self):
        self.action_done = []
        self.all_rewards = []
        self.total_reward = 0
        self.first_reward_set = False
        self.first_reward = 1
        self.state = CliqueTreeState(self.graph_path)
        return self.observation, {}

    def apply_model(self, model):
        self.reset()
        while self.state.get_number_of_cliques() >= self.stop_treshold:
            print(f"{self.state.get_number_of_cliques()=}")
            observation = torch.tensor(self.observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = model(observation).max(1).indices.view(1, 1)
            print(f"{action=}")
            if action in self.action_done:
                continue
            self.action_done.append(action)
            self.state.update(action)
            self.update_observation()
        return self.state.get_clique_array()
