from cliquetree_env import CliqueTreeEnv
from dqn import DQN
import torch


env_config = {"nneighbor": 2, "graph_path": "case9_molzahn.txt"}
env = CliqueTreeEnv(env_config)
state, info = env.reset()

model = DQN(len(state), env.action_space.n)
model.load_state_dict(torch.load("model"))

apply_env = CliqueTreeEnv(env_config)
cliques = apply_env.apply_model(model)
print(cliques)
print(f"NUMBER OF CLIQUES: {len(cliques)}")
