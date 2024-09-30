# MergeDQN

Code for experimenting with merging using DQN algorithms.

## Code structure

* `data/`: contains the data use as input for the algorithm. (To generate those file see: [OPFSDP](https://github.com/charlyalizadeh/OPFSDP.jl)
* `cliquetree_env.py`: gymnasium environment for the merging
* `dqn.py`: DQN class [source](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* `train_model.py`: code for the training [source](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* `merge.py`: code use to merge two cliques
* `apply_model.py`: code to apply an already trained model to a decomposition
