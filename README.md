# Adaptive Routing with Hierarchical Reinforcement Learning on Dragonfly Networks

* hierarchical.ipynb is for Q-hierarchical routing test

* shortest.ipynb is for Minimal and Global routing test

* VAL.ipynb is for VAL routing test

* qroute.ipynb is for Q-routing test

* qadapt.ipynb is for Q-adaptive routing test

* The routing algorithms is stored in the models folder

* Trained models is stored in the dump_dragonfly folder

* dragonfly_generator.py provides useful functions for generating and reading different dragonfly topologies

## Model training and testing

To train Q-routing, Q-adaptive, and Q-hierarchical, initialize agent with static=False, i.e. 

```agent = Qroute(network, static=False)```

To test, initialize agent with static=True