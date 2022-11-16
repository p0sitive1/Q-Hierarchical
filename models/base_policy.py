import numpy as np
import pickle


class Policy:
    """
    Policy creates the basis for all learning agents
    """
    mode = None
    attrs = {'links'}

    def __init__(self, network):
        self.links = {k: np.array(v, dtype=np.int)
                      for k, v in network.links.items()}
        self.action_idx = {node:
                           {a: i for i, a in enumerate(neighbors)}
                           for node, neighbors in self.links.items()}
        self.group_num = network.group_num

    def choose(self, source, dest, packet):
        """ choose decides which path would the `source` agent choose to `dest` """
        pass

    def get_info(self, source, dest, action):
        """ necessary information for training """
        return {}

    def learn(self, rewards, lr={}):
        """ learn and update tables from rewards given """
        if lr is None:
            lr = {}
        pass

    def clean(self):
        """ called by the environment when it's resetted """
        pass

    def receive(self, source, dest):
        """ [optional] define what the agent should do when a packet is received by a node """
        pass

    def send(self, source, dest):
        """ [optional] define what the agent should do when a packet is sent by a node """
        pass

    def reset(self):
        """ [optional] reset the agent """
        pass

    def drop_penalty(self, penalty):
        """ [optional] penalty when a packet is dropped """
        pass

    def store(self, filename):
        """ store attrs by pickle """
        f = open(filename, 'wb')
        pickle.dump({k: self.__dict__[k] for k in self.attrs}, f)
        f.close()


    def load(self, filename):
        """ load attrs by pickle """
        with open(filename, 'rb') as f:
            for k, v in pickle.load(f).items():
                self.__dict__[k] = v