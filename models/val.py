import numpy as np
from models.base_policy import Policy
from models.shortest_alt import *

class VAL(Policy):
    """
    VAL routing
    algorithm is found in shortest_alt.py
    """
    def __init__(self, network):
        super().__init__(network)
        self.network = network
        self.fshort = ShortestL(network)

    def choose(self, source, dest, packet):
        # print(source, dest, packet)
        return self.fshort._choose_inter(source, dest, packet)