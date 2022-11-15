import numpy as np
from base_policy import Policy
from shortest_alt import *

class VAL(Policy):
    def __init__(self, network):
        super().__init__(network)
        self.network = network
        self.fshort = ShortestL(network)

    def _choose(self, source, dest, packet):
        if self.network.nodes[source].group == self.network.nodes[packet.source].group and packet.rand_dest is None:
            rand_dest = np.random.choice(list(self.network.nodes.keys()))
            while self.network.nodes[rand_dest].group == self.network.nodes[source].group:
                rand_dest = np.random.choice(list(self.network.nodes.keys()))
            
            randc = list(self.network.nodes[source].outQueuesInter.keys())
            randc.append(dest)
            rand_dest = np.random.choice(randc)
            packet.flag = False
            packet.rand_dest = rand_dest
            # print(packet, packet.rand_dest)
        elif self.network.nodes[source].group == self.network.nodes[packet.rand_dest].group:
            packet.flag = True

        if not packet.flag:
            return self.fshort.choose(source, packet.rand_dest, packet)
        else:
            return self.fshort.choose(source, dest, packet)

    def choose(self, source, dest, packet):
        # print(source, dest, packet)
        return self.fshort._choose_inter(source, dest, packet)