import numpy as np
from models.base_policy import Policy

class ShortestL(Policy):
    """
    Minimal routing used on generated path
    This ensures the minimal path to be [intra-group, inter-group, intra-group]

    VAL routing also created here
    """
    def __init__(self, network):
        super().__init__(network)
        self.network = network

    def choose(self, source, dest, packet):
        """
        regular minimal path
        """
        if source == packet.source:
            path = self.short(packet.source, packet.dest)  # from p source to p dest
            packet.short_p = path
        
        return packet.short_p.pop(0)

    def choose_no_pop(self, source, dest, packet):
        """
        For testing purposes
        """
        if source == packet.source:
            path = self.short(packet.source, packet.dest)  # from p source to p dest
            packet.short_p = path

        return packet.short_p[0]

    def _choose_inter(self, source, dest, packet):
        """
        Generate path for VAL routing
        """
        if source == packet.source:
            randc = np.random.choice(list(self.network.nodes[source].outQueuesInter.keys()))
            randn = np.random.choice(list(self.network.nodes.keys()))
            while self.network.nodes[randn].group == self.network.nodes[source].group:
                randn = np.random.choice(list(self.network.nodes.keys()))
            rand_dest = np.random.choice([randc, randn], p=[0.95, 0.05])
            rpath = self.short(packet.source, rand_dest)  # from p source to rand_dest
            dpath = self.short(rand_dest, packet.dest)  # from intermid to p dest
            path = rpath + dpath
            packet.short_p = path
        
        return packet.short_p.pop(0)

    
    def short(self, src, dst):
        """
        Generate minimal path for 36 node network
        """
        path = []
        if src//4 != dst//4:
            gap = dst//4-src//4
            if gap<0:
                gap+=9
            for i in range(4):
                if gap in range((3-i%4)*2+1,(3-i%4)*2+3):
                    if i == src%4:
                        path.append(dst-dst%4+3-i%4)
                    else:
                        path.append(src-src%4+i)
                        path.append(dst-dst%4+3-i%4)
        if dst not in path:
            path.append(dst)
        return path
    
    def short_(self, src, dst):
        """
        Generate minimal path for 264 node network
        """
        path = []
        if src//8 != dst//8:
            gap = dst//8-src//8
            if gap<0:
                gap+=33
            for i in range(8):
                if gap in range((7-i%8)*4+1,(7-i%8)*4+5):
                    if i == src%8:
                        path.append(dst-dst%8+7-i%8)
                    else:
                        path.append(src-src%8+i)
                        path.append(dst-dst%8+7-i%8)
        if dst not in path:
            path.append(dst)
        return path
