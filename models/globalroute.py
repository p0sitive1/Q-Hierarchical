import numpy as np

from base_policy import Policy


class Shortest(Policy):
    """ 
    Minimal routing based on Dijkstra algorithm. 
    However, this routing does not ensure the minimal path to be [intra-group, inter-group, intra-group], therefore not used
    To see the used minimal routing, refer to shortest_alt.py
    """
    attrs = Policy.attrs | set(['distance', 'choice', 'mask'])

    def __init__(self, network, multiway=False, random=False):
        super().__init__(network)
        self.multiway = multiway
        self.random = random
        self.distance = np.full((len(self.links), len(self.links)), np.inf)
        self.mask = np.ones_like(self.distance, dtype=np.bool)
        self.choice = {n: np.zeros((len(self.links), len(v)), dtype=np.bool)
                       for n, v in self.links.items()}
        for x, neighbors in self.links.items():
            self.distance[x, x] = 0
            self.mask[x, x] = False
            for y in neighbors:
                self.distance[x, y] = 1
                self.choice[x][y, self.action_idx[x][y]] = True
                self.mask[x, y] = False
        self.unit = lambda x: 1  # regard unit distance as 1
        self._calc_distance()

        self.check_action = {x: np.zeros(
            (len(self.links), len(ys)))
            for x, ys in self.links.items()
        }

        self.check_utilization = {x: 0 for x, _ in self.links.items()}

    def choose(self, source, dest, packet=None):
        """ Return the action with shortest distance and the distance """
        choices = self.links[source][self.choice[source][dest]]

        ch = choices[0]
        ch = int(np.where(self.links[source]==ch)[0])
        self.check_action[source][dest][ch] += 1
        self.check_utilization[self.links[source][ch]] += 1
        return np.random.choice(choices) if self.random else choices[0]

    def _calc_distance(self):
        self.distance[self.mask] = np.inf
        changing = True
        while changing:
            changing = False
            for x, neighbors in self.links.items():
                for y in neighbors:
                    for z in self.links.keys():
                        new_dis = self.distance[y, z] + self.unit(y)
                        if self.distance[x, z] > new_dis:
                            self.distance[x, z] = new_dis
                            self.choice[x][z, :].fill(False)
                            self.choice[x][z, self.action_idx[x][y]] = True
                            changing = True
                        if self.multiway and np.isfinite(new_dis) and self.distance[x, z] == new_dis:
                            self.choice[x][z, self.action_idx[x][y]] = True


class GlobalRoute(Shortest):
    """
    Global (real time) routing
    """
    def __init__(self, network, multiway=False, random=False):
        super().__init__(network, multiway=multiway, random=random)
        self.mask.fill(True)
        np.fill_diagonal(self.mask, False)
        self.queue_size = np.zeros(len(self.links), dtype=np.int)
        self.unit = lambda x: 1+self.queue_size[x]
        self._calc_distance()

    def receive(self, source, dest):
        self.queue_size[source] += 1

    def send(self, source, dest):
        self.queue_size[source] -= 1

    def learn(self, rewards, lr={}):
        # for i, node in self.nodes.items():
        #     self.queue_size[i] = len(node.queue)
        for choice in self.choice.values():
            choice.fill(False)
        self._calc_distance()