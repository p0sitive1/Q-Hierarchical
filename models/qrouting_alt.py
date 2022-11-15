import numpy as np
from base_policy import Policy
from shortest_alt import *
from val import *


class Qroute(Policy):
    attrs = Policy.attrs | {'Qtable', 'discount', 'threshold'}

    def __init__(self, network, initQ=0, discount=0.99, threshold=0.1, static=False, pre=False):
        super().__init__(network)
        self.discount = discount
        self.threshold = threshold
        self.static = static
        self.pre = pre
        self.epsilon = 0.1
        self.network = network
        self.fshort = ShortestL(network)
        self.fval = VAL(network)

        self.Qtable = dict()
        for node in self.network.nodes.keys():
            self.Qtable[node] = dict()
            for x in self.network.nodes.keys():
                self.Qtable[node][x] = dict()
                for a in self.network.nodes[node].outQueuesInter.keys():
                    self.Qtable[node][x][a] = 0
                for b in self.network.nodes[node].outQueuesIntra.keys():
                    self.Qtable[node][x][b] = 0

    def choose(self, source, dest, packet=None, idx=False):
        choices = self.Qtable[source][dest]
        choices = list(choices.items())
        choices.sort(key=lambda x: x[1])
        Choice = choices[0][0]

        if self.pre and not self.static:
            # return self.fshort._choose(source, dest, packet)
            return self.fval.choose(source, dest, packet)
        else:
            return Choice

    def get_info(self, source, action, packet):
        return {'max_Q_y': np.max(self.Qtable[action][packet.dest].values())}

    def _extract(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = -info['q_y'] - info['t_y']
        n = -info['n_y'] - info['t_y']
        return r, info, x, y, d, n

    def _update_qtable(self, r, x, y, d, max_Q_y, lr, n):
        old_score = self.Qtable[x][d][y]
        self.Qtable[x][d][y] = lr * n + (1 - lr) * old_score

    def _update(self, reward, lr={'q': 0.01}):
        " update agent once/one turn "
        r, info, x, y, d, n = self._extract(reward)
        self._update_qtable(r, x, y, d, info['max_Q_y'], lr['q'], n)

    def learn(self, rewards, lr={}):
        if not self.static:
            for reward in rewards:
                self._update(reward, lr if lr else self._update.__defaults__[0])
        else:
            pass