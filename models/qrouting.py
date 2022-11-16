import numpy as np
from models.base_policy import Policy
from models.shortest import *


class Qroute(Policy):
    """
    Q-routing
    """
    attrs = Policy.attrs | {'Qtable', 'discount', 'threshold'}

    def __init__(self, network, initQ=0, discount=0.99, threshold=0.1, static=False, pre=False):
        super().__init__(network)
        self.discount = discount
        self.threshold = threshold
        self.static = static
        self.pre = pre
        self.epsilon = 0.1
        self.network = network
        self.fshort = Shortest(network)
        self.Qtable = {x: np.random.normal(initQ, 1, (len(self.links), len(ys))) for x, ys in self.links.items()}
        for x, table in self.Qtable.items():
            # Q_x(z, x) = 0, forall z in x.neighbors
            table[x] = 0
            # Q_x(z, y) = -1 if z == y else 0
            table[self.links[x]] = -np.eye(table.shape[1])

    def choose(self, source, dest, packet=None, idx=False):
        scores = self.Qtable[source][dest]
        if idx: # only for agent updating
            return np.argmax(scores), scores.max()
        else:
            return self.links[source][np.argmax(scores)]

    def get_info(self, source, action, packet):
        return {'max_Q_y': self.Qtable[action][packet.dest].max()}

    def _extract(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = -info['q_y'] - info['t_y']
        return r, info, x, y, d

    def _update_qtable(self, r, x, y, d, max_Q_y, lr):
        y_idx = self.action_idx[x][y]
        old_score = self.Qtable[x][d][y_idx]
        self.Qtable[x][d][y_idx] += lr * (r + self.discount * max_Q_y - old_score)

    def _update(self, reward, lr={'q': 0.01}):
        " update agent once/one turn "
        r, info, x, y, d = self._extract(reward)
        self._update_qtable(r, x, y, d, info['max_Q_y'], lr['q'])

    def learn(self, rewards, lr={}):
        if not self.static:
            for reward in rewards:
                self._update(reward, lr if lr else self._update.__defaults__[0])
        else:
            pass