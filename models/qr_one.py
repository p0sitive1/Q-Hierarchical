import numpy as np

from base_policy import Policy


class Qroute(Policy):
    """
    Q-routing
    """
    attrs = Policy.attrs | set(['Qtable', 'discount', 'epsilon'])

    def __init__(self, network, net_name='6x6', memory_capa=100000, initQ=0, discount=0.99, epsilon = 0.1, static=False):
        super().__init__(network)
        
        self.memory_counter = 0
        self.capa = memory_capa
        self.state = 3 * 36
        self.static = static
        self.memory = np.zeros((memory_capa, 2 * self.state + 4))
        
        self.discount = discount
        self.epsilon = epsilon

        self.Qtable = {x: np.random.normal(
            initQ, 1, (len(self.links), len(ys)))
            for x, ys in self.links.items()}
        for x, table in self.Qtable.items():
            # Q_x(z, x) = 0, forall z in x.neighbors
            table[x] = 0
            # Q_x(z, y) = -1 if z == y else 0
            table[self.links[x]] = -np.eye(table.shape[1])

    def choose(self, source, dest, packet=None, idx=False):
        scores = self.Qtable[source][dest]
        # score_max = scores.max()
        # choice = np.random.choice(np.argwhere(scores == score_max).flatten())
        if np.random.uniform() < self.epsilon:
            # exploration
            choice = int(np.random.randint(0,len(scores),1))
            return (choice, scores[choice]) if idx else self.links[source][choice]
        else: # greedy
            choice = np.argmax(scores)
            return (choice, scores.max()) if idx else self.links[source][choice]

    def get_info(self, source, action, packet):
        return {'max_Q_y': self.Qtable[action][packet.dest].max()}

    def _extract(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = - info['q_y'] - info['t_y']

        state = np.zeros(self.state)
        state[3 * x], state[3 * d + 2] = 1, 1
        for neighbor in self.links[x]:
            state[3 * neighbor + 1] = 1
            
        state_ = np.zeros(self.state)
        state_[3 * y], state_[3 * d + 2] = 1, 1
        for neighbor in self.links[y]:
            state_[3 * neighbor + 1] = 1
            
        a_ = self.choose(y,d)
        
        if info['max_Q_y'] == 0:
            is_dest = 0
        else:
            is_dest = 1

        transition = np.hstack((state, y, r, state_, a_, is_dest))
        index = self.memory_counter % self.capa
        self.memory[index, :] = transition
        self.memory_counter += 1

        return r, info, x, y, d

    def _update_qtable(self, r, x, y, d, max_Q_y, lr):
        y_idx = self.action_idx[x][y]
        old_score = self.Qtable[x][d][y_idx]
        self.Qtable[x][d][y_idx] += lr * (r + self.discount * max_Q_y - old_score)

    def _update(self, reward, lr={'q': 0.1}):
        " update agent once/one turn "
        r, info, x, y, d = self._extract(reward)
        self._update_qtable(r, x, y, d, info['max_Q_y'], lr['q'])

    def learn(self, rewards, lr={}):
        if not self.static:
            for reward in rewards:
                self._update(reward, lr if lr else self._update.__defaults__[0])
        else:
            pass
