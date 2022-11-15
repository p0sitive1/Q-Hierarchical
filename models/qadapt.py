import numpy as np
from base_policy import Policy
from shortest_alt import ShortestL
from val import VAL
import random
import math


class Qadapt(Policy):
    attrs = Policy.attrs | set(['QTable', 'discount', 'epsilon'])

    def __init__(self, network, initQ=0, discount=0.99, epsilon=0.1, threshold=0.1, static=False, pre=False):
        super().__init__(network)
        self.discount = discount
        self.threshold = threshold
        self.network = network
        self.epsilon = epsilon
        self.static = static
        self.QTable = dict()
        self.fshortest = ShortestL(network)
        self.fval = VAL(network)
        self.pre = pre
        self.thres = 0.8

        for g in range(self.network.group_num):
            self.QTable[g] = dict()
            for n in self.network.nodes.keys():
                self.QTable[g][n] = dict()
                for x in self.network.nodes[n].outQueuesIntra.keys():
                    self.QTable[g][n][x] = 0.1
                for y in self.network.nodes[n].outQueuesInter.keys():
                    self.QTable[g][n][y] = 0.1

    def choose(self, source, dest, packet):
        sourceg = self.network.nodes[source].group
        destg = self.network.nodes[dest].group
        psg = self.network.nodes[packet.source].group
        if self.pre and not self.static:
            return self.fval.choose(source, dest, packet)

        if sourceg == destg:
            return self.fshortest.short(source, dest)[0]

        elif sourceg == psg:
            # Find best port using table
            Choices = list(self.QTable[destg][source].items())
            Choices.sort(key=lambda x: x[1])
            best = Choices[0][0]

            shortv = self.fshortest.short(source, dest)[0]
            Qshort = self.QTable[destg][source][shortv]
            Qbest = self.QTable[destg][source][best]

            QV = (Qshort - Qbest) / Qshort
            if QV < self.thres:
                temp = shortv
            else:
                temp = best

            choices = list(self.network.nodes[source].outQueuesInter.keys())
            # choices += list(self.network.nodes[source].outQueuesIntra.keys())
            rand_out = np.random.choice(choices)
            choices = [temp, rand_out]
            choice = np.random.choice(choices, p=[0.1, 0.9])
            return choice
            return temp
        elif packet.flag and sourceg != destg and sourceg != psg:
            # If source is the first inter group
            packet.flag == False
            if dest in self.network.nodes[source].outQueuesInter.keys():
                return self.fshortest.short(source, dest)[0]
            else:
                # select random local port as best port
                return self.fshortest.short(source, dest)[0]
                # Choices = list(self.QTable[destg][source].keys())
                Choices = list(self.network.nodes[source].outQueuesIntra.keys())
                best = np.random.choice(Choices)

                shortv = self.fshortest.short(source, dest)[0]
                Qshort = self.QTable[destg][source][shortv]
                Qbest = self.QTable[destg][source][best]

                QV = (Qshort - Qbest) / Qshort
                if QV < self.thres:
                    temp = shortv
                else:
                    temp = best

                choices = list(self.network.nodes[source].outQueuesInter.keys())
                # choices += list(self.network.nodes[source].outQueuesIntra.keys())
                rand_out = np.random.choice(choices)
                choices = [temp, rand_out]
                choice = np.random.choice(choices, p=[0.1, 0.9])
                return choice
                return temp
        else:
            return self.fshortest.choose(source, dest, packet)

    def learn(self, rewards, lr={}):
        if not self.static:
            for reward in rewards:
                self._update(reward, lr if lr else self._update.__defaults__[0])
        else:
            pass
    
    def get_info(self, source, action, packet):
        output = {}
        output['min_Q'] = np.min(list(self.QTable[self.network.nodes[action].group][source].values()))
        return output

    def _extract(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = info['q_y'] + info['t_y']
        t = info['w_y'] + info['t_y']
        n = info['n_y'] + info['t_y']
        p = reward.packet
        return r, info, x, y, d, t, n
    
    def _update(self, reward, lr={'q': 0.01}):
        r, info, x, y, d, t, n = self._extract(reward)
        self._update_table(r, info['min_Q'], x, y, d, lr['q'], n)

    def _update_table(self, r, info, x, y, d, lr, n):
        sourceg = self.network.nodes[x].group
        destg = self.network.nodes[d].group

        old_var = self.QTable[destg][x][y]
        self.QTable[destg][x][y] = lr * n + (1 - lr) * old_var
