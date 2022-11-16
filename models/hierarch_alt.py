import numpy as np
from base_policy import Policy
from shortest_alt import ShortestL
from val import VAL
from shortest import *
import random
import math


class Hierarch(Policy):
    """
    Q-hierarchical routing
    """
    attrs = Policy.attrs | set(['inTable', 'outTable', 'discount', 'epsilon'])

    def __init__(self, network, initQ=0, discount=0.99, epsilon=0.1, threshold=0.1, static=False, pre=False):
        super().__init__(network)
        self.discount = discount
        self.threshold = threshold
        self.network = network
        self.epsilon = epsilon
        self.static = static
        self.pre = pre
        self.update_counter = 0
        self.x = 0
        self.setup = 1000000
        # used for pre training
        self.fshortest = ShortestL(network)
        self.fval = VAL(network)
        self.globalr = GlobalRoute(network)
        
        # generate tables
        self.outTable = dict()
        for group in range(self.group_num):
            outs = list()
            for node in self.network.nodes.values():
                if node.group == group:
                    outs += list(node.outQueuesInter.keys())
            outs.sort()

            self.outTable[group] = dict()
            for i in range(self.group_num):
                if i != group:
                    self.outTable[group][i] = dict()
                    for j in outs:
                        self.outTable[group][i][j] = 500

        self.inTable = dict()
        for node in self.network.nodes.keys():
            intra = self.network.nodes[node].outQueuesIntra.keys()
            inter = self.network.nodes[node].outQueuesInter.keys()
            self.inTable[node] = dict()
            outs = list()
            for tmp in self.network.nodes.values():
                if tmp.group == self.network.nodes[node].group:
                    outs += list(tmp.outQueuesInter.keys())
            outs.sort()
            for l in outs:
                self.inTable[node][l] = dict()
                for i in intra:
                    self.inTable[node][l][i] = 500
                for j in inter:
                    self.inTable[node][l][j] = 500
            for x in intra:
                self.inTable[node][x] = dict()
                for y in intra:
                    self.inTable[node][x][y] = 500

        # print(self.inTable[0])
        # print(self.outTable[0])
    
    def choose_(self, source, dest, packet):
        if self.network.nodes[source].group == self.network.nodes[packet.source].group and packet.rand_dest is None:
            rand_dest = np.random.choice(list(self.network.nodes.keys()))
            while self.network.nodes[rand_dest].group == self.network.nodes[source].group:
                rand_dest = np.random.choice(list(self.network.nodes.keys()))
            
            rand_dest = np.random.choice(list(self.network.nodes[source].outQueuesInter.keys()))
            packet.flag = False
            packet.rand_dest = rand_dest
        elif self.network.nodes[source].group == self.network.nodes[packet.rand_dest].group:
            packet.flag = True

        if not packet.flag:
            return self.fval.choose(source, packet.rand_dest, packet)
        else:
            return self.choose_hie(source, dest, packet)


    def choose(self, source, dest, packet):
        if False:
            # explore, not used
            firstChoice = np.random.choice([0, 1])
            if firstChoice == 1:
                return np.random.choice(list(self.network.nodes[source].outQueuesIntra.keys()))
            else:
                return np.random.choice(list(self.network.nodes[source].outQueuesInter.keys()))
            # go shortest
            # return self.fshortest.choose(source, dest, packet)
        else:
            if self.pre and not self.static:
                # pre training
                return self.fval.choose(source, dest, packet)
            # greedy
            # whether the current group is the dest group
            if self.network.nodes[packet.dest].group == self.network.nodes[source].group:
                # check bottom of the intable
                choices = self.inTable[source][dest]
                choices = list(choices.items())
                choices.sort(key=lambda x: x[1])
                inChoice = choices[0][0]

                if self.pre and not self.static:
                    # return self.fshortest.choose(source, dest, packet)
                    return self.fshortest.choose(source, dest, packet)
            else:
            # source and dest in different groups 
                # check outtable
                outChoices = self.outTable[self.network.nodes[source].group][self.network.nodes[dest].group]
                outChoices = list(outChoices.items())
                outChoices.sort(key=lambda x: x[1])
                outChoice = outChoices[0][0]
                
                # First step randomness
                if packet.flag:
                    choices = list(self.network.nodes[source].outQueuesInter.keys())
                    choi = list()
                    for choice in choices:
                        choi.append(self.outTable[self.network.nodes[source].group][self.network.nodes[dest].group][choice])
                    choices.append(outChoice)
                    choi.append(self.outTable[self.network.nodes[source].group][self.network.nodes[dest].group][outChoice])
                    W = np.array(choi)
                    xW = 1 / W
                    norm = xW / sum(xW)
                    tempChoice = np.random.choice(choices, p=norm)
                    tempQ = self.outTable[self.network.nodes[source].group][self.network.nodes[dest].group][tempChoice]
                    bestQ = self.outTable[self.network.nodes[source].group][self.network.nodes[dest].group][outChoice]
                    if (tempQ - bestQ) / bestQ <= 1:
                        outChoice = tempChoice
                    packet.flag = False

                if self.pre and not self.static:
                    return self.fshortest.choose(source, outChoice, packet)
                    # return self.fval.choose(source, outChoice, packet)

                inChoices = self.inTable[source][outChoice]
                inChoices = list(inChoices.items())
                inChoices.sort(key=lambda x: x[1])
                inChoice = inChoices[0][0]

            return inChoice

    def learn(self, rewards, lr={}):
        if not self.static:
            for reward in rewards:
                self._update(reward, lr if lr else self._update.__defaults__[0])
        else:
            pass

    def get_info(self, source, action, packet):
        output = {}
        output['min_Q'] = np.min(list(self.inTable[source][action].values()))
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
        self._update_intable(r, info['min_Q'], x, y, d, lr['q'], n)
        self._update_outtable(r, info['min_Q'], x, y, d, lr['q'], t)

    def _update_outtable(self, r, info, x, y, d, lr, t):
        #print(f"packet from {x} to {y} dest{d} waited total{r} waited ingroup {t}")
        if self.network.nodes[x].group != self.network.nodes[y].group and self.network.nodes[x].group != self.network.nodes[d].group:
            if self.network.nodes[y].group == self.network.nodes[d].group:
                if y == d:
                    nextChoice = 0
                else:
                    nextChoices = list(self.inTable[y][d].values())
                    nextChoice = np.min(nextChoices)

                old_var = self.outTable[self.network.nodes[x].group][self.network.nodes[d].group][y]
                self.outTable[self.network.nodes[x].group][self.network.nodes[d].group][y] = lr * (t + nextChoice) + (1 - lr) * old_var
            else:
                nextChoices = list(self.outTable[self.network.nodes[y].group][self.network.nodes[d].group].values())
                nextChoice = np.min(nextChoices)

                old_var = self.outTable[self.network.nodes[x].group][self.network.nodes[d].group][y]
                self.outTable[self.network.nodes[x].group][self.network.nodes[d].group][y] = lr * (t + nextChoice) + (1 - lr) * old_var
        else:
            pass

    def _update_intable(self, r, info, x, y, d, lr, n):
        # print(f"packet from {x} to {y} dest{d} waited total{r} waited innode {n}")
        if self.network.nodes[x].group == self.network.nodes[d].group and self.network.nodes[x].group == self.network.nodes[y].group:
            # if False:
            # x and y and dest are in same group
            if y == d:
                old_score = 0
            else:
                temp = list(self.inTable[y][d].values())
                old_score = np.min(temp)

            old_var = self.inTable[x][d][y]
            self.inTable[x][d][y] = lr * (n + old_score) + (1 - lr) * old_var
        elif self.network.nodes[x].group == self.network.nodes[y].group and self.network.nodes[x].group != self.network.nodes[d].group:
            # x and y are in same group, not d
            outChoices = self.outTable[self.network.nodes[x].group][self.network.nodes[d].group]
            outChoices = list(outChoices.items())
            outChoices.sort(key=lambda x: x[1])
            outChoice = outChoices[0][0]

            temp = list(self.inTable[y][outChoice].values())
            old_score = np.min(temp)

            old_var = self.inTable[x][outChoice][y]
            self.inTable[x][outChoice][y] = lr * (n + old_score) + (1 - lr) * old_var
        elif self.network.nodes[y].group == self.network.nodes[d].group and self.network.nodes[x].group != self.network.nodes[y].group:
            # y and d are in same group, not x
            outChoices = self.outTable[self.network.nodes[x].group][self.network.nodes[d].group]
            outChoices = list(outChoices.items())
            outChoices.sort(key=lambda x: x[1])
            outChoice = outChoices[0][0]
                
            if y == d:
                old_score = 0
            else:
                temp = list(self.inTable[y][d].values())
                old_score = np.min(temp)

            old_var = self.inTable[x][outChoice][y]
            self.inTable[x][outChoice][y] = lr * (n + old_score) + (1 - lr) * old_var
        elif self.network.nodes[x].group == self.network.nodes[d].group and self.network.nodes[x].group != self.network.nodes[y].group:
            # x and d are in same group, y in different group
            temp = list(self.outTable[self.network.nodes[y].group][self.network.nodes[d].group].values())
            old_score = np.min(temp)

            old_var = self.inTable[x][d][y]
            self.inTable[x][d][y] = lr * (n + old_score) + (1 - lr) * old_var
        else:
            # x and y and d are all not in the same group
            outChoices = self.outTable[self.network.nodes[x].group][self.network.nodes[d].group]
            outChoices = list(outChoices.items())
            outChoices.sort(key=lambda x: x[1])
            outChoice = outChoices[0][0]

            temp = list(self.outTable[self.network.nodes[y].group][self.network.nodes[d].group].values())
            old_score = np.min(temp)

            old_var = self.inTable[x][outChoice][y]
            self.inTable[x][outChoice][y] = lr * (n + old_score) + (1 - lr) * old_var
