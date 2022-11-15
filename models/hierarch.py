import numpy as np
from base_policy import Policy
from shortest import Shortest


class Hierarch(Policy):
    attrs = Policy.attrs | set(['inTable', 'outTable', 'discount', 'epsilon'])

    def __init__(self, network, initQ=0, discount=0.99, epsilon=0.1, threshold=0.1, static=False):
        super().__init__(network)
        self.discount = discount
        self.threshold = threshold
        self.network = network
        self.epsilon = epsilon
        self.static = static
        self.outTable = {}
        self.update_counter = 0
        self.x = 0
        self.setup = 1000000
        self.fshortest = Shortest(network)
        for i in range(self.group_num):
            outs = 0
            for node in self.network.nodes.values():
                if node.group == i:
                    outs += len(node.outQueuesInter.keys())
            # self.outTable[i] = np.random.normal(initQ, 1, (self.group_num-1, outs)) # randomized using normal distri
            self.outTable[i] = np.zeros((self.group_num-1, outs))  # all zeros
            # self.outTable[i] = np.full((self.group_num-1, outs), -10.0)

        self.inTable = {}
        for node in self.network.nodes.values():
            ins = len(self.outTable[node.group]) + len(node.outQueuesIntra)
            outs = len(node.outQueuesInter) + len(node.outQueuesIntra)
            # self.inTable[node.ID] = np.random.normal(initQ, 1, (ins, outs)) # randomized using normal distri
            self.inTable[node.ID] = np.zeros((ins, outs))  # all zeros
            # self.inTable[node.ID] = np.full((ins, outs), -10.0)

        
        ###########################re-implement tables######################
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
                        self.outTable[group][i][j] = 0

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
                    self.inTable[node][l][i] = 0
                for j in inter:
                    self.inTable[node][l][j] = 0
            for x in intra:
                self.inTable[node][x] = dict()
                for y in intra:
                    self.inTable[node][x][y] = 0

        # print(self.inTable[0])
        # print(self.outTable[0])

    def choose(self, source, dest, packet):
        # if self.x < self.setup:
        #     self.x += 1
        #     return self.fshortest.choose(source, dest, packet)
        if np.random.uniform() < self.epsilon:
            # if False:
            # explore
            firstChoice = np.random.choice([0, 1])
            if firstChoice == 1:
                return np.random.choice(list(self.network.nodes[source].outQueuesIntra.keys()))
            else:
                return np.random.choice(list(self.network.nodes[source].outQueuesInter.keys()))
            # go shortest
            # return self.fshortest.choose(source, dest, packet)
        else:
            # greedy
            # whether the current group is the dest group
            if self.network.nodes[packet.dest].group == self.network.nodes[source].group:
                # whether the current node is the dest node
                # done using external logic
                idx = list(
                    self.network.nodes[source].outQueuesIntra.keys()).index(dest)
                lenintra = len(
                    self.network.nodes[source].outQueuesIntra.keys())
                choices = self.inTable[source][idx][0:lenintra]
                inChoice = np.argmin(choices)
                choices = self.network.nodes[source].outQueuesIntra.keys()
                inChoice = list(choices).index(dest)
                return self.fshortest.choose(source, dest)
            else:
                if self.network.nodes[dest].group < self.network.nodes[source].group:
                    outchoices = self.outTable[self.network.nodes[source]
                                               .group][self.network.nodes[dest].group]
                else:
                    outchoices = self.outTable[self.network.nodes[source]
                                               .group][self.network.nodes[dest].group-1]
                outChoice = np.argmin(outchoices)

                # # ignore intable
                # outs = list()
                # for node in self.network.nodes.values():
                #     if node.group == self.network.nodes[source].group:
                #         outs += list(node.outQueuesInter.keys())
                # outs.sort()
                # idx = outs[outChoice]
                # # print(outs)
                # #print(packet, source, dest, outChoice, idx)
                # return self.fshortest.choose(source, idx, packet)

                choices = self.inTable[source][outChoice]
                inChoice = np.argmin(choices)

            if inChoice >= len(self.network.nodes[source].outQueuesIntra):
                return list(self.network.nodes[source].outQueuesInter.keys())[inChoice-len(self.network.nodes[source].outQueuesIntra)]
            else:
                return list(self.network.nodes[source].outQueuesIntra.keys())[inChoice]

    def learn(self, rewards, lr={}):
        if not self.static:
            for reward in rewards:
                self._update(
                    reward, lr if lr else self._update.__defaults__[0])
        else:
            pass

    def get_info(self, source, action, packet):
        output = {}
        if action in self.network.nodes[source].outQueuesIntra.keys():
            idx = list(
                self.network.nodes[source].outQueuesIntra.keys()).index(action)
        else:
            idx = list(self.network.nodes[source].outQueuesInter.keys()).index(
                action) + len(self.network.nodes[source].outQueuesIntra)
        output['min_Q'] = self.inTable[source][idx].min()
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

    def _update(self, reward, lr={'q': 0.1}):
        r, info, x, y, d, t, n = self._extract(reward)
        self._update_intable(r, info['min_Q'], x, y, d, lr['q'], n)
        self._update_outtable(r, info['min_Q'], x, y, d, lr['q'], t)

    def _update_outtable(self, r, info, x, y, d, lr, t):
        # return
        #print(f"packet from {x} to {y} dest{d} waited total{r} waited ingroup{t}")
        if self.network.nodes[x].group != self.network.nodes[y].group and self.network.nodes[x].group != self.network.nodes[d].group:
            if self.network.nodes[y].group == self.network.nodes[d].group:
                if y == d:
                    nextChoice = 0
                else:
                    temp = list(self.network.nodes[y].outQueuesIntra.keys()).index(
                        d) + len(self.outTable[0][0])
                    lenintra = len(self.network.nodes[y].outQueuesIntra.keys())
                    nextChoice = np.min(self.inTable[y][temp][0:lenintra])
                # nextChoice = 0
                if self.network.nodes[x].group < self.network.nodes[d].group:
                    x_idx = self.network.nodes[d].group - 1
                else:
                    x_idx = self.network.nodes[d].group

                outs = list()
                for node in self.network.nodes.values():
                    if node.group == self.network.nodes[x].group:
                        outs += list(node.outQueuesInter.keys())

                outs.sort()
                y_idx = outs.index(y)
                old_var = self.outTable[self.network.nodes[x].group][x_idx][y_idx]
                self.outTable[self.network.nodes[x].group][x_idx][y_idx] = lr * \
                    (t + nextChoice) + (1 - lr) * old_var
            else:
                if self.network.nodes[y].group < self.network.nodes[d].group:
                    nextChoice = np.min(
                        self.outTable[self.network.nodes[y].group][self.network.nodes[d].group-1])
                else:
                    nextChoice = np.min(
                        self.outTable[self.network.nodes[y].group][self.network.nodes[d].group])

                if self.network.nodes[x].group < self.network.nodes[d].group:
                    x_idx = self.network.nodes[d].group - 1
                else:
                    x_idx = self.network.nodes[d].group

                outs = list()
                for node in self.network.nodes.values():
                    if node.group == self.network.nodes[x].group:
                        outs += list(node.outQueuesInter.keys())

                outs.sort()
                y_idx = outs.index(y)
                old_var = self.outTable[self.network.nodes[x].group][x_idx][y_idx]
                #print(t, nextChoice, old_var)
                self.outTable[self.network.nodes[x].group][x_idx][y_idx] = lr * \
                    (t + nextChoice) + (1 - lr) * old_var
        else:
            pass

    def _update_intable(self, r, info, x, y, d, lr, n):
        # return
        # print(f"packet from {x} to {y} dest{d} waited total{r} waited innode{n}")
        # find which column to update
        if self.network.nodes[x].group == self.network.nodes[y].group:
            y_idx = list(self.network.nodes[x].outQueuesIntra.keys()).index(y)
        else:
            y_idx = list(self.network.nodes[x].outQueuesInter.keys()).index(
                y) + len(self.network.nodes[x].outQueuesIntra)

        # find which row to update
        if self.network.nodes[x].group == self.network.nodes[d].group and self.network.nodes[x].group == self.network.nodes[y].group:
            # if False:
            # x and y and dest are in same group
            x_idx = list(self.network.nodes[x].outQueuesIntra.keys()).index(d) + len(self.outTable[0][0])
            if y < d:
                tempx = list(self.network.nodes[y].outQueuesIntra.keys()).index(d) + len(self.outTable[0][0])
                old_score = np.min(self.inTable[d][tempx])
            elif y > d:
                tempx = list(self.network.nodes[y].outQueuesIntra.keys()).index(
                    d) + len(self.outTable[0][0]) - 1
                old_score = np.min(self.inTable[d][tempx])
            else:
                # y = d
                old_score = 0.0
            old_var = self.inTable[x][x_idx][y_idx]
            self.inTable[x][x_idx][y_idx] = lr * \
                (n + old_score) + (1 - lr) * old_var
        else:
            if self.network.nodes[x].group == self.network.nodes[y].group:
                # x and y are in same group
                if self.network.nodes[d].group < self.network.nodes[x].group:
                    outchoices = self.outTable[self.network.nodes[x]
                                               .group][self.network.nodes[d].group]
                else:
                    outchoices = self.outTable[self.network.nodes[x]
                                               .group][self.network.nodes[d].group-1]
                x_idx = np.argmin(outchoices)
                old_score = np.min(self.inTable[y][x_idx])
                old_var = self.inTable[x][x_idx][y_idx]
                # print(x_idx, y_idx)
                # print(n, old_score, old_var)
                #print(lr * (n + old_score) + (1 - lr) * old_var)
                self.inTable[x][x_idx][y_idx] = lr * \
                    (n + old_score) + (1 - lr) * old_var
            elif self.network.nodes[x].group != self.network.nodes[y].group:
                # x and y are in different group
                if self.network.nodes[d].group < self.network.nodes[x].group:
                    outchoices = self.outTable[self.network.nodes[x]
                                               .group][self.network.nodes[d].group]
                else:
                    outchoices = self.outTable[self.network.nodes[x]
                                               .group][self.network.nodes[d].group-1]

                if self.network.nodes[d].group == self.network.nodes[y].group and d != y:
                    temp = list(
                        self.network.nodes[y].outQueuesIntra.keys()).index(d)
                    # old_score = np.min(self.inTable[y][temp + len(self.outTable[0][0])])
                    old_score = 0.0
                else:
                    outs = list()
                    for node in self.network.nodes.values():
                        if node.group == self.network.nodes[x].group:
                            outs += list(node.outQueuesInter.keys())
                    outs.sort()
                    var = outs[np.argmin(outchoices)]
                    varg = var // 4
                    if self.network.nodes[d].group < varg:
                        old_score = np.min(
                            self.outTable[varg][self.network.nodes[d].group])
                    else:
                        old_score = np.min(
                            self.outTable[varg][self.network.nodes[d].group-1])
                x_idx = np.argmin(outchoices)
                old_var = self.inTable[x][x_idx][y_idx]
                # print(x_idx, y_idx)
                # print(n, old_score)
                #print(n, old_score, old_var)
                #print(lr * (n + old_score) + (1 - lr) * old_var)
                self.inTable[x][x_idx][y_idx] = lr * \
                    (n + old_score) + (1 - lr) * old_var
