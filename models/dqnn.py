import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from base_policy import Policy
import logging


def clip_r(r, mini=-5):
    return r if r > mini else mini

class config:
    def __init__(self, n_node=36, node_input=3, given_num=6, node_output=8, num_head=2, learning_rate_critic=4e-5, learning_rate_actor=4e-7, reward_dacay=0.99, batch=256, memory_capacity=2000, target_replace_iter=30, tau=0.05):
        self.number_of_node = n_node
        self.node_input = node_input
        self.node_output = node_output
        self.n_state = node_input * n_node
        self.number_of_head = num_head
        self.given_num = given_num

        self.learning_rate = learning_rate_critic
        self.learning_rate_critic = learning_rate_critic
        self.learning_rate_actor = learning_rate_actor
        self.reward_dacay = reward_dacay
        self.tau = tau

        self.batch = batch
        self.memory_capacity = memory_capacity
        self.target_replace_iter = target_replace_iter

    def __str__(self):
        print('Network Topology Information')
        print('Number of Node:', self.number_of_node)
        print('Input dimension', self.node_input)
        print('-----------------------------------')
        print('Hyper Parameter')
        print('Learning rate:', self.learning_rate)
        print('Reward_decay:', self.reward_dacay)
        print('Memory capacity:', self.memory_capacity)
        print('Batch size:', self.batch)
        print('Tau:', self.tau)

        return '-----------------------------------'

class DQN(nn.Module):
    def __init__(self, number_of_node, dim_input, dim_output, dim_action=36):
        super().__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input

        self.fc1 = nn.Linear(number_of_node * dim_input, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, dim_action)

    def forward(self, X, adj):
        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value

class DQN_4(nn.Module):
    def __init__(self, number_of_node, dim_input, given_num, dim_output, dim_action=36):
        super().__init__()
        self.number_of_node = number_of_node
        self.dim_input = dim_input

        self.fc1 = nn.Linear(number_of_node * dim_input * given_num, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, dim_action)

    def forward(self, X, adj):
        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        q_value = self.out(X)  # dim = (N_ACTION,1)
        return q_value

# Build the policy training
class DRL(Policy):
    def __str__(self):
        print(self.config)
        print('Memory shape', self.memory.shape)
        print('-----------------------------------')
        print('Network shape', self.eval_net)
        print('-----------------------------------')
        print('Optimizer', self.optimizer)
        return '-----------------------------------'

    def __init__(self, network, net_name='6x6', epsilon=0, static=False):
        super().__init__(network)
        self.config = config()
        self.static = static
        self.epsilon = epsilon
        self.network = network
        self.adj = self.network.adj

        # initialize the memory
        self.build_memory()

    def pre_training(self, pre_memory, pre_time):
        self.learn_step_counter = 0
        for i in tqdm(range(pre_time)):
            s, a, r, s_, a_, is_dest = self._batch_memory(pre_memory)
            self._update_parameter(s, a, r, s_, a_, is_dest, pre=True)
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def pre_train(self, pre_mem, pre_time):
        self.learn_step_counter = 0
        result = self.network.train(duration=pre_time, lambd=1, pre=True)  # pre train parameters
        self.target_net.load_state_dict(self.eval_net.state_dict())
        return result

    def choose(self, source, dest, target=False, packet=None, idx=False):
        x = self._get_state(source, dest)
        x = torch.tensor(x, dtype=torch.float).view(1, -1)
        scores = self.eval_net.forward(x, self.adj).view(-1, 1)[self.links[source]]
        if np.random.uniform() < self.epsilon:
            # exploration
            choice = int(np.random.randint(0, len(scores), 1))
            return (choice, scores[choice]) if idx else self.links[source][choice]
        else:
            # greedy
            choice = int(torch.argmax(scores))
            return (choice, scores.max()) if idx else self.links[source][choice]

    def learn(self, rewards, pre=False):
        if not self.static:
            for reward in rewards:
                self._store_memory(reward)
                # self._update_immediately(reward)
                if self.memory_counter % self.config.memory_capacity == 0 and self.memory_counter != 0:
                    s, a, r, s_, a_, is_dest = self._batch_memory(self.memory)
                    self._update_parameter(s, a, r, s_, a_, is_dest, pre=pre)
        else:
            None

    def reset_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=learning_rate)

    def _get_state(self, source, dest):
        x = np.zeros(self.config.n_state)
        for neighbor in self.links[source]:
            # 0 for source, 1 for action, 2 for destination
            x[3 * source], x[3 * neighbor + 1], x[3 * dest + 2] = 1, 1, 1
        return x

    def build_model(self, eval_NN, target_NN):
        self.eval_net = eval_NN
        self.target_net = target_NN

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=self.config.learning_rate)
        self.loss_func = nn.MSELoss()   # MSE loss

        logging.debug("The network structure is: ", self.eval_net)
        logging.debug("The learning rate is now: ", self.config.learning_rate)

    def build_memory(self,):
        self.loss = []
        # for timing the change of the target network
        self.learn_step_counter = 0
        # counter of the memory
        self.memory_counter = 0
        self.memory = np.zeros((self.config.memory_capacity, 2 * self.config.n_state + 4))
        self.pre_memory_counter = 0
        self.pre_memory = np.zeros((self.config.memory_capacity, 2 * self.config.n_state + 4))

    def _store_memory(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = - info['q_y'] - info['t_y']
        r = clip_r(r, -10)
        s = self._get_state(x, d)
        s_ = self._get_state(y, d)
        a_ = 0

        if y == d:
            is_dest = 0
        else:
            is_dest = 1

        transition = np.hstack((s, y, r, s_, a_, is_dest))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.config.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def _store_pre_memory(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = - info['q_y'] - info['t_y']
        r = clip_r(r, -10)
        s = self._get_state(x, d)
        s_ = self._get_state(y, d)
        a_ = 0

        if y == d:
            is_dest = 0
        else:
            is_dest = 1

        transition = np.hstack((s, y, r, s_, a_, is_dest))
        # 如果记忆库满了, 就覆盖老数据
        index = self.pre_memory_counter % self.config.memory_capacity
        self.pre_memory[index, :] = transition
        self.pre_memory_counter += 1

    def _batch_memory(self, memory):
        sample_index = np.random.choice(memory.shape[0], self.config.batch,
                                        replace=False)
        b_memory = memory[sample_index, :]
        n_s = self.config.n_state
        s = torch.FloatTensor(b_memory[:, :n_s])
        a = torch.LongTensor(b_memory[:, n_s:n_s+1].astype(int))
        r = torch.FloatTensor(b_memory[:, n_s+1:n_s+2])
        s_ = torch.FloatTensor(b_memory[:, n_s+2:2*n_s+2])
        a_ = torch.LongTensor(b_memory[:, 2*n_s+2:2*n_s+3].astype(int))
        is_dest = torch.FloatTensor(b_memory[:, n_s*2+3:])
        # a, a_ --> LongTensor
        # s, s_, r, is_dest --> FloatTensor
        return s, a, r, s_, a_, is_dest

    def _update_parameter(self, s, a, r, s_, a_, is_dest, pre=False):
        tau = self.config.tau
        if self.learn_step_counter % self.config.target_replace_iter == 0:
            for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        # if self.learn_step_counter % self.config.target_replace_iter == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        q_eval = self.eval_net(s, self.adj).gather(1, a)

        # q_next must be max_action
        if pre:
            q_next = self.target_net(s_, self.adj).gather(1, a_)
        else:
            ava_act = s_[:, 1::3]
            q_next = self.target_net(s_, self.adj)
            q_next = torch.where(ava_act > 0, q_next, torch.Tensor([-1e12]))
            q_next = q_next.max(1).values.view(-1, 1)

        q_target = r + self.config.reward_dacay * q_next * is_dest

        loss = self.loss_func(q_eval, q_target.detach())

        self.loss.append(float(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store(self, filename):
        torch.save(self.eval_net.state_dict(), filename)

    def load(self, filename):
        self.eval_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(torch.load(filename))


class DRL_4(Policy):
    def __str__(self):
        print(self.config)
        print('Memory shape', self.memory.shape)
        print('-----------------------------------')
        print('Network shape', self.eval_net)
        print('-----------------------------------')
        print('Optimizer', self.optimizer)
        return '-----------------------------------'

    def __init__(self, network, net_name='6x6',
                 epsilon=0, static=False):
        super().__init__(network)
        self.config = config()
        self.static = static
        self.epsilon = epsilon
        self.network = network
        self.adj = self.network.adj

        # initialize the memory
        self.build_memory()

    def pre_training(self, pre_memory, pre_time):
        self.learn_step_counter = 0
        for i in tqdm(range(pre_time)):
            s, a, r, s_, a_, is_dest = self._batch_memory(pre_memory)
            self._update_parameter(s, a, r, s_, a_, is_dest, pre=True)
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def pre_train(self, pre_mem, pre_time):
        self.learn_step_counter = 0
        result = self.network.train(duration=pre_time, lambd=1, pre=True)  # pre train parameters
        self.target_net.load_state_dict(self.eval_net.state_dict())
        return result

    def choose(self, packets, source, dest, target=False, idx=False):
        x = self._get_state(source, dest, packets)
        x = torch.tensor(x, dtype=torch.float).view(1, -1)
        # x = torch.flatten(x)
        scores = self.eval_net.forward(x, self.adj).view(-1, 1)[self.links[source]]
        if np.random.uniform() < self.epsilon:
            # exploration
            choice = int(np.random.randint(0, len(scores), 1))
            return (choice, scores[choice]) if idx else self.links[source][choice]
        else:
            # greedy
            choice = int(torch.argmax(scores))
            return (choice, scores.max()) if idx else self.links[source][choice]

    def learn(self, rewards, pre=False):
        if not self.static:
            for reward in rewards:
                self._store_memory(reward)
                # self._update_immediately(reward)
                if self.memory_counter % self.config.memory_capacity == 0 and self.memory_counter != 0:
                    s, a, r, s_, a_, is_dest = self._batch_memory(self.memory)
                    self._update_parameter(s, a, r, s_, a_, is_dest, pre=pre)
        else:
            None

    def reset_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=learning_rate)

    def _get_state(self, source, dest, info=[]):
        x = np.zeros((self.config.given_num, self.config.n_state))
        for i in range(len(info)):
            p_info = info[i]
            p_source = p_info[0]
            p_dest = p_info[1]
            p_choice = p_info[2]
            x[i + 1][3 * p_source], x[i + 1][3 * p_dest + 1], x[i + 1][3 * p_choice + 2] = 1, 1, 1

        x[0][3 * source], x[0][3 * dest + 1] = 1, 1
        return x

    def build_model(self, eval_NN, target_NN):
        self.eval_net = eval_NN
        self.target_net = target_NN

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=self.config.learning_rate)
        self.loss_func = nn.MSELoss()   # MSE loss

        logging.debug("The network structure is: ", self.eval_net)
        logging.debug("The learning rate is now: ", self.config.learning_rate)

    def build_memory(self,):
        self.loss = []
        # for timing the change of the target network
        self.learn_step_counter = 0
        # counter of the memory
        self.memory_counter = 0
        self.memory = np.zeros((self.config.memory_capacity, 2 * self.config.given_num * self.config.n_state + 4))
        self.pre_memory_counter = 0
        self.pre_memory = np.zeros((self.config.memory_capacity, 2 * self.config.given_num * self.config.n_state + 4))

    def _store_memory(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = - info['q_y'] - info['t_y']
        r = clip_r(r, -10)
        s = self._get_state(x, d)
        s_ = self._get_state(y, d)
        a_ = 0

        s = s.flatten()
        s_ = s_.flatten()

        if y == d:
            is_dest = 0
        else:
            is_dest = 1

        transition = np.hstack((s, y, r, s_, a_, is_dest))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.config.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def _store_pre_memory(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = - info['q_y'] - info['t_y']
        r = clip_r(r, -10)
        s = self._get_state(x, d)
        s_ = self._get_state(y, d)
        a_ = 0

        s = s[0]
        s_ = s_[0]

        if y == d:
            is_dest = 0
        else:
            is_dest = 1

        transition = np.hstack((s, y, r, s_, a_, is_dest))
        # 如果记忆库满了, 就覆盖老数据
        index = self.pre_memory_counter % self.config.memory_capacity
        self.pre_memory[index, :] = transition
        self.pre_memory_counter += 1

    def _batch_memory(self, memory):
        sample_index = np.random.choice(memory.shape[0], self.config.batch,
                                        replace=False)
        b_memory = memory[sample_index, :]
        n_s = self.config.n_state * self.config.given_num
        s = torch.FloatTensor(b_memory[:, :n_s])
        a = torch.LongTensor(b_memory[:, n_s:n_s+1].astype(int))
        r = torch.FloatTensor(b_memory[:, n_s+1:n_s+2])
        s_ = torch.FloatTensor(b_memory[:, n_s+2:2*n_s+2])
        a_ = torch.LongTensor(b_memory[:, 2*n_s+2:2*n_s+3].astype(int))
        is_dest = torch.FloatTensor(b_memory[:, n_s*2+3:])
        # a, a_ --> LongTensor
        # s, s_, r, is_dest --> FloatTensor
        return s, a, r, s_, a_, is_dest

    def _update_parameter(self, s, a, r, s_, a_, is_dest, pre=False):
        tau = self.config.tau
        if self.learn_step_counter % self.config.target_replace_iter == 0:
            for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        # if self.learn_step_counter % self.config.target_replace_iter == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        q_eval = self.eval_net(s, self.adj).gather(1, a)

        # q_next must be max_action
        if pre:
            q_next = self.target_net(s_, self.adj).gather(1, a_)
        else:
            ava_act = s_[:, 1::3]
            q_next = self.target_net(s_, self.adj)
            # q_next = torch.where(ava_act > 0, q_next, torch.Tensor([-1e12]))
            q_next = q_next.max(1).values.view(-1, 1)

        q_target = r + self.config.reward_dacay * q_next * is_dest

        loss = self.loss_func(q_eval, q_target.detach())

        self.loss.append(float(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store(self, filename):
        torch.save(self.eval_net.state_dict(), filename)

    def load(self, filename):
        self.eval_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(torch.load(filename))