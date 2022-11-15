import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DRL(nn.Module):
    def __init__(self, lr, node_num, input_dims, output_dims, n_actions=36):
        super(DRL, self).__init__()
        self.node_num = node_num
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims * self.node_num, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, state):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(self, network, gamma=0.09, epsilon=1.0, lr=0.003, batch_size=64, max_mem_size=100000, eps_end=0.01,
                 eps_dec=5e-4):
        self.links = network.links
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.adj = network.adj
        self.node_num = len(self.adj)
        self.input_dims = 3
        self.output_dims = 8

        self.Q_eval = DRL(lr=self.lr, node_num=self.node_num, input_dims=self.input_dims, output_dims=self.output_dims)

        self.state_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def _get_state(self, source, dest):
        # source neighbors queue info
        x = np.zeros(self.node_num * self.input_dims)
        for neighbor in self.links[source]:
            x[3 * source], x[3 * neighbor + 1], x[3 * dest + 2] = 1, 1, 1
        return x

    def choose_action(self, source, dest):
        state = self._get_state(source, dest)
        state = torch.tensor(state, dtype=torch.float32).view(1, -1).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state, self.adj)
        if np.random.uniform() > self.epsilon:
            choice = torch.argmax(actions).item()
            return self.links[source][choice]
        else:
            choice = int(np.random.randint(0, len(actions), 1))
            return self.links[source][choice]

    def choose_rand(self, source, dest):
        neighbors = self.links[source]
        choice = int(np.random.randint(0, len(neighbors), 1))
        return self.links[source][choice]

    def learn(self, rewards):
        for reward in rewards:
            self.store_transition(reward=reward)
