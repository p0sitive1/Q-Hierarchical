from heapq import heappop, nsmallest
from operator import ilshift
from struct import pack
import numpy as np
from Agent import *
from matplotlib import pyplot as plt
from models.base_policy import Policy
import random
from tqdm import tqdm


class Clock:
    def __init__(self, now):
        self.t = now

    def __str__(self):
        return str(self.t)


class Event:
    def __init__(self, packet, from_node, to_node, time):
        self.from_node = from_node
        self.to_node = to_node
        self.packet = packet
        self.time = time

    def __str__(self):
        return f"From {self.from_node} to {self.to_node} at {self.time}"

    def __repr__(self):
        return f"From {self.from_node} to {self.to_node} at {self.time}"

    def __lt__(self, other):
        return self.time < other.time


class Reward:
    def __init__(self, source, packet, action, agent_info={}):
        self.source = source
        self.dest = packet.dest
        self.action = action
        self.packet = packet
        self.agent_info = agent_info

    def __repr__(self):
        return f"Reward<{self.source}->{self.dest} by action {self.action}>"


class Packet:
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest
        self.trans_time = 0
        self.hops = 0
        self.birtht = 0
        self.endt = 0
        self.path = []
        self.cur_group = None
        self.cur_wait = self.birtht
        self.node_wait = self.birtht
        self.flag = True  # used for VAL & shortest_alt
        self.rand_dest = None  # used for VAL
        self.short_p = []  # used for shortest_alt

    def __str__(self):
        return f"<{self.source} -> {self.dest}>"

    def __repr__(self):
        return f"<{self.source} -> {self.dest}>"

    def return_path(self):
        return self.path


class Node:
    """
    Node object
    ID: identifier for current node
    inQueue: store incoming packets
    outQueuesInter: store outgoing inter-group packets
    outQueuesIntra: store outgoing intra-group packets
    group: identifier for current group
    """
    def __init__(self, ID, outs, network, clock, group):
        self.ID = ID
        self.outs = outs
        self.inQueue = []
        self.outQueuesInter = {}
        self.outQueuesIntra = {}
        self.network = network
        self.clock = clock
        self.send_info = []
        self.group = group

    def __str__(self):
        return f"Node {self.ID} in group {self.group}, inQueue {self.inQueue}, outQueues intergroup {self.outQueuesInter}, outQueues intragroup {self.outQueuesIntra}"

    def __repr__(self):
        return f"Node {self.ID} in group {self.group}, inQueue {self.inQueue}, outQueues intergroup {self.outQueuesInter}, outQueues intragroup {self.outQueuesIntra}"

    @property
    def agent(self):
        return self.network.agent

    @property
    def pre_agent(self):
        return self.network.pre_agent

    def _build_info(self, agent_info, packet, action):
        # set the environment rewards
        # q: queuing delay; t: transmission delay
        agent_info['q_y'] = self.clock.t - packet.birtht
        agent_info['w_y'] = self.clock.t - packet.cur_wait
        agent_info['n_y'] = self.clock.t - packet.node_wait
        agent_info['t_y'] = 1
        return agent_info

    def refresh(self, group_pop):
        for i in self.outs:
            if self.group * group_pop <= i.ID < self.group * group_pop + group_pop:
                self.outQueuesIntra[i.ID] = []
            else:
                self.outQueuesInter[i.ID] = []

    def receive(self, packet):
        """
        node receives packet *packet*

        :param packet:
        """
        packet.node_wait = self.clock.t
        if packet.cur_group != self.group:
            packet.cur_group = self.group
            packet.cur_wait = self.clock.t
        if self.ID == packet.dest:
            # packet reached destination
            self.network.end_packet(packet)
        else:
            self.inQueue.append(packet)
            self.agent.receive(self.ID, packet.dest)

    def send(self):
        """
        send first packets in inQueue to outQueue according to action
        """
        rewards = []
        if len(self.inQueue) > 0:
            packet = self.inQueue.pop(0)
            dest = packet.dest
            # action = self.agent.choose_rand(self.ID, dest)  # choose random for testing
            if False:
                action = self.agent.choose(self.send_info, self.ID, dest) # pass info for older testing, no longer used
            else:
                action = self.agent.choose(self.ID, dest, packet=packet)  # choose queue
            packet.path.append(action)
            if action in self.outQueuesInter:
                self.outQueuesInter[action].append(packet)
            else:
                self.outQueuesIntra[action].append(packet)
            self.send_packet()
            self.agent.send(self.ID, dest)
            agent_info = self.agent.get_info(self.ID, action, packet)
            agent_info = self._build_info(agent_info, packet, action)
            rewards.append(Reward(self.ID, packet, action, agent_info))
        return rewards

    def send_packet(self):
        """
        send packets in all queue
        """
        for q in self.outQueuesInter.items():
            while len(q[1]) > 0:
                dest = q[0]
                packet = q[1].pop(0)
                packet.trans_time = self.network.transtime
                packet.hops += 1
                packet.node_wait = self.clock.t
                self.network.event_queue.append(Event(packet, self.ID, dest, self.clock.t + packet.trans_time))
                if len(self.send_info) >= 5:
                    self.send_info.pop(0)
                self.send_info.append([self.ID, packet.dest, dest])
        for q in self.outQueuesIntra.items():
            while len(q[1]) > 0:
                dest = q[0]
                packet = q[1].pop(0)
                packet.trans_time = self.network.transtime
                packet.hops += 1
                packet.node_wait = self.clock.t
                self.network.event_queue.append(Event(packet, self.ID, dest, self.clock.t + packet.trans_time))
                if len(self.send_info) >= 5:
                    self.send_info.pop(0)
                self.send_info.append([self.ID, packet.dest, dest])


class Network:
    """
    Network object
    nodes: stores all node in network
    links: stores all links in network
    """
    def __init__(self, adj, bandwidth=1, transtime=1, group_num=9, drop=False):
        self.bandwidth = bandwidth
        self.transtime = transtime
        self.nodes = {}
        self.links = {}
        self.total_packets = 0
        self.active_packets = 0
        self.drop_packets = 0
        self.end_packets = 0
        self.clock = Clock(0)
        self.adj = adj
        self.group_num = group_num
        self.generate(adj)
        self.event_queue = []
        self.agent = Policy(self)
        self.pre_agent = Policy(self)
        self.delays = []
        self.hops = 0
        self.route_time = 0
        self.is_drop = drop
        self.drop = []
        self.stored_packets = []
        self.groups = {}
        self.grouping()
        self.step_route_time = []

    def generate(self, adj):
        """
        generate network from adjacency matrix *adj*

        :param adj:
        """
        size = len(adj)
        nodes = [Node(i, [], self, self.clock, i//(size//self.group_num)) for i in range(len(adj))]
        ids = range(len(adj))
        self.nodes = dict(zip(ids, nodes))
        for i in range(size):
            for j in range(size):
                if adj[i][j] == 1:
                    a, b = 0, 0
                    for node in self.nodes.values():
                        if node.ID == i:
                            a = node
                        if node.ID == j:
                            b = node
                    a.outs.append(b)
                    a.refresh(len(self.nodes)//self.group_num)
                    self.links[a.ID] = self.links.get(a.ID, [])
                    self.links[a.ID].append(b.ID)
    
    def grouping(self):
        """
        Create groupings
        """
        self.groups = {}
        for i in range(self.group_num):
            self.groups[i] = []
            for node in self.nodes.values():
                if node.group == i:
                    self.groups[i].append(node)
            

    def bind(self, agent):
        """
        Bind agent to network
        """
        self.agent = agent

    def reset(self):
        """
        Reset network
        """
        self.nodes = {}
        self.links = {}
        self.clock = Clock(0)
        self.hops = 0
        self.route_time = 0
        self.delays = []
        self.event_queue = []
        self.total_packets = 0
        self.active_packets = 0
        self.drop_packets = 0
        self.end_packets = 0
        self.generate(self.adj)
        self.hops = 0
        self.route_time = 0
        self.drop = []
        self.agent.clean()
        self.pre_agent.clean()
        self.stored_packets = []
        self.groups = {}
        self.grouping()
        self.step_route_time = []

    def print_network(self):
        """
        display network topology
        """
        print(f"Number of nodes: {len(self.nodes)}")
        nlinks = 0
        for _, var in self.links.items():
            nlinks += len(var)
        print(f"Number of links: {nlinks}")
        for node in self.nodes.values():
            print(f"Node ID: {node.ID}")
            print(f"links to: ", end="")
            for out in node.outs:
                print(out.ID, end=" ")
            print()

    def new_packet(self, lambd):
        """
        create n packets under uniform traffic

        :param n:
        :return list of packets:
        """
        packets = []
        for _ in range(np.random.poisson(lambd)):
            source = np.random.randint(0, len(self.nodes))
            dest = np.random.randint(0, len(self.nodes))
            while dest == source:
                dest = np.random.randint(0, len(self.nodes))
            birtht = self.clock.t
            packet = Packet(source, dest)
            packet.birtht = birtht
            packet.cur_wait = birtht
            packet.node_wait = birtht
            packets.append(packet)
        return packets

    def new_packet_adv(self, lambd, h):
        """
        create n packets under adversarial traffic with adv+h

        :param n:
        :return list of packets:
        """
        packets = []
        g_num = 4
        for _ in range(np.random.poisson(lambd)):
            source = np.random.randint(0, len(self.nodes))
            sg = self.nodes[source].group
            dg = sg + h
            if dg >= self.group_num:
                dg -= self.group_num
            drange = np.arange(dg*g_num, dg*g_num + 4, 1)
            dest = np.random.choice(drange)
            birtht = self.clock.t
            packet = Packet(source, dest)
            packet.birtht = birtht
            packet.cur_wait = birtht
            packet.node_wait = birtht
            packets.append(packet)
        return packets

    def inject(self, packets):
        """
        inject a list of packets *packets* into the network

        :param packets:
        """
        self.active_packets += len(packets)
        self.total_packets += len(packets)
        for packet in packets:
            self.nodes[packet.source].receive(packet)

    def end_packet(self, packet):
        """
        remove packet *packet* from the network
        """
        self.active_packets -= 1
        self.end_packets += 1
        packet.endt = self.clock.t
        self.delays.append(packet.endt - packet.birtht)
        self.route_time += self.clock.t - packet.birtht
        self.hops += packet.hops
        path = packet.return_path()
        self.stored_packets.append(packet)
        # print(f"path {path} for packet {packet}")
        del packet
    
    def destroy_packets(self):
        """
        remove all active packets from the network
        """
        self.total_packets -= self.active_packets
        self.active_packets = 0
        for node in self.nodes.values():
            for packet in node.inQueue:
                del packet

    def step(self, duration):
        """
        Advance the simulation by *duration*

        :param duration:
        :return:
        """
        rewards = []
        for node in self.nodes.values():
            event = node.send()
            if event:
                rewards += event

        route_time = []
        end_time = self.clock.t + duration
        next_event = nsmallest(1, self.event_queue)
        while len(next_event) > 0 and next_event[0].time <= end_time:
            e = heappop(self.event_queue)
            next_event = nsmallest(1, self.event_queue)
            if self.is_drop and e.packet.hops >= 10:
                # drop the packet if too many hops
                self.drop_packets += 1
                self.active_packets -= 1
                self.agent.drop_penalty(e)
                self.drop.append([e.packet.source, e.packet.dest])
                del e.packet
                continue
            self.clock.t = e.time
            self.nodes[e.to_node].receive(e.packet)
            if e.packet.dest == e.to_node:
                # received
                temp = self.clock.t - e.packet.birtht
                route_time.append(temp)

        self.clock.t = end_time
        if route_time != []:
            # print(route_time)
            route_time = np.average(route_time)
            self.step_route_time.append(route_time)
        else:
            self.step_route_time.append(0)
        return rewards

    def train(self, duration, lambd, slot=1, freq=1, lr={}, hop=False, drop=False, arrive=False, inject=True, adv=False, alting=False):
        """
        train the network for a given duration under load lambda

        hop: whether to record average hops
        drop: whether to record drop rate
        arrive: whether to record arrival rate
        inject: whether to inject packets every time slot
        adv: set to False for uniform traffic pattern, set to integer n for adv+n traffic pattern
        alting: test changing traffic load
        """
        step_num = int(duration / slot)
        result = {'route_time': np.zeros(step_num)}
        if hop:
            result['hop'] = np.zeros(step_num)
        if drop:
            result['drop'] = np.zeros(step_num)
        if arrive:
            result['arrive'] = np.zeros(step_num)
        self.offset = 0
        for i in tqdm(range(step_num)):
            if alting and (i == 10000 or i == 20000):
                self.reset()
            if alting:
                if i < 10000:
                    packets = self.new_packet_adv(4 * slot, 1)
                if i >= 10000 and i < 20000:
                    packets = self.new_packet_adv(10 * slot, 1)
                if i >= 20000:
                    packets = self.new_packet_adv(4 * slot, 1)
            else:
                if adv is not False:
                    packets = self.new_packet_adv(lambd * slot, adv)
                else:
                    packets = self.new_packet(lambd * slot)
            if inject:
                self.inject(packets)
            for _ in range(freq):
                r = self.step(slot)
                if r:
                    self.agent.learn(r)
            result['route_time'][i] = self.ave_route_time
            if hop:
                result['hop'][i] = self.ave_hops
            if drop:
                result['drop'][i] = self.drop_rate
            if arrive:
                result['arrive'][i] = self.complete_rate
        return result
    
    def train_one_load(self, duration, lambd, slot=1, freq=1, lr={}, droprate=False, hop=False, arrive=False, adv=False):
        """
        used for older testing, use train() instead
        """
        step_num = int(duration / slot)
        result = {'route_time': np.zeros(step_num)}
        if droprate:
            result['drop'] = np.zeros(step_num)
        if hop:
            result['hop'] = np.zeros(step_num)
        if arrive:
            result['arrive'] = np.zeros(step_num)
        for i in tqdm(range(step_num)):
            if not adv:
                self.inject(self.new_packet(lambd*slot))
            if adv:
                self.inject(self.new_packet_adv(lambd*slot, 1))
            for _ in range(freq):
                r = self.step(slot)
                if r is not None:
                    if lr:
                        self.agent.learn(r, lr=lr)
                    else:
                        self.agent.learn(r)
            result['route_time'][i] = self.ave_route_time
            if droprate:
                result['drop'][i] = self.drop_rate
            if hop:
                result['hop'][i] = self.ave_hops
            if arrive:
                result['arrive'][i] = self.complete_rate
        return result


    def print_node_info(self):
        """
        Output all node info
        """
        print(f"Total packets: {self.total_packets}, active packets: {self.active_packets}, ended packets: {self.end_packets}")
        for node in self.nodes.values():
            print(f"{node}")

    @property
    def ave_hops(self):
        return self.hops / self.end_packets if self.end_packets > 0 else 0

    @property
    def ave_route_time(self):
        return self.route_time / self.end_packets if self.end_packets > 0 else 0

    @property
    def drop_rate(self):
        return self.drop_packets / self.total_packets if self.total_packets > 0 else 0

    @property
    def complete_rate(self):
        return self.end_packets / self.total_packets if self.total_packets > 0 else 0
