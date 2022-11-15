import pickle

with open("/home/rm2022/DQN_routing/DRL/dump_dragonfly/uniform/qroute/10.0.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
