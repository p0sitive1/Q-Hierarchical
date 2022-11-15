import numpy as np
import sys

def generate_dragonfly(router_per_p, connections):
    colleague = router_per_p
    connection = connections
    group = colleague*connection+1
    node = group*colleague

    topology = np.zeros((node,node))
    for i in range(node):
        for j in range(node):
            if i==j:
                topology[i,j]=0
            elif i//colleague == j//colleague:
                topology[i,j]=1
            elif ((j//colleague-i//colleague)+group)%group in range( ((colleague-1-i%colleague)*connection)+1, ((colleague-1-i%colleague)*connection)+connection+1) and j%colleague == colleague-1-i%colleague:
                topology[i,j]=1

    return topology


def read_from_file(filename):
    f = open(filename, "r")
    adj = list()
    lines = f.readlines()
    for line in lines:
        line = line[1:-2]
        line = line.split(", ")
        temp = list()
        for item in line:
            item = float(item)
            temp.append(item)
        adj.append(temp)
    f.close()
    return adj


np.set_printoptions(threshold=sys.maxsize)
f = open("topo.txt", "w")
for l in generate_dragonfly(8, 4):
    f.write(repr(list(l)))
    f.write("\n")
f.close()
