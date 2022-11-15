from ast import Interactive
from sympy import Q
from builtins import list
from locale import normalize
from re import S, X
from cProfile import label, run
import pickle
from bs4 import ResultSet
import numpy as np
from matplotlib import pyplot as plt
from env import *
from models.shortest import *
# from qr_one import Qroute
from models.dqnn import *
from tqdm import tqdm
from models.globalroute import *
from models.hierarch_alt import *
from models.val import *
from models.shortest_alt import *
from models.qrouting_alt import *
from models.qadapt import *


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


if __name__ == "__main__":
    # 6x6 test set
    # adj = [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    # ]

    # dragonfly
    adj = [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
           [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        ]

    # large dargonfly
    # adj = read_from_file("/home/rm2022/topo.txt")

    # network = Network(adj, group_num=33, drop=True)
    network = Network(adj, group_num=9, drop=False)
    # network.print_network()
    # network.print_node_info()

    duration = 10000
    slot = 1
    # loads = np.arange(0.25, 16.25, 0.25)
    # loads = np.arange(20, 100, 2.5)

    # loads = np.arange(20, 40, 2.5)
    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_adv4_fin.txt', 'a')

    #-------------------global route-------------------#
    # agent = GlobalRoute(network)
    # network.bind(agent)
    # network.reset()
    # # np.random.seed(221)

    # # result = network.train(duration=5000, lambd=60, adv=1)
    # # print(f"load {60} completed with {result['route_time'][-1]}")

    # gr_ave_result = list()
    # for load in loads:
    #     network.reset()
    #     result = network.train(duration=5000, lambd=load, adv=1)
    #     gr_ave_result.append(result['route_time'][-1])
    #     print(f"load {load} completed")

    # f = open('/home/rm2022/DQN_routing/DRL/global.txt', "w")
    # f.write(repr(gr_ave_result))
    # f.write("\n")

    #--------------------shortest----------------------#
    # agent = ShortestL(network)
    # network.bind(agent)
    # network.reset()

    # # result = network.train(duration=10000, lambd=30, slot=slot, hop=True, adv=4)
    # # print(f"completed with result {result['route_time'][-1]}")

    # loads = np.arange(0.25, 16.25, 0.25)
    # short_ave_result = list()
    # short_x_var = list()
    # for load in loads:
    #     run_res = list()
    #     for i in range(1):
    #         result = network.train(duration=10000, lambd=load, slot=slot, hop=True, adv=False)
    #         # run_res.append(result["route_time"][-1])
    #         run_res.append(result["hop"][-1])
    #         network.reset()
    #     short_ave_result.append(np.min(run_res))
    #     short_x_var.append(load)
    #     print(f"load {load} completed with result {np.min(run_res)}")

    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_hops.txt', "w")
    # f.write(repr(short_ave_result))
    # f.write("\n")

    #-----------------qroute training------------------#
    # agent = Qroute(network)
    # network.bind(agent)
    # network.reset()

    # q_ave_result = list()
    # q_x_var = list()
    # loads = [3, 8, 11]
    # for load in loads:
    #     run_res = list()
    #     # network.agent.load(f'/home/rm2022/DQN_routing/DRL/dump_dragonfly/uniform/qroute/{load}.pkl')
    #     # network.agent.load(f'/home/rm2022/DQN_routing/DRL/{load}.pkl')
    #     result = network.train(duration=duration*10, lambd=load, slot=slot, adv=False)
    #     run_res.append(result["route_time"][-1])
    #     # network.agent.store(f'/home/rm2022/DQN_routing/DRL/dump_dragonfly/uniform/qroute/{load}.pkl')
    #     network.agent.store(f'/home/rm2022/DQN_routing/DRL/test{load}.pkl')
    #     network.reset()
    #     # q_ave_result.append(np.min(run_res))
    #     # q_x_var.append(load)
    #     print(f"load {load} completed")

    #------------------qroute testing-------------------#
    # agent = Qroute(network, static=True)
    # network.bind(agent)
    # network.reset()
    # q_ave_result = list()
    # q_x_var = list()

    # # duration = 4000
    # for load in loads:
    #     run_res = list()
    #     for i in range(5):
    #         network.agent.load(f'/home/rm2022/DQN_routing/DRL/12.0.pkl')
    #         result = network.train(duration=duration, lambd=load, slot=slot, hop=True, adv=True)
    #         run_res.append(result["route_time"][-1])
    #         # run_res.append(result["hop"][-1])
    #         network.reset()
    #     q_ave_result.append(np.min(run_res))
    #     q_x_var.append(load)
    #     print(f"load {load} completed with {np.min(run_res)}")

    # f.write(repr(q_ave_result))
    # f.write("\n")
    # f.close()

    # agent = Qroute(network)
    # network.bind(agent)
    # network.reset()
    # network.agent.load(f'/home/rm2022/DQN_routing/DRL/adv11.0.pkl')
    # result = network.train(duration=100000, lambd=11, adv=True)
    # network.agent.store(f'/home/rm2022/DQN_routing/DRL/adv11.0.pkl')

    # plt.plot(result['route_time'])
    # plt.savefig(f"/home/rm2022/DQN_routing/DRL/qadv.png")
    # plt.close()

    # agent = Qroute(network, static=True)
    # agent = Shortest(network)
    # network.bind(agent)
    # network.reset()

    # network.agent.load(f'/home/rm2022/DQN_routing/DRL/adv15.0.pkl')
    # result = network.train(duration=5000, lambd=10.5, adv=True)
    # print(result['route_time'][-1])

    #-----------------shortest & qroute path match-----------#
    # agent = ShortestL(network)
    # network.bind(agent)
    # network.reset()
    # for load in [13.0]:
    #     # network.inject(packets)
    #     result = network.train(duration=10000, lambd=load, slot=slot, adv=1)
    #     shortest_p = network.stored_packets

    # agent = Qadapt(network, static=True)
    # network.bind(agent)
    # network.reset()
    # for load in [13.0]:
    #     network.agent.load(f'/home/rm2022/DQN_routing/DRL/qadapt12adv.pkl')
    #     # network.agent.load(f'/home/rm2022/DQN_routing/DRL/qadapt12.pkl')
    #     result = network.train(duration=10000, lambd=load, slot=slot, adv=1)
    #     adapt_p = network.stored_packets

    # agent = Qroute(network, static=True)
    # network.bind(agent)
    # network.reset()
    # for load in [13.0]:
    #     network.agent.load(f'/home/rm2022/qroute_retest.pkl')
    #     # network.agent.load(f'/home/rm2022/DQN_routing/DRL/qadapt12.pkl')
    #     result = network.train(duration=10000, lambd=load, slot=slot, adv=1)
    #     print(result["route_time"][-1])
    #     q_p = network.stored_packets

    # agent = VAL(network)
    # network.bind(agent)
    # network.reset()
    # for load in [13.0]:
    #     result = network.train(duration=10000, lambd=load, slot=slot, adv=1)
    #     val_p = network.stored_packets

    # agent = Hierarch(network, static=True)
    # network.bind(agent)
    # network.reset()
    # for load in [13.0]:
    #     network.agent.load(f'/home/rm2022/DQN_routing/DRL/hieadv1_retest.pkl')
    #     # network.agent.load(f'/home/rm2022/DQN_routing/DRL/hie12in.pkl')
    #     result = network.train(duration=10000, lambd=load, slot=1, freq=1, adv=1)
    #     print(result['route_time'][-1])
    #     hie_p = network.stored_packets

    # class Links:
    #     def __init__(self, source, dest):
    #         self.source = source
    #         self.dest = dest
    #         self.used = 0
    #         self.type = 0

    #     def __str__(self):
    #         return f"link from {self.source} to {self.dest}, used {self.used} times"

    #     def __repr__(self):
    #         return f"link from {self.source} to {self.dest}, used {self.used} times"

    # def calc_freq(inputl):
    #     log = list()
    #     for link in network.links.items():
    #         for i in link[1]:
    #             l = Links(link[0], i)
    #             temp = l.source - (l.source % 4)
    #             if temp <= l.dest < temp + 4:
    #                 l.type = 0
    #             else:
    #                 l.type = 1
    #             log.append(l)

    #     for sp in set(inputl):
    #         # path = list()
    #         path = sp.path
    #         # print(path)
    #         sep = list()
    #         source = sp.source
    #         sep.append([source, path[0]])
    #         for i in range(len(path)-1):
    #             sep.append([path[i], path[i+1]])

    #         for route in sep:
    #             for link in log:
    #                 if (route[0] == link.source and route[1] == link.dest):
    #                     link.used += 1

    #     stored = list()
    #     for item in log:
    #         for check in log:
    #             if item.source == check.dest and item.dest == check.source:
    #                 item.used += check.used
    #                 check.used = 0
    #                 del check
    #                 stored.append(item)

    #     log = list()
    #     for item in stored:
    #         if item.used != 0:
    #             log.append(item)

    #     values = dict()
    #     interv = dict()
    #     intrav = dict()
    #     for item in log:
    #         # values[f"{item.source} to {item.dest}"] = item.used
    #         if item.type == 1:
    #             interv[f"{item.source} to {item.dest}"] = item.used
    #         elif item.type == 0:
    #             intrav[f"{item.source} to {item.dest}"] = item.used

    #     # total = np.sum(values.values())
    #     intert = np.sum(interv.values())
    #     intrat = np.sum(intrav.values())

    #     return interv, intrav

    # def graph_c(listvar):
    #     output = list()
    #     length = len(listvar)
    #     maxv = max(listvar)
    #     minv = min(listvar)
    #     rang = maxv-minv
    #     addv = 1
    #     # startv = minv
    #     startv = 0
    #     for i in range(maxv):
    #     # for i in range(0, 1,):
    #         tot = 0
    #         startv += addv
    #         for j in listvar:
    #             if j <= startv:
    #                 tot += 1
    #         output.append(tot/length)

    #     return output

    # s = calc_freq(shortest_p)
    # a = calc_freq(adapt_p)
    # v = calc_freq(val_p)
    # h = calc_freq(hie_p)
    # g = calc_freq(q_p)

    # # print(network.links)
    # def get_zeros(s):
    #     s = list(s)
        
    #     for i in range(36):
    #         for j in range(36):
    #             if i != j:
    #                 if i // 4 != j // 4:
    #                     # intergroup
    #                     if f"{i} to {j}" in s[0].keys() or f"{j} to {i}" in s[0].keys():
    #                         # exists
    #                         pass
    #                     else:
    #                         if j in network.links[i] or i in network.links[j]:
    #                             s[0][f"{i} to {j}"] = 0
    #                 else:
    #                     # intragroup
    #                     if f"{i} to {j}" in s[1].keys() or f"{j} to {i}" in s[1].keys():
    #                         pass
    #                     else:
    #                         if j in network.links[i] or i in network.links[j]:
    #                             s[1][f"{i} to {j}"] = 0
    #     return s

    # s = get_zeros(s)
    # a = get_zeros(a)
    # v = get_zeros(v)
    # h = get_zeros(h)
    # g = get_zeros(g)

    # s0, s1 = s[0].values(), s[1].values()
    # a0, a1 = a[0].values(), a[1].values()
    # v0, v1 = v[0].values(), v[1].values()
    # h0, h1 = h[0].values(), h[1].values()
    # g0, g1 = g[0].values(), g[1].values()

    # sout0 = graph_c(s0)
    # aout0 = graph_c(a0)
    # vout0 = graph_c(v0)
    # hout0 = graph_c(h0)
    # gout0 = graph_c(g0)

    # sout1 = graph_c(s1)
    # aout1 = graph_c(a1)
    # vout1 = graph_c(v1)
    # hout1 = graph_c(h1)
    # gout1 = graph_c(g1)

    # def normalize(arr, t_min, t_max):
    #     norm_arr = []
    #     diff = t_max - t_min
    #     diff_arr = max(arr) - min(arr)    
    #     for i in arr:
    #         temp = (((i - min(arr))*diff)/diff_arr) + t_min
    #         norm_arr.append(temp)
    #     return norm_arr

    # # plt.hist(sout, cumulative=True, histtype='step', label="Shortest")
    # # plt.hist(qout, cumulative=True, histtype='step', label="Qroute")
    # # plt.hist(gout, cumulative=True, histtype='step', label="Global")

    # # plt.plot(range(len(sout)), sout, label="Shortest")
    # # plt.plot(range(len(qout)), qout, label="Qroute")
    # # plt.plot(range(len(gout)), gout, label="Global")

    # # plt.ylim(0.95, 1)
    # # plt.xlim(0, 10000)
    # # plt.plot(range(len(sout0)), sout0, label="MIN intergroup")
    # # #plt.plot(range(len(sout1)), sout1, label="MIN intragroup")
    # # plt.legend(loc='lower right')
    # # plt.title("Under load 10")
    # # plt.savefig(f"/home/rm2022/DQN_routing/DRL/compshort10high.png")
    # # plt.close()

    # # plt.ylim(0.95, 1)
    # # plt.xlim(0, 10000)
    # # plt.plot(range(len(aout0)), aout0, label="Q-adaptive intergroup")
    # # #plt.plot(range(len(aout1)), aout1, label="Q-adaptive intragroup")
    # # plt.legend(loc='lower right')
    # # plt.title("Under load 10")
    # # plt.savefig(f"/home/rm2022/DQN_routing/DRL/compqadapt10high.png")
    # # plt.close()
    
    # # plt.ylim(0.95, 1)
    # # plt.xlim(0, 10000)
    # # plt.plot(range(len(hout0)), hout0, label="Q-hierarchical intergroup")
    # # #plt.plot(range(len(hout1)), hout1, label="Q-hierarchical intragroup")
    # # plt.legend(loc='lower right')
    # # plt.title("Under load 10")
    # # plt.savefig(f"/home/rm2022/DQN_routing/DRL/comphie10high.png")
    # # plt.close()
    
    # # plt.ylim(0.95, 1)
    # # plt.xlim(0, 10000)
    # # plt.plot(range(len(vout0)), vout0, label="VAL intergroup")
    # # #plt.plot(range(len(vout1)), vout1, label="VAL intragroup")
    # # plt.legend(loc='lower right')
    # # plt.title("Under load 10")
    # # plt.savefig(f"/home/rm2022/DQN_routing/DRL/compVAL10high.png")
    # # plt.close()

    # rs = normalize(np.arange(0, len(sout0), 1), 0, 1)
    # rv = normalize(np.arange(0, len(vout0), 1), 0, 1)
    # rg = normalize(np.arange(0, len(gout0), 1), 0, 1)
    # ra = normalize(np.arange(0, len(aout0), 1), 0, 1)
    # rh = normalize(np.arange(0, len(hout0), 1), 0, 1)

    # f = open(ff"/home/rm2022/DQN_routing/DRL/usage_data.txt", "w")
    # f.write(f"{repr(sout0)}\n")
    # f.write(f"{repr(vout0)}\n")
    # f.write(f"{repr(gout0)}\n")
    # f.write(f"{repr(aout0)}\n")
    # f.write(f"{repr(gout0)}\n")
    # f.close()

    # fig = plt.figure(figsize=(10, 4.5))
    # ax = plt.axes()
    # ax.plot(rs, sout0, label="MIN", color="green", linewidth="2.5")
    # ax.plot(rs[:len(vout0)], vout0, label="VAL", color="blue", linewidth="2.5")
    # ax.plot(rs[:len(gout0)], gout0, label="Q-Routing", color="purple", linewidth="2.5")
    # ax.plot(rs[:len(aout0)], aout0, label="Q-Adaptive", color="grey", linewidth="2.5")
    # ax.plot(rs[:len(hout0)], hout0, label="Q-Hierarchical", color="red", linewidth="2.5")
    # ax.legend()
    # ax.tick_params(axis='both', labelsize=15)
    # # plt.title("Inter Under load 13")
    # ax.set_ylabel("Inter-group Link Percentage", size=19)
    # ax.set_xlabel("Normalized Link Usage", size=19)
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(2)

    # # increase tick width
    # ax.tick_params(width=2)
    # plt.savefig(f"/home/rm2022/DQN_routing/DRL/apparently_good_inter_adv.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # plt.xlim(0, 10000)
    # plt.plot(range(len(sout1)), sout1, label="MIN", color="green")
    # plt.plot(range(len(vout1)), vout1, label="VAL", color="blue")
    # plt.plot(range(len(gout1)), gout1, label="Q-Routing", color="purple")
    # plt.plot(range(len(aout1)), aout1, label="Q-Adaptive", color="grey")
    # plt.plot(range(len(hout1)), hout1, label="Q-Hierarchical", color="red")
    # plt.legend()
    # # plt.title("Under load 13")
    # plt.ylabel("Percentage of Intragroup Links")
    # plt.xlabel("Link Usage")
    # plt.savefig(f"/home/rm2022/DQN_routing/DRL/apparently_good_intra_adv.png")
    # plt.close()

    # print(values.values())

    # print(np.var(np.array(list(values.values()))))
    # plt.figure(figsize=(10, 18))
    # plt.barh(range(len(values.keys())), values.values(), color ='red', height = 0.4)
    # plt.yticks(range(len(values.values())), values.keys())

    # plt.savefig(f"/home/rm2022/DQN_routing/DRL/global_link_traffic.png")
    # plt.close()

    # agent = Qroute(network, static=True)
    # network.bind(agent)
    # network.reset()
    # for load in [12.0]:
    #     network.agent.load(f'/home/rm2022/DQN_routing/DRL/dump_dragonfly/uniform/qroute/{load}.pkl')
    #     # network.inject(shortest_p)
    #     result = network.train(duration=1000, lambd=load, slot=slot, hop=True, adv=False)
    #     adapt_p = network.stored_packets

    # tot, mat = 0, 0
    # for sp in set(shortest_p):
    #     for qp in set(adapt_p):
    #         if sp.source == qp.source and sp.dest == qp.dest:
    #             # print(sp, sp.path, qp.path)
    #             if len(sp.path) == len(qp.path):
    #                 tot += 1
    #                 mat += 1
    #             else:
    #                 tot += 1

    # print(f"{mat/tot*100}% matching under load 12.0")

    #----------------------------------shortest path usage---------------#
    # agent = Shortest(network)
    # network.bind(agent)
    # network.reset()

    # test_p = list()
    # for i in range(36):
    #     for j in range(i+1, 36, 1):
    #         p = Packet(i, j)
    #         test_p.append(p)

    # network.inject(test_p)
    # result = network.train(duration=10000, lambd=0, slot=1, inject=False, adv=False)
    # result_p = network.stored_packets

    # rp = calc_freq(result_p)

    # r0, r1 = rp[0].values(), rp[1].values()
    # r2 = list()
    # for i in list(r0):
    #     r2.append(i)
    # for i in list(r1):
    #     r2.append(i)
    # r2.sort()

    # ro0 = graph_c(r0)
    # ro1 = graph_c(r1)
    # ro2 = graph_c(r2)

    # plt.plot(range(len(ro0)), ro0, label="Shortest intergroup")
    # plt.plot(range(len(ro1)), ro1, label="Shortest intragroup")
    # plt.plot(range(len(ro2)), ro2, label="Shortest total")
    # plt.legend(loc='upper left')
    # plt.title("Under load controlled")
    # plt.savefig(f"/home/rm2022/DQN_routing/DRL/compshortctrl.png")
    # plt.close()

    #------------------deep q route---------------------#
    # pre training
    # network.bind(Qroute(network, memory_capa=50000))
    # network.agent.epsilon = 0
    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/dump_dragonfly/adv+1/qroute/adv8.0.pkl")

    # result = network.train_one_load(50000, 8, lr={'q': 0.4}, adv=True)
    # network.reset()

    # plt.plot(result['route_time'])
    # plt.savefig(f"/home/rm2022/DQN_routing/DRL/resultpre.png")
    # plt.close()
    # m = network.agent.memory
    # print(m)

    # agent = DRL(network, static=True)
    # network.bind(agent)
    # network.reset()

    # net = DQN(36, 3, 8)
    # network.agent.build_model(net, net)

    # lr = 5e-3
    # network.agent.reset_optimizer(lr)
    # network.agent.config.memory_capacity = 50000
    # network.agent.config.batch = 64
    # network.agent.build_memory()

    # print(network.agent)

    # pre_time = 1000000
    # network.agent.pre_training(m, pre_time)
    # network.agent.store(f"/home/rm2022/DQN_routing/DRL/pre5000bk")

    # load trained policy
    # network.agent.load(f'/home/rm2022/DQN_routing/DRL/dump_dragonfly/uniform/dqn/l15')
    # experiment_range = np.arange(0.25, 16.25, 0.25)
    # d_ave_result = list()
    # duration=10000
    # for i in tqdm(experiment_range):
    #     results = list()
    #     for j in range(5):
    #         network.reset()
    #         result = network.train(duration=duration, slot=slot, lambd=i)
    #         results.append(result['route_time'][-1])
    #     print(f"{i} finished and average time is {np.min(results)}")
    #     d_ave_result.append(np.min(results))

    # f = open(f"/home/rm2022/DQN_routing/DRL/testout_dragonfly.txt", "a")
    # f.write(repr(d_ave_result))
    # f.close()
    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/pre5000bk")

    # dres = list()
    # network.reset()
    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/dump_dragonfly/uniform/dqn/l15")
    # result = network.train(duration=10000, slot=slot, lambd=13.75, adv=False)
    # # network.agent.store(f"/home/rm2022/DQN_routing/DRL/11.5")
    # # results.append(result['route_time'][-1])
    # print(f"finsihed with {result['route_time'][-1]}")
    # # dres.append(np.min(results))

    # # f.write(repr(dres))
    # # f.close()
    # plt.plot(result['route_time'])
    # plt.savefig(f"/home/rm2022/DQN_routing/DRL/dadv.png")
    # plt.close()

    #------------------deep q route with queue info---------------------#
    # pre training
    # agent = DRL(network, static=False)
    # network.bind(agent)
    # network.reset()

    # net = DQN(36, 3, 8)
    # network.agent.build_model(net, net)

    # lr = 5e-3
    # network.agent.reset_optimizer(lr)
    # network.agent.config.memory_capacity = 10000
    # network.agent.config.batch = 64
    # network.agent.build_memory()

    # network.agent.load(f' 3.2')
    # result = network.train(duration=10000, slot=slot, lambd=2)
    # print(f"finished and average time is {result['route_time'][-1]}")
    # plt.plot(result['route_time'])
    # plt.savefig(f"pretou.png")
    # plt.close()

    # m = network.agent.memory
    # pm = network.agent.pre_memory

    # network.bind(Qroute(network, memory_capa=50000))
    # network.agent.epsilon = 0
    # # network.agent.load(f" dump/l1.pkl")
    # # network.agent.load(f" 1.0")
    # network.reset()
    # network.agent.load(f" dump_6x6/qroute/2.0.pkl")
    # result = network.train(50000, 2,  lr={'q': 0.4}, pre=False)
    # plt.plot(result['route_time'])
    # plt.savefig(f" pretou.png")
    # plt.close()
    # network.reset()
    # m = network.agent.memory
    # print(m)

    # agent = DRL_4(network, static=True)
    # network.bind(agent)
    # network.reset()

    # net = DQN_4(36, 3, 6, 8)
    # network.agent.build_model(net, net)

    # lr = 5e-3
    # network.agent.reset_optimizer(lr)
    # network.agent.config.memory_capacity = 50000
    # network.agent.config.batch = 64
    # network.agent.build_memory()

    # print(network.agent)

    # pre_time = 50000
    # network.agent.pre_training(pm, pre_time)
    # network.agent.store(f" 3pre5000")

    # load trained policy
    # load = 1
    # network.agent.load(f' l0.2')
    # network.agent.load(f' 3pre5000')
    # experiment_range = np.arange(3, 4, 0.25)
    # dur = 1000
    # network.agent.load(f' l3')
    # results = list()
    # for i in experiment_range:
    #     network.reset()
    #     # network.destroy_packets()
    #     # network.agent.load(f' l3')
    #     result = network.train(duration=5000, slot=slot, lambd=i, pre=True)
    #     # if result['route_time'][-1] < 8:
    #     # network.agent.store(f" l3")
    #     results.append(result['route_time'][-1])
    #     print(f"{i} finished and average time is {result['route_time'][-1]}"

    #-----------------------Hierarchy structure--------------#
    # agent = Hierarch(network, pre=True)
    # network.bind(agent)
    # network.reset()

    # result = network.train(duration=10000, slot=1, freq=1, lambd=10, adv=1)
    # network.agent.store(f"/home/rm2022/DQN_routing/DRL/hieadv1_retest.pkl")

    # agent = Hierarch(network)
    # network.bind(agent)
    # network.reset()

    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/hieadv1_retest.pkl")
    # for i in list(network.agent.inTable[9].items()):
    #     print(i)
    # for j in list(network.agent.outTable[2].items()):
    #     print(j)
    # result = network.train(duration=100000, slot=1, freq=1, lambd=10, adv=1)
    # network.agent.store(f"/home/rm2022/DQN_routing/DRL/hieadv1_retest.pkl")
    # # for i in range(50):
    # #     network.reset()
    # #     network.agent.load(f"/home/rm2022/DQN_routing/DRL/hielargeadv_alt.pkl")
    # #     result = network.train(duration=1000, slot=1, freq=1, lambd=80, adv=1)
    # #     network.agent.store(f"/home/rm2022/DQN_routing/DRL/hielargeadv_alt.pkl")

    # agent = Hierarch(network, static=True)
    # network.bind(agent)
    # network.reset()

    # np.set_printoptions(suppress=True)
    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/hieadv1_retest.pkl")

    # for i in list(network.agent.inTable[9].items()):
    #     print(i)
    # for j in list(network.agent.outTable[2].items()):
    #     print(j)
    # result = network.train(duration=10000, slot=1, freq=1, lambd=10, adv=1)
    # ps = network.stored_packets
    # for p in ps[-30:]:
    #     print(p, p.path)
    # print(result['route_time'][-1])
    # plt.plot(result['route_time'])
    # plt.savefig(f"/home/rm2022/DQN_routing/DRL/hie1.png")
    # plt.close()

    # FULL TEST
    # for load in loads:
    #     network.reset()
    #     result = network.train(duration=1000000, slot=1, freq=1, lambd=load, inject=True)
    #     network.agent.store(f"outload{load}.pkl")
    #     print(f"load {load} complete")

#     agent = Hierarch(network, static=True)
#     network.bind(agent)
#     network.reset()

#     hierarch_res = list()
#     # loads = np.arange(0.25, 7.25, 0.25)
#     loads = np.arange(0.25, 16.25, 0.25)
#     for load in loads:
#         run_res = list()
#         for i in range(1):
#             network.reset()
#             # network.agent.load(f"/home/rm2022/DQN_routing/DRL/hielargeadv4_alt.pkl")
#             network.agent.load(f"/home/rm2022/DQN_routing/DRL/hie12in.pkl")
#             result = network.train(duration=10000, slot=1, freq=1, lambd=load, inject=True, adv=False, hop=True)
#             print(f"load {load} complete with result {result['hop'][-1]}")
#             run_res.append(result['hop'][-1])
#         hierarch_res.append(np.min(run_res))

# #     # line 1 - uniform
#     f = open("/home/rm2022/DQN_routing/DRL/hie.txt", "a")
#     f.write(repr(hierarch_res))
#     f.close()

    #-----------------------VAL-----------------------#
    # agent = VAL(network)
    # network.bind(agent)
    # network.reset()

    # # loads = np.arange(20, 100, 2.5)
    # loads = np.arange(0.25, 16.25, 0.25)

    # val_res = list()
    # for load in loads:
    #     run_res = list()
    #     for i in range(1):
    #         network.reset()
    #         result = network.train(duration=10000, slot=1, freq=1, lambd=load, inject=True, hop=True, adv=False)
    #         run_res.append(result['hop'][-1])
    #     val_res.append(np.min(run_res))
    #     print(f"load {load} completed with result {np.min(run_res)}")

    # f = open(f"/home/rm2022/DQN_routing/DRL/val.txt", "a")
    # f.write(repr(val_res))
    # f.close()

    #-------------------Q adapt--------------------#
    # agent = Qadapt(network, pre=True)
    # network.bind(agent)
    # network.reset()

    # result = network.train(duration=10000, slot=1, freq=1, lambd=40, inject=True, adv=4)
    # network.agent.store(f"/home/rm2022/DQN_routing/DRL/qadaptlargeadv4.pkl")

    # agent = Qadapt(network)
    # network.bind(agent)
    # network.reset()

    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/qadaptlargeadv4.pkl")
    # result = network.train(duration=100000, slot=1, freq=1, lambd=40, inject=True, adv=4)
    # network.agent.store(f"/home/rm2022/DQN_routing/DRL/qadaptlargeadv4.pkl")

    # agent = Qadapt(network, static=True)
    # network.bind(agent)
    # network.reset()

    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/qadaptlargeadv4.pkl")
    # for j in list(network.agent.QTable[2].items()):
    #     print(j)
    # result = network.train(duration=10000, slot=1, freq=1, lambd=40, inject=True, adv=4)
    # plt.plot(result['route_time'])
    # plt.savefig(f"adapted.png")
    # plt.close()
    # print(f"completed with {result['route_time'][-1]}")

    # agent = Qadapt(network, static=True)
    # network.bind(agent)
    # network.reset()

    # loads = np.arange(0.25, 16.25, 0.25)
    # f = open(f"/home/rm2022/DQN_routing/DRL/qadapt.txt", "a")
    # qadapt_res = list()
    # for load in loads:
    #     for i in range(1):
    #         run_res = list()
    #         network.reset()
    #         # network.agent.load(f"/home/rm2022/DQN_routing/DRL/qadaptlargeadv4.pkl")
    #         network.agent.load(f"/home/rm2022/DQN_routing/DRL/qadapt12.pkl")
    #         result = network.train(duration=10000, slot=1, freq=1, lambd=load, inject=True, adv=False, hop=True)
    #         run_res.append(result['hop'][-1])
    #     qadapt_res.append(np.min(run_res))
    #     print(f"{load} completed with {np.min(run_res)}")
    
    # f.write(repr(qadapt_res))
    # f.write("\n")
    # f.close()

    #---------------------misc------------------------#
    # agent = Qroute(network, pre=True)
    # network.bind(agent)
    # network.reset()

    # result = network.train(duration=10000, slot=1, freq=1, lambd=9, adv=1)
    # network.agent.store("qroute_retest.pkl")

    # agent = Qroute(network)
    # network.bind(agent)
    # network.reset()

    # network.agent.load("qroute_retest.pkl")
    # result = network.train(duration=100000, slot=1, freq=1, lambd=9, adv=1)
    # network.agent.store("qroute_retest.pkl")

    # agent = Qroute(network, static=True)
    # network.bind(agent)
    # network.reset()

    # network.agent.load("qroute_retest.pkl")
    # # print(network.agent.Qtable[0])
    # result = network.train(duration=10000, slot=1, freq=1, lambd=9, adv=1)
    # print(f"done with {result['route_time'][-1]}")

    # agent = Qroute(network, static=True)
    # network.bind(agent)
    # network.reset()

    # loads = np.arange(0.25, 16.25, 0.25)
    # f = open(f"/home/rm2022/DQN_routing/DRL/qroutealt.txt", "a")
    # qresult = list()
    # for load in loads:
    #     # network.agent.load("uni1adv.pkl")
    #     # print(network.agent.Qtable[0])
    #     for i in range(1):
    #         run_res = list()
    #         network.reset()
    #         # network.agent.load(f'uni85adv4s.pkl')
    #         # network.agent.load(f'largeadv4.pkl')  # large topo
    #         # network.agent.load(f"/home/rm2022/DQN_routing/DRL/qadapt10adv")
    #         network.agent.load("qroute_retest.pkl")
    #         result = network.train(duration=10000, slot=1, freq=1, lambd=load, adv=1, hop=True)
    #         run_res.append(result['hop'][-1])
    #     print(f"{load} done with {result['hop'][-1]}")
    #     qresult.append(np.min(run_res))
        
    # f.write(repr(qresult))
    # f.write("\n")
    # f.close()


    #-------------------------hie changing load----------------#
    # agent = Hierarch(network, pre=True)
    # network.bind(agent)
    # network.reset()

    # result = network.train(duration=10000, slot=1, freq=1, lambd=10, adv=False)
    # network.agent.store(f"/home/rm2022/DQN_routing/DRL/hie_retest.pkl")

    # agent = Hierarch(network)
    # network.bind(agent)
    # network.reset()

    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/hie_retest.pkl")
    # result = network.train(duration=100000, slot=1, freq=1, lambd=10, adv=False)
    # network.agent.store(f"/home/rm2022/DQN_routing/DRL/hie_retest.pkl")

    # agent = Hierarch(network, static=True)
    # network.bind(agent)
    # network.reset()

    # # network.agent.load(f"/home/rm2022/DQN_routing/DRL/hieadv1_retest.pkl")
    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/hie_retest.pkl")
    # result = network.train(duration=10000, slot=1, freq=1, lambd=10, adv=False)
    # print(f"{result['route_time'][-1]}")

    # agent = Hierarch(network)
    # network.bind(agent)
    # network.reset()

    # network.agent.load(f"/home/rm2022/DQN_routing/DRL/hieadv1_retest.pkl")
    # # network.agent.load(f"/home/rm2022/DQN_routing/DRL/hie_retest.pkl")
    # # result = network.train(duration=30000, slot=1, freq=1, lambd=10, adv=1, alting=True)
    # # # result = network.train(duration=10000, slot=1, freq=1, lambd=10, adv=1, alting=False)
    # # # print(result["route_time"][-1])
    # # # network.agent.store(f"/home/rm2022/DQN_routing/DRL/hieadv1_retest.pkl")
    # # # print(network.step_route_time)
    # # # plt.plot(network.step_route_time)

    # # f = open(f"/home/rm2022/DQN_routing/DRL/stored_time_adv.txt", "w")
    # # f.write(repr(list(result['route_time'])))
    # # f.close()
    
    # f = open(f"/home/rm2022/DQN_routing/DRL/stored_time.txt", "r")
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # fig = plt.figure(figsize=(7, 5))
    # ax = plt.axes()
    # ax.plot(np.arange(0, 29.941, 0.001), data[0], linewidth='2.5')
    # ax.set_xlabel("Time (s)", size=19)
    # ax.set_ylabel("Average Packet Delay (ms)", size=19)
    # ax.tick_params(axis='both', labelsize=15)
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(2)

    # # increase tick width
    # ax.tick_params(width=2)
    # plt.savefig(f"/home/rm2022/DQN_routing/DRL/hie_alting_x.png", dpi=300, bbox_inches='tight')
    # plt.close()

    #-------------------Plot result------------------#

    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_adv.txt', 'w')
    # f.write(repr(short_ave_result))
    # f.write("\n")
    # f.write(repr(gr_ave_result))
    # f.write("\n")
    # f.write(repr(q_ave_result))
    # f.write("\n")
    # f.write(repr(d_ave_result))
    # f.write("\n")
    # f.close()
    plt.xlabel("Network load")
    plt.ylabel("Average packet delay")
    plt.ylim(0, 100)
    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_adv.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # plt.plot(loads, data[0], label="Shortest adv")
    # plt.plot(loads, data[1], label="Global Route adv")
    # plt.plot(loads, data[2], label="Qroute adv")
    # plt.plot(loads, data[3], label="DQN adv")
    # plt.plot(loads, data[5], label="Hierarch adv")
    # plt.plot(loads, data[4], label="VAL adv")
    # plt.legend()
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_adv.png")

    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # plt.plot(loads, data[0], label="Shortest")
    # plt.plot(loads, data[1], label="Global Route")
    # plt.plot(loads, data[2], label="Qroute")
    # plt.plot(loads, data[3], label="DQN")
    # plt.plot(loads, data[4], label="Hierarch")
    # plt.plot(loads, data[5], label="VAL")
    # # plt.plot(loads, data[4], label="DQN with queue info")
    # plt.legend()
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_w_h.png")

    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_adv4.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # plt.plot(loads, data[1], label="Shortest adv4")
    # plt.plot(loads, data[0], label="Global Route adv4")
    # plt.plot(loads, data[2], label="Qroute adv4")
    # plt.plot(loads, data[3], label="DQN adv4")
    # plt.plot(loads, data[5], label="Hierarch adv4")
    # plt.plot(loads, data[4], label="VAL adv4")
    # # plt.plot(loads, data[4], label="DQN with queue info")
    # plt.legend()
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_adv4.png")

    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_adv2.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # plt.plot(loads, data[0], label="Shortest adv2")
    # plt.plot(loads, data[1], label="VAL adv2")
    # plt.legend()
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_adv2.png")


    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_fin.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # plt.figure(figsize=(7, 5))
    # plt.ylim(0, 100)
    # plt.xlabel("Network load")
    # plt.ylabel("Average packet delay")
    # loads = np.arange(0.25, 16.25, 0.25)
    # plt.plot(loads, data[0], label="MIN", color="green", marker="o", markersize=4)
    # plt.plot(loads, data[1], label="Q-routing", color="red", marker=".", markersize=4)
    # plt.plot(loads, data[2], label="VAL", color="blue", marker="^", markersize=4)
    # plt.plot(loads, data[3], label="Global", color="orange", marker="*", markersize=4)
    # plt.plot(loads, data[4], label="Q-Hierarchical", color="purple", marker="X", markersize=4)
    # plt.plot(loads, data[5], label="Q-Adaptive", color="grey", marker="+", markersize=4)
    # plt.legend()
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_fin.png")
    # plt.close()

    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_fin.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # plt.figure(figsize=(7, 5))
    # plt.ylim(0, 100)
    # plt.xlim(14.5, 15.5)
    # loads = np.arange(0.25, 16.25, 0.25)
    # plt.plot(loads, data[0], label="MIN", color="green", marker="o", markersize=4)
    # plt.plot(loads, data[1], label="Q-routing", color="purple", marker=".", markersize=4)
    # # plt.plot(loads, data[2], label="VAL", color="blue", marker="^", markersize=4)
    # plt.plot(loads, data[3], label="Global", color="orange", marker="*", markersize=4)
    # plt.plot(loads, data[5], label="Q-Adaptive", color="grey", marker="+", markersize=4)
    # plt.plot(loads, data[4], label="Q-Hierarchical", color="red", marker="X", markersize=4)
    # plt.legend()
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_fin_mini.png")
    # plt.close()

#========================================Small Topo=============================================# 

    def process(input):
        out = list()
        for i in range(len(input)):
            if i % 2 == 0:
                out.append(input[i])
        return out

    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_fin.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # fig = plt.figure(figsize=(7, 5))
    # ax1 = plt.axes()
    # ax2 = plt.axes([0.19, 0.40, 0.33, 0.33])

    # ax1.set_ylim(0, 100)
    # loads = np.arange(0.25, 16.25, 0.5)
    # ax1.set_xlabel("Network load", size=19)
    # ax1.set_ylabel("Average packet delay", size=19)
    # ax1.plot(loads, process(data[0]), label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[3]), label="Global", color="orange", marker="*", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[2]), label="VAL", color="blue", marker="^", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[1]), label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[5]), label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[4]), label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    # # ax1.legend(ncol=2, prop={'size': 12})
    # ax2.set_xlim(14.5, 15.5)
    # ax2.set_ylim(0, 100)
    # # ax2.set_yticks([0, 25, 50, 75, 100], labelsize=5)
    # ax1.set_xticks(np.arange(0, 17, 2))
    # # ax2.set_xticks(np.arange(14.5, 16, 0.25))
    # ax2.tick_params(axis='both', labelsize=11)
    # ax1.tick_params(axis='both', labelsize=15)
    # ax2.plot(loads, process(data[0]), label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    # ax2.plot(loads, process(data[3]), label="Global", color="orange", marker="*", markersize=6, linewidth='2.5')
    # ax2.plot(loads, process(data[1]), label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    # # plt.plot(loads, data[2], label="VAL", color="blue", marker="^", markersize=4)
    # ax2.plot(loads, process(data[5]), label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    # ax2.plot(loads, process(data[4]), label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    # for axis in ['top','bottom','left','right']:
    #     ax1.spines[axis].set_linewidth(2)
    #     ax2.spines[axis].set_linewidth(1.5)

    # # increase tick width
    # ax1.tick_params(width=2)
    # ax2.tick_params(width=1.5)
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_fin_mini.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_adv_fin.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(7, 5))
    # ax = plt.axes()
    # ax.set_ylim(0, 100)
    # ax.set_xlabel("Network load", size=19)
    # ax.set_ylabel("Average packet delay", size=19)
    # loads = np.arange(0.25, 16.25, 0.5)
    # ax.set_xticks(np.arange(0, 17, 2))
    # ax.tick_params(axis='both', labelsize=15)
    # ax.plot(loads, process(data[0]), label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    # ax.plot(loads, process(data[1]), label="Global", color="orange", marker="*", markersize=6, linewidth='2.5')
    # ax.plot(loads, process(data[2]), label="VAL", color="blue", marker="^", markersize=6, linewidth='2.5')
    # ax.plot(loads, process(data[3]), label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    # ax.plot(loads, process(data[5]), label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    # ax.plot(loads, process(data[4]), label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    # # plt.legend(prop={'size': 12})
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(2)

    # # increase tick width
    # ax.tick_params(width=2)
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_adv1_fin.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_adv4_fin.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # fig = plt.figure(figsize=(7, 5))
    # ax1 = plt.axes()
    # ax2 = plt.axes([0.19, 0.40, 0.33, 0.33])

    # ax1.set_ylim(0, 100)
    # loads = np.arange(0.25, 16.25, 0.5)
    # ax1.set_xlabel("Network load", size=19)
    # ax1.set_ylabel("Average packet delay", size=19)
    # ax1.plot(loads, process(data[0]), label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[1]), label="Global", color="orange", marker="*", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[2]), label="VAL", color="blue", marker="^", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[3]), label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[5]), label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    # ax1.plot(loads, process(data[4]), label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    # # ax1.legend(ncol=2, prop={'size': 12})
    # ax2.set_xlim(8, 10)
    # ax2.set_ylim(0, 100)
    # ax1.set_xticks(np.arange(0, 17, 2))
    # # ax2.set_yticks([0, 25, 50, 75, 100], labelsize=5)
    # ax2.tick_params(axis='both', labelsize=11)
    # ax1.tick_params(axis='both', labelsize=15)
    # ax2.plot(loads, process(data[0]), label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    # ax2.plot(loads, process(data[1]), label="Global", color="orange", marker="*", markersize=6, linewidth='2.5')
    # ax2.plot(loads, process(data[2]), label="VAL", color="blue", marker="^", markersize=6, linewidth='2.5')
    # ax2.plot(loads, process(data[3]), label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    # ax2.plot(loads, process(data[5]), label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    # ax2.plot(loads, process(data[4]), label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    # for axis in ['top','bottom','left','right']:
    #     ax1.spines[axis].set_linewidth(2)
    #     ax2.spines[axis].set_linewidth(1.5)

    # # increase tick width
    # ax1.tick_params(width=2)
    # ax2.tick_params(width=1.5)
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_adv4_fin_mini.png", dpi=300, bbox_inches='tight')
    # plt.close()

# # =====================================Big topo=========================================#

    # f = open('/home/rm2022/DQN_routing/DRL/testout_large_dragonfly.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # loads = np.arange(40, 102, 2)
    # plt.xlim(40, 100)

    # fig = plt.figure(figsize=(7, 5))
    # ax1 = plt.axes()
    # ax2 = plt.axes([0.19, 0.40, 0.22, 0.33])

    # ax1.set_xlim(40, 100)
    # ax1.set_ylim(0, 100)
    # ax1.set_xlabel("Network load", size=19)
    # ax1.set_ylabel("Average packet delay", size=19)
    # ax1.plot(loads, data[0], label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    # ax1.plot(loads, data[1], label="VAL", color="blue", marker="^", markersize=6, linewidth='2.5')
    # ax1.plot(loads, data[2], label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    # ax1.plot(loads, data[4], label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    # ax1.plot(loads, data[3], label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    # # ax1.legend(ncol=2, prop={'size': 12})
    # ax2.set_xlim(92.5, 97)
    # ax2.set_ylim(0, 100)
    # # ax2.set_yticks([0, 25, 50, 75, 100], labelsize=5)
    # ax2.tick_params(axis='both', labelsize=11)
    # ax1.tick_params(axis='both', labelsize=15)
    # ax2.plot(loads, data[0], label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    # ax2.plot(loads, data[1], label="VAL", color="blue", marker="^", markersize=6, linewidth='2.5')
    # ax2.plot(loads, data[2], label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    # ax2.plot(loads, data[4], label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    # ax2.plot(loads, data[3], label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    # for axis in ['top','bottom','left','right']:
    #     ax1.spines[axis].set_linewidth(2)
    #     ax2.spines[axis].set_linewidth(1.5)

    # # increase tick width
    # ax1.tick_params(width=2)
    # ax2.tick_params(width=1.5)
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_large_dragonfly.png", dpi=300, bbox_inches='tight')


    # f = open('/home/rm2022/DQN_routing/DRL/testout_large_dragonfly_adv1.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # fig = plt.figure(figsize=(7, 5))
    # ax = plt.axes()
    # loads = np.arange(20, 100, 2.5)
    # ax.set_ylim(0, 100)
    # ax.set_xlabel("Network load", size=19)
    # ax.set_ylabel("Average packet delay", size=19)
    # ax.tick_params(axis='both', labelsize=15)
    # ax.plot(loads, data[0], label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    # ax.plot(loads, data[1], label="VAL", color="blue", marker="^", markersize=6, linewidth='2.5')
    # ax.plot(loads, data[2], label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    # ax.plot(loads, data[4], label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    # ax.plot(loads, data[3], label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    # # plt.legend(loc="upper left", prop={'size': 12})
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(2)

    # # increase tick width
    # ax.tick_params(width=2)
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_large_dragonfly_adv1.png", dpi=300, bbox_inches='tight')


    # f = open('/home/rm2022/DQN_routing/DRL/testout_large_dragonfly_adv4.txt', 'r')
    # lines = f.readlines()
    # data = list()
    # for line in lines:
    #     line = line[1:-2]
    #     line = line.split(", ")
    #     temp = list()
    #     for item in line:
    #         item = float(item)
    #         temp.append(item)
    #     data.append(temp)
    # f.close()

    # fig = plt.figure(figsize=(7, 5))
    # ax = plt.axes()
    # ax.set_ylim(0, 100)
    # ax.set_xlim(20, 70)
    # ax.set_xlabel("Network load", size=19)
    # ax.set_ylabel("Average packet delay", size=19)
    # ax.tick_params(axis='both', labelsize=15)
    # loads = np.arange(20, 100, 2.5)
    # plt.plot(np.arange(20, 40, 2.5), data[0], label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    # plt.plot(loads, data[1], label="VAL", color="blue", marker="^", markersize=6, linewidth='2.5')
    # plt.plot(loads, data[2], label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    # plt.plot(loads, data[4], label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    # plt.plot(loads, data[3], label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    # # plt.legend(loc="upper left", prop={'size': 12})
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(2)

    # # increase tick width
    # ax.tick_params(width=2)
    # plt.savefig("/home/rm2022/DQN_routing/DRL/output_large_dragonfly_adv4.png", dpi=300, bbox_inches='tight')

# #=============================================hops==================================================#

    f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_hops_adv1.txt', 'r')
    lines = f.readlines()
    data = list()
    for line in lines:
        line = line[1:-2]
        line = line.split(", ")
        temp = list()
        for item in line:
            item = float(item)
            temp.append(item)
        data.append(temp)
    f.close()

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes()
    ax.set_ylim(2, 3.8)
    loads = np.arange(0.25, 16.25, 0.5)
    ax.plot(loads, process(data[0]), label="MIN", color="green", marker="o", markersize=6, linewidth='2.5')
    ax.plot(loads, process(data[1]), label="VAL", color="blue", marker="^", markersize=6, linewidth='2.5')
    ax.plot(loads, process(data[4]), label="Q-routing", color="purple", marker=".", markersize=6, linewidth='2.5')
    ax.plot(loads, process(data[2]), label="Q-Adaptive", color="grey", marker="+", markersize=6, linewidth='2.5')
    ax.plot(loads, process(data[3]), label="Q-Hierarchical", color="red", marker="X", markersize=6, linewidth='2.5')
    ax.legend(loc="upper left")
    ax.set_yticks(np.arange(2, 3.8, 0.3), labelsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel("Network Load", size=19)
    ax.set_ylabel("Average Packet Hop", size=19)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    # increase tick width
    ax.tick_params(width=2)
    plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_hops_adv1.png", dpi=300, bbox_inches='tight')


#     f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_hops_adv4.txt', 'r')
#     lines = f.readlines()
#     data = list()
#     for line in lines:
#         line = line[1:-2]
#         line = line.split(", ")
#         temp = list()
#         for item in line:
#             item = float(item)
#             temp.append(item)
#         data.append(temp)
#     f.close()

#     plt.figure(figsize=(7, 5))
#     plt.ylim(2, 4.25)
#     loads = np.arange(0.25, 16.25, 0.25)
#     plt.plot(loads, data[0], label="MIN", color="green", marker="o", markersize=4)
#     plt.plot(loads, data[1], label="VAL", color="blue", marker="^", markersize=4)
#     # plt.plot(loads, data[2], label="Q-routing", color="purple", marker=".", markersize=4)
#     plt.plot(loads, data[2], label="Q-Adaptive", color="grey", marker="+", markersize=4)
#     plt.plot(loads, data[3], label="Q-Hierarchical", color="red", marker="X", markersize=4)
#     plt.legend(loc="upper left")
#     plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_hops_adv4.png", dpi=300)

#     f = open('/home/rm2022/DQN_routing/DRL/testout_dragonfly_hops.txt', 'r')
#     lines = f.readlines()
#     data = list()
#     for line in lines:
#         line = line[1:-2]
#         line = line.split(", ")
#         temp = list()
#         for item in line:
#             item = float(item)
#             temp.append(item)
#         data.append(temp)
#     f.close()

#     plt.figure(figsize=(7, 5))
#     plt.ylim(2, 10)
#     loads = np.arange(0.25, 16.25, 0.25)
#     plt.plot(loads, data[0], label="MIN", color="green", marker="o", markersize=4)
#     plt.plot(loads, data[1], label="VAL", color="blue", marker="^", markersize=4)
#     # plt.plot(loads, data[2], label="Q-routing", color="purple", marker=".", markersize=4)
#     plt.plot(loads, data[2], label="Q-Adaptive", color="grey", marker="+", markersize=4)
#     plt.plot(loads, data[3], label="Q-Hierarchical", color="red", marker="X", markersize=4)
#     plt.legend(loc="upper left")
#     plt.savefig("/home/rm2022/DQN_routing/DRL/output_dragonfly_hops.png", dpi=300)
