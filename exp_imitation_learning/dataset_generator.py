import sys
sys.path.append("/DB/rhome/gjliu/ML-EE228/2048-api")
from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
import game2048.agents
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent, getBoardFormExpect, MyRnnAgent
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import time
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset


display1 = Display()
display2 = IPythonDisplay()
#生成数据集
for i in range(1):
    game = Game(4, score_to_win=64, random=False)
    display1.display(game)
    agent2 = getBoardFormExpect(game, display=display2)
    agent2.writeBoard()
    print(i)
