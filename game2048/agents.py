import numpy as np
import csv, os
import sys
from time import sleep
import torchvision.transforms as transforms
import torch
import sys
sys.path.append("/DB/rhome/gjliu/ML-EE228/2048-api/exp_imitation_learning")
from train import RNN

dataSetFilename = 'Train_special.csv'
PATH = '/DATA5_DB8/data/gjliu/game2048/dataset/'
dataSetFilepath = os.path.join(PATH,dataSetFilename)
class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction



class getBoardFormExpect(Agent):
    ''' to generate dataset '''
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def writeBoard(self, root = dataSetFilepath, max_iter=np.inf, verbose=False):
        with open(root, "a") as dataSetFile:
            wrt = csv.writer(dataSetFile)
            if   not (os.path.exists(root)):
                wrt.writerow(["11","12","13","14","21","22","23","24","31","32","33","34","41","42","43","44","label"])
            n_iter = 0
            while (n_iter < max_iter) and (not self.game.end):
                direction = self.step()
                board = np.where(self.game.board == 0, 1, self.game.board)
                board = np.log2(board)
                board = board.flatten()
                board = board.tolist()
                oneRow = np.int32(np.append(board, direction))
                wrt.writerow(oneRow)
                self.game.move(direction)
    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyRnnAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

        self.model = torch.load('/DATA5_DB8/data/gjliu/game2048/model/model_rnn_256_dataup/rnn_256_model_24.pkl', map_location='cpu')
        self.model.eval()

    def step(self):

        board = np.where(self.game.board == 0, 1, self.game.board)
        board = np.log2(board)
        board = board[:, :, np.newaxis]
        board = board/ 11.0
        trans = transforms.Compose([transforms.ToTensor()])
        board = trans(board)
        board = board.type(torch.float)
        out = self.model(board)
        direction = torch.max(out, 1)[1]
        return int(direction)

class MyOneHotAgent(Agent):
    ''' agent for onehot_input model'''

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.model = torch.load('/DATA5_DB8/data/gjliu/game2048/model/model_onehot/rnn_onehot_model_final_2.pkl', map_location='cpu')
        self.model.eval()

    def onehot(self,board):
        ret = np.zeros(shape=(4,4,12),dtype=bool)
        for r in range(4):
            for c in range(4):
                ret[c,r,board[r,c]] = 1
        return ret
    
    def step(self):

        board = np.where(game.board == 0, 1, game.board)
        board = np.log2(board)
        board = board.astype(np.int64)
        board = self.onehot(board)
        board = board.astype(np.int)
        trans = transforms.Compose([transforms.ToTensor()])
        board = trans(board)

        board = board.type(torch.float)
        board = board[np.newaxis,:, :,: ]

        out = self.model(board)
        direction = torch.max(out, 1)[1]
        return int(direction)