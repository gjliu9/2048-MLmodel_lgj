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

# trainfilertoread = '/DB/rhome/gjliu/ML-EE228/2048-api/Train.csv'
# testfiletoread = 'Test.csv'

class DealDataset_0(Dataset):
    '''basic DealDataset'''
    def __init__(self, root,  transform=None):
        super().__init__()
        Data0 = pd.read_csv(root).values
        self.board = Data0[:, 0:-1]
        self.direc = Data0[:, -1]
        self.len = len(Data0)
        self.transform = transform
        self.idx = 0

    def __getitem__(self, index):

        board = self.board[index].reshape((4, 4))
        board = board[:, :, np.newaxis]

        board = board/11.0
        direc = self.direc[index]
        if self.transform is not None:
            board = self.transform(board)
            board = board.type(torch.float)
        return board, direc

    def __len__(self):

        return self.len


class DealDataset_enhanced(Dataset):
    '''datasetdeal with enhancement'''
    def __init__(self, root,  transform=None):
        super().__init__()
        Data0 = pd.read_csv(root).values
        self.board = Data0[:, 0:-1]
        self.direc = Data0[:, -1]
        self.len = len(Data0)
        self.transform = transform
        self.idx = 0

    def __getitem__(self, index):

        #board = self.board[index].reshape((4, 4))
        num = int(index/8)
        typ = int(index%8)
        board = self.board[num].reshape((4, 4))
        
        #---------------数据增强----------
        board1 = self.rote_90_nishiz(board)
        board2 = self.rote_90_nishiz(board1)
        board3 = self.rote_90_nishiz(board2)
        
        board4 = self.vvv(board)
        board5 = self.rote_90_nishiz(board4)
        board6 = self.rote_90_nishiz(board5)
        board7 = self.rote_90_nishiz(board6)
        
        #-----------------end----------

        if typ == 0:
            direc = self.direc[num]

        elif typ == 1:
            direc = (self.direc[num]+1)%4
            board = board1
        elif typ == 2:
            direc = (self.direc[num]+2)%4
            board = board2
        
        elif typ == 3:
            direc = (self.direc[num]+3)%4
            board = board3
        elif typ == 4:
            direc = (self.direc[num]+2*((self.direc[num])%2))%4
            board = board4
        elif typ == 5:
            direc = (self.direc[num]+2*((self.direc[num])%2)+1)%4
            board = board5
        
        elif typ == 6:
            direc = (self.direc[num]+2*((self.direc[num])%2)+2)%4
            board = board6
        elif typ == 7:
            direc = (self.direc[num]+2*((self.direc[num])%2)+3)%4
            board = board7
            

        board = board[:, :, np.newaxis]
        board = board/11.0
        #direc = self.direc[index]
        if self.transform is not None:
            board = self.transform(board)
            board = board.type(torch.float)#更改过board = board.type(torch.float)
        return board, direc


    def __len__(self):
        
        lenn = 8*self.len

        return lenn
    
    def rote_90_nishiz(self,board):
        '''逆时针旋转90'''
        boardnew=np.zeros((4,4))
        boardnew = boardnew.astype(np.int64)

        for i in range(4):
            for j in range(4):
                boardnew[i,j]=board[j,3-i]
        return boardnew
        
    def vvv(self,board):
        '''上下对称'''
        boardnew=np.zeros((4,4))
        boardnew = boardnew.astype(np.int64)
        
        for i in range(4):
            for j in range(4):
                boardnew[i,j]=board[3-i,j]
        return boardnew

class DealDataset_onehot(Dataset):
    '''one hot datasetdeal'''
    def __init__(self, root,  transform=None):
        super().__init__()
        Data0 = pd.read_csv(root).values
        self.board = Data0[:, 0:-1]
        self.direc = Data0[:, -1]
        self.len = len(Data0)
        self.transform = transform
        self.idx = 0

    def __getitem__(self, index):

        board = self.board[index].reshape((4, 4))

        board = self.onehot(board)
        board = board.astype(np.int)

        direc = self.direc[index]
        if self.transform is not None:
            board = self.transform(board)
            board = board.type(torch.float)#更改过board = board.type(torch.float)
            
        board = board[  np.newaxis,:,:,:]
        return board, direc

    def onehot(self,board):
        ret = np.zeros(shape=(4,4,12),dtype=bool)
        for r in range(4):
            for c in range(4):
                ret[c,r,board[r,c]] = 1
        return ret
    def __len__(self):

        return self.len