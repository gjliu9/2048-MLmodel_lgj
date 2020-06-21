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
from model import RNN, RNN_onehot
from datadeal import DealDataset_0, DealDataset_enhanced, DealDataset_onehot

NUM_EPOCHS = 1
BATCH_SIZE = 64
INPUT_SIZE = 4
LR = 0.001
rnn_size=256
trainfilertoread = '/DATA5_DB8/data/gjliu/game2048/dataset/Train2.csv'



class TrainModel():

    def __init__(self):

        self.model_2048 = RNN(rnn_size)

    def trainModel(self):

        trainDataset = DealDataset_enhanced(root=trainfilertoread, transform=transforms.Compose(transforms=[transforms.ToTensor()]))
        train_loader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model_2048.parameters(),lr=LR)
        for epoch in range(NUM_EPOCHS):


            for index, (board, direc) in enumerate(train_loader):
                board, direc = Variable(board), Variable(direc)

                if torch.cuda.is_available():
                    board, direc = board.cuda(), direc.cuda()
                    self.model_2048.cuda()

                board = board.view(-1,4,4)
                out = self.model_2048(board)
                loss = criterion(out, direc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if index % 50 == 0:
                    out = self.model_2048(board)
                    pred = torch.max(out, 1)[1]
                    train_correct = (pred == direc).sum().item()
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss,
                          '| test accuracy: %.4f' % (train_correct/(BATCH_SIZE * 1.0)))
            torch.save(self.model_2048, 'rnn_model_' + str(epoch) + '.pkl')
        torch.save(self.model_2048, 'rnn_model_final.pkl')

def main():
    trian1 = TrainModel()
    trian1.trainModel()

if __name__ == '__main__':
    main()